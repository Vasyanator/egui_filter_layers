/*
File: state.rs

Purpose:
Persistent per-layer state (F7), the backend-neutral half: the public configuration and error
types, the per-context reset requests, and the registry every backend keeps its state targets
in (lifetime, reset, eviction, one step per frame). Which draw reads or writes which state
texture is NOT decided here (`plan.rs`); the GPU objects are the backends' (`T` below).

Key structures:
- `StateOptions`, `StateFormat` (public): resolution and storage format of a layer's state.
- `StateError` (public): why a stateful layer / offline run is refused; `StateFailure` (public):
  a state format the installed backend cannot render (`failed_states`).
- `StateKey` (crate): `(ViewportId, Id)` — one state per layer identity and viewport.
- `StateRegistry<T>` / `StateEntry<T>` / `StateRun` (crate): per-key targets `T`, which of the
  two holds the latest state and its used size, last-painted clock, stepped-this-frame latch.

Key functions:
- `reset_state()` / `state_count()` / `failed_states()` (public, re-exported at the crate root).
- `take_reset()` (crate): consumes a pending reset request for an id (`FilterLayer::paint`).
- `StateRegistry::begin()` / `finish()` / `evict()`.

Notes:
State is keyed by viewport AND id: egui counts frames per viewport (`cumulative_frame_nr`), and
eframe renders several viewports through one renderer, so one id painted in two viewports gets
two independent states (documented). Within one viewport and frame, only the FIRST paint of an
id steps the state; later paints of that id in the same frame run their display passes over the
state as it is (their state passes are skipped, debug-logged once per key). Eviction: an entry
not painted for more than `EVICT_FRAMES` frames AND more than `EVICT_SECONDS` seconds is freed
(both, so an idle app — no frames — and a fast one — many frames per second — keep their state).
*/

use std::collections::HashMap;

use egui::{Context, Id, ViewportId};

use crate::backend::installed;

/// Storage format of a layer's persistent state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StateFormat {
    /// Half-float RGBA (the default): signed values, about 3 decimal digits, unbounded range —
    /// what simulations need. It must be renderable AND filterable on the device (every desktop
    /// GL 3.3 / Vulkan / Metal / DX12 device; WebGL2 / GLES only with `EXT_color_buffer_float`);
    /// where it is not, stateful layers are refused (`failed_states`), never silently downgraded.
    #[default]
    Rgba16Float,
    /// 8-bit normalized RGBA (`0..=1`, 1/255 steps): explicit opt-in for effects that pack their
    /// state into 8 bits (trails, masks). Renderable everywhere.
    Rgba8Unorm,
}

impl std::fmt::Display for StateFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rgba16Float => "RGBA16F",
            Self::Rgba8Unorm => "RGBA8",
        })
    }
}

/// How a layer's persistent state is stored ([`crate::FilterLayer::state`],
/// `OfflineOptions::with_state`). `Default`: full resolution, [`StateFormat::Rgba16Float`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateOptions {
    /// Resolution relative to the capture, in `[MIN_PASS_SCALE, 1]`.
    scale: f32,
    /// Storage format.
    format: StateFormat,
}

impl Default for StateOptions {
    fn default() -> Self {
        Self { scale: 1.0, format: StateFormat::Rgba16Float }
    }
}

impl StateOptions {
    /// The state holds `ceil(capture * scale)` texels per axis (`state_size()` in WGSL), spanning
    /// the whole capture. A smaller state is cheaper for smooth fields (water heights). `scale`
    /// is clamped to `[MIN_PASS_SCALE, 1]`; a non-finite value means `1`.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = if scale.is_finite() { scale.clamp(crate::pass::MIN_PASS_SCALE, 1.0) } else { 1.0 };
        self
    }

    /// The storage format (default [`StateFormat::Rgba16Float`]).
    #[must_use]
    pub fn with_format(mut self, format: StateFormat) -> Self {
        self.format = format;
        self
    }

    /// The state's resolution relative to the capture.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The storage format.
    #[must_use]
    pub fn format(&self) -> StateFormat {
        self.format
    }
}

/// Why a layer (or an offline run) with a state pass cannot run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateError {
    /// The chain has a state pass but the layer has no identity: `FilterLayer::id` is required,
    /// because the state must be found again next frame.
    #[error("effect `{effect}` (pass {pass}) is a state pass, but the layer has no id; call FilterLayer::id(..) so its state can be kept between frames")]
    MissingId {
        /// The first state pass's effect label.
        effect: String,
        /// Its index in the chain.
        pass: usize,
    },
    /// The device cannot render (or bilinearly filter) the state format.
    #[error("the {format} state format is not renderable and filterable on this device ({reason}); opt in to StateFormat::Rgba8Unorm if the effect packs its state into 8 bits")]
    UnsupportedFormat {
        /// The requested format.
        format: StateFormat,
        /// What the backend found.
        reason: String,
    },
}

/// A state format the installed backend refused (see [`failed_states`]). Layers asking for it
/// draw nothing extra; the entry stays for the backend's lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StateFailure {
    /// The refused format.
    pub format: StateFormat,
    /// Why ([`StateError::UnsupportedFormat`]).
    pub error: StateError,
}

/// Resets the persistent state of the layer `id` on `ctx`: the next paint of that id (in any
/// viewport, the first one to paint it) starts from zero. Cheap; call it from UI code.
pub fn reset_state(ctx: &Context, id: Id) {
    ctx.data_mut(|data| {
        let pending = data.get_temp_mut_or_default::<PendingResets>(Id::new(PENDING_RESETS));
        if !pending.0.contains(&id) {
            pending.0.push(id);
        }
    });
}

/// How many layer states the installed backend currently holds (a debug / leak-check accessor;
/// evicted and reset states are not counted twice). 0 without a backend.
#[must_use]
pub fn state_count(ctx: &Context) -> usize {
    installed(ctx).map_or(0, |backend| backend.state_count())
}

/// State formats the installed backend refused (the device cannot render or filter them); layers
/// asking for them draw nothing extra. Each is also logged once at `error` level.
#[must_use]
pub fn failed_states(ctx: &Context) -> Vec<StateFailure> {
    installed(ctx).map_or_else(Vec::new, |backend| backend.failed_states())
}

/// Key of the pending reset requests in a context's temporary memory.
const PENDING_RESETS: &str = "egui_filter_layers::pending_state_resets";

/// Layer ids whose state must be reset at their next paint (`reset_state`).
#[derive(Clone, Debug, Default)]
struct PendingResets(Vec<Id>);

/// Consumes a pending [`reset_state`] request for `id`; returns whether there was one.
pub(crate) fn take_reset(ctx: &Context, id: Id) -> bool {
    ctx.data_mut(|data| {
        let pending = data.get_temp_mut_or_default::<PendingResets>(Id::new(PENDING_RESETS));
        let before = pending.0.len();
        pending.0.retain(|pending| *pending != id);
        pending.0.len() != before
    })
}

/// Frames an entry may go unpainted before eviction (together with [`EVICT_SECONDS`]).
pub(crate) const EVICT_FRAMES: u32 = 120;
/// Seconds an entry may go unpainted before eviction (together with [`EVICT_FRAMES`]).
pub(crate) const EVICT_SECONDS: f64 = 2.0;

/// One layer state: its identity and viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StateKey {
    /// The viewport that painted it.
    pub viewport: ViewportId,
    /// `FilterLayer::id`.
    pub id: Id,
}

/// One entry: the backend's two state targets and the bookkeeping around them.
#[derive(Debug)]
pub(crate) struct StateEntry<T> {
    /// The backend's objects (two state textures).
    pub targets: T,
    /// Which target (0 / 1) holds the latest state.
    current: u8,
    /// Size in texels of the latest state content; `None` = no content yet (created or reset:
    /// the plan clears `current` before anything reads it).
    used: Option<[u32; 2]>,
    /// The format the targets were made in (a change recreates the entry).
    format: StateFormat,
    /// Frame (per viewport) of the last paint.
    last_frame: u32,
    /// Time (seconds) of the last paint.
    last_time: f64,
    /// Frame whose paint stepped the state (`None` before the first).
    stepped: Option<u32>,
    /// A repeated same-frame paint was logged.
    logged_repeat: bool,
}

/// What one paint of an entry does (the input of `plan::StatePlanInput`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateRun {
    /// Which target holds the latest state.
    pub current: u8,
    /// Its content's size; `None` = clear it first (new or reset).
    pub previous: Option<[u32; 2]>,
    /// Whether this paint steps the state (false: the id was already stepped this frame).
    pub step: bool,
}

/// Every layer state of one backend (one per install / renderer), keyed by [`StateKey`].
#[derive(Debug)]
pub(crate) struct StateRegistry<T> {
    /// The entries.
    entries: HashMap<StateKey, StateEntry<T>>,
}

impl<T> Default for StateRegistry<T> {
    fn default() -> Self {
        Self { entries: HashMap::new() }
    }
}

impl<T> StateRegistry<T> {
    /// Number of entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Starts a paint of `key` at `frame` / `time`: creates the entry (`make`) when missing or
    /// when its `format` changed (the replaced targets are returned for the caller to free),
    /// applies a `reset`, and decides whether this paint steps the state (the first of the frame).
    /// Returns the entry and the run description.
    pub(crate) fn begin(&mut self, key: StateKey, format: StateFormat, frame: u32, time: f64, reset: bool, make: impl FnOnce() -> T) -> (&mut StateEntry<T>, StateRun, Option<T>) {
        let replaced = match self.entries.get(&key) {
            Some(entry) if entry.format != format => self.entries.remove(&key).map(|entry| entry.targets),
            _ => None,
        };
        let entry = self.entries.entry(key).or_insert_with(|| StateEntry { targets: make(), current: 0, used: None, format, last_frame: frame, last_time: time, stepped: None, logged_repeat: false });
        if reset {
            entry.used = None;
        }
        entry.last_frame = frame;
        entry.last_time = time;
        let step = entry.stepped != Some(frame);
        if !step && !entry.logged_repeat {
            entry.logged_repeat = true;
            log::debug!(
                "egui_filter_layers: the stateful layer {:?} was painted twice in one frame of viewport {:?}; the repeat shows the state without stepping it. Logged once.",
                key.id,
                key.viewport
            );
        }
        let run = StateRun { current: entry.current, previous: entry.used, step };
        (entry, run, replaced)
    }

    /// Evicts every entry not painted for more than [`EVICT_FRAMES`] frames and more than
    /// [`EVICT_SECONDS`] seconds before `frame` / `time` of `viewport` (other viewports count
    /// their own frames, so only their time is compared), returning their targets to free.
    pub(crate) fn evict(&mut self, viewport: ViewportId, frame: u32, time: f64) -> Vec<T> {
        let stale: Vec<StateKey> = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                let old_frames = key.viewport != viewport || frame.wrapping_sub(entry.last_frame) > EVICT_FRAMES;
                old_frames && time - entry.last_time > EVICT_SECONDS
            })
            .map(|(key, _)| *key)
            .collect();
        stale.into_iter().filter_map(|key| self.entries.remove(&key)).map(|entry| entry.targets).collect()
    }

    /// Removes every entry, returning the targets to free.
    #[cfg_attr(not(feature = "glow"), allow(dead_code))] // Only glow frees GL objects explicitly.
    pub(crate) fn drain(&mut self) -> Vec<T> {
        self.entries.drain().map(|(_, entry)| entry.targets).collect()
    }
}

impl<T> StateEntry<T> {
    /// Records the outcome of a run that planned `step`: the latest state is now in target
    /// `current`, `used` texels large, and (when it stepped) this frame's step is done.
    pub(crate) fn finish(&mut self, run: StateRun, frame: u32, current: u8, used: [u32; 2]) {
        self.current = current;
        self.used = Some(used);
        if run.step {
            self.stepped = Some(frame);
        }
    }

    /// Forgets the content (the next run clears it), e.g. after a failed run.
    pub(crate) fn invalidate(&mut self) {
        self.used = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: &str) -> StateKey {
        StateKey { viewport: ViewportId::ROOT, id: Id::new(id) }
    }

    #[test]
    fn a_new_entry_is_cleared_then_stepped_once_per_frame() {
        let mut registry: StateRegistry<u32> = StateRegistry::default();
        let (entry, run, replaced) = registry.begin(key("a"), StateFormat::Rgba16Float, 10, 1.0, false, || 7);
        assert_eq!((entry.targets, run, replaced), (7, StateRun { current: 0, previous: None, step: true }, None));
        entry.finish(run, 10, 1, [8, 4]);
        // The same frame again: no step, the latest content is read.
        let (entry, again, _) = registry.begin(key("a"), StateFormat::Rgba16Float, 10, 1.0, false, || 99);
        assert_eq!(again, StateRun { current: 1, previous: Some([8, 4]), step: false });
        entry.finish(again, 10, 1, [8, 4]);
        let (_, next, _) = registry.begin(key("a"), StateFormat::Rgba16Float, 11, 1.1, false, || 99);
        assert_eq!(next, StateRun { current: 1, previous: Some([8, 4]), step: true });
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn reset_and_format_changes_start_from_zero() {
        let mut registry: StateRegistry<u32> = StateRegistry::default();
        let (entry, run, _) = registry.begin(key("a"), StateFormat::Rgba16Float, 1, 0.0, false, || 1);
        entry.finish(run, 1, 1, [4, 4]);
        let (entry, reset, _) = registry.begin(key("a"), StateFormat::Rgba16Float, 2, 0.0, true, || 2);
        assert_eq!((entry.targets, reset.previous, reset.step), (1, None, true), "reset keeps the targets, drops the content");
        entry.finish(reset, 2, 0, [4, 4]);
        let (entry, changed, replaced) = registry.begin(key("a"), StateFormat::Rgba8Unorm, 3, 0.0, false, || 3);
        assert_eq!((entry.targets, changed.previous, replaced), (3, None, Some(1)), "a new format recreates the targets");
    }

    #[test]
    fn viewports_and_ids_have_their_own_states() {
        let mut registry: StateRegistry<u32> = StateRegistry::default();
        let other = StateKey { viewport: ViewportId::from_hash_of("other"), id: Id::new("a") };
        let (entry, run, _) = registry.begin(key("a"), StateFormat::Rgba16Float, 5, 0.0, false, || 1);
        entry.finish(run, 5, 1, [2, 2]);
        let (_, run, _) = registry.begin(other, StateFormat::Rgba16Float, 5, 0.0, false, || 2);
        assert_eq!(run, StateRun { current: 0, previous: None, step: true }, "the same frame number of another viewport steps its own state");
        registry.begin(key("b"), StateFormat::Rgba16Float, 5, 0.0, false, || 3);
        assert_eq!(registry.len(), 3);
    }

    /// Eviction needs BOTH more than 120 frames and more than 2 s without a paint; an idle app
    /// (time passes, frames do not) and a fast one (frames pass, little time) keep their states.
    #[test]
    fn entries_are_evicted_after_frames_and_seconds() {
        let mut registry: StateRegistry<u32> = StateRegistry::default();
        registry.begin(key("old"), StateFormat::Rgba16Float, 0, 0.0, false, || 1);
        registry.begin(key("fresh"), StateFormat::Rgba16Float, 0, 0.0, false, || 2);
        assert!(registry.evict(ViewportId::ROOT, 500, 1.5).is_empty(), "many frames, little time");
        assert!(registry.evict(ViewportId::ROOT, 100, 60.0).is_empty(), "idle: much time, few frames");
        registry.begin(key("fresh"), StateFormat::Rgba16Float, 121, 2.5, false, || 2);
        assert_eq!(registry.evict(ViewportId::ROOT, 121, 2.5), vec![1]);
        assert_eq!(registry.len(), 1);
        // Frame counters wrap.
        let mut wrapping: StateRegistry<u32> = StateRegistry::default();
        wrapping.begin(key("a"), StateFormat::Rgba16Float, u32::MAX - 5, 0.0, false, || 1);
        assert!(wrapping.evict(ViewportId::ROOT, 5, 10.0).is_empty(), "11 frames across the wrap");
        // Another viewport's frame numbers are not comparable: its entries age by time only.
        let mut viewports: StateRegistry<u32> = StateRegistry::default();
        viewports.begin(StateKey { viewport: ViewportId::from_hash_of("other"), id: Id::new("a") }, StateFormat::Rgba16Float, 1000, 0.0, false, || 1);
        assert!(viewports.evict(ViewportId::ROOT, 1000, 1.0).is_empty());
        assert_eq!(viewports.evict(ViewportId::ROOT, 1000, 3.0), vec![1]);
        assert_eq!(viewports.drain(), Vec::<u32>::new());
    }

    #[test]
    fn reset_requests_are_consumed_once_per_id() {
        let ctx = Context::default();
        let (a, b) = (Id::new("a"), Id::new("b"));
        assert!(!take_reset(&ctx, a));
        reset_state(&ctx, a);
        reset_state(&ctx, a);
        assert!(!take_reset(&ctx, b));
        assert!(take_reset(&ctx, a));
        assert!(!take_reset(&ctx, a), "consumed");
        assert_eq!(state_count(&ctx), 0);
        assert!(failed_states(&ctx).is_empty());
    }

    #[test]
    fn options_clamp_the_scale() {
        assert!((StateOptions::default().scale() - 1.0).abs() < f32::EPSILON);
        assert_eq!(StateOptions::default().format(), StateFormat::Rgba16Float);
        assert!((StateOptions::default().with_scale(0.25).scale() - 0.25).abs() < f32::EPSILON);
        assert!((StateOptions::default().with_scale(3.0).scale() - 1.0).abs() < f32::EPSILON);
        assert!((StateOptions::default().with_scale(f32::NAN).scale() - 1.0).abs() < f32::EPSILON);
        assert!((StateOptions::default().with_scale(0.0).scale() - crate::pass::MIN_PASS_SCALE).abs() < f32::EPSILON);
        assert_eq!(StateOptions::default().with_format(StateFormat::Rgba8Unorm).format(), StateFormat::Rgba8Unorm);
    }
}
