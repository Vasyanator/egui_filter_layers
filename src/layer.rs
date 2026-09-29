/*
File: layer.rs

Purpose:
`ShaderLayer`, the user-facing shape: one or more entries (`entry.rs`: a rect + a mask + a
`FilterChain` + textures + state options; `ShaderLayer::new` builds one), painted into an
`egui::Painter`. Resolves the entries into a backend-neutral `LayerJob` and hands it to whichever
backend is installed on the painter's context.

Key structures:
- `ShaderLayer`: entries, overlap mode (`Overlap`), id.

Key functions:
- `ShaderLayer::paint()`.
- `ShaderLayer::from_entries()` / `with_entry()` / `overlap()` / `entries()`: multi-entry layers
  (R3); the layer-level builders below apply to every entry, the accessors read the first.
- `ShaderLayer::texture()` / `check_textures()`: user textures (F8) and the check `paint` runs.
- `ShaderLayer::id()` / `state()` / `check_state()`: persistent state (F7) and its check.
- `ShaderLayer::mask()` / `corner_radius()` / `regions()` / `check_mask()`: the mask (F9) and
  regions (R1); a shape mask and every region are tessellated in `paint` (`resolve_mask`, GUI
  thread).
- `frame_input()`: the per-frame input (`fl.delta_time`, `fl.frame`, `fl.pointer`) read from the
  context on the GUI thread.

Notes:
Never panics and never draws a fake: with no backend installed it logs one warning per
`egui::Context` (latched in the context's temporary memory, see `warn_not_installed_once`) and
paints nothing, leaving the backdrop unfiltered. A failed backend is skipped the same way, and
neither case requests repaints for animated chains. A layer whose user textures cannot serve its
effects (`TextureError`: an invalid slot, or a slot an effect reads with nothing attached) is
skipped too, its error logged once per context and message (`log_texture_error_once`), and so
is a layer with a state pass but no id (`StateError::MissingId`, same latch), and one whose mask
shape contains a paint callback or a malformed mesh, or which has too many regions (`MaskError`,
same latch). Any failing entry skips the whole layer (the log names the entry). A pending
`reset_state` request for the layer's id travels in its job as a ticket (`state::pending_reset`);
the backend claims it where the state registry runs, so a paint that draws nothing keeps it.
*/

use std::sync::Arc;

use egui::{Context, CornerRadius, Id, Painter, Rect, TextureId};

use crate::backend::{FrameInput, LayerJob, PointerInput, TextureError, installed};
use crate::effect::TEXTURE_SLOTS;
use crate::entry::{Entry, Overlap};
use crate::mask::{JobMask, Mask, MaskError, Region, tessellate, tessellate_regions};
use crate::pass::FilterChain;
use crate::state::{StateError, StateOptions, pending_reset};

/// Sets `ctx`'s "no backend installed" latch; returns whether it was unset (i.e. whether to warn).
fn first_not_installed_warning(ctx: &Context) -> bool {
    ctx.data_mut(|data| {
        let warned = data.get_temp_mut_or_default::<bool>(Id::new("egui_shader_layers::warned_not_installed"));
        !std::mem::replace(warned, true)
    })
}

/// Logs the "no backend installed" warning the first time it is called for `ctx`.
///
/// The latch lives in `ctx`'s temporary memory (not a process-wide static): there is no install
/// to own it, and each `Context` is its own UI. It is not persisted, so a restored or reset
/// memory (`Memory` replaced by the integration) may warn again.
fn warn_not_installed_once(ctx: &Context) {
    if first_not_installed_warning(ctx) {
        log::warn!(
            "egui_shader_layers: a ShaderLayer was painted but no backend is installed on this egui::Context; \
             call egui_shader_layers::install_glow (or the wgpu installer) at startup. Layers draw nothing until then."
        );
    }
}

/// Sets `ctx`'s latch for `message`; returns whether it was unset (i.e. whether to log).
fn first_layer_error(ctx: &Context, message: &str) -> bool {
    ctx.data_mut(|data| !std::mem::replace(data.get_temp_mut_or_default::<bool>(Id::new(("egui_shader_layers::layer_error", message))), true))
}

/// Logs `error` (a layer skipped for its textures) the first time this message is seen on `ctx`.
/// The latch is per message in `ctx`'s temporary memory, like `warn_not_installed_once`: the same
/// misuse repeats every frame, a different one (another effect or slot) is logged once more.
fn log_texture_error_once(ctx: &Context, error: &TextureError) {
    let message = error.to_string();
    if first_layer_error(ctx, &message) {
        log::error!(
            "egui_shader_layers: a ShaderLayer was skipped (it draws nothing extra) because its textures cannot serve its effects.\n\
             Error: {message}\n\
             Fix: attach a texture to every slot the chain's effects read (`ShaderLayer::texture(slot, id)`, slots 0..{TEXTURE_SLOTS}); \
             `ShaderLayer::check_textures` reports this without painting. Logged once per message."
        );
    }
}

/// Logs `error` (a stateful layer skipped) once per context and message, like
/// `log_texture_error_once`.
fn log_state_error_once(ctx: &Context, error: &StateError) {
    let message = error.to_string();
    if first_layer_error(ctx, &message) {
        log::error!(
            "egui_shader_layers: a ShaderLayer was skipped (it draws nothing extra).\n\
             Error: {message}\n\
             Fix: give every layer whose chain has a state pass a stable identity with `ShaderLayer::id(egui::Id)`; \
             `ShaderLayer::check_state` reports this without painting. Logged once per message."
        );
    }
}

/// Logs `error` (a layer skipped for the mask of its entry `entry`) once per context and message.
fn log_mask_error_once(ctx: &Context, entry: usize, error: &MaskError) {
    let message = format!("layer entry {entry}: {error}");
    if first_layer_error(ctx, &message) {
        log::error!(
            "egui_shader_layers: a ShaderLayer was skipped (it draws nothing extra) because its mask cannot be rasterized.\n\
             Error: {message}\n\
             Fix: build the mask from paths, circles, rects, meshes or text; `ShaderLayer::check_mask` reports this without painting. Logged once per message."
        );
    }
}

/// The job mask of a layer painted on `ctx`: a shape is tessellated here, on the GUI thread, at the
/// context's pixels per point with its tessellation options (feathering) and font atlas size (text
/// uvs), without pre-rasterized discs (`mask::tessellate`).
fn resolve_mask(ctx: &Context, mask: Mask, rect: Rect) -> JobMask {
    let tessellation = || {
        let options = ctx.options(|options| options.tessellation_options);
        // The method path does not satisfy `ctx.fonts`'s higher-ranked `FontsView<'_>` bound.
        #[allow(clippy::redundant_closure_for_method_calls)]
        let font_size = ctx.fonts(|fonts| fonts.font_image_size());
        (options, font_size)
    };
    match mask {
        Mask::RoundedRect(radius) => JobMask::RoundedRect(radius),
        Mask::Shape(shape) => {
            let (options, font_size) = tessellation();
            JobMask::Coverage(Arc::new(tessellate(shape, rect, ctx.pixels_per_point(), options, font_size)))
        }
        Mask::Regions(regions) => {
            let (options, font_size) = tessellation();
            JobMask::Regions(Arc::new(tessellate_regions(regions, rect, ctx.pixels_per_point(), options, font_size)))
        }
    }
}

/// The per-frame input of `ctx`'s current pass: `stable_dt` clamped to
/// `0..=FrameInput::MAX_DELTA_TIME` (non-finite reads as 0), `cumulative_frame_nr` truncated to
/// its low 32 bits (wrapping, documented for `fl.frame`), and the latest pointer position (screen
/// points) with the primary button state.
fn frame_input(ctx: &Context) -> FrameInput {
    // Read outside `ctx.input`: both take the context lock, which must not be re-entered.
    let frame_nr = ctx.cumulative_frame_nr();
    let frame = u32::try_from(frame_nr & u64::from(u32::MAX)).unwrap_or(0);
    ctx.input(|input| {
        let dt = input.stable_dt;
        let delta_time = if dt.is_finite() { dt.clamp(0.0, FrameInput::MAX_DELTA_TIME) } else { 0.0 };
        let pointer = input.pointer.latest_pos().map(|pos| PointerInput { pos, pressed: input.pointer.primary_down() });
        FrameInput { delta_time, frame, pointer }
    })
}

/// The chain [`ShaderLayer::chain`] returns for a layer without entries.
static EMPTY_CHAIN: FilterChain = FilterChain::new();
/// The mask [`ShaderLayer::layer_mask`] returns for a layer without entries (the default).
static DEFAULT_MASK: Mask = Mask::RoundedRect(CornerRadius::ZERO);

/// A filter applied to everything painted beneath `rect`.
///
/// "Beneath" means earlier in egui's paint order: lower layers and the shapes painted earlier into
/// the same layer. Anything painted after the layer (later shapes, higher `Order` areas such as
/// popups and tooltips) is drawn on top, unfiltered.
///
/// ```no_run
/// # fn demo(ui: &mut egui::Ui, rect: egui::Rect) {
/// use egui_shader_layers::{ShaderLayer, presets};
/// ShaderLayer::new(rect, presets::gaussian_blur(6.0))
///     .corner_radius(12)
///     .paint(ui.painter());
/// # }
/// ```
///
/// A layer holds one or more ENTRIES ([`Entry`]: rect, chain, mask, textures, state options);
/// [`ShaderLayer::new`] builds one, [`ShaderLayer::from_entries`] / [`ShaderLayer::with_entry`]
/// several, which then share one paint callback (and on wgpu one selection and tessellation of
/// the content beneath) while each captures only its own padded rect. The layer-level builders
/// ([`ShaderLayer::mask`], [`ShaderLayer::corner_radius`], [`ShaderLayer::regions`],
/// [`ShaderLayer::texture`], [`ShaderLayer::state`]) apply to EVERY entry present when they are
/// called; the accessors ([`ShaderLayer::chain`], [`ShaderLayer::layer_mask`],
/// [`ShaderLayer::texture_at`], [`ShaderLayer::state_options`]) return the FIRST entry's.
#[derive(Clone, Debug)]
pub struct ShaderLayer {
    /// The entries, in paint order.
    entries: Vec<Entry>,
    /// How overlapping entries see each other.
    overlap: Overlap,
    /// The layer's identity (required by stateful chains, F7).
    id: Option<Id>,
}

impl ShaderLayer {
    /// A layer filtering `rect` (points) through `chain` (a `Pass`, `Vec<Pass>` or `FilterChain`):
    /// one [`Entry`].
    #[must_use]
    pub fn new(rect: Rect, chain: impl Into<FilterChain>) -> Self {
        Self::from_entries([Entry::new(rect, chain)])
    }

    /// A layer of several entries, painted in this order (see [`Overlap`] for what a later entry
    /// filters where entries overlap). [`ShaderLayer::rect`] is the union of their rects. An
    /// empty list paints nothing.
    #[must_use]
    pub fn from_entries(entries: impl IntoIterator<Item = Entry>) -> Self {
        Self { entries: entries.into_iter().collect(), overlap: Overlap::default(), id: None }
    }

    /// Appends `entry` (painted after the existing ones); the layer's rect grows to include it.
    ///
    /// The entry is taken as it is: layer-level builders called BEFORE this one
    /// ([`ShaderLayer::mask`], [`ShaderLayer::corner_radius`], [`ShaderLayer::regions`],
    /// [`ShaderLayer::texture`], [`ShaderLayer::state`]) do not reach it — e.g.
    /// `.corner_radius(8).with_entry(e)` leaves `e` with its own mask. Call them after, or set
    /// them on the [`Entry`].
    #[must_use]
    pub fn with_entry(mut self, entry: Entry) -> Self {
        self.entries.push(entry);
        self
    }

    /// How overlapping entries see each other (default [`Overlap::Independent`]).
    #[must_use]
    pub fn overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        self
    }

    /// The overlap mode.
    #[must_use]
    pub fn overlap_mode(&self) -> Overlap {
        self.overlap
    }

    /// The entries, in paint order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Gives the layer a stable identity (e.g. `ui.id().with("water")`), REQUIRED when a chain
    /// has a state pass ([`crate::Effect::new_state`]): the layer's persistent state is found
    /// again by this id (and the painting viewport) every frame; every stateful entry keeps its
    /// own state under it, keyed by its index in the entry list (an earlier entry that draws
    /// nothing for a while does not shift it). Paint one id once per frame: a second paint in the same frame shows the
    /// state without stepping it. A state not painted for 120 frames of its viewport and 2
    /// seconds is freed; [`crate::reset_state`] zeroes it (every entry's) at the next paint the
    /// backend runs.
    #[must_use]
    pub fn id(mut self, id: Id) -> Self {
        self.id = Some(id);
        self
    }

    /// The layer's identity, if any.
    #[must_use]
    pub fn layer_id(&self) -> Option<Id> {
        self.id
    }

    /// How the persistent state is stored (resolution, format), for EVERY entry; ignored by
    /// chains without a state pass. Default: full resolution, RGBA16F.
    #[must_use]
    pub fn state(mut self, options: StateOptions) -> Self {
        self.entries.iter_mut().for_each(|entry| entry.set_state(options));
        self
    }

    /// The first entry's state options (the default for a layer without entries).
    #[must_use]
    pub fn state_options(&self) -> StateOptions {
        self.entries.first().map(Entry::state_options).unwrap_or_default()
    }

    /// Whether the layer can keep its states: an entry whose chain has a state pass needs the
    /// layer's id. [`ShaderLayer::paint`] runs this and skips the layer on an error (logged once
    /// per context).
    ///
    /// # Errors
    /// [`StateError::MissingId`] naming the first such entry and its first state pass.
    pub fn check_state(&self) -> Result<(), StateError> {
        self.entries.iter().enumerate().try_for_each(|(index, entry)| entry.check_state_at(index, self.id.is_some()))
    }

    /// Attaches the egui texture `id` at user texture `slot` (`0..TEXTURE_SLOTS`) of EVERY entry,
    /// which effects read with `tex(slot, uv)` (the texture's own uv and sampler, from the
    /// `TextureOptions` it was loaded with), `tex_size(slot)` and `tex_load(slot, px)`. Any egui
    /// texture works: a `TextureHandle` from `Context::load_texture` (`handle.id()`), a native
    /// texture registered with the renderer. Replaces an earlier texture at that slot; use
    /// [`Entry::texture`] for per-entry textures.
    ///
    /// Keep the texture alive while the layer is painted: a freed `id` makes the backend skip the
    /// layer and report it in [`crate::failed_textures`]. A `slot` out of range makes the layer
    /// fail [`ShaderLayer::check_textures`] (and so draw nothing).
    #[must_use]
    pub fn texture(mut self, slot: usize, id: TextureId) -> Self {
        self.entries.iter_mut().for_each(|entry| entry.set_texture(slot, id));
        self
    }

    /// The texture attached at `slot` of the first entry, if any.
    #[must_use]
    pub fn texture_at(&self, slot: usize) -> Option<TextureId> {
        self.entries.first().and_then(|entry| entry.texture_at(slot))
    }

    /// Whether the attached textures can serve the chains: in every entry, every slot given to
    /// a `texture` builder exists, and every slot an effect reads
    /// ([`crate::Effect::reads_texture`]) has a texture. [`ShaderLayer::paint`] runs this and
    /// skips the layer on an error (logged once per context). It cannot see whether a texture id
    /// is still alive; the backend reports that ([`crate::failed_textures`]).
    ///
    /// # Errors
    /// [`TextureError::InvalidSlot`] or [`TextureError::NotAttached`], naming the entry.
    pub fn check_textures(&self) -> Result<(), TextureError> {
        self.entries.iter().enumerate().try_for_each(|(index, entry)| entry.check_textures_at(index))
    }

    /// Restricts EVERY entry to a rounded rect (anti-aliased edge): shorthand for
    /// `mask(Mask::RoundedRect(corner_radius))`, replacing any earlier mask.
    #[must_use]
    pub fn corner_radius(self, corner_radius: impl Into<CornerRadius>) -> Self {
        self.mask(Mask::RoundedRect(corner_radius.into()))
    }

    /// Restricts EVERY entry to `mask` (replacing any earlier mask or corner radius); outside it,
    /// and outside the entry's rect, the backdrop stays untouched. A [`Mask::Shape`] is in layer
    /// points (the space of the layer's rect, so it follows layer transforms), its alpha is the
    /// coverage, and it is tessellated when the layer is painted (with egui's anti-aliasing
    /// feather of one pixel at the context's pixels per point; under a zoomed layer transform the
    /// feather scales with the zoom). Text galleys work in the UI (not offline). Use
    /// [`Entry::mask`] for per-entry masks.
    #[must_use]
    pub fn mask(mut self, mask: Mask) -> Self {
        if let Some((last, others)) = self.entries.split_last_mut() {
            for entry in others {
                entry.set_mask(mask.clone());
            }
            last.set_mask(mask);
        }
        self
    }

    /// Splits the layer into REGIONS: shorthand for `mask(Mask::Regions(..))`, replacing any
    /// earlier mask (of every entry). The union of the regions is the mask; every pass reads the id
    /// of the region that owns a pixel with the prelude's `region()` (see [`Mask::Regions`]). More
    /// than [`crate::MAX_REGIONS`] regions are refused ([`MaskError::TooManyRegions`]).
    #[must_use]
    pub fn regions(self, regions: impl IntoIterator<Item = Region>) -> Self {
        self.mask(Mask::Regions(regions.into_iter().collect()))
    }

    /// The first entry's mask (the default rounded rect for a layer without entries).
    #[must_use]
    pub fn layer_mask(&self) -> &Mask {
        self.entries.first().map_or(&DEFAULT_MASK, Entry::layer_mask)
    }

    /// Whether every entry's mask can be rasterized ([`Mask::check`]: no paint callbacks, no
    /// malformed meshes, at most [`crate::MAX_REGIONS`] regions). [`ShaderLayer::paint`] runs this
    /// and skips the layer on an error (logged once per context, naming the entry).
    ///
    /// # Errors
    /// [`MaskError::Callback`], [`MaskError::InvalidMesh`] or [`MaskError::TooManyRegions`] of the
    /// first entry that fails.
    pub fn check_mask(&self) -> Result<(), MaskError> {
        self.check_mask_entry().map_err(|(_, error)| error)
    }

    /// [`ShaderLayer::check_mask`] with the index of the entry that failed.
    fn check_mask_entry(&self) -> Result<(), (usize, MaskError)> {
        self.entries.iter().enumerate().try_for_each(|(index, entry)| entry.check_mask().map_err(|error| (index, error)))
    }

    /// The filtered rect, in points: the union of the entries' rects (`Rect::NOTHING` without
    /// entries).
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.entries.iter().fold(Rect::NOTHING, |union, entry| union.union(entry.rect()))
    }

    /// The first entry's chain (an empty chain for a layer without entries).
    #[must_use]
    pub fn chain(&self) -> &FilterChain {
        self.entries.first().map_or(&EMPTY_CHAIN, Entry::chain)
    }

    /// Adds the layer to `painter` at the current position in its paint list.
    ///
    /// Entries with an empty chain or a non-positive rect are left out (the others keep their
    /// index: a stateful entry keeps its state while an earlier entry collapses to nothing and
    /// back); a layer with none left does nothing. Requests a repaint when a chain is animated. With no backend installed on
    /// the painter's context, logs a warning once per context and paints nothing; with a failed
    /// backend, paints nothing (the failure was logged when it happened). Neither case requests
    /// repaints — see [`crate::status`]. A layer ANY of whose entries fails
    /// [`ShaderLayer::check_textures`], [`ShaderLayer::check_state`] or
    /// [`ShaderLayer::check_mask`] draws nothing (one layer, one decision; logged once).
    pub fn paint(self, painter: &Painter) {
        if !self.entries.iter().any(Entry::is_drawable) {
            return;
        }
        let ctx = painter.ctx();
        let Some(backend) = installed(ctx) else {
            warn_not_installed_once(ctx);
            return;
        };
        // A failed backend would ignore the job anyway; skipping here also stops an animated
        // chain from forcing a repaint every frame for nothing.
        if !backend.status().is_usable() {
            return;
        }
        // Refused before any backend work: an effect reading an empty slot would otherwise sample
        // a dummy texture, a silent wrong result.
        if let Err(error) = self.check_textures() {
            log_texture_error_once(ctx, &error);
            return;
        }
        if let Err(error) = self.check_state() {
            log_state_error_once(ctx, &error);
            return;
        }
        if let Err((entry, error)) = self.check_mask_entry() {
            log_mask_error_once(ctx, entry, &error);
            return;
        }
        // Non-drawable entries are left out of the job, but each kept entry carries its USER
        // index: its state key and attributions must not shift when an earlier entry collapses.
        let entries: Vec<(usize, Entry)> = self.entries.into_iter().enumerate().filter(|(_, entry)| entry.is_drawable()).collect();
        // An effect reading `fl.time` must keep moving in an otherwise idle app.
        if entries.iter().any(|(_, entry)| entry.chain().is_animated()) {
            ctx.request_repaint();
        }
        // f64 seconds -> f32: effects only need a smooth phase, and f32 keeps millisecond
        // precision for ~2.3 hours of uptime, which animated effects tolerate.
        #[allow(clippy::cast_possible_truncation)] // Intentional precision reduction, see above.
        let time = ctx.input(|input| input.time) as f32;
        let frame_input = frame_input(ctx);
        let stateful = entries.iter().any(|(_, entry)| entry.is_stateful());
        // Handed only to a stateful layer's paint, so a request waits for the layer it targets;
        // the backend claims it where the state registry actually runs.
        let reset_state = if stateful { self.id.and_then(|id| pending_reset(ctx, id)) } else { None };
        let viewport = ctx.viewport_id();
        let rect = entries.iter().fold(Rect::NOTHING, |union, (_, entry)| union.union(entry.rect()));
        let entries = entries.into_iter().map(|(index, entry)| entry.resolve(index, |mask, rect| resolve_mask(ctx, mask, rect))).collect();
        let job = LayerJob { rect, entries, overlap: self.overlap, time, frame_input, id: self.id, viewport, reset_state };
        backend.paint_layer(painter, job);
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use egui::{LayerId, pos2, vec2};

    use super::*;
    use crate::backend::{Backend, BackendKind, BackendStatus, EffectFailure, TextureFailure, register};
    use crate::presets;

    /// A backend with a fixed status that counts `paint_layer` calls.
    struct Counting {
        status: BackendStatus,
        painted: AtomicUsize,
    }

    impl Backend for Counting {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            self.status.clone()
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, _job: LayerJob) {
            self.painted.fetch_add(1, Ordering::Relaxed);
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Paints one animated layer per pass on a fresh context with `status`, for a few passes
    /// (egui itself repaints during the first ones); returns (paint calls, whether the LAST pass
    /// asked for an immediate repaint).
    fn paint_animated_with(status: BackendStatus) -> (usize, bool) {
        const PASSES: usize = 4;
        let ctx = Context::default();
        let backend = Arc::new(Counting { status, painted: AtomicUsize::new(0) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let mut repaint = false;
        for _ in 0..PASSES {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
                ShaderLayer::new(Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)), presets::ripple(2.0, 10.0, 1.0)).paint(&painter);
            });
            repaint = output.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|viewport| viewport.repaint_delay.is_zero());
            // The font atlas upload is irrelevant here; epaint asserts deltas are handled.
            output.textures_delta.clear();
        }
        (backend.painted.load(Ordering::Relaxed) / PASSES, repaint)
    }

    #[test]
    fn a_failed_backend_gets_no_jobs_and_no_repaints() {
        let (painted, repaint) = paint_animated_with(BackendStatus::Failed { backend: BackendKind::Glow, error: "test".to_owned() });
        assert_eq!(painted, 0);
        assert!(!repaint, "an animated chain must not spin the event loop on a failed backend");

        let (painted, repaint) = paint_animated_with(BackendStatus::Installed { backend: BackendKind::Glow });
        assert_eq!(painted, 1);
        assert!(repaint);
    }

    /// A backend that keeps the frame input of every job.
    struct Recording {
        inputs: std::sync::Mutex<Vec<FrameInput>>,
    }

    impl Backend for Recording {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, job: LayerJob) {
            self.inputs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(job.frame_input);
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Headless runs: `delta_time` stays within `0..=0.1` whatever egui reports, `frame` counts
    /// completed frames, and the pointer is the latest position (screen points) with the primary
    /// button state.
    #[test]
    fn paint_captures_the_frame_input() {
        let ctx = Context::default();
        let backend = Arc::new(Recording { inputs: std::sync::Mutex::new(Vec::new()) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let frames: [(f64, Vec<egui::Event>); 3] = [
            (0.0, Vec::new()),
            // A 5 s gap: egui's dt is large, the uniform is clamped.
            (5.0, vec![egui::Event::PointerMoved(pos2(12.0, 7.0))]),
            (5.016, vec![egui::Event::PointerButton { pos: pos2(12.0, 7.0), button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }]),
        ];
        for (time, events) in frames {
            let raw = egui::RawInput { time: Some(time), events, ..Default::default() };
            let mut output = ctx.run_ui(raw, |ui| {
                let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
                ShaderLayer::new(Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)), presets::gaussian_blur(1.0)).paint(&painter);
            });
            output.textures_delta.clear();
        }
        let inputs = backend.inputs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        // egui may run extra passes per frame; each pass paints once.
        assert!(inputs.len() >= 3, "{inputs:?}");
        assert!(inputs.iter().all(|input| (0.0..=FrameInput::MAX_DELTA_TIME).contains(&input.delta_time)), "{inputs:?}");
        let (first, last) = (inputs[0], inputs[inputs.len() - 1]);
        assert_eq!(first.frame, 0);
        assert_eq!(first.pointer, None);
        assert_eq!(last.frame, 2, "two frames completed before the third");
        assert_eq!(last.pointer, Some(PointerInput { pos: pos2(12.0, 7.0), pressed: true }));
        assert!(inputs.iter().any(|input| input.pointer == Some(PointerInput { pos: pos2(12.0, 7.0), pressed: false })), "the move frame: present, not pressed");
    }

    /// F8: a layer whose effect reads a texture slot with nothing attached (or with an invalid
    /// slot) never reaches the backend; attaching the texture makes it paint, with the ids in the job.
    #[test]
    fn texture_misuse_skips_the_layer_before_the_backend() {
        let Ok(effect) = crate::Effect::new("textured", "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(2u, uv); }") else { panic!("must compile") };
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let id = TextureId::User(7);
        let layer = ShaderLayer::new(rect, crate::Pass::new(&effect));
        assert_eq!(layer.check_textures(), Err(TextureError::NotAttached { entry: 0, effect: "textured".to_owned(), pass: 0, slot: 2 }));
        assert_eq!(layer.clone().texture(0, id).check_textures().map_err(|error| matches!(error, TextureError::NotAttached { slot: 2, .. })), Err(true));
        assert_eq!(layer.clone().texture(2, id).texture(3, id).check_textures(), Err(TextureError::InvalidSlot { entry: 0, slot: 3 }));
        assert_eq!(layer.clone().texture(2, id).texture_at(2), Some(id));

        let ctx = Context::default();
        let backend = Arc::new(Counting { status: BackendStatus::Installed { backend: BackendKind::Glow }, painted: AtomicUsize::new(0) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        for attach in [false, true] {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
                let layer = ShaderLayer::new(rect, crate::Pass::new(&effect));
                if attach { layer.texture(2, id) } else { layer }.paint(&painter);
            });
            output.textures_delta.clear();
            assert_eq!(backend.painted.swap(0, Ordering::Relaxed) > 0, attach, "attached: {attach}");
        }
    }

    /// A backend that keeps `(id, reset claimed, stateful)` of every job; it claims a job's reset
    /// ticket (as a real backend does where its state registry runs) only while `claim` is set.
    struct Jobs {
        jobs: std::sync::Mutex<Vec<(Option<Id>, bool, bool)>>,
        claim: std::sync::atomic::AtomicBool,
    }

    impl Backend for Jobs {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, job: LayerJob) {
            let stateful = job.state_key(0).is_some();
            let claimed = self.claim.load(Ordering::Relaxed) && job.claim_reset();
            self.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((job.id, claimed, stateful));
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// F7: a stateful chain without an id never reaches the backend (`MissingId`); with one it
    /// does, and a `reset_state` request is claimed by the first paint of that id the backend
    /// runs; a paint that never reaches the backend's state registry leaves it pending.
    #[test]
    fn stateful_layers_need_an_id_and_consume_resets() {
        let Ok(state) = crate::Effect::new_state("sim", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state(uv); }") else { panic!("must compile") };
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let (id, other) = (Id::new("water"), Id::new("other"));
        assert_eq!(ShaderLayer::new(rect, crate::Pass::new(&state)).check_state(), Err(StateError::MissingId { effect: "sim".to_owned(), pass: 0, entry: 0 }));
        assert_eq!(ShaderLayer::new(rect, crate::Pass::new(&state)).id(id).check_state(), Ok(()));
        assert_eq!(ShaderLayer::new(rect, presets::invert(1.0)).check_state(), Ok(()), "no state pass, no id needed");

        let ctx = Context::default();
        let backend = Arc::new(Jobs { jobs: std::sync::Mutex::new(Vec::new()), claim: std::sync::atomic::AtomicBool::new(true) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let frame = |paint: &dyn Fn(&Painter)| {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| paint(&Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING)));
            output.textures_delta.clear();
            std::mem::take(&mut *backend.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
        };
        assert!(frame(&|painter| ShaderLayer::new(rect, crate::Pass::new(&state)).paint(painter)).is_empty(), "refused before the backend");
        crate::reset_state(&ctx, id);
        // A stateless layer with the id and a stateful one with another id leave the request.
        let jobs = frame(&|painter| {
            ShaderLayer::new(rect, presets::invert(1.0)).id(id).paint(painter);
            ShaderLayer::new(rect, crate::Pass::new(&state)).id(other).paint(painter);
        });
        assert_eq!(jobs, [(Some(id), false, false), (Some(other), false, true)]);
        // A paint whose run returns before the state registry (layer out of view) claims nothing.
        backend.claim.store(false, Ordering::Relaxed);
        let jobs = frame(&|painter| ShaderLayer::new(rect, crate::Pass::new(&state)).id(id).paint(painter));
        assert_eq!(jobs, [(Some(id), false, true)]);
        backend.claim.store(true, Ordering::Relaxed);
        let jobs = frame(&|painter| {
            ShaderLayer::new(rect, crate::Pass::new(&state)).id(id).paint(painter);
            ShaderLayer::new(rect, crate::Pass::new(&state)).id(id).paint(painter);
        });
        assert_eq!(jobs, [(Some(id), true, true), (Some(id), false, true)], "claimed by the first run of the id, after the unclaimed paint");
        let jobs = frame(&|painter| ShaderLayer::new(rect, crate::Pass::new(&state)).id(id).paint(painter));
        assert_eq!(jobs, [(Some(id), false, true)], "claimed once");
    }

    /// A backend that keeps the mask of every job.
    struct Masks {
        masks: std::sync::Mutex<Vec<JobMask>>,
    }

    impl Backend for Masks {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, job: LayerJob) {
            self.masks.lock().unwrap_or_else(std::sync::PoisonError::into_inner).extend(job.entries.into_iter().map(|entry| entry.mask));
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// F9: `corner_radius` is the rounded-rect mask; a shape mask reaches the backend tessellated
    /// (feathered, in layer points); a mask with a paint callback never reaches it.
    #[test]
    fn masks_are_resolved_or_refused_before_the_backend() {
        let rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(100.0, 100.0));
        let circle = egui::Shape::circle_filled(pos2(60.0, 60.0), 30.0, egui::Color32::WHITE);
        let layer = ShaderLayer::new(rect, presets::invert(1.0));
        assert_eq!(layer.clone().corner_radius(4).layer_mask(), &Mask::RoundedRect(CornerRadius::same(4)));
        assert_eq!(layer.clone().corner_radius(4).mask(Mask::Shape(circle.clone())).layer_mask(), &Mask::Shape(circle.clone()), "the last mask wins");
        let callback = egui::Shape::Callback(egui::PaintCallback { rect, callback: Arc::new(()) });
        assert_eq!(layer.clone().mask(Mask::Shape(callback.clone())).check_mask(), Err(MaskError::Callback));

        let ctx = Context::default();
        let backend = Arc::new(Masks { masks: std::sync::Mutex::new(Vec::new()) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
            layer.clone().corner_radius(6).paint(&painter);
            layer.clone().mask(Mask::Shape(circle.clone())).paint(&painter);
            layer.clone().mask(Mask::Shape(callback.clone())).paint(&painter);
            layer.clone().regions([Region::rounded_rect(rect, 0), Region::shape(circle.clone())]).paint(&painter);
            layer.clone().regions(vec![Region::rounded_rect(rect, 0); crate::MAX_REGIONS + 1]).paint(&painter);
        });
        output.textures_delta.clear();
        let masks = std::mem::take(&mut *backend.masks.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        assert_eq!(masks.len(), 3, "the callback mask and 256 regions are refused: {masks:?}");
        let Some(regions) = masks[2].regions() else { panic!("regions are tessellated per region: {:?}", masks[2]) };
        assert_eq!(regions.regions.len(), 2);
        assert!(regions.regions.iter().all(|region| !region.meshes.is_empty()));
        assert_eq!(layer.clone().regions(vec![Region::rounded_rect(rect, 0); 256]).check_mask(), Err(MaskError::TooManyRegions { count: 256, max: 255 }));
        assert_eq!(masks[0], JobMask::RoundedRect(CornerRadius::same(6)));
        let Some(meshes) = masks[1].coverage() else { panic!("a shape mask is tessellated: {:?}", masks[1]) };
        let bounds = meshes.meshes.iter().fold(Rect::NOTHING, |bounds, mesh| bounds.union(mesh.calc_bounds()));
        // Radius 30 plus half a feather pixel (ppp 1), in layer points.
        assert!((bounds.width() - 61.0).abs() < 0.01 && (bounds.center() - pos2(60.0, 60.0)).length() < 0.01, "{bounds:?}");
    }

    /// A backend that keeps `(rect, entry rects, overlap)` of every job.
    struct EntryJobs {
        jobs: std::sync::Mutex<Vec<(Rect, Vec<Rect>, Overlap)>>,
    }

    impl Backend for EntryJobs {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, job: LayerJob) {
            let rects = job.entries.iter().map(|entry| entry.rect).collect();
            self.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((job.rect, rects, job.overlap));
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// R3: `new` is one entry; `rect()` is the union; the layer-level builders reach every entry
    /// present, the accessors read the first; checks name the failing entry; `paint` sends one job
    /// with every drawable entry (empty chains left out) and skips the whole layer when one entry
    /// fails a check.
    #[test]
    fn entries_share_one_job_and_one_decision() {
        let a = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let b = Rect::from_min_size(pos2(50.0, 20.0), vec2(10.0, 30.0));
        assert_eq!(ShaderLayer::new(a, presets::invert(1.0)).entries().len(), 1);
        let id = TextureId::User(9);
        let layer = ShaderLayer::from_entries([Entry::new(a, presets::invert(1.0)), Entry::new(b, presets::grayscale(1.0))]).corner_radius(3).texture(1, id).state(StateOptions::default().with_scale(0.5));
        assert_eq!(layer.rect(), a.union(b));
        assert!(layer.entries().iter().all(|entry| entry.layer_mask() == &Mask::RoundedRect(CornerRadius::same(3)) && entry.texture_at(1) == Some(id) && (entry.state_options().scale() - 0.5).abs() < f32::EPSILON));
        assert_eq!((layer.texture_at(1), layer.layer_mask()), (Some(id), &Mask::RoundedRect(CornerRadius::same(3))));
        assert_eq!(layer.chain().len(), 1);
        let empty = ShaderLayer::from_entries([]);
        assert!(empty.chain().is_empty() && empty.layer_mask() == &Mask::default() && empty.rect() == Rect::NOTHING && empty.texture_at(0).is_none());
        let callback = egui::Shape::Callback(egui::PaintCallback { rect: a, callback: Arc::new(()) });
        let bad = layer.clone().with_entry(Entry::new(a, presets::invert(1.0)).mask(Mask::Shape(callback)));
        assert_eq!(bad.check_mask(), Err(MaskError::Callback));
        let Ok(textured) = crate::Effect::new("textured", "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(2u, uv); }") else { panic!("must compile") };
        let missing = layer.clone().with_entry(Entry::new(b, crate::Pass::new(&textured)));
        assert_eq!(missing.check_textures(), Err(TextureError::NotAttached { entry: 2, effect: "textured".to_owned(), pass: 0, slot: 2 }));

        let ctx = Context::default();
        let backend = Arc::new(EntryJobs { jobs: std::sync::Mutex::new(Vec::new()) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
            layer.clone().with_entry(Entry::new(a, FilterChain::new())).overlap(Overlap::Stacked).paint(&painter);
            bad.clone().paint(&painter);
            missing.clone().paint(&painter);
            ShaderLayer::from_entries([Entry::new(a, FilterChain::new())]).paint(&painter);
        });
        output.textures_delta.clear();
        let jobs = std::mem::take(&mut *backend.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        assert_eq!(jobs, [(a.union(b), vec![a, b], Overlap::Stacked)], "one job, the empty-chain entry left out; failing layers and a layer of empty chains send none");
    }

    /// `(user index, state key)` of every entry of one job.
    type JobKeys = Vec<(usize, Option<crate::state::StateKey>)>;

    /// A backend that keeps the [`JobKeys`] of every job.
    struct EntryKeys {
        jobs: std::sync::Mutex<Vec<JobKeys>>,
    }

    impl Backend for EntryKeys {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn failed_states(&self) -> Vec<crate::state::StateFailure> {
            Vec::new()
        }
        fn state_count(&self) -> usize {
            0
        }
        fn paint_layer(&self, _painter: &Painter, job: LayerJob) {
            let keys = job.entries.iter().enumerate().map(|(position, entry)| (entry.index, job.state_key(position))).collect();
            self.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(keys);
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// R3: an entry left out of the job (empty chain, zero-size rect) does not shift the later
    /// entries: the kept entry carries its USER index, its state key is the layer key `.with(user
    /// index)` whether or not the earlier entry draws, and check errors name the user index.
    #[test]
    fn a_non_drawable_entry_keeps_the_user_indices() {
        let Ok(state) = crate::Effect::new_state("sim", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state(uv); }") else { panic!("must compile") };
        let Ok(textured) = crate::Effect::new("textured", "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(2u, uv); }") else { panic!("must compile") };
        let a = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let collapsed = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 0.0));
        let b = Rect::from_min_size(pos2(50.0, 20.0), vec2(10.0, 30.0));
        let id = Id::new("entries keep indices");
        let stateful = Entry::new(b, crate::Pass::new(&state));
        let before = |a_rect: Rect, a_chain: FilterChain| ShaderLayer::from_entries([Entry::new(a_rect, a_chain), stateful.clone()]);
        assert_eq!(before(collapsed, presets::invert(1.0).into()).check_state(), Err(StateError::MissingId { effect: "sim".to_owned(), pass: 0, entry: 1 }));
        let missing = ShaderLayer::from_entries([Entry::new(a, FilterChain::new()), Entry::new(b, crate::Pass::new(&textured))]);
        assert_eq!(missing.check_textures(), Err(TextureError::NotAttached { entry: 1, effect: "textured".to_owned(), pass: 0, slot: 2 }));

        let ctx = Context::default();
        let backend = Arc::new(EntryKeys { jobs: std::sync::Mutex::new(Vec::new()) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
            before(a, presets::invert(1.0).into()).id(id).paint(&painter);
            before(collapsed, presets::invert(1.0).into()).id(id).paint(&painter);
            before(a, FilterChain::new()).id(id).paint(&painter);
        });
        output.textures_delta.clear();
        let jobs = std::mem::take(&mut *backend.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        let key = crate::plan::entry_state_key(crate::state::StateKey { viewport: ctx.viewport_id(), id }, 1);
        assert_eq!(jobs, [vec![(0, None), (1, Some(key))], vec![(1, Some(key))], vec![(1, Some(key))]], "the stateful entry keeps index 1 and its key");
    }

    /// The "no backend installed" latch is per `Context`: set once per context, independent between contexts.
    #[test]
    fn not_installed_warning_latches_per_context() {
        let first = Context::default();
        let second = Context::default();
        assert!(first_not_installed_warning(&first));
        assert!(!first_not_installed_warning(&first));
        assert!(first_not_installed_warning(&second));
    }
}
