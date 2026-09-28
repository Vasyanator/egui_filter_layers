/*
File: impulses.rs

Purpose:
F3: a small, user-owned collection of timed "impulses" (a click, a tap, a drop) that an effect
turns into a transient animation — the input of the `presets::click_ripples` preset, and of any
user effect that declares the same named parameters.

Key structures:
- `Impulses` (public): the bounded collection; ages impulses against a caller-supplied clock
  (egui's `input.time` in the UI, any clock offline), drops expired and overflowing ones, packs
  the live ones into a `Pass`'s named parameters.
- `Impulse`, `ImpulsePack` (public): one impulse, and the packed shader-ready form.

Key functions:
- `Impulses::push()` / `push_point()`: add an impulse in layer-rect uv / at a point.
- `Impulses::retire()`, `is_alive()`, `pack()`, `apply()`.
- `Impulses::request_repaint_if_alive()`: the only function that touches an `egui::Context`.

Notes:
State lives in the caller (a field of the app), never in egui memory or a global: the helper is
plain data, usable offline, and a repaint is requested only while something is alive, so an
idle app does not spin. Positions are relative to the LAYER RECT (0..1 across it), not to the
capture: the capture's padding changes with the chain, the rect does not; the shader maps them
with `fl.rect_uv`.
*/

use std::collections::VecDeque;

use egui::{Pos2, Rect};

use crate::effect::ParamError;
use crate::pass::Pass;

/// Most impulses an [`Impulses`] holds and packs: the length of the `impulses` array parameter.
pub const IMPULSE_SLOTS: usize = 16;

/// [`IMPULSE_SLOTS`] as the `u32` the shader counts with (a small constant: the value is exact).
const IMPULSE_SLOTS_U32: u32 = 16;

/// Named parameter receiving the packed impulses: `impulses: array<vec4<f32>, 16>` in WGSL, each
/// `vec4(uv.x, uv.y, age_seconds, strength)` with `uv` relative to the layer rect.
pub const IMPULSES_PARAM: &str = "impulses";

/// Named parameter receiving how many leading elements of [`IMPULSES_PARAM`] are live:
/// `impulse_count: u32` in WGSL.
pub const IMPULSE_COUNT_PARAM: &str = "impulse_count";

/// One impulse: where and when it started and how strong it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Impulse {
    /// Position relative to the layer rect: `(0, 0)` its top-left corner, `(1, 1)` its
    /// bottom-right, y down (values outside `0..1` lie outside the rect).
    pub uv: [f32; 2],
    /// Start time in seconds on the clock passed to [`Impulses::push`].
    pub started: f64,
    /// Effect-defined strength (for `click_ripples`: a multiplier of the amplitude, 1 = full).
    pub strength: f32,
}

/// The live impulses packed for a shader: element `i < count` is
/// `[uv.x, uv.y, age_seconds, strength]`, the rest are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpulsePack {
    /// The packed impulses, oldest first.
    pub impulses: [[f32; 4]; IMPULSE_SLOTS],
    /// How many leading elements of `impulses` are live (`0..=16`).
    pub count: u32,
}

/// A bounded, user-owned collection of timed impulses feeding an effect such as
/// [`presets::click_ripples`](crate::presets::click_ripples).
///
/// Keep it in your app state and drive it with one clock — egui's `ctx.input(|i| i.time)` in the
/// UI, frame time offline. Every impulse lives `life` seconds. When full, pushing drops the
/// OLDEST pushed impulse first. The collection never touches an `egui::Context` except in
/// [`Impulses::request_repaint_if_alive`], which the UI calls once per frame so animation keeps
/// running while (and only while) an impulse is alive.
///
/// ```
/// use egui_filter_layers::{FilterLayer, Impulses, presets};
///
/// struct App { ripples: Impulses }
///
/// fn ui(app: &mut App, ui: &mut egui::Ui, rect: egui::Rect) {
///     let now = ui.input(|input| input.time);
///     let response = ui.interact(rect, ui.id().with("ripples"), egui::Sense::click());
///     if let Some(pos) = response.interact_pointer_pos().filter(|_| response.clicked()) {
///         app.ripples.push_point(pos, rect, 1.0, now);
///     }
///     app.ripples.retire(now);
///     app.ripples.request_repaint_if_alive(ui.ctx(), now);
///     FilterLayer::new(rect, presets::click_ripples(&app.ripples, now, 6.0, 24.0, 120.0)).paint(ui.painter());
/// }
/// # let _ = App { ripples: Impulses::new(1.5) };
/// ```
#[derive(Clone, Debug)]
pub struct Impulses {
    /// Seconds every impulse lives; a non-finite or non-positive value keeps none alive.
    life: f32,
    /// Most impulses kept, `1..=IMPULSE_SLOTS`.
    capacity: usize,
    /// Impulses in push order (front = oldest pushed).
    items: VecDeque<Impulse>,
}

impl Impulses {
    /// An empty collection whose impulses live `life_seconds` each, holding up to
    /// [`IMPULSE_SLOTS`] impulses. A non-finite or non-positive life keeps nothing alive.
    #[must_use]
    pub fn new(life_seconds: f32) -> Self {
        let life = if life_seconds.is_finite() { life_seconds.max(0.0) } else { 0.0 };
        Self { life, capacity: IMPULSE_SLOTS, items: VecDeque::with_capacity(IMPULSE_SLOTS) }
    }

    /// Limits the collection to `capacity` impulses, clamped to `1..=IMPULSE_SLOTS`; excess
    /// impulses already held are dropped oldest first.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.clamp(1, IMPULSE_SLOTS);
        self.trim();
        self
    }

    /// Seconds every impulse lives (0 when the life given to [`Impulses::new`] was invalid).
    #[must_use]
    pub fn life(&self) -> f32 {
        self.life
    }

    /// Most impulses held at once (`1..=16`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The impulses currently held (expired ones included until [`Impulses::retire`]), oldest
    /// pushed first.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Impulse> {
        self.items.iter()
    }

    /// Adds an impulse at `uv` (relative to the layer rect, see [`Impulse::uv`]) started at
    /// `now` seconds. When the collection is full the oldest pushed impulse is dropped.
    ///
    /// Returns `false` and stores nothing when `uv`, `strength` or `now` is not finite (it would
    /// poison every pixel of the effect with NaN).
    pub fn push(&mut self, uv: [f32; 2], strength: f32, now: f64) -> bool {
        if !(uv[0].is_finite() && uv[1].is_finite() && strength.is_finite() && now.is_finite()) {
            return false;
        }
        self.items.push_back(Impulse { uv, started: now, strength });
        self.trim();
        true
    }

    /// Adds an impulse at `pos`, given in the same coordinates as `layer_rect` (the rect passed
    /// to [`FilterLayer::new`](crate::FilterLayer::new), e.g. a `Response`'s pointer position).
    ///
    /// Returns `false` and stores nothing when `layer_rect` has no positive area or any value is
    /// not finite (see [`Impulses::push`]).
    pub fn push_point(&mut self, pos: Pos2, layer_rect: Rect, strength: f32, now: f64) -> bool {
        match rect_uv(pos, layer_rect) {
            Some(uv) => self.push(uv, strength, now),
            None => false,
        }
    }

    /// Drops every impulse that has expired at `now` (age >= life).
    pub fn retire(&mut self, now: f64) {
        let life = f64::from(self.life);
        self.items.retain(|impulse| age(impulse, now) < life);
    }

    /// Whether any impulse is alive at `now` (age < life), whether or not
    /// [`Impulses::retire`] ran.
    #[must_use]
    pub fn is_alive(&self, now: f64) -> bool {
        let life = f64::from(self.life);
        self.items.iter().any(|impulse| age(impulse, now) < life)
    }

    /// Requests an egui repaint when an impulse is alive at `now` (see [`Impulses::is_alive`]),
    /// so the animation advances in an otherwise idle app. Call once per frame in the UI. Nothing
    /// is requested once every impulse has expired, so an idle app stops repainting.
    pub fn request_repaint_if_alive(&self, ctx: &egui::Context, now: f64) {
        if self.is_alive(now) {
            ctx.request_repaint();
        }
    }

    /// The impulses alive at `now`, oldest first, as `[uv.x, uv.y, age_seconds, strength]`. An
    /// impulse started after `now` (a clock that went backwards) counts as age 0.
    #[must_use]
    pub fn pack(&self, now: f64) -> ImpulsePack {
        let life = f64::from(self.life);
        let mut pack = ImpulsePack { impulses: [[0.0; 4]; IMPULSE_SLOTS], count: 0 };
        let alive = self.items.iter().filter_map(|impulse| Some((impulse, age(impulse, now))).filter(|&(_, age)| age < life));
        let mut count = 0_usize;
        for (slot, (impulse, age)) in pack.impulses.iter_mut().zip(alive) {
            // Ages are below `life` (an f32), so the f64 -> f32 narrowing loses only precision
            // the shader cannot use anyway.
            #[allow(clippy::cast_possible_truncation)] // Intentional precision reduction, see above.
            let age = age as f32;
            *slot = [impulse.uv[0], impulse.uv[1], age, impulse.strength];
            count += 1;
        }
        // `count <= IMPULSE_SLOTS` by the zip above; the fallback is unreachable.
        pack.count = u32::try_from(count).unwrap_or(IMPULSE_SLOTS_U32);
        pack
    }

    /// Writes [`Impulses::pack`] at `now` into `pass`'s named parameters [`IMPULSES_PARAM`]
    /// (`array<vec4<f32>, 16>`) and [`IMPULSE_COUNT_PARAM`] (`u32`).
    ///
    /// # Errors
    /// The [`ParamError`] of [`Pass::set_slice`] / [`Pass::set`] when the pass's effect does not
    /// declare both members with those types; the pass may then hold the new array but the old
    /// count.
    pub fn apply(&self, pass: &mut Pass, now: f64) -> Result<(), ParamError> {
        let pack = self.pack(now);
        pass.set_slice(IMPULSES_PARAM, &pack.impulses)?;
        pass.set(IMPULSE_COUNT_PARAM, pack.count)
    }

    /// Drops the oldest pushed impulses beyond the capacity.
    fn trim(&mut self) {
        while self.items.len() > self.capacity {
            self.items.pop_front();
        }
    }
}

/// Seconds since `impulse` started, 0 for one started in the future.
fn age(impulse: &Impulse, now: f64) -> f64 {
    (now - impulse.started).max(0.0)
}

/// `pos` relative to `rect` (`(0, 0)` = `rect.min`, `(1, 1)` = `rect.max`), or `None` for a rect
/// without positive area or non-finite input.
fn rect_uv(pos: Pos2, rect: Rect) -> Option<[f32; 2]> {
    if !rect.is_positive() || !rect.is_finite() {
        return None;
    }
    let uv = (pos - rect.min) / rect.size();
    (uv.x.is_finite() && uv.y.is_finite()).then_some([uv.x, uv.y])
}

#[cfg(test)]
mod tests {
    use egui::{pos2, vec2};

    use super::*;
    use crate::Effect;

    /// Element-wise equality within a few ulps (packed values are exact up to f64 -> f32).
    fn same(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 4.0 * f32::EPSILON)
    }

    #[test]
    fn impulses_age_and_expire_after_their_life() {
        let mut impulses = Impulses::new(2.0);
        assert!(!impulses.is_alive(0.0));
        assert!(impulses.push([0.25, 0.5], 1.0, 10.0));
        assert!(impulses.is_alive(10.0) && impulses.is_alive(11.99));
        assert!(!impulses.is_alive(12.0), "age == life is expired");
        let pack = impulses.pack(11.5);
        assert_eq!(pack.count, 1);
        assert!(same(pack.impulses[0], [0.25, 0.5, 1.5, 1.0]));
        assert_eq!(impulses.pack(12.0).count, 0, "expired impulses are not packed even before retire");
        impulses.retire(11.0);
        assert_eq!(impulses.iter().len(), 1);
        impulses.retire(12.0);
        assert_eq!(impulses.iter().len(), 0);
        assert!(impulses.push([0.0, 0.0], 1.0, 20.0));
        assert!(impulses.pack(19.0).impulses[0][2].abs() < f32::EPSILON, "a clock that went backwards gives age 0");
    }

    #[test]
    fn a_full_collection_drops_the_oldest_first() {
        let mut impulses = Impulses::new(100.0);
        for i in 0..20_u8 {
            assert!(impulses.push([f32::from(i), 0.0], 1.0, f64::from(i)));
        }
        assert_eq!(impulses.iter().len(), IMPULSE_SLOTS);
        let pack = impulses.pack(20.0);
        assert_eq!(pack.count, 16);
        assert!((pack.impulses[0][0] - 4.0).abs() < f32::EPSILON, "impulses 0..4 were dropped");
        assert!((pack.impulses[15][0] - 19.0).abs() < f32::EPSILON);

        let mut small = impulses.clone().with_capacity(3);
        assert_eq!(small.capacity(), 3);
        assert_eq!(small.iter().map(|impulse| impulse.uv[0]).collect::<Vec<_>>(), [17.0, 18.0, 19.0]);
        assert!(small.push([99.0, 0.0], 1.0, 21.0));
        assert_eq!(small.iter().map(|impulse| impulse.uv[0]).collect::<Vec<_>>(), [18.0, 19.0, 99.0]);
        assert_eq!(Impulses::new(1.0).with_capacity(0).capacity(), 1);
        assert_eq!(Impulses::new(1.0).with_capacity(1000).capacity(), IMPULSE_SLOTS);
    }

    #[test]
    fn packing_keeps_order_and_zeroes_the_tail() {
        let mut impulses = Impulses::new(1.0);
        assert!(impulses.push([0.1, 0.2], 0.5, 0.0));
        assert!(impulses.push([0.3, 0.4], 2.0, 0.5));
        let pack = impulses.pack(0.75);
        assert_eq!(pack.count, 2);
        assert!(same(pack.impulses[0], [0.1, 0.2, 0.75, 0.5]));
        assert!(same(pack.impulses[1], [0.3, 0.4, 0.25, 2.0]));
        assert!(pack.impulses[2..].iter().all(|slot| same(*slot, [0.0; 4])));
        // The first expired, the second alive: the live one moves to slot 0.
        let pack = impulses.pack(1.25);
        assert_eq!(pack.count, 1);
        assert!(same(pack.impulses[0], [0.3, 0.4, 0.75, 2.0]));
    }

    #[test]
    fn invalid_input_is_rejected_not_stored() {
        let mut impulses = Impulses::new(1.0);
        assert!(!impulses.push([f32::NAN, 0.0], 1.0, 0.0));
        assert!(!impulses.push([0.0, 0.0], f32::INFINITY, 0.0));
        assert!(!impulses.push([0.0, 0.0], 1.0, f64::NAN));
        let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(100.0, 50.0));
        assert!(!impulses.push_point(pos2(0.0, 0.0), Rect::from_min_size(pos2(0.0, 0.0), vec2(0.0, 10.0)), 1.0, 0.0));
        assert_eq!(impulses.iter().len(), 0);
        assert!(impulses.push_point(pos2(35.0, 70.0), rect, 1.0, 0.0));
        assert_eq!(impulses.iter().next().map(|impulse| impulse.uv), Some([0.25, 1.0]));
        assert!(!Impulses::new(f32::NAN).is_alive(0.0) && Impulses::new(-1.0).life() == 0.0);
        let mut dead = Impulses::new(0.0);
        assert!(dead.push([0.5, 0.5], 1.0, 0.0));
        assert!(!dead.is_alive(0.0), "zero life keeps nothing alive");
    }

    #[test]
    fn apply_writes_the_named_parameters() {
        let effect = Effect::new("impulse reader", "struct Params { impulses: array<vec4<f32>, 16>, impulse_count: u32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return p.impulses[p.impulse_count % 16u]; }").unwrap_or_else(|error| panic!("{error}"));
        let mut pass = Pass::new(&effect);
        let mut impulses = Impulses::new(1.0);
        assert!(impulses.push([0.5, 0.25], 1.0, 0.0));
        assert_eq!(impulses.apply(&mut pass, 0.5), Ok(()));
        assert_eq!(pass.get(IMPULSE_COUNT_PARAM), Some(crate::ParamValue::U32(1)));
        let slots = pass.get_slice(IMPULSES_PARAM).unwrap_or_default();
        assert_eq!(slots.len(), IMPULSE_SLOTS);
        assert_eq!(slots[0], crate::ParamValue::Vec4([0.5, 0.25, 0.5, 1.0]));
        assert_eq!(slots[1], crate::ParamValue::Vec4([0.0; 4]));

        let other = Effect::new("no impulses", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }").unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(impulses.apply(&mut Pass::new(&other), 0.5), Err(ParamError::Unknown { .. })));
    }

    /// Runs a few egui passes on a fresh headless context, calling `request_repaint_if_alive`
    /// at `now` in each; returns whether the LAST pass asked for an immediate repaint (egui
    /// repaints during its first passes on its own).
    fn repaint_requested(impulses: &Impulses, now: f64) -> bool {
        let ctx = egui::Context::default();
        let mut repaint = false;
        for _ in 0..4 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                impulses.request_repaint_if_alive(ui.ctx(), now);
            });
            repaint = output.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|viewport| viewport.repaint_delay.is_zero());
            // The font atlas upload is irrelevant here; epaint asserts deltas are handled.
            output.textures_delta.clear();
        }
        repaint
    }

    #[test]
    fn repaints_are_requested_only_while_an_impulse_is_alive() {
        let mut impulses = Impulses::new(1.0);
        assert!(!repaint_requested(&impulses, 0.0), "empty: an idle app must not spin");
        assert!(impulses.push([0.5, 0.5], 1.0, 0.0));
        assert!(repaint_requested(&impulses, 0.5), "alive: keeps animating");
        assert!(!repaint_requested(&impulses, 1.0), "expired: stops");
    }
}
