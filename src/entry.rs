/*
File: entry.rs

Purpose:
One filtered area of a multi-entry `ShaderLayer` (R3): its own rect, chain, mask, user textures
and state options (`Entry`), and how entries that overlap see each other (`Overlap`). The layer
(`layer.rs`) holds the list; a single-entry layer is what `ShaderLayer::new` builds.

Key structures:
- `Entry` (public): rect + chain + mask + textures + state options, with the same builders the
  layer has; its checks name the entry's index in the layer.
- `Overlap` (public): `Independent` (every entry filters the original backdrop) or `Stacked`
  (a later entry filters the result of the earlier ones).

Key functions:
- `Entry::check_textures_at()` / `check_state_at()`: the layer's checks for one entry (index
  given by the layer).
- `Entry::resolve()`: the entry turned into its backend-neutral `EntryJob` (mask tessellated by
  the caller's function, on the GUI thread).

Notes:
The ORDER in which entries are captured, run, handed off and composited is not decided here or in
any backend: `plan::schedule` is its single owner.
*/

use egui::{CornerRadius, Rect, TextureId};

use crate::backend::{EntryJob, TextureError};
use crate::effect::TEXTURE_SLOTS;
use crate::mask::{JobMask, Mask, MaskError};
use crate::pass::FilterChain;
use crate::plan::check_textures;
use crate::state::{StateError, StateOptions};

/// How the entries of one [`crate::ShaderLayer`] that overlap see each other. Entries are always
/// composited in list order (a later entry lands over an earlier one, blended by its own mask
/// coverage); the mode decides what a later entry FILTERS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Overlap {
    /// Every entry filters the backdrop beneath the layer, as if the other entries did not exist.
    /// Where entries overlap, the later entry's filtered backdrop covers the earlier one's.
    #[default]
    Independent,
    /// A later entry filters the backdrop WITH the earlier entries' results already on it (a blur
    /// entry over an invert entry blurs the inverted pixels), like separate layers painted one
    /// after the other — on every backend, including wgpu, where separate layers cannot see each
    /// other.
    Stacked,
}

/// One filtered area of a [`crate::ShaderLayer`]: its own rect (points), chain, mask, user
/// textures and state options. A layer built with [`crate::ShaderLayer::new`] has exactly one
/// entry; [`crate::ShaderLayer::from_entries`] / [`crate::ShaderLayer::with_entry`] build layers of
/// several. Every entry captures only ITS OWN padded rect (never the union of the entries), runs
/// its own chain and is composited in list order.
///
/// ```no_run
/// # fn frame(ui: &mut egui::Ui) {
/// use egui::{Rect, pos2, vec2};
/// use egui_shader_layers::{Entry, ShaderLayer, presets};
/// let left = Rect::from_min_size(pos2(20.0, 20.0), vec2(120.0, 80.0));
/// let right = Rect::from_min_size(pos2(400.0, 20.0), vec2(120.0, 80.0));
/// ShaderLayer::from_entries([
///     Entry::new(left, presets::gaussian_blur(6.0)).corner_radius(8),
///     Entry::new(right, presets::invert(1.0)),
/// ])
/// .paint(ui.painter());
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Entry {
    /// The filtered rect, in points (the layer's painter space).
    rect: Rect,
    /// The passes to run.
    chain: FilterChain,
    /// The mask (rounded rect by default).
    mask: Mask,
    /// The user texture attached at each slot.
    textures: [Option<TextureId>; TEXTURE_SLOTS],
    /// The first out-of-range slot given to [`Entry::texture`], reported by the texture check.
    invalid_slot: Option<usize>,
    /// How the persistent state is stored (stateful chains only).
    state: StateOptions,
}

impl Entry {
    /// An entry filtering `rect` (points) through `chain` (a `Pass`, `Vec<Pass>` or `FilterChain`),
    /// with the plain rect as mask, no textures and default state options.
    #[must_use]
    pub fn new(rect: Rect, chain: impl Into<FilterChain>) -> Self {
        Self { rect, chain: chain.into(), mask: Mask::default(), textures: [None; TEXTURE_SLOTS], invalid_slot: None, state: StateOptions::default() }
    }

    /// Restricts the entry to `mask` (replacing any earlier mask or corner radius); see
    /// [`crate::ShaderLayer::mask`].
    #[must_use]
    pub fn mask(mut self, mask: Mask) -> Self {
        self.mask = mask;
        self
    }

    /// Restricts the entry to a rounded rect: shorthand for `mask(Mask::RoundedRect(..))`.
    #[must_use]
    pub fn corner_radius(self, corner_radius: impl Into<CornerRadius>) -> Self {
        self.mask(Mask::RoundedRect(corner_radius.into()))
    }

    /// Attaches the egui texture `id` at user texture `slot` of THIS entry's effects; see
    /// [`crate::ShaderLayer::texture`]. An out-of-range slot fails the layer's texture check.
    #[must_use]
    pub fn texture(mut self, slot: usize, id: TextureId) -> Self {
        self.set_texture(slot, id);
        self
    }

    /// How this entry's persistent state is stored (stateful chains only); see
    /// [`crate::ShaderLayer::state`]. Every stateful entry keeps its own state.
    #[must_use]
    pub fn state(mut self, options: StateOptions) -> Self {
        self.state = options;
        self
    }

    /// The filtered rect, in points.
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// The chain this entry runs.
    #[must_use]
    pub fn chain(&self) -> &FilterChain {
        &self.chain
    }

    /// The entry's mask.
    #[must_use]
    pub fn layer_mask(&self) -> &Mask {
        &self.mask
    }

    /// The texture attached at `slot`, if any.
    #[must_use]
    pub fn texture_at(&self, slot: usize) -> Option<TextureId> {
        self.textures.get(slot).copied().flatten()
    }

    /// The state options.
    #[must_use]
    pub fn state_options(&self) -> StateOptions {
        self.state
    }

    /// Whether the entry has a state pass.
    #[must_use]
    pub(crate) fn is_stateful(&self) -> bool {
        self.chain.passes().iter().any(|pass| pass.effect().is_state())
    }

    /// Whether this entry draws anything: a non-empty chain over a
    /// positive rect.
    #[must_use]
    pub(crate) fn is_drawable(&self) -> bool {
        !self.chain.is_empty() && self.rect.is_positive()
    }

    /// The texture check of [`crate::ShaderLayer::check_textures`] for this entry, the layer's
    /// entry `index`.
    ///
    /// # Errors
    /// [`TextureError::InvalidSlot`] or [`TextureError::NotAttached`], naming `index`.
    pub(crate) fn check_textures_at(&self, index: usize) -> Result<(), TextureError> {
        if let Some(slot) = self.invalid_slot {
            return Err(TextureError::InvalidSlot { entry: index, slot });
        }
        check_textures(self.chain.passes(), self.textures.map(|texture| texture.is_some()), index)
    }

    /// The state check of [`crate::ShaderLayer::check_state`] for this entry, the layer's entry
    /// `index`, on a layer with (`has_id`) or without an id.
    ///
    /// # Errors
    /// [`StateError::MissingId`] naming the first state pass and `index`.
    pub(crate) fn check_state_at(&self, index: usize, has_id: bool) -> Result<(), StateError> {
        match self.chain.passes().iter().position(|pass| pass.effect().is_state()) {
            Some(pass) if !has_id => Err(StateError::MissingId { effect: self.chain.passes()[pass].effect().label().to_owned(), pass, entry: index }),
            Some(_) | None => Ok(()),
        }
    }

    /// The mask check ([`Mask::check`]).
    ///
    /// # Errors
    /// As [`Mask::check`].
    pub(crate) fn check_mask(&self) -> Result<(), MaskError> {
        self.mask.check()
    }

    /// Replaces the mask (the layer-level builders apply to every entry).
    pub(crate) fn set_mask(&mut self, mask: Mask) {
        self.mask = mask;
    }

    /// Attaches a texture (the layer-level builder applies to every entry).
    pub(crate) fn set_texture(&mut self, slot: usize, id: TextureId) {
        match self.textures.get_mut(slot) {
            Some(entry) => *entry = Some(id),
            None => {
                self.invalid_slot.get_or_insert(slot);
            }
        }
    }

    /// Replaces the state options (the layer-level builder applies to every entry).
    pub(crate) fn set_state(&mut self, options: StateOptions) {
        self.state = options;
    }

    /// The entry as the backend sees it; `resolve_mask` turns the mask into its job form (a shape
    /// tessellated on the GUI thread, in the entry's rect).
    pub(crate) fn resolve(self, resolve_mask: impl FnOnce(Mask, Rect) -> JobMask) -> EntryJob {
        let padding_points = self.chain.padding_points();
        EntryJob { rect: self.rect, mask: resolve_mask(self.mask, self.rect), passes: self.chain.into_passes(), padding_points, textures: self.textures, state: self.state }
    }
}

#[cfg(test)]
mod tests {
    use egui::{pos2, vec2};

    use super::*;
    use crate::presets;

    #[test]
    fn checks_name_the_entry() {
        let Ok(textured) = crate::Effect::new("textured", "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(1u, uv); }") else { panic!("must compile") };
        let Ok(state) = crate::Effect::new_state("sim", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state(uv); }") else { panic!("must compile") };
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let entry = Entry::new(rect, crate::Pass::new(&textured));
        assert_eq!(entry.check_textures_at(3), Err(TextureError::NotAttached { entry: 3, effect: "textured".to_owned(), pass: 0, slot: 1 }));
        assert_eq!(entry.clone().texture(1, TextureId::User(1)).check_textures_at(3), Ok(()));
        assert_eq!(entry.clone().texture(7, TextureId::User(1)).check_textures_at(2), Err(TextureError::InvalidSlot { entry: 2, slot: 7 }));
        let stateful = Entry::new(rect, FilterChain::from(presets::invert(1.0)).then(crate::Pass::new(&state)));
        assert!(stateful.is_stateful());
        assert_eq!(stateful.check_state_at(1, false), Err(StateError::MissingId { effect: "sim".to_owned(), pass: 1, entry: 1 }));
        assert_eq!(stateful.check_state_at(1, true), Ok(()));
        let mut edited = Entry::new(rect, presets::invert(1.0));
        edited.set_texture(0, TextureId::User(4));
        edited.set_texture(5, TextureId::User(4));
        assert_eq!((edited.texture_at(0), edited.rect()), (Some(TextureId::User(4)), rect), "set_texture keeps the entry");
        assert_eq!(edited.check_textures_at(0), Err(TextureError::InvalidSlot { entry: 0, slot: 5 }));
        assert!(!Entry::new(rect, FilterChain::new()).is_drawable() && edited.is_drawable());
    }
}
