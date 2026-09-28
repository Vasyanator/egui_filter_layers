/*
File: wgpu_backend/textures.rs

Purpose:
Resolves the egui `TextureId`s the beneath meshes sample into bind groups the mesh pipeline can
use, at the end of an egui pass — before egui-wgpu has applied the frame's texture deltas.

Key structures:
- `TextureCopies`: textures uploaded in full that egui-wgpu has not applied yet (`Kept`), their
  lazily created GPU copies, and the bind-group layout / samplers needed to bind them.

Key functions:
- `TextureCopies::observe()`: records one pass's `TexturesDelta` (called for EVERY pass).
- `snapshot_textures()`: per texture id, egui-wgpu's own bind group or the kept copy's.
- `resolve_user_textures()`: the same resolution for a layer's attached user textures (F8), with
  unresolvable ids reported instead of skipped.

Notes:
- egui-wgpu applies a frame's deltas (all passes appended) only when it paints, after every
  pass has ended. A texture uploaded in FULL (new, replaced, or the font atlas after it grew) is
  therefore not usable from the renderer during the passes that follow, possibly several of them
  in a multi-pass frame (`request_discard`). Such a texture is kept here: its full delta and
  every partial delta after it, from whichever pass carried them. The GPU copy is created on
  first use and later partial deltas (glyphs rasterised in a later pass) are written into it, so
  every layer and pass shares one upload.
- A kept texture is dropped once egui-wgpu holds a different `wgpu::Texture` for its id than
  when the full upload was seen (egui-wgpu creates a new texture for every full upload), or when
  the id is freed. This needs no notion of frame or viewport, which the end-of-pass hook cannot
  observe reliably (`end_pass` pops the viewport stack before plugins run).
- A texture never uploaded in full since egui-wgpu last applied it, or only patched
  (`ImageDelta::pos` is `Some`), is bound through egui-wgpu's bind group: egui-wgpu writes
  partial updates into that same texture before the frame's command buffer runs.
- Anything else missing (unknown id, a font atlas whose renderer copy does not match the CPU
  atlas size) is skipped for that frame, debug-logged once per install
  (`Shared::logged`). A user texture a layer ATTACHES is different: its id must resolve, or the
  layer is not planned and the id is reported (`Shared::fail_texture`, `failed_textures()`).
  User textures are sampled in their own uv, so the font atlas size check does not apply there.
- egui-wgpu's `Renderer` is read-locked only briefly; eframe write-locks it only while it paints,
  never across the egui pass, so this cannot deadlock.
*/

use std::collections::{HashMap, HashSet};

use egui::TextureId;
use egui::epaint::textures::{TextureOptions, TexturesDelta};
use egui::epaint::{ClippedPrimitive, ImageData, ImageDelta, Primitive};

use super::gpu::{egui_sampler_descriptor, egui_texture_entries};
use super::{LoggedOnce, Shared};
use crate::effect::TEXTURE_SLOTS;

/// The GPU copy of a kept texture.
struct KeptGpu {
    /// The copy (`Rgba8Unorm`, like egui-wgpu's).
    texture: wgpu::Texture,
    /// Its size in texels (the full delta's size).
    size: [u32; 2],
    /// Bind group for the mesh pipeline's group 1.
    bind_group: wgpu::BindGroup,
    /// How many of `Kept::deltas` have been written into `texture`.
    written: usize,
}

/// A texture uploaded in full that egui-wgpu has not applied yet.
struct Kept {
    /// egui-wgpu's texture for the id when the full upload was seen (`None`: not allocated).
    /// Once the renderer holds anything else, it has applied the upload and this entry is dropped.
    baseline: Option<wgpu::Texture>,
    /// The last full delta and every partial delta after it, in upload order (images are shared
    /// `Arc`s, so keeping them is cheap).
    deltas: Vec<ImageDelta>,
    /// Created on first use by a filter layer.
    gpu: Option<KeptGpu>,
}

/// Textures uploaded in full but not yet applied by egui-wgpu, and what binding them needs.
#[derive(Default)]
pub(super) struct TextureCopies {
    /// Same descriptor as egui-wgpu's texture bind-group layout (created on first use).
    layout: Option<wgpu::BindGroupLayout>,
    /// Samplers by egui texture options, created like egui-wgpu's.
    samplers: HashMap<TextureOptions, wgpu::Sampler>,
    /// Kept textures by id.
    kept: HashMap<TextureId, Kept>,
}

/// Index of the last delta of `deltas` that replaces the whole texture.
fn last_full_upload(deltas: &[ImageDelta]) -> Option<usize> {
    deltas.iter().rposition(|delta| delta.pos.is_none())
}

/// Size `[w, h]` of an image delta, if it fits a texture.
fn delta_size(delta: &ImageDelta) -> Option<[u32; 2]> {
    let [w, h] = delta.image.size();
    Some([u32::try_from(w).ok()?, u32::try_from(h).ok()?])
}

/// egui-wgpu's current `wgpu::Texture` for `id` (`None` when unknown or a user bind group).
fn renderer_texture(renderer: &egui_wgpu::Renderer, id: TextureId) -> Option<wgpu::Texture> {
    renderer.texture(&id).and_then(|texture| texture.texture.clone())
}

/// Writes `delta` into `texture` (`size` texels) at its position (the origin for a full delta).
/// Deltas that do not fit are skipped (they would be a wgpu validation error; egui never produces them).
fn write_delta(queue: &wgpu::Queue, texture: &wgpu::Texture, size: [u32; 2], delta: &ImageDelta) {
    let (Some([w, h]), [x, y]) = (delta_size(delta), delta.pos.unwrap_or([0, 0])) else {
        return;
    };
    let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
        return;
    };
    if w == 0 || h == 0 || x.saturating_add(w) > size[0] || y.saturating_add(h) > size[1] {
        return;
    }
    let ImageData::Color(image) = &delta.image;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&image.pixels),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

impl TextureCopies {
    /// Records one pass's texture delta. Must see every pass (also passes egui will discard), in
    /// order: a full upload in pass 1 is still missing from the renderer in pass 2.
    ///
    /// Drops kept textures egui-wgpu has applied since, starts keeping textures uploaded in full
    /// by `delta`, appends partial deltas to kept ones, and forgets freed ids.
    pub(super) fn observe(&mut self, state: &egui_wgpu::RenderState, delta: &TexturesDelta) {
        if self.kept.is_empty() && delta.set.iter().all(|(_, deltas)| last_full_upload(deltas).is_none()) {
            return;
        }
        let renderer = state.renderer.read();
        self.kept.retain(|id, kept| renderer_texture(&renderer, *id) == kept.baseline);
        for (id, deltas) in &delta.set {
            if let Some(start) = last_full_upload(deltas) {
                // A second full upload before egui-wgpu applied the first one keeps the first
                // baseline: the renderer still holds that texture.
                let baseline = self.kept.remove(id).map_or_else(|| renderer_texture(&renderer, *id), |kept| kept.baseline);
                self.kept.insert(*id, Kept { baseline, deltas: deltas[start..].to_vec(), gpu: None });
            } else if let Some(kept) = self.kept.get_mut(id) {
                kept.deltas.extend_from_slice(deltas);
            }
        }
        drop(renderer);
        for id in &delta.free {
            self.kept.remove(id);
        }
    }

    /// Bind group of the kept texture `id`, creating its GPU copy on first use and writing the
    /// partial deltas recorded since. `None` when `id` is not kept or its image is empty.
    fn kept_bind_group(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: TextureId) -> Option<wgpu::BindGroup> {
        let kept = self.kept.get_mut(&id)?;
        if kept.gpu.is_none() {
            let full = kept.deltas.first()?;
            let size = delta_size(full)?;
            if size[0] == 0 || size[1] == 0 {
                return None;
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("egui_filter_layers texture copy"),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let layout = self.layout.get_or_insert_with(|| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_filter_layers egui texture"), entries: &egui_texture_entries() }));
            // Like egui-wgpu, the sampler follows the full upload's options.
            let sampler = self.samplers.entry(full.options).or_insert_with(|| device.create_sampler(&egui_sampler_descriptor(full.options)));
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_filter_layers texture copy"),
                layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) }],
            });
            kept.gpu = Some(KeptGpu { texture, size, bind_group, written: 0 });
        }
        let gpu = kept.gpu.as_mut()?;
        // Writes queued now land before the frame's command buffers (queue writes precede the
        // submit), in upload order, so later glyphs overwrite earlier atlas contents correctly.
        for delta in kept.deltas.get(gpu.written..).unwrap_or_default() {
            write_delta(queue, &gpu.texture, gpu.size, delta);
        }
        gpu.written = kept.deltas.len();
        Some(gpu.bind_group.clone())
    }
}

/// Every texture the meshes of `primitives` sample.
pub(super) fn primitive_textures(primitives: &[ClippedPrimitive]) -> HashSet<TextureId> {
    primitives
        .iter()
        .filter_map(|primitive| match &primitive.primitive {
            Primitive::Mesh(mesh) => Some(mesh.texture_id),
            Primitive::Callback(_) => None,
        })
        .collect()
}

/// A bind group for every texture of `used`, or none for textures that cannot be resolved this
/// frame (see the file header). `font_size` is the CPU font atlas size.
pub(super) fn snapshot_textures(shared: &Shared, used: HashSet<TextureId>, font_size: [usize; 2]) -> HashMap<TextureId, wgpu::BindGroup> {
    let font_size = [u32::try_from(font_size[0]).unwrap_or(u32::MAX), u32::try_from(font_size[1]).unwrap_or(u32::MAX)];
    let state = &shared.render_state;
    let mut copies = shared.texture_copies();
    let mut textures = HashMap::with_capacity(used.len());
    let mut skipped = Vec::new();
    let renderer = state.renderer.read();
    for id in used {
        let bind_group = copies.kept_bind_group(&state.device, &state.queue, id).or_else(|| {
            let current = renderer.texture(&id)?;
            // Glyph UVs are normalised by the CPU atlas size: a renderer atlas of another size
            // (an upload this crate never observed) would sample the wrong texels.
            let stale_atlas = id == TextureId::default() && current.texture.as_ref().is_some_and(|gpu| [gpu.width(), gpu.height()] != font_size);
            (!stale_atlas).then(|| current.bind_group.clone())
        });
        match bind_group {
            Some(bind_group) => {
                textures.insert(id, bind_group);
            }
            None => skipped.push(id),
        }
    }
    drop(renderer);
    if !skipped.is_empty() && LoggedOnce::first(&shared.logged.skipped_texture) {
        log::debug!(
            "egui_filter_layers (wgpu): texture(s) {skipped:?} cannot be resolved this frame (unknown to the renderer, or a font \
             atlas whose size does not match the CPU atlas); meshes using them are left out of the filter backdrop for this frame. Logged once."
        );
    }
    textures
}

/// The bind group of every attached user texture of a layer (`ids[i]`), resolved like
/// [`snapshot_textures`] (the kept copy of a texture uploaded in full this frame, else egui-wgpu's
/// own bind group, native textures included). Unlike beneath meshes, an attached texture that
/// cannot be resolved is not skipped: the first such id is returned as the error (the caller
/// records it and does not plan the layer).
pub(super) fn resolve_user_textures(shared: &Shared, ids: [Option<TextureId>; TEXTURE_SLOTS]) -> Result<[Option<wgpu::BindGroup>; TEXTURE_SLOTS], TextureId> {
    if ids.iter().all(Option::is_none) {
        return Ok([const { None }; TEXTURE_SLOTS]);
    }
    let state = &shared.render_state;
    let mut copies = shared.texture_copies();
    let renderer = state.renderer.read();
    let mut resolved = [const { None }; TEXTURE_SLOTS];
    for (slot, id) in resolved.iter_mut().zip(ids) {
        let Some(id) = id else {
            continue;
        };
        let bind_group = copies.kept_bind_group(&state.device, &state.queue, id).or_else(|| renderer.texture(&id).map(|texture| texture.bind_group.clone()));
        *slot = Some(bind_group.ok_or(id)?);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Color32, ColorImage};

    fn delta(pos: Option<[usize; 2]>) -> ImageDelta {
        let image = ColorImage::new([2, 2], vec![Color32::RED; 4]);
        match pos {
            Some(pos) => ImageDelta::partial(pos, image, TextureOptions::LINEAR),
            None => ImageDelta::full(image, TextureOptions::LINEAR),
        }
    }

    #[test]
    fn only_whole_uploads_need_an_own_copy() {
        assert_eq!(last_full_upload(&[delta(Some([0, 0]))]), None);
        assert_eq!(last_full_upload(&[delta(None), delta(Some([1, 1])), delta(None), delta(Some([0, 0]))]), Some(2));
        assert_eq!(delta_size(&delta(None)), Some([2, 2]));
    }

}
