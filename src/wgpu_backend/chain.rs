/*
File: wgpu_backend/chain.rs

Purpose:
The ONE owner of how a `FilterChain` is executed on wgpu: given an input texture that holds the
capture (at its origin) and the layer's `LayerGeometry`, run every pass ping-pong into
intermediates and prepare the composite pass that writes the result, masked by the rounded rect,
into any render target. Holds no egui state: callers bring `&wgpu::Device`, `&wgpu::Queue` and a
`CommandEncoder`, so both the UI paint callback (`render.rs`) and a standalone runner on its own
device use exactly the same objects, formats, uniform layout and pass order.

Key structures:
- `ChainResources`: per-device objects (sampler, effect / composite layouts, intermediate format,
  uniform stride, effect pipeline cache); builds composite pipelines for any output format.
- `ChainTargets`: per-concurrent-run ping-pong intermediates and the uniform buffer.
- `ChainSource`, `ChainRecording`, `ChainJob`, `ChainWindow`: the inputs of one recorded run.

Key functions:
- `ChainResources::create()`, `effect_pipeline()`, `composite_pipeline()` (replace),
  `composite_over_pipelines()` (blended, `CompositeOver`), `record()`.
- `chain_uniforms()`: pure serialization of every pass's and the composite's uniform block.
- `draw_composite()` / `draw_composite_over()`: the composite draw(s) into a caller-owned render
  pass (replace: offline; blended over the pass's pixels: the UI).

Notes:
Uniform slot `i` (at `uniform_stride * i`) holds pass `i`; slot `n` (the pass count) the
composite. Pass `i` reads the source (i = 0) or pass `i - 1`'s output and writes
`ping_pong[i % 2]`. Textures are stored top-down on wgpu, so `flip_y` is false.
A run computes a WINDOW of the capture (`ChainWindow`): the UI always uses the whole capture; the
offline tiler runs one window per tile. Fragments derive their capture uv from their pixel
position (prelude frag mode 1), and every texture holds its window at its origin, so a pixel's
arithmetic does not depend on which window computed it.
*/

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use super::gpu::{GrowBuffer, Target, begin_pass, pass_pipeline, px_ratio, renderable_and_filterable, sampler_entry, texture_entry, texture_scale, uniform_entry, with_error_scope};
use crate::effect::{BINDING_ORIGINAL_TEXTURE, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_UNIFORMS, CompiledModule, composite_add_linear_module, composite_add_module, composite_erase_module, composite_linear_module, composite_module};
use crate::geometry::{LayerGeometry, PxRect, px_f32};
use crate::pass::{FrameUniforms, Params, Pass, UNIFORM_BLOCK_SIZE, pack_uniforms};

/// Blend of the blended composite's erase draw: `dst * (1 - src.a)` on colour and alpha (the
/// shader outputs `(0, 0, 0, coverage)`, so the source term is zero).
const ERASE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
};

/// Blend of the blended composite's add draw: `dst + src` on colour and alpha.
const ADD_BLEND: wgpu::BlendState = wgpu::BlendState { color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add }, alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add } };

/// The two pipelines of the blended composite (see `ChainResources::composite_over_pipelines`).
///
/// A single blended draw cannot be exact: `mix(dst, f, c)` needs `1 - c` as the dst factor and
/// `c * f.a` as the alpha addend, which differ when `f.a < 1`. A constant blend factor is per draw,
/// not per pixel. Dual-source blending would do it in one draw, but it is not universally
/// available (WebGL2 lacks it; it is an optional feature on WebGPU, Vulkan and GL), and the
/// composite must work wherever egui-wgpu does. Erase (`dst * (1 - c)`) then add (`+ c * f`) is exact, and with
/// `c = 1` the result is exactly `f`, as with the replace composite.
#[derive(Debug)]
pub(crate) struct CompositeOver {
    /// Scales the pass's pixels by `1 - coverage` (all four channels); discards `coverage == 0`.
    erase: wgpu::RenderPipeline,
    /// Adds `coverage * filtered` (linearised on `*Srgb` targets); discards `coverage == 0`.
    add: wgpu::RenderPipeline,
}

/// A compiled effect pipeline and the module it was built from (to evict it once dropped).
#[derive(Debug)]
struct CachedPipeline {
    /// Dies with the last `Effect` clone.
    module: Weak<CompiledModule>,
    /// The pipeline (rendering into the intermediate format).
    pipeline: wgpu::RenderPipeline,
}

/// Every device-wide object the chain executor needs; one per device (and per install in the UI).
#[derive(Debug)]
pub(crate) struct ChainResources {
    /// Whether intermediates are RGBA16F.
    float_intermediates: bool,
    /// Format of the ping-pong targets.
    intermediate_format: wgpu::TextureFormat,
    /// Byte distance between uniform blocks in a run's uniform buffer (device alignment).
    uniform_stride: u64,
    /// Linear, clamp-to-edge sampler for pass inputs.
    sampler: wgpu::Sampler,
    /// Layout of the effect passes' group 0.
    effect_layout: wgpu::BindGroupLayout,
    /// Pipeline layout shared by every effect pipeline.
    effect_pipeline_layout: wgpu::PipelineLayout,
    /// Layout of the composite pass's group 0.
    composite_layout: wgpu::BindGroupLayout,
    /// Pipeline layout of every composite pipeline.
    composite_pipeline_layout: wgpu::PipelineLayout,
    /// Effect pipelines by `CompiledModule::id`.
    pipelines: HashMap<u64, CachedPipeline>,
}

impl ChainResources {
    /// Builds the sampler and layouts, and picks the intermediate format (RGBA16F when the
    /// adapter can render to and filter it, else RGBA8).
    ///
    /// # Errors
    /// The device's validation message when it rejects any object.
    pub(crate) fn create(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Result<Self, String> {
        let float_intermediates = renderable_and_filterable(adapter, wgpu::TextureFormat::Rgba16Float);
        let intermediate_format = if float_intermediates { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8Unorm };
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let block = u64::try_from(UNIFORM_BLOCK_SIZE).unwrap_or(u64::MAX);
        let uniform_stride = block.div_ceil(alignment) * alignment;

        with_error_scope(device, || {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("egui_filter_layers sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let effect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_filter_layers effect"),
                entries: &[uniform_entry(BINDING_UNIFORMS, UNIFORM_BLOCK_SIZE), texture_entry(BINDING_SOURCE_TEXTURE), sampler_entry(BINDING_SAMPLER)],
            });
            let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_filter_layers composite"),
                entries: &[uniform_entry(BINDING_UNIFORMS, UNIFORM_BLOCK_SIZE), texture_entry(BINDING_SOURCE_TEXTURE), sampler_entry(BINDING_SAMPLER), texture_entry(BINDING_ORIGINAL_TEXTURE)],
            });
            let effect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_filter_layers effect"), bind_group_layouts: &[Some(&effect_layout)], immediate_size: 0 });
            let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_filter_layers composite"), bind_group_layouts: &[Some(&composite_layout)], immediate_size: 0 });
            Self { float_intermediates, intermediate_format, uniform_stride, sampler, effect_layout, effect_pipeline_layout, composite_layout, composite_pipeline_layout, pipelines: HashMap::new() }
        })
    }

    /// Whether intermediates are RGBA16F (else RGBA8, which quantizes every pass).
    #[must_use]
    pub(crate) fn float_intermediates(&self) -> bool {
        self.float_intermediates
    }

    /// Format of the ping-pong intermediates.
    #[must_use]
    pub(crate) fn intermediate_format(&self) -> wgpu::TextureFormat {
        self.intermediate_format
    }

    /// The pipeline of `module`, building and caching it on first use.
    ///
    /// # Errors
    /// The device's validation message when it rejects the shader or pipeline.
    pub(crate) fn effect_pipeline(&mut self, device: &wgpu::Device, module: &Arc<CompiledModule>) -> Result<wgpu::RenderPipeline, String> {
        if let Some(entry) = self.pipelines.get(&module.id) {
            return Ok(entry.pipeline.clone());
        }
        let pipeline = with_error_scope(device, || pass_pipeline(device, &self.effect_pipeline_layout, module, self.intermediate_format, None, wgpu::MultisampleState::default(), None))?;
        log::debug!("egui_filter_layers (wgpu): built pipeline for effect `{}`", module.label);
        self.pipelines.insert(module.id, CachedPipeline { module: Arc::downgrade(module), pipeline: pipeline.clone() });
        Ok(pipeline)
    }

    /// Drops pipelines of effects whose last clone was dropped.
    pub(crate) fn evict_dropped(&mut self) {
        self.pipelines.retain(|_, entry| entry.module.strong_count() > 0);
    }

    /// A REPLACE composite pipeline writing into `format` with the given multisample / depth state
    /// (they must match the render pass `draw_composite` records into). `*Srgb` formats get the
    /// linear output variant (the hardware re-encodes on write); the pipeline writes every pixel of
    /// the region (the original capture outside the mask), no blend. Exact only when the capture
    /// holds everything the target holds there (offline).
    ///
    /// # Errors
    /// A message when the crate's composite module is broken or the device rejects the pipeline.
    #[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // The replace composite serves the offline renderer only.
    pub(crate) fn composite_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat, multisample: wgpu::MultisampleState, depth_stencil: Option<wgpu::DepthStencilState>) -> Result<wgpu::RenderPipeline, String> {
        let composite = if format.is_srgb() { composite_linear_module() } else { composite_module() }.map_err(|error| format!("the composite pass does not compile (crate bug): {error}"))?;
        with_error_scope(device, || pass_pipeline(device, &self.composite_pipeline_layout, &composite, format, None, multisample, depth_stencil))
    }

    /// The two pipelines of the BLENDED composite into `format` (multisample / depth state as for
    /// [`Self::composite_pipeline`]): drawn in order by `draw_composite_over`, they turn every
    /// pixel into `mix(dst, filtered, coverage)` on all four channels and leave pixels outside the
    /// mask untouched. For targets that hold content the capture lacks (the UI pass: paint
    /// callbacks, lower filter layers). `*Srgb` formats blend in linear, like egui-wgpu does.
    ///
    /// # Errors
    /// As [`Self::composite_pipeline`].
    pub(crate) fn composite_over_pipelines(&self, device: &wgpu::Device, format: wgpu::TextureFormat, multisample: wgpu::MultisampleState, depth_stencil: Option<wgpu::DepthStencilState>) -> Result<CompositeOver, String> {
        let crate_bug = |error: crate::effect::EffectError| format!("the composite pass does not compile (crate bug): {error}");
        let erase = composite_erase_module().map_err(crate_bug)?;
        let add = if format.is_srgb() { composite_add_linear_module() } else { composite_add_module() }.map_err(crate_bug)?;
        with_error_scope(device, || CompositeOver {
            erase: pass_pipeline(device, &self.composite_pipeline_layout, &erase, format, Some(ERASE_BLEND), multisample, depth_stencil.clone()),
            add: pass_pipeline(device, &self.composite_pipeline_layout, &add, format, Some(ADD_BLEND), multisample, depth_stencil),
        })
    }

    /// Records every pass of `job` on `rec.encoder` and returns the composite bind group for
    /// `draw_composite`, or `None` when the job has no passes, `pipelines` does not match
    /// `passes` one to one, or the window is empty. Uniforms are uploaded with
    /// `rec.queue.write_buffer`, so they take effect at the next submission of `rec.queue`;
    /// `targets` must not be reused by another run before this run's composite has been drawn.
    /// Every pass renders `job.window.rect` (at its target's origin); pixels of the window
    /// within the chain's reach of a window edge that is not a capture edge are only as exact as
    /// the data the window holds there (the offline tiler overlaps windows by the padding).
    pub(crate) fn record(&self, rec: &mut ChainRecording<'_>, targets: &mut ChainTargets, job: &ChainJob<'_>) -> Option<wgpu::BindGroup> {
        let pass_count = job.passes.len();
        if pass_count == 0 || job.pipelines.len() != pass_count {
            return None;
        }
        let capture = job.geometry.capture.size().map(|extent| u32::try_from(extent).unwrap_or(1));
        let window = job.window.rect;
        let (Ok(window_w), Ok(window_h), Ok(window_x), Ok(window_y)) = (u32::try_from(window.w), u32::try_from(window.h), u32::try_from(window.x), u32::try_from(window.y)) else {
            return None;
        };
        if window_w == 0 || window_h == 0 {
            return None;
        }
        let needed = [window_w.max(job.window.min_target_size[0]), window_h.max(job.window.min_target_size[1])];
        for (index, slot) in targets.ping_pong.iter_mut().enumerate() {
            Target::ensure(slot, rec.device, self.intermediate_format, None, if index == 0 { "egui_filter_layers ping" } else { "egui_filter_layers pong" }, needed);
        }
        let [Some(first), Some(second)] = &targets.ping_pong else {
            return None;
        };
        let intermediates = [first, second];

        let stride = self.uniform_stride;
        let slot_bytes = usize::try_from(stride).unwrap_or(UNIFORM_BLOCK_SIZE);
        // wgpu textures are stored top-down, like the effect `uv`; fragments take their capture
        // uv from their pixel position, so any window of the capture computes identical pixels.
        let mut frame = FrameUniforms::for_layer(&job.geometry, job.time, false);
        frame.frag_from_position = true;
        let origin_in = |allocated: [u32; 2]| [px_ratio(window_x, allocated[0]), px_ratio(window_y, allocated[1])];
        let mapping = ChainMapping {
            source: TextureMapping { scale: texture_scale(capture, job.source.allocated), offset: origin_in(job.source.allocated) },
            intermediates: [TextureMapping { scale: first.scale(capture), offset: origin_in(first.size) }, TextureMapping { scale: second.scale(capture), offset: origin_in(second.size) }],
            pass_frag_offset: [px_f32(window.x), px_f32(window.y)],
            composite_frag_offset: [-px_f32(job.window.composite_origin[0]), -px_f32(job.window.composite_origin[1])],
        };
        let uniforms = chain_uniforms(frame, job.passes, &job.geometry.composite_params(), slot_bytes, mapping);
        let uniform_buffer = targets.uniforms.write(rec.device, rec.queue, &uniforms).clone();
        let uniform_at = |slot: usize| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniform_buffer, offset: stride * u64::try_from(slot).unwrap_or(0), size: std::num::NonZeroU64::new(u64::try_from(UNIFORM_BLOCK_SIZE).unwrap_or(0)) })
        };

        // Filter passes, ping-pong: pass `i` reads the source (i = 0) or pass `i - 1`'s output.
        let [w, h] = [px_f32(window.w), px_f32(window.h)];
        for (index, pipeline) in job.pipelines.iter().enumerate() {
            let input = if index == 0 { job.source.view } else { &intermediates[(index - 1) % 2].view };
            let bind_group = rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_filter_layers pass"),
                layout: &self.effect_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: BINDING_UNIFORMS, resource: uniform_at(index) },
                    wgpu::BindGroupEntry { binding: BINDING_SOURCE_TEXTURE, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: BINDING_SAMPLER, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = begin_pass(rec.encoder, &intermediates[index % 2].view, "egui_filter_layers filter pass", wgpu::Color::TRANSPARENT);
            pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Some(rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_filter_layers composite"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: BINDING_UNIFORMS, resource: uniform_at(pass_count) },
                wgpu::BindGroupEntry { binding: BINDING_SOURCE_TEXTURE, resource: wgpu::BindingResource::TextureView(&intermediates[(pass_count - 1) % 2].view) },
                wgpu::BindGroupEntry { binding: BINDING_SAMPLER, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: BINDING_ORIGINAL_TEXTURE, resource: wgpu::BindingResource::TextureView(job.source.view) },
            ],
        }))
    }
}

/// The ping-pong intermediates and uniform buffer of ONE chain run. A run's composite reads
/// them, so runs whose composites are pending at the same time need separate `ChainTargets`.
#[derive(Debug)]
pub(crate) struct ChainTargets {
    /// Ping-pong pass outputs (the `ChainResources` intermediate format), grow-only.
    ping_pong: [Option<Target>; 2],
    /// Every pass's `FlUniforms` and the composite's, at `uniform_stride` steps.
    uniforms: GrowBuffer,
}

impl ChainTargets {
    /// Empty targets; textures and the buffer are allocated on first use.
    pub(crate) fn new() -> Self {
        Self { ping_pong: [None, None], uniforms: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_filter_layers uniforms") }
    }
}

/// The chain's input: a texture whose top-left region holds the run's window of the capture.
/// It is the first pass's `src()` and the composite's original (shown outside the mask).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainSource<'a> {
    /// A filterable float view of the texture (single-sampled, 2D).
    pub view: &'a wgpu::TextureView,
    /// The texture's allocated size in pixels (may exceed the capture: grow-only textures).
    pub allocated: [u32; 2],
}

/// The GPU handles one chain run is recorded with.
pub(crate) struct ChainRecording<'a> {
    /// Device creating targets and bind groups.
    pub device: &'a wgpu::Device,
    /// Queue receiving the uniform upload.
    pub queue: &'a wgpu::Queue,
    /// Encoder the passes are recorded on.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Which part of the capture one run computes, and where its composite lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChainWindow {
    /// Capture-pixel rect the source texture holds at its origin and every pass renders at its
    /// target's origin. The whole capture (`0, 0, capture.w, capture.h`) without tiling.
    pub rect: PxRect,
    /// Where capture pixel `(0, 0)` lies in the composite's render target, in target pixels
    /// (the UI: the capture's screen position; an offline tile: minus its output origin).
    pub composite_origin: [i32; 2],
    /// Minimum allocation of the ping-pong targets (`[0, 0]`: just the window).
    pub min_target_size: [u32; 2],
}

impl ChainWindow {
    /// The whole capture of `geometry`, composited with capture pixel `(0, 0)` at
    /// `composite_origin` of the target.
    #[must_use]
    pub(crate) fn whole(geometry: &LayerGeometry, composite_origin: [i32; 2]) -> Self {
        Self { rect: PxRect { x: 0, y: 0, w: geometry.capture.w, h: geometry.capture.h }, composite_origin, min_target_size: [0, 0] }
    }
}

/// What one chain run computes.
pub(crate) struct ChainJob<'a> {
    /// The captured input (its window at the texture origin).
    pub source: ChainSource<'a>,
    /// Pixel geometry of the layer (capture size, UVs, mask) — always the WHOLE layer, also when
    /// a window of it is computed.
    pub geometry: LayerGeometry,
    /// The part of the capture computed by this run.
    pub window: ChainWindow,
    /// `fl.time`, seconds. (`fl.pixels_per_point` is `geometry.pixels_per_layer_point`.)
    pub time: f32,
    /// The passes (their parameters).
    pub passes: &'a [Pass],
    /// `passes[i]`'s pipeline from `ChainResources::effect_pipeline`, same order and length.
    pub pipelines: &'a [wgpu::RenderPipeline],
}

/// How capture uv maps into one texture's storage uv: `clamp(uv) * scale - offset` (the
/// prelude's `internal_*_scale` / `internal_*_offset`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TextureMapping {
    /// Capture size / allocated size.
    pub scale: [f32; 2],
    /// The window origin / allocated size (0 for a whole-capture window).
    pub offset: [f32; 2],
}

/// Storage mappings of the chain's textures and the fragment offsets of one run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainMapping {
    /// Of the source texture.
    pub source: TextureMapping,
    /// Of `ping_pong[0]` and `ping_pong[1]`.
    pub intermediates: [TextureMapping; 2],
    /// Added to a pass fragment's position (target pixels) to get capture pixels: the window origin.
    pub pass_frag_offset: [f32; 2],
    /// Same for the composite's fragments: minus `ChainWindow::composite_origin`.
    pub composite_frag_offset: [f32; 2],
}

/// Serializes the uniform blocks of one run: slot `i` = pass `i` (its `src_*` mapping is its
/// input's), slot `passes.len()` = the composite (`src_*` of the last pass output, `aux_*` of the
/// source, `composite_params`), each at a multiple of `slot_bytes` (>= the block size).
#[must_use]
pub(crate) fn chain_uniforms(mut frame: FrameUniforms, passes: &[Pass], composite_params: &Params, slot_bytes: usize, mapping: ChainMapping) -> Vec<u8> {
    let pass_count = passes.len();
    let mut uniforms = vec![0_u8; slot_bytes * (pass_count + 1)];
    frame.frag_offset = mapping.pass_frag_offset;
    for (index, pass) in passes.iter().enumerate() {
        let input = if index == 0 { mapping.source } else { mapping.intermediates[(index - 1) % 2] };
        frame.src_scale = input.scale;
        frame.src_offset = input.offset;
        let offset = slot_bytes * index;
        uniforms[offset..offset + UNIFORM_BLOCK_SIZE].copy_from_slice(&pack_uniforms(&frame, pass.params()));
    }
    let last = mapping.intermediates[pass_count.saturating_sub(1) % 2];
    frame.src_scale = last.scale;
    frame.src_offset = last.offset;
    frame.aux_scale = mapping.source.scale;
    frame.aux_offset = mapping.source.offset;
    frame.frag_offset = mapping.composite_frag_offset;
    let offset = slot_bytes * pass_count;
    uniforms[offset..offset + UNIFORM_BLOCK_SIZE].copy_from_slice(&pack_uniforms(&frame, composite_params));
    uniforms
}

/// Draws the REPLACE composite of a recorded run into `render_pass` over `region` (target
/// pixels, top-left origin). `pipeline` must come from `ChainResources::composite_pipeline` for
/// the pass's format / sample count / depth format, `bind_group` from `ChainResources::record`.
#[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // The replace composite serves the offline renderer only.
pub(crate) fn draw_composite(render_pass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup, region: PxRect) {
    render_pass.set_viewport(px_f32(region.x), px_f32(region.y), px_f32(region.w), px_f32(region.h), 0.0, 1.0);
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

/// Draws the BLENDED composite of a recorded run into `render_pass` over `region`: the erase
/// draw, then the add draw, with the same bind group (from `ChainResources::record`). Pixels the
/// rounded mask does not cover keep what the pass already holds there.
pub(crate) fn draw_composite_over(render_pass: &mut wgpu::RenderPass<'_>, pipelines: &CompositeOver, bind_group: &wgpu::BindGroup, region: PxRect) {
    render_pass.set_viewport(px_f32(region.x), px_f32(region.y), px_f32(region.w), px_f32(region.h), 0.0, 1.0);
    render_pass.set_bind_group(0, bind_group, &[]);
    // Order matters: the add draw's `+ c * f` must land on the already-scaled `dst * (1 - c)`.
    render_pass.set_pipeline(&pipelines.erase);
    render_pass.draw(0..3, 0..1);
    render_pass.set_pipeline(&pipelines.add);
    render_pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Effect;
    use crate::pass::{PARAM_SLOTS, offsets};

    /// Whether the two `f32`s at `offset` equal `expected` (all test values are exact in f32).
    fn holds(bytes: &[u8], offset: usize, expected: [f32; 2]) -> bool {
        let read = |at: usize| f32::from_ne_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        (read(offset) - expected[0]).abs() < f32::EPSILON && (read(offset + 4) - expected[1]).abs() < f32::EPSILON
    }

    #[test]
    fn uniform_slots_follow_the_ping_pong_order() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(0, [f32::from(u8::try_from(index).unwrap_or(0)); 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        let mut composite: Params = [[0.0; 4]; PARAM_SLOTS];
        composite[2] = [9.0; 4];
        let map = |scale: [f32; 2], offset: [f32; 2]| TextureMapping { scale, offset };
        let mapping = ChainMapping {
            source: map([0.5, 0.25], [0.125, 0.0]),
            intermediates: [map([0.75, 0.75], [0.25, 0.5]), map([1.0, 0.5], [0.0, 0.25])],
            pass_frag_offset: [16.0, 32.0],
            composite_frag_offset: [-4.0, -8.0],
        };
        let slot = 512;
        let bytes = chain_uniforms(FrameUniforms::default(), &passes, &composite, slot, mapping);
        assert_eq!(bytes.len(), slot * 4);
        // Pass 0 reads the source, pass 1 ping (0), pass 2 pong (1); composite reads pass 2's output (ping).
        assert!(holds(&bytes, offsets::SRC_SCALE, [0.5, 0.25]));
        assert!(holds(&bytes, slot + offsets::SRC_SCALE, [0.75, 0.75]));
        assert!(holds(&bytes, 2 * slot + offsets::SRC_SCALE, [1.0, 0.5]));
        assert!(holds(&bytes, 3 * slot + offsets::SRC_SCALE, [0.75, 0.75]));
        assert!(holds(&bytes, 3 * slot + offsets::AUX_SCALE, [0.5, 0.25]));
        assert!(holds(&bytes, 2 * slot + offsets::AUX_SCALE, [0.0, 0.0]), "passes keep the frame's aux scale");
        assert!(holds(&bytes, 2 * slot + offsets::PARAMS, [2.0, 2.0]));
        assert!(holds(&bytes, 3 * slot + offsets::PARAMS + 2 * 16, [9.0, 9.0]));
        // Offsets travel with their textures; fragments of passes vs the composite differ.
        assert!(holds(&bytes, offsets::SRC_OFFSET, [0.125, 0.0]));
        assert!(holds(&bytes, 2 * slot + offsets::SRC_OFFSET, [0.0, 0.25]));
        assert!(holds(&bytes, 3 * slot + offsets::SRC_OFFSET, [0.25, 0.5]));
        assert!(holds(&bytes, 3 * slot + offsets::AUX_OFFSET, [0.125, 0.0]));
        assert!(holds(&bytes, slot + offsets::FRAG_OFFSET, [16.0, 32.0]));
        assert!(holds(&bytes, 3 * slot + offsets::FRAG_OFFSET, [-4.0, -8.0]));
    }
}
