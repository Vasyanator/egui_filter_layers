/*
File: wgpu_backend/chain.rs

Purpose:
The ONE wgpu executor of a `FilterChain`: given an input texture that holds the capture (at its
origin) and the layer's `LayerGeometry`, it runs the draws of the backend-neutral `plan::ChainPlan`
(which slot each draw reads and writes, its viewport and uniform block are decided there, shared
with glow) and prepares the composite that writes the result, masked by the rounded rect, into any
render target. Holds no egui state: callers bring `&wgpu::Device`, `&wgpu::Queue` and a
`CommandEncoder`, so both the UI paint callback (`render.rs`) and a standalone runner on its own
device use exactly the same objects, formats, uniform layout and pass order.

Key structures:
- `ChainResources`: per-device objects (linear and nearest samplers, the ONE `@group(0)` bind
  group layout of every effect and composite pipeline, dummies for bindings a draw leaves unused,
  intermediate format, uniform stride, effect pipeline cache); builds composite pipelines for any
  output format.
- `ChainTargets`: per-concurrent-run ping-pong intermediates and the uniform buffer.
- `ChainSource`, `ChainRecording`, `ChainJob`, `ChainWindow`: the inputs of one recorded run.

Key functions:
- `ChainResources::create()`, `effect_pipeline()`, `composite_pipeline()` (replace),
  `composite_over_pipelines()` (blended, `CompositeOver`), `record()`.
- `frame_uniforms()`: the wgpu flags on the layer's `FrameUniforms` (top-down storage, position
  fragment mode).
- `chain_uniforms()`: the plan's uniform blocks laid out at the device's uniform stride.
- `draw_composite()` / `draw_composite_over()`: the composite draw(s) into a caller-owned render
  pass (replace: offline; blended over the pass's pixels: the UI).

Notes:
Uniform slot `i` (at `uniform_stride * i`) holds plan step `i`; slot `n` (the step count) the
composite. Every draw binds all eight `@group(0)` entries of the prelude's layout; a binding the
plan leaves empty gets a dummy (transparent texture, white coverage, a 16-byte zero buffer).
Textures are stored top-down on wgpu, so `flip_y` is false.
A run computes a WINDOW of the capture (`ChainWindow`): the UI always uses the whole capture; the
offline tiler runs one window per tile. Fragments derive their capture uv from their pixel
position (prelude frag mode 1), and every texture holds its window at its origin, so a pixel's
arithmetic does not depend on which window computed it.
*/

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use super::gpu::{GrowBuffer, Target, begin_pass, pass_pipeline, renderable_and_filterable, sampler_entry, texture_entry, uniform_entry, with_error_scope};
use crate::effect::{
    BINDING_COVERAGE_TEXTURE, BINDING_NEAREST_SAMPLER, BINDING_ORIGINAL_TEXTURE, BINDING_PARAMS, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, CompiledModule, composite_add_linear_module, composite_add_module,
    composite_erase_module, composite_linear_module, composite_module,
};
use crate::geometry::{LayerGeometry, PxRect, px_f32};
use crate::pass::{FrameUniforms, Pass, UNIFORM_BLOCK_SIZE};
use crate::plan::{Bindings, ChainPlan, Slot, SlotMappings, TextureMapping, composite_uniforms, plan, step_uniforms};

/// Size of the zero buffer bound at `BINDING_PARAMS` (the smallest valid uniform binding).
const DUMMY_PARAMS_SIZE: u64 = 16;

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
    /// Linear, clamp-to-edge sampler (`fl_sampler`).
    sampler: wgpu::Sampler,
    /// Nearest, clamp-to-edge sampler (`fl_nearest_sampler`).
    nearest_sampler: wgpu::Sampler,
    /// The `@group(0)` layout of the prelude, shared by every effect and composite pipeline.
    layout: wgpu::BindGroupLayout,
    /// Pipeline layout of every effect and composite pipeline.
    pipeline_layout: wgpu::PipelineLayout,
    /// Bound where a draw has no texture: 1x1 transparent (original, state).
    transparent: wgpu::TextureView,
    /// Bound where a draw has no mask coverage texture: 1x1 opaque white.
    white: wgpu::TextureView,
    /// Bound at `BINDING_PARAMS` when a module declares no block there: 16 zero bytes.
    dummy_params: wgpu::Buffer,
    /// Effect pipelines by `CompiledModule::id`.
    pipelines: HashMap<u64, CachedPipeline>,
}

impl ChainResources {
    /// Builds the samplers, the layout and the dummies (the white one is uploaded with
    /// `queue.write_texture`, i.e. before the next submission of `queue`), and picks the
    /// intermediate format (RGBA16F when the adapter can render to and filter it, else RGBA8).
    ///
    /// # Errors
    /// The device's validation message when it rejects any object.
    pub(crate) fn create(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter) -> Result<Self, String> {
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
            let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("egui_filter_layers nearest sampler"), ..Default::default() });
            // A layout may hold entries a shader does not use, so this one layout serves every
            // module; the params block is unsized (`None`): a module may declare any size there.
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_filter_layers group 0"),
                entries: &[
                    uniform_entry(BINDING_UNIFORMS, Some(UNIFORM_BLOCK_SIZE)),
                    texture_entry(BINDING_SOURCE_TEXTURE),
                    sampler_entry(BINDING_SAMPLER),
                    texture_entry(BINDING_ORIGINAL_TEXTURE),
                    uniform_entry(BINDING_PARAMS, None),
                    texture_entry(BINDING_STATE_TEXTURE),
                    texture_entry(BINDING_COVERAGE_TEXTURE),
                    sampler_entry(BINDING_NEAREST_SAMPLER),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_filter_layers group 0"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            // Zero-initialized by wgpu (textures and buffers are cleared before first use).
            let transparent = dummy_texture(device, "egui_filter_layers transparent dummy");
            let white_texture = dummy_texture(device, "egui_filter_layers white dummy");
            queue.write_texture(white_texture.as_image_copy(), &[u8::MAX; 4], wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) }, wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
            let dummy_params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("egui_filter_layers params dummy"), size: DUMMY_PARAMS_SIZE, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
            Self {
                float_intermediates,
                intermediate_format,
                uniform_stride,
                sampler,
                nearest_sampler,
                layout,
                pipeline_layout,
                transparent: transparent.create_view(&wgpu::TextureViewDescriptor::default()),
                white: white_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                dummy_params,
                pipelines: HashMap::new(),
            }
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
        let pipeline = with_error_scope(device, || pass_pipeline(device, &self.pipeline_layout, module, self.intermediate_format, None, wgpu::MultisampleState::default(), None))?;
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
        with_error_scope(device, || pass_pipeline(device, &self.pipeline_layout, &composite, format, None, multisample, depth_stencil))
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
            erase: pass_pipeline(device, &self.pipeline_layout, &erase, format, Some(ERASE_BLEND), multisample, depth_stencil.clone()),
            add: pass_pipeline(device, &self.pipeline_layout, &add, format, Some(ADD_BLEND), multisample, depth_stencil),
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
        if job.pipelines.len() != job.passes.len() {
            return None;
        }
        let window = job.window.rect;
        let plan = plan(job.passes.len(), window, job.window.composite_origin)?;
        // `plan` rejected empty and negative windows, so these conversions cannot fail.
        let (Ok(window_w), Ok(window_h), Ok(window_x), Ok(window_y)) = (u32::try_from(window.w), u32::try_from(window.h), u32::try_from(window.x), u32::try_from(window.y)) else {
            return None;
        };
        let needed = [window_w.max(job.window.min_target_size[0]), window_h.max(job.window.min_target_size[1])];
        for (index, slot) in targets.ping_pong.iter_mut().enumerate() {
            Target::ensure(slot, rec.device, self.intermediate_format, None, if index == 0 { "egui_filter_layers ping" } else { "egui_filter_layers pong" }, needed);
        }
        let [Some(first), Some(second)] = &targets.ping_pong else {
            return None;
        };

        let capture = job.geometry.capture.size().map(|extent| u32::try_from(extent).unwrap_or(1));
        let origin = [window_x, window_y];
        let maps = SlotMappings { source: TextureMapping::of(capture, job.source.allocated, origin), ping_pong: [TextureMapping::of(capture, first.size, origin), TextureMapping::of(capture, second.size, origin)] };
        let stride = self.uniform_stride;
        let slot_bytes = usize::try_from(stride).unwrap_or(UNIFORM_BLOCK_SIZE);
        let uniforms = chain_uniforms(&frame_uniforms(&job.geometry, job.time), &plan, job.passes, &job.geometry, &maps, slot_bytes);
        let uniform_buffer = targets.uniforms.write(rec.device, rec.queue, &uniforms).clone();
        let view_of = |slot: Slot| match slot {
            Slot::Source => job.source.view,
            Slot::Ping => &first.view,
            Slot::Pong => &second.view,
        };
        let bind_group = |slot: usize, bindings: Bindings| {
            let uniforms = wgpu::BufferBinding { buffer: &uniform_buffer, offset: stride * u64::try_from(slot).unwrap_or(0), size: std::num::NonZeroU64::new(u64::try_from(UNIFORM_BLOCK_SIZE).unwrap_or(0)) };
            let original = bindings.original.map_or(&self.transparent, view_of);
            rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_filter_layers draw"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: BINDING_UNIFORMS, resource: wgpu::BindingResource::Buffer(uniforms) },
                    wgpu::BindGroupEntry { binding: BINDING_SOURCE_TEXTURE, resource: wgpu::BindingResource::TextureView(view_of(bindings.src)) },
                    wgpu::BindGroupEntry { binding: BINDING_SAMPLER, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: BINDING_ORIGINAL_TEXTURE, resource: wgpu::BindingResource::TextureView(original) },
                    wgpu::BindGroupEntry { binding: BINDING_PARAMS, resource: self.dummy_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: BINDING_STATE_TEXTURE, resource: wgpu::BindingResource::TextureView(&self.transparent) },
                    wgpu::BindGroupEntry { binding: BINDING_COVERAGE_TEXTURE, resource: wgpu::BindingResource::TextureView(&self.white) },
                    wgpu::BindGroupEntry { binding: BINDING_NEAREST_SAMPLER, resource: wgpu::BindingResource::Sampler(&self.nearest_sampler) },
                ],
            })
        };

        for (index, step) in plan.steps.iter().enumerate() {
            let pipeline = job.pipelines.get(step.pass)?;
            let group = bind_group(index, step.bindings);
            let mut pass = begin_pass(rec.encoder, view_of(step.output), "egui_filter_layers filter pass", wgpu::Color::TRANSPARENT);
            pass.set_viewport(0.0, 0.0, px_f32(step.used_px[0]), px_f32(step.used_px[1]), 0.0, 1.0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        Some(bind_group(plan.steps.len(), plan.composite.bindings))
    }
}

/// A 1x1 `Rgba8Unorm` texture bound where a draw has no texture of its own.
fn dummy_texture(device: &wgpu::Device, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
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

/// The layer's `FrameUniforms` with the wgpu flags: textures are stored top-down, like the effect
/// `uv` (`flip_y` false), and fragments take their capture uv from their pixel position, so any
/// window of the capture computes identical pixels.
#[must_use]
pub(crate) fn frame_uniforms(geometry: &LayerGeometry, time: f32) -> FrameUniforms {
    let mut frame = FrameUniforms::for_layer(geometry, time, false);
    frame.frag_from_position = true;
    frame
}

/// The uniform blocks of one run laid out for one buffer: slot `i` (at `slot_bytes * i`, a
/// multiple >= the block size) = `plan.steps[i]`, slot `plan.steps.len()` = the composite. The
/// blocks themselves come from `plan.rs`.
#[must_use]
pub(crate) fn chain_uniforms(frame: &FrameUniforms, plan: &ChainPlan, passes: &[Pass], geometry: &LayerGeometry, maps: &SlotMappings, slot_bytes: usize) -> Vec<u8> {
    let step_count = plan.steps.len();
    let mut uniforms = vec![0_u8; slot_bytes * (step_count + 1)];
    for (index, step) in plan.steps.iter().enumerate() {
        // `plan` indexes `passes` it was built from; a mismatch leaves the slot zero.
        if let Some(pass) = passes.get(step.pass) {
            let offset = slot_bytes * index;
            uniforms[offset..offset + UNIFORM_BLOCK_SIZE].copy_from_slice(&step_uniforms(frame, step, maps, pass));
        }
    }
    let offset = slot_bytes * step_count;
    uniforms[offset..offset + UNIFORM_BLOCK_SIZE].copy_from_slice(&composite_uniforms(frame, plan, maps, geometry));
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

    /// Every step's block and the composite's land at their stride slot, byte-identical to the
    /// blocks `plan.rs` produces.
    #[test]
    fn uniform_slots_hold_the_plan_blocks_in_step_order() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(0, [f32::from(u8::try_from(index).unwrap_or(0)); 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        let window = PxRect { x: 0, y: 0, w: 40, h: 30 };
        let Some(plan) = plan(passes.len(), window, [5, 6]) else { panic!("planned") };
        let geometry = LayerGeometry { region: window, capture: window, rect_uv: [0.0, 0.0, 1.0, 1.0], region_uv: [0.0, 0.0, 1.0, 1.0], mask_rect_px: [0.0, 0.0, 40.0, 30.0], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let maps = SlotMappings { source: TextureMapping::of([40, 30], [64, 64], [0, 0]), ping_pong: [TextureMapping::of([40, 30], [40, 30], [0, 0]), TextureMapping::of([40, 30], [128, 32], [0, 0])] };
        let frame = frame_uniforms(&geometry, 0.5);
        let slot = 512;
        let bytes = chain_uniforms(&frame, &plan, &passes, &geometry, &maps, slot);
        assert_eq!(bytes.len(), slot * 4);
        for (index, step) in plan.steps.iter().enumerate() {
            assert_eq!(&bytes[slot * index..slot * index + UNIFORM_BLOCK_SIZE], &step_uniforms(&frame, step, &maps, &passes[index])[..], "slot {index}");
        }
        assert_eq!(&bytes[slot * 3..slot * 3 + UNIFORM_BLOCK_SIZE], &composite_uniforms(&frame, &plan, &maps, &geometry)[..]);
        assert!(bytes[UNIFORM_BLOCK_SIZE..slot].iter().all(|&byte| byte == 0), "stride padding stays zero");
    }
}
