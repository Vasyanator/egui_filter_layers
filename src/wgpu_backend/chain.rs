/*
File: wgpu_backend/chain.rs

Purpose:
The ONE wgpu executor of a `FilterChain`: given an input texture that holds the capture (at its
origin) and the layer's `LayerGeometry`, it runs the draws of the backend-neutral `plan::ChainPlan`
(which slot each draw reads and writes, its viewport and uniform block are decided there, shared
with glow) and prepares the composite that writes the result, masked by the rounded rect or a
rasterized shape (F9), into any
render target. Holds no egui state: callers bring `&wgpu::Device`, `&wgpu::Queue` and a
`CommandEncoder`, so both the UI paint callback (`render.rs`) and a standalone runner on its own
device use exactly the same objects, formats, uniform layout and pass order.

Key structures:
- `ChainResources`: per-device objects (the linear sampler, the ONE `@group(0)` bind group layout
  of every effect and composite pipeline, the user texture group layout (egui-wgpu's texture
  layout, groups 1..=3 of effect pipelines), dummies for bindings a draw leaves unused,
  intermediate format, uniform offset alignment, effect pipeline cache); builds composite
  pipelines for any output format.
- `ChainTargets`: per-concurrent-run ping-pong intermediates, the uniform buffer and the shape
  mask's coverage and region-id targets (F9, R1).
- `ChainSource`, `ChainRecording`, `ChainJob`, `ChainWindow`, `ChainState`: the inputs of one
  recorded run (`ChainState`: a stateful layer's two state textures, owned by the caller and kept
  across runs, F7).

Key functions:
- `ChainResources::create()`, `effect_pipeline()`, `composite_pipeline()` (replace),
  `composite_over_pipelines()` (blended, `CompositeOver`), `record()`.
- `frame_uniforms()`: the wgpu flags on the layer's `FrameUniforms` (top-down storage, position
  fragment mode).
- `chain_uniforms()`: the plan's blocks (binding 0 `fl`, binding 4 `p`) of every draw laid out in
  one buffer at the device's offset alignment (`ChainUniforms`, `DrawUniforms`).
- `draw_composite()` / `draw_composite_over()`: the composite draw(s) into a caller-owned render
  pass (replace: offline; blended over the pass's pixels: the UI).

Notes:
One uniform buffer per run holds, for draw `i` (plan step `i`; draw `n` = the composite), its
`fl` block and then its named-parameter block, each at an offset aligned to the device's
`min_uniform_buffer_offset_alignment` (`[fl_0, p_0, fl_1, p_1, …]`). Every draw binds all eight
`@group(0)` entries of the prelude's layout; a texture binding the plan leaves empty gets a dummy
(transparent texture, white coverage, white region ids = `REGION_NONE`). Effect pipelines have three more groups, the user texture
slots (`@group(1 + i)`): each step binds the caller's bind group of the attached texture (egui's
own in the UI, the renderer's in offline runs) where the plan says so, a dummy group elsewhere.
Composite pipelines read no user texture and use the group-0-only layout.
Shape mask (F9): a job with `coverage` (packed meshes, their transform, a texture group per
draw) plans `with_coverage`; `record` rasterizes it (`coverage.rs`, before the passes) into the
run's `ChainTargets` coverage texture and binds it at binding 6 of the composite only.
Regions (R1): a job whose coverage carries region ranges plans `with_regions`; the same step also
runs the id passes into the run's region-id texture, bound at binding 7 of every pass step.
State (F7): the caller owns a layer's two state textures (`ChainState`); `record` grows them to the
plan's size KEEPING their content (`Target::ensure_kept`, a copy recorded first), clears the one
the plan says, and maps `Slot::StateA` / `StateB` to them. A state pass's pipeline renders into
the state format, so the pipeline cache is keyed by module AND output format.
Textures are stored top-down on wgpu, so `flip_y` is false.
A run computes a WINDOW of the capture (`ChainWindow`): the UI always uses the whole capture; the
offline tiler runs one window per tile. Fragments derive their capture uv from their pixel
position (prelude frag mode 1), and every texture holds its window at its origin, so a pixel's
arithmetic does not depend on which window computed it.
*/

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use super::coverage::{CoverageJob, CoverageRasterizer, CoverageTargets};
use super::gpu::{GrowBuffer, Target, begin_pass, egui_texture_entries, pass_pipeline, renderable_and_filterable, sampler_entry, texture_entry, uniform_entry, with_error_scope};
use crate::effect::{
    BINDING_COVERAGE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_PARAMS, BINDING_REGION_TEXTURE, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, CompiledModule, TEXTURE_SLOTS, USER_SAMPLER_BINDING, USER_TEXTURE_BINDING, USER_TEXTURE_GROUP_BASE,
    composite_add_linear_module, composite_add_module, composite_erase_module, composite_linear_module, composite_module,
};
use crate::geometry::{LayerGeometry, PxRect, px_f32};
use crate::pass::{FrameUniforms, InputUniforms, Pass, UNIFORM_BLOCK_SIZE};
use crate::state::StateFormat;
use crate::plan::{Allocations, Bindings, ChainPlan, Slot, StatePlan, StatePlanInput, composite_params_block, composite_uniforms, params_block, plan, step_uniforms};

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
    /// Alignment of every block's offset in a run's uniform buffer
    /// (`min_uniform_buffer_offset_alignment`).
    uniform_alignment: u64,
    /// Linear, clamp-to-edge sampler (`fl_sampler`; also the dummy user group's sampler).
    sampler: wgpu::Sampler,
    /// The `@group(0)` layout of the prelude, shared by every effect and composite pipeline.
    layout: wgpu::BindGroupLayout,
    /// Layout of each user texture group (`@group(1..=3)`): egui-wgpu's texture layout.
    user_layout: wgpu::BindGroupLayout,
    /// Pipeline layout of every composite pipeline (`@group(0)` only).
    pipeline_layout: wgpu::PipelineLayout,
    /// Pipeline layout of every effect pipeline: `@group(0)` and the three user texture groups.
    effect_pipeline_layout: wgpu::PipelineLayout,
    /// Bound at a user texture group the plan leaves empty (transparent 1x1, linear sampler).
    dummy_user: wgpu::BindGroup,
    /// Bound where a draw has no texture: 1x1 transparent (state; an original the plan leaves empty).
    transparent: wgpu::TextureView,
    /// Bound where a draw has no mask coverage texture (full coverage) and no region-id texture
    /// (`.r = 1`: id 255, `REGION_NONE`): 1x1 opaque white.
    white: wgpu::TextureView,
    /// Effect pipelines by `CompiledModule::id` and output format (intermediate / state).
    pipelines: HashMap<(u64, wgpu::TextureFormat), CachedPipeline>,
    /// The shape-mask coverage rasterizer (F9).
    coverage: CoverageRasterizer,
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
        let uniform_alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));

        with_error_scope(device, || {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("egui_filter_layers sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            // A layout may hold entries a shader does not use, so this one layout serves every
            // module; the params block is unsized (`None`): a module may declare any size there
            // (wgpu checks the bound range against the pipeline's struct size at draw time).
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
                    texture_entry(BINDING_REGION_TEXTURE),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_filter_layers group 0"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            // egui-wgpu's texture layout, so egui's own per-texture bind groups bind here.
            let user_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_filter_layers user texture"), entries: &egui_texture_entries() });
            let effect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("egui_filter_layers effect"),
                bind_group_layouts: &[Some(&layout), Some(&user_layout), Some(&user_layout), Some(&user_layout)],
                immediate_size: 0,
            });
            // Zero-initialized by wgpu (textures and buffers are cleared before first use).
            let transparent = dummy_texture(device, "egui_filter_layers transparent dummy");
            let transparent = transparent.create_view(&wgpu::TextureViewDescriptor::default());
            let dummy_user = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_filter_layers empty user texture"),
                layout: &user_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: USER_TEXTURE_BINDING, resource: wgpu::BindingResource::TextureView(&transparent) },
                    wgpu::BindGroupEntry { binding: USER_SAMPLER_BINDING, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            let white_texture = dummy_texture(device, "egui_filter_layers white dummy");
            queue.write_texture(white_texture.as_image_copy(), &[u8::MAX; 4], wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) }, wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
            let white = white_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let coverage = CoverageRasterizer::create(device, &user_layout, &white, &sampler);
            Self {
                float_intermediates,
                intermediate_format,
                uniform_alignment,
                sampler,
                layout,
                user_layout,
                pipeline_layout,
                effect_pipeline_layout,
                dummy_user,
                transparent,
                white,
                pipelines: HashMap::new(),
                coverage,
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

    /// The pipeline of `module` rendering into `format` (the intermediate format for display
    /// passes, [`state_texture_format`] for state passes), building and caching it on first use.
    ///
    /// # Errors
    /// The device's validation message when it rejects the shader or pipeline.
    pub(crate) fn effect_pipeline(&mut self, device: &wgpu::Device, module: &Arc<CompiledModule>, format: wgpu::TextureFormat) -> Result<wgpu::RenderPipeline, String> {
        if let Some(entry) = self.pipelines.get(&(module.id, format)) {
            return Ok(entry.pipeline.clone());
        }
        let pipeline = with_error_scope(device, || pass_pipeline(device, &self.effect_pipeline_layout, module, format, None, wgpu::MultisampleState::default(), None))?;
        log::debug!("egui_filter_layers (wgpu): built pipeline for effect `{}` ({format:?})", module.label);
        self.pipelines.insert((module.id, format), CachedPipeline { module: Arc::downgrade(module), pipeline: pipeline.clone() });
        Ok(pipeline)
    }

    /// The pipelines of every pass of `passes`, in order: display passes render into the
    /// intermediate format, state passes into `state_format`. Stops at the first failure.
    ///
    /// # Errors
    /// The module whose pipeline the device rejected, and why.
    pub(crate) fn pass_pipelines(&mut self, device: &wgpu::Device, passes: &[Pass], state_format: StateFormat) -> Result<Vec<wgpu::RenderPipeline>, (Arc<CompiledModule>, String)> {
        let state_format = state_texture_format(state_format);
        passes
            .iter()
            .map(|pass| {
                let module = pass.effect().module();
                let format = if pass.effect().is_state() { state_format } else { self.intermediate_format };
                self.effect_pipeline(device, module, format).map_err(|error| (Arc::clone(module), error))
            })
            .collect()
    }

    /// Why `format` cannot hold a state on this device (`None`: it can). RGBA16F needs the
    /// adapter to render AND filter it — exactly when intermediates are RGBA16F.
    #[must_use]
    pub(crate) fn state_format_problem(&self, format: StateFormat) -> Option<String> {
        match format {
            StateFormat::Rgba16Float if !self.float_intermediates => Some("the adapter cannot render to and linearly filter Rgba16Float textures".to_owned()),
            StateFormat::Rgba16Float | StateFormat::Rgba8Unorm => None,
        }
    }

    /// The layout of a user texture group (egui-wgpu's texture layout: texture at
    /// `USER_TEXTURE_BINDING`, its sampler at `USER_SAMPLER_BINDING`). Callers that own their
    /// textures (offline) create their `ChainJob::user_textures` bind groups with it.
    #[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // The UI binds egui's own bind groups.
    pub(crate) fn user_layout(&self) -> &wgpu::BindGroupLayout {
        &self.user_layout
    }

    /// The shape-mask coverage rasterizer (its white texture group serves untextured meshes).
    #[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // Only offline runs need the white group.
    pub(crate) fn coverage(&self) -> &CoverageRasterizer {
        &self.coverage
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
    /// Every full-resolution pass renders `job.window.rect` (at its target's origin), a scaled
    /// pass the plan's smaller viewport; pixels of the window within the chain's reach of a
    /// window edge that is not a capture edge are only as exact as the data the window holds
    /// there (the offline tiler overlaps windows by the padding). A chain with a scaled pass over
    /// a partial window is not planned (`None`; tiling callers refuse such chains first).
    ///
    /// A stateful job (`job.state`) needs `state` (the layer's two state textures, grown here
    /// keeping their content); the returned [`StatePlan`] says where the latest state is afterwards.
    pub(crate) fn record(&self, rec: &mut ChainRecording<'_>, targets: &mut ChainTargets, job: &ChainJob<'_>, state: Option<&mut ChainState<'_>>) -> Option<(wgpu::BindGroup, Option<StatePlan>)> {
        if job.pipelines.len() != job.passes.len() {
            return None;
        }
        let attached = job.user_textures.map(|texture| texture.is_some());
        let plan = plan(job.passes, job.geometry.capture.size(), job.window.rect, job.window.composite_origin, attached, job.state)?;
        // A shape mask (F9): the plan's coverage step rasterizes it into the run's coverage
        // texture before any pass (it only depends on the mask), and the composite reads it. A
        // region mask (R1) also rasterizes the region ids there, which every pass step reads.
        let plan = match &job.coverage {
            Some(coverage) => plan.with_regions(u32::try_from(coverage.regions.len()).ok()?),
            None => plan,
        };
        let coverage_output = match (&job.coverage, plan.coverage.as_ref()) {
            (Some(coverage), Some(step)) => Some(self.coverage.record(rec, &mut targets.coverage, coverage, step)?),
            _ => None,
        };
        let coverage_target = coverage_output.map(|output| output.coverage);
        let region_target = coverage_output.and_then(|output| output.region);
        // The state textures: grown to the plan's size keeping the latest content, the empty one
        // cleared (both recorded before any pass of this run).
        let state_targets: Option<&[Option<Target>; 2]> = match (plan.state, state) {
            (Some(state_plan), Some(state)) => {
                let format = state_texture_format(state.format);
                for (index, slot) in state.targets.iter_mut().enumerate() {
                    Target::ensure_kept(slot, rec.device, rec.encoder, format, if index == 0 { "egui_filter_layers state A" } else { "egui_filter_layers state B" }, state_plan.target_px);
                }
                if let Some(clear) = state_plan.clear
                    && let Some(target) = state.targets.get(usize::from(clear == Slot::StateB)).and_then(Option::as_ref)
                {
                    drop(begin_pass(rec.encoder, &target.view, "egui_filter_layers state clear", wgpu::Color::TRANSPARENT));
                }
                Some(&*state.targets)
            }
            (Some(_), None) => return None,
            (None, _) => None,
        };
        let state_views = match state_targets {
            Some([Some(a), Some(b)]) => Some([a, b]),
            Some(_) => return None,
            None => None,
        };
        // Both ping-pong targets hold every step's viewport (the plan's pool size) at least.
        let needed = [plan.target_px[0].max(job.window.min_target_size[0]), plan.target_px[1].max(job.window.min_target_size[1])];
        for (index, slot) in targets.ping_pong.iter_mut().enumerate() {
            Target::ensure(slot, rec.device, self.intermediate_format, None, if index == 0 { "egui_filter_layers ping" } else { "egui_filter_layers pong" }, needed);
        }
        let [Some(first), Some(second)] = &targets.ping_pong else {
            return None;
        };

        let allocations = Allocations { source: job.source.allocated, ping_pong: [first.size, second.size], state: state_views.map_or([[0, 0]; 2], |[a, b]| [a.size, b.size]), coverage: coverage_target.map_or([0, 0], |target| target.size), region: region_target.map_or([0, 0], |target| target.size) };
        let alignment = usize::try_from(self.uniform_alignment).unwrap_or(UNIFORM_BLOCK_SIZE);
        let uniforms = chain_uniforms(&frame_uniforms(&job.geometry, job.time).with_input(&job.input), &plan, job.passes, &job.geometry, &allocations, alignment);
        let uniform_buffer = targets.uniforms.write(rec.device, rec.queue, &uniforms.bytes).clone();
        let view_of = |slot: Slot| match (slot, state_views) {
            (Slot::Source, _) => job.source.view,
            (Slot::Ping, _) => &first.view,
            (Slot::Pong, _) => &second.view,
            (Slot::StateA, Some([a, _])) => &a.view,
            (Slot::StateB, Some([_, b])) => &b.view,
            // No plan binds a state slot without `plan.state` (checked above): unreachable, and
            // the neutral dummy is what a stateless draw binds anyway.
            (Slot::StateA | Slot::StateB, None) => &self.transparent,
            // Bound only by a plan `with_coverage`, whose texture was recorded above; the white
            // dummy (full coverage) is what a draw without a shape mask binds anyway.
            (Slot::Coverage, _) => coverage_target.map_or(&self.white, |target| &target.view),
            // Bound only by a plan `with_regions`, whose texture was recorded above; the white
            // dummy (`REGION_NONE`) is what a draw without regions binds anyway.
            (Slot::RegionId, _) => region_target.map_or(&self.white, |target| &target.view),
        };
        let bind_group = |draw: &DrawUniforms, bindings: Bindings| {
            let fl = wgpu::BufferBinding { buffer: &uniform_buffer, offset: draw.fl.start, size: std::num::NonZeroU64::new(draw.fl.end - draw.fl.start) };
            let params = wgpu::BufferBinding { buffer: &uniform_buffer, offset: draw.params.start, size: std::num::NonZeroU64::new(draw.params.end - draw.params.start) };
            let original = bindings.original.map_or(&self.transparent, view_of);
            let state = bindings.state.map_or(&self.transparent, view_of);
            let coverage = bindings.coverage.map_or(&self.white, view_of);
            let region = bindings.region.map_or(&self.white, view_of);
            rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_filter_layers draw"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: BINDING_UNIFORMS, resource: wgpu::BindingResource::Buffer(fl) },
                    wgpu::BindGroupEntry { binding: BINDING_SOURCE_TEXTURE, resource: wgpu::BindingResource::TextureView(view_of(bindings.src)) },
                    wgpu::BindGroupEntry { binding: BINDING_SAMPLER, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: BINDING_ORIGINAL_TEXTURE, resource: wgpu::BindingResource::TextureView(original) },
                    wgpu::BindGroupEntry { binding: BINDING_PARAMS, resource: wgpu::BindingResource::Buffer(params) },
                    wgpu::BindGroupEntry { binding: BINDING_STATE_TEXTURE, resource: wgpu::BindingResource::TextureView(state) },
                    wgpu::BindGroupEntry { binding: BINDING_COVERAGE_TEXTURE, resource: wgpu::BindingResource::TextureView(coverage) },
                    wgpu::BindGroupEntry { binding: BINDING_REGION_TEXTURE, resource: wgpu::BindingResource::TextureView(region) },
                ],
            })
        };

        for (step, draw) in plan.steps.iter().zip(&uniforms.draws) {
            let pipeline = job.pipelines.get(step.pass)?;
            let group = bind_group(draw, step.bindings);
            let mut pass = begin_pass(rec.encoder, view_of(step.output), "egui_filter_layers filter pass", wgpu::Color::TRANSPARENT);
            pass.set_viewport(0.0, 0.0, px_f32(step.used_px[0]), px_f32(step.used_px[1]), 0.0, 1.0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            for ((group, bind), texture) in (USER_TEXTURE_GROUP_BASE..).zip(step.bindings.user).zip(job.user_textures) {
                // The plan binds a slot only where the caller attached a bind group (`attached`
                // came from `job.user_textures`), so `texture` is `Some` whenever `bind` holds.
                pass.set_bind_group(group, texture.filter(|_| bind).unwrap_or(&self.dummy_user), &[]);
            }
            pass.draw(0..3, 0..1);
        }
        Some((bind_group(uniforms.draws.get(plan.steps.len())?, plan.composite.bindings), plan.state))
    }
}

/// The wgpu format of a state texture.
#[must_use]
pub(crate) fn state_texture_format(format: StateFormat) -> wgpu::TextureFormat {
    match format {
        StateFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        StateFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// A stateful layer's state textures for one run: the two targets (kept by the caller across
/// runs; `record` grows them) and their format.
#[derive(Debug)]
pub(crate) struct ChainState<'a> {
    /// `Slot::StateA` / `Slot::StateB`.
    pub targets: &'a mut [Option<Target>; 2],
    /// Their storage format.
    pub format: StateFormat,
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
    /// Every draw's `fl` and named-parameter blocks (`chain_uniforms`).
    uniforms: GrowBuffer,
    /// The shape-mask coverage texture and mesh buffers (F9).
    coverage: CoverageTargets,
}

impl ChainTargets {
    /// Empty targets; textures and the buffer are allocated on first use.
    pub(crate) fn new() -> Self {
        Self { ping_pong: [None, None], uniforms: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_filter_layers uniforms"), coverage: CoverageTargets::new() }
    }
}

/// The chain's input: a texture whose top-left region holds the run's window of the capture.
/// It is the first pass's `src()` and every draw's original (`original()`; the composite shows it
/// outside the mask). No draw of the run renders into it.
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
    /// `fl.delta_time`, `fl.frame`, `fl.pointer`, resolved for `geometry`'s capture.
    pub input: InputUniforms,
    /// The passes (their parameters).
    pub passes: &'a [Pass],
    /// `passes[i]`'s pipeline from `ChainResources::effect_pipeline`, same order and length.
    pub pipelines: &'a [wgpu::RenderPipeline],
    /// The bind group of the texture attached at each user texture slot, in the layout of
    /// [`ChainResources::user_layout`] (egui's own bind groups in the UI); `None` = nothing attached.
    pub user_textures: [Option<&'a wgpu::BindGroup>; TEXTURE_SLOTS],
    /// The layer's state (stateful chains; `record` then needs the `ChainState` too).
    pub state: Option<StatePlanInput>,
    /// A shape mask (F9): its meshes are rasterized into the run's coverage texture (and, for a
    /// region mask, R1, its region ids into the region-id texture); `None` = the rounded-rect
    /// mask of `geometry`.
    pub coverage: Option<CoverageJob<'a>>,
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

/// Where one draw's two uniform blocks lie in a run's uniform buffer (byte ranges).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DrawUniforms {
    /// The `fl` block (binding 0), `UNIFORM_BLOCK_SIZE` bytes.
    pub fl: std::ops::Range<u64>,
    /// The named-parameter block (binding 4), a multiple of 16 bytes.
    pub params: std::ops::Range<u64>,
}

/// The contents of one run's uniform buffer and where each draw's blocks lie.
#[derive(Clone, Debug)]
pub(crate) struct ChainUniforms {
    /// The buffer contents (padding between blocks is zero).
    pub bytes: Vec<u8>,
    /// `draws[i]` = plan step `i`; the last entry is the composite.
    pub draws: Vec<DrawUniforms>,
}

/// Lays out the blocks `plan.rs` produces for every draw of one run in one buffer: per draw, the
/// `fl` block then the named-parameter block, each at an offset that is a multiple of
/// `alignment` (the device's `min_uniform_buffer_offset_alignment`; values below 1 count as 1).
#[must_use]
pub(crate) fn chain_uniforms(frame: &FrameUniforms, plan: &ChainPlan, passes: &[Pass], geometry: &LayerGeometry, allocations: &Allocations, alignment: usize) -> ChainUniforms {
    let alignment = alignment.max(1);
    let mut bytes = Vec::new();
    // Appends `block` at the next aligned offset, returning its byte range.
    let mut push = |block: &[u8]| {
        let start = bytes.len().div_ceil(alignment) * alignment;
        bytes.resize(start, 0);
        bytes.extend_from_slice(block);
        let to_u64 = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
        to_u64(start)..to_u64(bytes.len())
    };
    let mut draws = Vec::with_capacity(plan.steps.len() + 1);
    for step in &plan.steps {
        // `plan` indexes the `passes` it was built from; a mismatch would be a crate bug, and the
        // draw then gets zero blocks of the right sizes rather than another pass's values.
        let (fl, params) = match passes.get(step.pass) {
            Some(pass) => (push(&step_uniforms(frame, plan, step, allocations, pass)), push(params_block(pass))),
            None => (push(&[0; UNIFORM_BLOCK_SIZE]), push(composite_params_block())),
        };
        draws.push(DrawUniforms { fl, params });
    }
    let fl = push(&composite_uniforms(frame, plan, allocations, geometry));
    let params = push(composite_params_block());
    draws.push(DrawUniforms { fl, params });
    ChainUniforms { bytes, draws }
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

    /// Every draw's `fl` block and named-parameter block land at aligned offsets, byte-identical
    /// to the blocks `plan.rs` produces, in the order `[fl_0, p_0, fl_1, p_1, …, fl_c, p_c]`.
    #[test]
    fn uniform_blocks_hold_the_plan_blocks_in_draw_order() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let named = Effect::new("named", "struct Params { color: vec4<f32>, lights: array<vec4<f32>, 20> }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return p.color + p.lights[3]; }");
        let Ok(named) = named else { panic!("{named:?}") };
        let mut passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(0, [f32::from(u8::try_from(index).unwrap_or(0)); 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        passes[1] = Pass::new(&named).with("color", [1.0, 2.0, 3.0, 4.0]).unwrap_or_else(|error| panic!("{error}"));
        let window = PxRect { x: 0, y: 0, w: 40, h: 30 };
        let Some(plan) = plan(&passes, window.size(), window, [5, 6], [false; TEXTURE_SLOTS], None) else { panic!("planned") };
        let geometry = LayerGeometry { region: window, capture: window, rect_uv: [0.0, 0.0, 1.0, 1.0], region_uv: [0.0, 0.0, 1.0, 1.0], mask_rect_px: [0.0, 0.0, 40.0, 30.0], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let allocations = Allocations { source: [64, 64], ping_pong: [[40, 30], [128, 32]], state: [[0, 0]; 2], coverage: [0, 0], region: [0, 0] };
        let frame = frame_uniforms(&geometry, 0.5);
        let alignment = 256;
        let uniforms = chain_uniforms(&frame, &plan, &passes, &geometry, &allocations, alignment);
        assert_eq!(uniforms.draws.len(), 4);
        let at = |range: &std::ops::Range<u64>| &uniforms.bytes[usize::try_from(range.start).unwrap_or(0)..usize::try_from(range.end).unwrap_or(0)];
        let mut previous_end = 0;
        for (index, draw) in uniforms.draws.iter().enumerate() {
            assert!(draw.fl.start % 256 == 0 && draw.params.start % 256 == 0, "draw {index}: aligned {draw:?}");
            assert!(draw.fl.start >= previous_end && draw.params.start >= draw.fl.end, "draw {index}: ordered, disjoint {draw:?}");
            previous_end = draw.params.end;
            let expected_fl = match plan.steps.get(index) {
                Some(step) => step_uniforms(&frame, &plan, step, &allocations, &passes[step.pass]),
                None => composite_uniforms(&frame, &plan, &allocations, &geometry),
            };
            assert_eq!(at(&draw.fl), &expected_fl[..], "draw {index}: fl block");
            let expected_params = plan.steps.get(index).map_or(composite_params_block(), |step| params_block(&passes[step.pass]));
            assert_eq!(at(&draw.params), expected_params, "draw {index}: params block");
        }
        assert_eq!(uniforms.draws[1].params.end - uniforms.draws[1].params.start, 16 + 20 * 16, "the named block keeps its layout size");
        assert_eq!(uniforms.draws[0].params.end - uniforms.draws[0].params.start, 16, "no Params: the empty block");
        assert!(uniforms.bytes[UNIFORM_BLOCK_SIZE..512].iter().all(|&byte| byte == 0), "alignment padding stays zero");
    }
}
