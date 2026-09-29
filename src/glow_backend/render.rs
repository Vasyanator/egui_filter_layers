/*
File: glow_backend/render.rs

Purpose:
One shader layer, executed inside an `egui_glow` paint callback: capture the backdrop from the
bound framebuffer, run the draws of the backend-neutral `plan::ChainPlan` (passes ping-pong in
offscreen targets, then the composite back into the caller's framebuffer). Which slot a draw reads
and writes, its viewport and its uniform blocks (`fl` at binding point 0, the named-parameter
block at binding point 1) come from `plan.rs` (shared with wgpu); this file maps slots to GL
objects, binds them on the texture units of `glsl.rs`, uploads the blocks and draws.

Key structures:
- `CallerState`: the framebuffer bindings (and sample count) egui left bound, saved first.
- `LayerOutcome`: what a layer run ended with, applied to `GlowState` by `paint_layer`.

Key functions:
- `paint_layer()`: saves the caller's bindings, runs the layer, applies the outcome, restores.

Notes (GL state contract, checked against egui_glow 0.36.2 `painter.rs`):
- egui_glow paints into the framebuffer bound when the callback starts (`intermediate_fbo()` is
  `None` today, but an integration may paint into its own FBO). Both DRAW and READ bindings are
  saved BEFORE anything else — resource creation and the RGBA16F probe rebind framebuffer 0 —
  and restored on every exit path by `paint_layer`.
- After every callback egui_glow re-runs `prepare_painting` (scissor ON, blend + func, colour
  mask, viewport, its program, TEXTURE0 active, its VAO, element buffer), so this file may change
  those freely. It does NOT restore the framebuffer, the other units' textures, renderbuffer or
  uniform-buffer bindings, so those are reset here (`cleanup`). No sampler object is ever bound:
  every texture samples through its own parameters (user textures through the ones egui_glow
  set from their `TextureOptions`).
- Shape masks (F9): the plan's coverage step runs right after the capture (`coverage.rs`: the
  mask's meshes into the coverage target, an additive union); only the composite binds it (unit 3).
  Its GL errors are the crate's own work (latch).
- User textures (F8) are egui_glow's own texture objects (resolved by `mod.rs` through
  `Painter::texture`), bound on units `USER_UNIT_BASE + i` for the draws the plan says; they are
  only read, never modified.
- The scissor box is set by egui to the callback's clip rect BEFORE the callback and left
  untouched here; scissor is disabled for the offscreen work (it also clips blits) and re-enabled
  for the composite so the write-back respects the clip rect.
- Capture always uses `glBlitFramebuffer` (copying an RGB default framebuffer into an RGBA
  texture with `glCopyTexSubImage2D` is illegal on GLES 3). The read buffer's format is queried
  (`ReadFormat`; cached for framebuffer 0): only normalized fixed-point buffers can be blitted
  into the RGBA8 capture. A multisampled framebuffer is first resolved into a renderbuffer of
  the SAME internal format with identical source/destination rects — both required by GLES 3 /
  WebGL2 for multisample blits — then blitted (single-sample, format-converting) into the
  capture. A buffer with no matching format skips only the layers painted into it.
- Persistent state (F7): a stateful layer's two state targets live in `GlowState::states` (keyed by
  viewport and id); they are grown keeping their content (`RenderTarget::ensure_kept`, a blit)
  and the plan's empty one cleared before the passes, and the entry records where the latest state
  is after a clean run (any other outcome forgets the content: the next run starts from zero).
  RGBA16F state needs the float intermediates probe to have passed (renderable; half-float
  filtering is core in GL 3.3 / GLES 3.0), else the layer is refused (`GlowState::fail_state`).
- GL errors after the passes are attributed to the effects: with one distinct effect in the
  chain it is disabled directly; otherwise the chain's effects become "suspects" and later runs
  check after every pass until the culprit is found. Errors in capture / composite (the crate's
  own work) latch the whole backend.
*/

use std::collections::HashSet;

use glow::HasContext as _;

use super::coverage::CoverageResources;
use super::{GlowState, StateTargets};
use super::glsl::{GROUP0_TEXTURE_BINDINGS, GlslTarget, USER_UNIT_BASE};
use super::resources::{GlError, PARAMS_BINDING_POINT, ParamsBuffer, ReadFormat, ReadFormatError, RenderTarget, ResolveTarget, Resources, TargetFormat, UNIFORM_BINDING_POINT};
use crate::backend::LayerJob;
use crate::effect::{BINDING_COVERAGE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_REGION_TEXTURE, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, TEXTURE_SLOTS};
use crate::geometry::{LayerGeometry, MaskTransform, PxRect, layer_geometry, mask_transform, transform_scale};
use crate::mask::PackedCoverage;
use crate::pass::{FrameUniforms, InputUniforms, UNIFORM_BLOCK_SIZE};
use crate::plan::{Allocations, Bindings, ChainPlan, Slot, StatePlan, StatePlanInput, composite_params_block, coverage_locals, composite_uniforms, params_block, plan, step_uniforms};
use crate::state::StateFormat;

/// Stand-in key for rejections that have no known format (no colour buffer, or a driver that
/// cannot report it), so they are still logged only once.
const UNKNOWN_FORMAT: ReadFormat = ReadFormat { bits: [0; 4], component_type: glow::NONE, encoding: glow::NONE };

/// GL state of the caller (egui) that this module changes and must hand back.
#[derive(Clone, Copy, Debug)]
struct CallerState {
    /// Draw framebuffer bound when the callback started (egui's target).
    draw: Option<glow::Framebuffer>,
    /// Read framebuffer bound when the callback started.
    read: Option<glow::Framebuffer>,
    /// `GL_SAMPLES` of `draw`, queried while it is still bound.
    samples: i32,
}

impl CallerState {
    /// Reads the current bindings. Must run before ANY framebuffer is bound by this module.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn save(gl: &glow::Context) -> Self {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); only state queries follow.
        unsafe {
            Self {
                draw: gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING),
                read: gl.get_parameter_framebuffer(glow::READ_FRAMEBUFFER_BINDING),
                samples: gl.get_parameter_i32(glow::SAMPLES),
            }
        }
    }

    /// Re-binds the saved framebuffers.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn restore(&self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the bindings restored are the caller's own, saved by `save` on this context.
        unsafe {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, self.draw);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, self.read);
        }
    }
}

/// `egui_glow`'s textures one layer samples, resolved by the paint callback (`mod.rs`).
#[derive(Clone, Copy, Debug)]
pub(super) struct LayerTextures<'a> {
    /// The texture of every attached user texture slot (`None` where none is attached).
    pub user: [Option<glow::Texture>; TEXTURE_SLOTS],
    /// A shape mask (F9): its packed meshes and the texture of each of its draws.
    pub coverage: Option<(&'a PackedCoverage, &'a [glow::Texture])>,
}

/// The per-layer inputs every GL step below needs.
struct LayerRun<'a> {
    /// Pixel geometry of the layer.
    geometry: LayerGeometry,
    /// The job being painted.
    job: &'a LayerJob,
    /// The job's per-frame input resolved for `geometry`'s capture (`fl.pointer` in capture uv).
    input: InputUniforms,
    /// Screen size in physical pixels.
    screen: [i32; 2],
    /// The caller's bindings (the capture source and composite destination is `caller.draw`).
    caller: CallerState,
    /// Whether the context is GLES / WebGL (affects how the default read buffer is named).
    embedded: bool,
    /// `egui_glow`'s texture of every attached user texture slot (resolved by the caller).
    user_textures: [Option<glow::Texture>; TEXTURE_SLOTS],
    /// A shape mask (F9): its packed meshes, `egui_glow`'s texture of each draw, and the
    /// layer-points -> capture-pixels transform.
    coverage: Option<(&'a PackedCoverage, &'a [glow::Texture], MaskTransform)>,
}

/// How one layer run ended; applied to `GlowState` by [`paint_layer`].
#[derive(Debug)]
enum LayerOutcome {
    /// Composited successfully; a stateful run says where the latest state now is.
    Done(Option<StatePlan>),
    /// A crate-owned step failed: latch the whole backend.
    Latch {
        /// The step, for the log.
        what: &'static str,
        /// The error.
        error: String,
    },
    /// The framebuffer cannot be captured; skip this layer (logged once per format).
    Rejected {
        /// The rejected read format, the key for "logged once".
        format: ReadFormat,
        /// Why.
        reason: String,
    },
    /// Pass `index` raised `error`: disable its effect.
    PassFailed {
        /// Index into `job.passes`.
        index: usize,
        /// The GL error.
        error: u32,
    },
    /// The passes raised `error` but several effects could have caused it.
    PassesFailed {
        /// The GL error.
        error: u32,
    },
}

/// Discards errors raised before this callback, logging them at debug level, so the error checks
/// below only see this layer's own. Bounded: some drivers keep one flag per error kind.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn drain_errors(gl: &glow::Context) {
    for _ in 0..8 {
        // SAFETY: forwarded contract.
        let error = unsafe { gl.get_error() };
        if error == glow::NO_ERROR {
            return;
        }
        log::debug!("egui_shader_layers (glow): discarding a GL error raised before the filter callback: 0x{error:04x}");
    }
}

/// Runs one shader layer. Never panics; failures are latched in `state`. The caller's framebuffer
/// bindings are saved before anything else and restored on every path.
///
/// `textures` are `egui_glow`'s textures of the job's attached user textures and of its shape
/// mask's draws (resolved by the caller).
///
/// # Safety
/// `gl` must be the current context on this thread, called from inside an `egui_glow` paint
/// callback (so egui restores its own state afterwards); every texture of `textures` is alive on it.
pub(super) unsafe fn paint_layer(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob, textures: LayerTextures<'_>) {
    if state.failure.is_some() || state.destroyed || job.passes.is_empty() {
        return;
    }
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract), inside an egui_glow paint callback (this fn's contract).
    unsafe {
        // FIRST, before resource creation / the RGBA16F probe, which bind framebuffer 0.
        let caller = CallerState::save(gl);
        paint_saved(gl, state, target, info, job, caller, textures);
        caller.restore(gl);
    }
}

/// [`paint_layer`] after the caller's state was saved; may return from anywhere.
///
/// # Safety
/// As [`paint_layer`].
unsafe fn paint_saved(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob, caller: CallerState, textures: LayerTextures<'_>) {
    let LayerTextures { user: user_textures, coverage } = textures;
    // The callback rect is the layer rect after any layer transform; the transform scale joins
    // pixels-per-point in `LayerGeometry::pixels_per_layer_point` (padding, radii, `fl.pixels_per_point`).
    let transform_scale = transform_scale(job.rect, info.viewport);
    let Some(geometry) = layer_geometry(info.viewport, info.clip_rect, info.pixels_per_point, info.screen_size_px, job.padding_points, job.mask.corner_radius(), transform_scale) else {
        return;
    };

    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the objects used belong to `state`, were created on this same context, and are alive until `destroy`.
    unsafe {
        state.evict_dropped(gl);
        if state.resources.is_none() {
            match Resources::create(gl, target) {
                Ok(resources) => {
                    log::info!("egui_shader_layers (glow): resources created ({target:?}, float intermediates: {})", resources.float_intermediates());
                    state.resources = Some(resources);
                }
                Err(error) => {
                    state.latch("creating GPU resources", &error);
                    return;
                }
            }
        }

        // Build (or fetch) every pass program BEFORE touching the framebuffer, so a failing
        // effect skips the layer without leaving a half-processed region behind.
        let mut programs = Vec::with_capacity(job.passes.len());
        for pass in &job.passes {
            match state.program_for(gl, target, pass.effect().module()) {
                Some(program) => programs.push(program),
                None => return,
            }
        }
        // The mask rasterizer is created with the first shape-masked layer (a failure is a crate
        // bug in the mesh shader or a driver refusal: latched like the composite's).
        if coverage.is_some()
            && let Some(resources) = state.resources.as_mut()
            && resources.coverage.is_none()
        {
            match CoverageResources::create(gl, target) {
                Ok(created) => resources.coverage = Some(created),
                Err(error) => {
                    state.latch("creating the shape-mask rasterizer", &error);
                    return;
                }
            }
        }
        let diagnose = job.passes.iter().any(|pass| state.suspects.contains(&pass.effect().module().id));
        let state_key = job.state_key();
        let state_format = job.state.format();
        let float_targets = state.resources.as_ref().is_some_and(Resources::float_intermediates);
        if state_key.is_some() && state_format == StateFormat::Rgba16Float && !float_targets {
            state.fail_state(state_format, "RGBA16F is not renderable on this context (GLES / WebGL2 without EXT_color_buffer_float)".to_owned());
            return;
        }
        let Some(mut resources) = state.resources.take() else {
            return;
        };
        let screen = [i32::try_from(info.screen_size_px[0]).unwrap_or(i32::MAX), i32::try_from(info.screen_size_px[1]).unwrap_or(i32::MAX)];
        let input = job.frame_input.resolve(info.pixels_per_point, geometry.capture);
        let coverage = coverage.map(|(packed, textures)| (packed, textures, mask_transform(job.rect, info.viewport, info.pixels_per_point, geometry.capture)));
        let run = LayerRun { geometry, job, input, screen, caller, embedded: target != GlslTarget::Desktop330, user_textures, coverage };

        // Unpainted states are freed at every paint (GL is current here), then this layer's
        // entry is looked up (a format change replaces its targets, freed too).
        let (frame, time) = (job.frame_input.frame, f64::from(job.time));
        GlowState::free_state_targets(gl, state.states.evict(job.viewport, frame, time));
        let mut entry = state_key.map(|key| state.states.begin(key, state_format, frame, time, job.claim_reset(), || [None, None]));
        if let Some((_, _, replaced)) = entry.as_mut() {
            GlowState::free_state_targets(gl, replaced.take());
        }
        drain_errors(gl);
        let outcome = {
            let mut layer_state = entry.as_mut().map(|(entry, run, _)| LayerState {
                targets: &mut entry.targets,
                format: if state_format == StateFormat::Rgba16Float { TargetFormat::Rgba16F } else { TargetFormat::Rgba8 },
                input: StatePlanInput { scale: job.state.scale(), current: run.current, previous_px: run.previous, step: run.step },
            });
            run_layer(gl, &mut resources, &run, &programs, diagnose, layer_state.as_mut())
        };
        if let Some((entry, run, _)) = entry {
            match &outcome {
                LayerOutcome::Done(Some(state_plan)) => entry.finish(run, frame, state_plan.final_index, state_plan.final_px),
                // A stateful run always plans its state; without a plan, or not run cleanly, the
                // content is unknown: start over from zero.
                LayerOutcome::Done(None) | LayerOutcome::Latch { .. } | LayerOutcome::Rejected { .. } | LayerOutcome::PassFailed { .. } | LayerOutcome::PassesFailed { .. } => entry.invalidate(),
            }
        }
        cleanup(gl);
        apply_outcome(gl, state, &mut resources, job, outcome);
        state.resources = Some(resources);
    }
}

/// Applies a [`LayerOutcome`] to the backend state (latches, effect failures, suspects).
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn apply_outcome(gl: &glow::Context, state: &mut GlowState, resources: &mut Resources, job: &LayerJob, outcome: LayerOutcome) {
    let modules = || job.passes.iter().map(|pass| pass.effect().module());
    match outcome {
        LayerOutcome::Done(_) => {
            // A clean run of the whole chain clears its suspects.
            for module in modules() {
                state.suspects.remove(&module.id);
            }
        }
        LayerOutcome::Latch { what, error } => state.latch(what, &error),
        LayerOutcome::Rejected { format, reason } => {
            if resources.rejected_formats.insert(format) {
                log::error!(
                    "egui_shader_layers (glow): shader layers cannot read this framebuffer and are skipped on it.\n\
                     Error: {reason}\n\
                     Effect: layers painted into this framebuffer draw nothing extra; other framebuffers are unaffected.\n\
                     Possible cause: a float / integer / unusual colour format (e.g. an HDR or RGB565 multisampled surface)."
                );
            }
        }
        LayerOutcome::PassFailed { index, error } => {
            if let Some(pass) = job.passes.get(index) {
                // SAFETY: forwarded contract.
                unsafe { state.fail_effect(gl, pass.effect().module(), "running on the GPU", &format!("GL error 0x{error:04x}")) };
            }
        }
        LayerOutcome::PassesFailed { error } => {
            let distinct: HashSet<u64> = modules().map(|module| module.id).collect();
            if let (1, Some(module)) = (distinct.len(), modules().next()) {
                // SAFETY: forwarded contract.
                unsafe { state.fail_effect(gl, module, "running on the GPU", &format!("GL error 0x{error:04x}")) };
            } else {
                log::warn!("egui_shader_layers (glow): a filter chain raised GL error 0x{error:04x}; checking its passes one by one on the next frames to find the effect responsible");
                state.suspects.extend(distinct);
            }
        }
    }
}

/// A stateful layer's state for one run: its two targets (kept in `GlowState::states`), their
/// format and what the registry knows of them.
struct LayerState<'a> {
    /// `Slot::StateA` / `Slot::StateB`.
    targets: &'a mut StateTargets,
    /// Storage format.
    format: TargetFormat,
    /// The plan's input.
    input: StatePlanInput,
}

/// Capture, passes and composite of one layer. Leaves bindings for `paint_layer` to restore.
/// A stateful job needs `layer_state`; its targets are grown (keeping content) and cleared here.
///
/// # Safety
/// `gl` must be current on this thread; `programs` matches `run.job.passes`.
unsafe fn run_layer(gl: &glow::Context, resources: &mut Resources, run: &LayerRun<'_>, programs: &[glow::Program], diagnose: bool, layer_state: Option<&mut LayerState<'_>>) -> LayerOutcome {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `resources` and `programs` were created on this context and are alive, and `programs` matches `run.job.passes` (caller contract).
    unsafe {
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::BLEND);

        match capture(gl, resources, run) {
            Ok(()) => {}
            Err(CaptureError::Rejected { format, reason }) => return LayerOutcome::Rejected { format, reason },
            Err(CaptureError::Gl(error)) => return LayerOutcome::Latch { what: "allocating the capture target", error: error.to_string() },
        }
        let capture_error = gl.get_error();
        if capture_error != glow::NO_ERROR {
            return LayerOutcome::Latch {
                what: "copying the backdrop out of the framebuffer",
                error: format!("GL error 0x{capture_error:04x} (framebuffer samples: {})", run.caller.samples),
            };
        }

        let capture = run.geometry.capture;
        // glow always computes the whole capture (fragment mode 0 interpolates over the viewport);
        // the composite's origin is the capture's screen position, as on the wgpu UI.
        let attached = run.user_textures.map(|texture| texture.is_some());
        let state_input = layer_state.as_ref().map(|state| state.input);
        let Some(plan) = plan(&run.job.passes, capture.size(), PxRect { x: 0, y: 0, w: capture.w, h: capture.h }, [capture.x, capture.y], attached, state_input) else {
            return LayerOutcome::Done(None);
        };
        let plan = if run.coverage.is_some() { plan.with_coverage() } else { plan };
        // Both ping-pong targets hold every step's viewport (the plan's pool size).
        let needed = plan.target_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
        let max = resources.max_size;
        for target in &mut resources.ping_pong {
            if let Err(error) = target.ensure_size(gl, needed, max) {
                return LayerOutcome::Latch { what: "allocating an intermediate target", error: error.to_string() };
            }
        }
        // State targets: grown keeping the latest content, the empty one cleared.
        let states: Option<&StateTargets> = match (plan.state, layer_state) {
            (Some(state_plan), Some(layer_state)) => {
                let needed = state_plan.target_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
                for slot in layer_state.targets.iter_mut() {
                    if let Err(error) = RenderTarget::ensure_kept(slot, gl, layer_state.format, needed, max) {
                        return LayerOutcome::Latch { what: "allocating a state target", error: error.to_string() };
                    }
                }
                if let Some(clear) = state_plan.clear
                    && let Some(target) = layer_state.targets.get(usize::from(clear == Slot::StateB)).and_then(Option::as_ref)
                {
                    target.clear(gl);
                }
                Some(&*layer_state.targets)
            }
            (Some(_), None) => return LayerOutcome::Done(None),
            (None, _) => None,
        };
        // The shape mask's coverage (F9), before the passes (it depends only on the mask).
        if let (Some(step), Some((packed, textures, transform))) = (plan.coverage, run.coverage) {
            let Some(coverage) = resources.coverage.as_mut() else {
                return LayerOutcome::Latch { what: "rasterizing the mask", error: "the mask rasterizer is missing (crate bug)".to_owned() };
            };
            if let Err(error) = coverage.draw(gl, packed, textures, &coverage_locals(&transform, &step), step.used_px, max) {
                return LayerOutcome::Latch { what: "allocating the mask coverage target", error: error.to_string() };
            }
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return LayerOutcome::Latch { what: "rasterizing the mask", error: format!("GL error 0x{error:04x}") };
            }
        }
        match run_passes(gl, resources, states, run, &plan, programs, diagnose) {
            Ok(Some((index, error))) => return LayerOutcome::PassFailed { index, error },
            Ok(None) => {}
            Err(error) => return LayerOutcome::Latch { what: "uploading a named-parameter block", error: error.to_string() },
        }
        let pass_error = gl.get_error();
        if pass_error != glow::NO_ERROR {
            // Nothing was written to the caller's framebuffer yet: skipping the composite leaves
            // the backdrop unfiltered for this layer.
            return LayerOutcome::PassesFailed { error: pass_error };
        }

        if let Err(error) = composite(gl, resources, states, run, &plan) {
            return LayerOutcome::Latch { what: "uploading the composite's parameter block", error: error.to_string() };
        }
        let error = gl.get_error();
        if error != glow::NO_ERROR {
            return LayerOutcome::Latch { what: "compositing the filtered region", error: format!("GL error 0x{error:04x}") };
        }
        LayerOutcome::Done(plan.state)
    }
}

/// Why [`capture`] did not copy the backdrop.
#[derive(Debug)]
enum CaptureError {
    /// The framebuffer's format cannot be captured; skip the layer.
    Rejected {
        /// The offending format.
        format: ReadFormat,
        /// Why.
        reason: String,
    },
    /// A GL object operation of ours failed; latch the backend.
    Gl(GlError),
}

/// The format of the caller's framebuffer read buffer (bound as READ by the caller of this).
///
/// Cached for framebuffer 0, whose format is fixed for the context's lifetime. `None` when the
/// driver could not answer: the capture then proceeds as if the buffer were RGBA8 (the only
/// option left; a real mismatch then surfaces as a capture GL error).
///
/// # Errors
/// A rejection when the framebuffer has no readable colour buffer.
///
/// # Safety
/// `gl` must be current on this thread with `run.caller.draw` bound as READ framebuffer.
unsafe fn read_format(gl: &glow::Context, resources: &mut Resources, run: &LayerRun<'_>) -> Result<Option<ReadFormat>, CaptureError> {
    let default_framebuffer = run.caller.draw.is_none();
    if default_framebuffer && let Some(format) = resources.default_read_format {
        return Ok(Some(format));
    }
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract), with `run.caller.draw` bound as READ framebuffer (caller contract).
    match unsafe { ReadFormat::query(gl, default_framebuffer, run.embedded) } {
        Ok(format) => {
            if default_framebuffer {
                log::debug!("egui_shader_layers (glow): default framebuffer read format {format:?} (samples: {})", run.caller.samples);
                resources.default_read_format = Some(format);
            }
            Ok(Some(format))
        }
        Err(ReadFormatError::NoColorBuffer(reason)) => {
            Err(CaptureError::Rejected { format: UNKNOWN_FORMAT, reason })
        }
        Err(ReadFormatError::Query(error)) => {
            log::debug!("egui_shader_layers (glow): the driver could not report the framebuffer format (GL error 0x{error:04x}); assuming RGBA8");
            Ok(None)
        }
    }
}

/// Copies the capture rect of the caller's framebuffer into `resources.capture` at `(0, 0)`.
///
/// # Errors
/// [`CaptureError::Rejected`] for a read buffer that cannot be captured (checked BEFORE any
/// blit, so no GL error is raised); [`CaptureError::Gl`] when a target cannot be allocated.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn capture(gl: &glow::Context, resources: &mut Resources, run: &LayerRun<'_>) -> Result<(), CaptureError> {
    let rect = run.geometry.capture;
    let screen = run.screen;
    let src = gl_rect(rect, screen[1]);
    let max = resources.max_size;
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `resources` were created on this context and are alive, and `run.caller` holds framebuffer names saved from this same context.
    unsafe {
        resources.capture.ensure_size(gl, rect.size(), max).map_err(CaptureError::Gl)?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, run.caller.draw);
        let format = read_format(gl, resources, run)?;
        if let Some(format) = format {
            if resources.rejected_formats.contains(&format) {
                return Err(CaptureError::Rejected { format, reason: "previously rejected".to_owned() });
            }
            if !format.is_normalized() {
                return Err(CaptureError::Rejected { format, reason: format!("the framebuffer is not normalized fixed-point ({format:?}), so it cannot be copied into an 8-bit capture") });
            }
        }
        if run.caller.samples > 0 {
            // GLES 3 / WebGL2: a multisample resolve needs identical rects and an identical
            // internal format on the draw side.
            let internal_format = match format.map(|format| format.resolve_format()) {
                Some(Ok(internal_format)) => internal_format,
                Some(Err(reason)) => return Err(CaptureError::Rejected { format: format.unwrap_or(UNKNOWN_FORMAT), reason }),
                None => glow::RGBA8,
            };
            if resources.resolve.is_none() {
                resources.resolve = Some(ResolveTarget::create(gl).map_err(CaptureError::Gl)?);
                log::debug!("egui_shader_layers (glow): framebuffer is multisampled ({} samples); resolving through a 0x{internal_format:04x} renderbuffer", run.caller.samples);
            }
            let Some(resolve) = resources.resolve.as_mut() else {
                return Ok(());
            };
            if let Err(error) = resolve.ensure(gl, internal_format, screen, max) {
                return Err(CaptureError::Rejected { format: format.unwrap_or(UNKNOWN_FORMAT), reason: error.to_string() });
            }
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, run.caller.draw);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(resolve.framebuffer));
            gl.blit_framebuffer(src[0], src[1], src[2], src[3], src[0], src[1], src[2], src[3], glow::COLOR_BUFFER_BIT, glow::NEAREST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(resolve.framebuffer));
        }
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(resources.capture.framebuffer));
        gl.blit_framebuffer(src[0], src[1], src[2], src[3], 0, 0, rect.w, rect.h, glow::COLOR_BUFFER_BIT, glow::NEAREST);
    }
    Ok(())
}

/// `[x0, y0, x1, y1]` of a top-left-origin rect in GL's bottom-left framebuffer coordinates.
fn gl_rect(rect: PxRect, screen_h: i32) -> [i32; 4] {
    let y0 = rect.gl_y(screen_h);
    [rect.x, y0, rect.x + rect.w, y0 + rect.h]
}

/// Uploads one draw's uniform block and binds the buffer to the `fl` binding point.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn upload_uniforms(gl: &glow::Context, buffer: glow::Buffer, bytes: &[u8; UNIFORM_BLOCK_SIZE]) {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `buffer` is our live uniform buffer, allocated with `UNIFORM_BLOCK_SIZE` bytes, exactly the length of `bytes`.
    unsafe {
        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(buffer));
        gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, bytes);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, Some(buffer));
    }
}

/// Uploads one draw's named-parameter block (`plan::params_block`, a multiple of 16 bytes and at
/// least the program's block size) at offset 0 of the grow-only params buffer and binds the whole
/// buffer to [`PARAMS_BINDING_POINT`]. The buffer is re-allocated (`glBufferData`) only when the
/// block outgrows it (`ParamsBuffer::grown_capacity`), otherwise written with `glBufferSubData`;
/// a buffer larger than the program's block is valid on GL 3.3, GLES 3 and WebGL2.
///
/// # Errors
/// [`GlError::CreateResource`] when the block does not fit `glBufferData`'s `i32` size (never for
/// the 16 KiB-capped blocks `Effect::new` accepts).
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn upload_params(gl: &glow::Context, buffer: &mut ParamsBuffer, bytes: &[u8]) -> Result<(), GlError> {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `buffer.buffer` is our live params buffer, allocated with at least `bytes.len()` bytes before the sub-data write.
    unsafe {
        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(buffer.buffer));
        if let Some(capacity) = ParamsBuffer::grown_capacity(buffer.capacity, bytes.len()) {
            let size = i32::try_from(capacity).map_err(|_| GlError::CreateResource { resource: "uniform buffer", detail: format!("a {capacity}-byte named-parameter block exceeds GL's size range") })?;
            gl.buffer_data_size(glow::UNIFORM_BUFFER, size, glow::DYNAMIC_DRAW);
            buffer.capacity = capacity;
        }
        gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, bytes);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, PARAMS_BINDING_POINT, Some(buffer.buffer));
    }
    Ok(())
}

/// The layer's `FrameUniforms` with the GL flags: texture storage is bottom-up (the prelude flips
/// so `uv` stays top-down) and fragments interpolate their uv over the viewport (mode 0).
fn frame_uniforms(geometry: &LayerGeometry, time: f32) -> FrameUniforms {
    FrameUniforms::for_layer(geometry, time, true)
}

/// The GL target a plan slot names, or `None` for a state slot without state targets (no plan
/// binds one: a state slot exists only with `plan.state`, whose targets `run_layer` created) and
/// for the region-id slot (glow runs no region plan yet).
fn target_of<'a>(resources: &'a Resources, states: Option<&'a StateTargets>, slot: Slot) -> Option<&'a RenderTarget> {
    match slot {
        Slot::Source => Some(&resources.capture),
        Slot::Ping => Some(&resources.ping_pong[0]),
        Slot::Pong => Some(&resources.ping_pong[1]),
        Slot::StateA => states.and_then(|states| states[0].as_ref()),
        Slot::StateB => states.and_then(|states| states[1].as_ref()),
        Slot::Coverage => resources.coverage.as_ref().map(|coverage| &coverage.target),
        // No region-id target on glow yet: region masks are refused before any run
        // (`GlowState::refuse_regions`), so no plan here binds this slot.
        Slot::RegionId => None,
    }
}

/// The allocated sizes of the capture, the ping-pong and the state targets (the plan maps each
/// step's content, held at its target's `(0, 0)` corner, through them).
fn allocations(resources: &Resources, states: Option<&StateTargets>) -> Allocations {
    let state = |index: usize| states.and_then(|states| states[index].as_ref()).map_or([0, 0], RenderTarget::allocated_px);
    let coverage = resources.coverage.as_ref().map_or([0, 0], |coverage| coverage.target.allocated_px());
    Allocations { source: resources.capture.allocated_px(), ping_pong: [resources.ping_pong[0].allocated_px(), resources.ping_pong[1].allocated_px()], state: [state(0), state(1)], coverage, region: [0, 0] }
}

/// Binds every texture of one draw on its texture units (`glsl.rs`): the `@group(0)` textures on
/// units `0..5` (the plan's slots where it names one, the dummies elsewhere), and user texture
/// slot `i` on unit `USER_UNIT_BASE + i` (the attached texture where the plan binds it, the
/// transparent dummy elsewhere). Leaves `TEXTURE0` active.
///
/// # Safety
/// `gl` must be current on this thread; `resources` and `user_textures` alive on it.
unsafe fn bind_textures(gl: &glow::Context, resources: &Resources, states: Option<&StateTargets>, bindings: Bindings, user_textures: &[Option<glow::Texture>; TEXTURE_SLOTS]) {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every texture belongs to `resources` or is one of egui_glow's live user textures (caller contract).
    unsafe {
        for ((unit, bind), texture) in (USER_UNIT_BASE..).zip(bindings.user).zip(user_textures) {
            // The plan binds a slot only where a texture is attached (`attached` came from
            // `user_textures`), so `texture` is `Some` whenever `bind` holds.
            let texture = texture.filter(|_| bind).unwrap_or(resources.transparent);
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        }
        // A slot the plan binds always has a target (see `target_of`); the dummies stand only for
        // bindings the plan leaves empty.
        let texture_of = |slot: Option<Slot>| slot.and_then(|slot| target_of(resources, states, slot)).map_or(resources.transparent, |target| target.texture);
        for (unit, binding) in (0_u32..).zip(GROUP0_TEXTURE_BINDINGS) {
            let texture = match binding {
                BINDING_SOURCE_TEXTURE => texture_of(Some(bindings.src)),
                BINDING_ORIGINAL_TEXTURE => texture_of(bindings.original),
                BINDING_STATE_TEXTURE => texture_of(bindings.state),
                // Full coverage (white) unless the plan binds a shape mask's coverage.
                BINDING_COVERAGE_TEXTURE => bindings.coverage.and_then(|slot| target_of(resources, states, slot)).map_or(resources.white, |target| target.texture),
                // White = id 255 = `REGION_NONE` unless the plan binds a region-id texture.
                BINDING_REGION_TEXTURE => bindings.region.and_then(|slot| target_of(resources, states, slot)).map_or(resources.white, |target| target.texture),
                // `GROUP0_TEXTURE_BINDINGS` holds nothing else.
                _ => resources.transparent,
            };
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        }
        gl.active_texture(glow::TEXTURE0);
    }
}

/// Runs every pass step of `plan` into its ping-pong or state target (`target_of`).
///
/// With `diagnose`, checks `glGetError` after every pass and stops at the first failing one,
/// returning `(pass index, error)`; otherwise the caller checks once after all passes.
///
/// # Safety
/// `gl` must be current on this thread; targets must be at least `plan.target_px`; `programs`
/// matches `run.job.passes`; no GL error may be pending on entry.
unsafe fn run_passes(gl: &glow::Context, resources: &mut Resources, states: Option<&StateTargets>, run: &LayerRun<'_>, plan: &ChainPlan, programs: &[glow::Program], diagnose: bool) -> Result<Option<(usize, u32)>, GlError> {
    let allocations = allocations(resources, states);
    let frame = frame_uniforms(&run.geometry, run.job.time).with_input(&run.input);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); targets are at least the plan's pool size and `programs` matches `run.job.passes` (caller contract); all objects are alive on this context.
    unsafe {
        gl.bind_vertex_array(Some(resources.vertex_array));
        for step in &plan.steps {
            // `plan` was built from `run.job.passes`, which `programs` matches one to one.
            let (Some(pass), Some(program), Some(output)) = (run.job.passes.get(step.pass), programs.get(step.pass), target_of(resources, states, step.output)) else {
                continue;
            };
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(output.framebuffer));
            gl.viewport(0, 0, step.used_px[0], step.used_px[1]);
            gl.use_program(Some(*program));
            bind_textures(gl, resources, states, step.bindings, &run.user_textures);
            upload_uniforms(gl, resources.uniform_buffer, &step_uniforms(&frame, plan, step, &allocations, pass));
            upload_params(gl, &mut resources.params, params_block(pass))?;
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            if diagnose {
                let error = gl.get_error();
                if error != glow::NO_ERROR {
                    return Ok(Some((step.pass, error)));
                }
            }
        }
    }
    Ok(None)
}

/// Writes the last pass output back over the layer region of the caller's framebuffer, masked
/// by the rounded rect (outside the mask the original capture is written, i.e. unchanged).
///
/// # Errors
/// As [`upload_params`].
///
/// # Safety
/// `gl` must be current on this thread; the passes of `plan` have run.
unsafe fn composite(gl: &glow::Context, resources: &mut Resources, states: Option<&StateTargets>, run: &LayerRun<'_>, plan: &ChainPlan) -> Result<(), GlError> {
    let geometry = &run.geometry;
    let bytes = composite_uniforms(&frame_uniforms(geometry, run.job.time).with_input(&run.input), plan, &allocations(resources, states), geometry);
    let region = gl_rect(geometry.region, run.screen[1]);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the passes ran (caller contract), so the plan's composite source holds their output; all objects are alive on this context.
    unsafe {
        run.caller.restore(gl);
        gl.viewport(region[0], region[1], geometry.region.w, geometry.region.h);
        gl.enable(glow::SCISSOR_TEST);
        gl.use_program(Some(resources.composite));
        bind_textures(gl, resources, states, plan.composite.bindings, &run.user_textures);
        upload_uniforms(gl, resources.uniform_buffer, &bytes);
        upload_params(gl, &mut resources.params, composite_params_block())?;
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
    }
    Ok(())
}

/// Unbinds every object of ours that egui's `prepare_painting` does not rebind itself.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn cleanup(gl: &glow::Context) {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); only unbinding (binding name 0) happens here.
    unsafe {
        // Units 0..5 (`@group(0)`) and the user texture units after them, highest first.
        let units = USER_UNIT_BASE + u32::try_from(TEXTURE_SLOTS).unwrap_or(0);
        for unit in (0..units).rev() {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        // The loop ends on unit 0, which egui_glow expects active.
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, None);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, PARAMS_BINDING_POINT, None);
        gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_vertex_array(None);
        gl.use_program(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// glow and wgpu upload the same uniform bytes for the same plan and mappings, except the two
    /// backend flags (`internal_flip_y`, `internal_frag_mode`): every wiring decision is
    /// `plan.rs`'s. The UI's wgpu window (`ChainWindow::whole`) also plans exactly like glow.
    #[cfg(feature = "wgpu")]
    #[test]
    fn glow_and_wgpu_upload_identical_blocks() {
        use crate::effect::Effect;
        use crate::pass::{Pass, offsets};
        use crate::wgpu_backend::chain::{ChainWindow, chain_uniforms, frame_uniforms as wgpu_frame_uniforms};
        use egui::{CornerRadius, Rect, pos2, vec2};

        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let mut passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(1, [f32::from(u8::try_from(index).unwrap_or(0)) + 0.5; 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        let named = Effect::new("named", "struct Params { tint: vec3<f32>, gain: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * vec4<f32>(p.tint * p.gain, 1.0); }");
        let Ok(named) = named else { panic!("{named:?}") };
        passes[2] = Pass::new(&named).with("tint", [0.25, 0.5, 0.75]).and_then(|pass| pass.with("gain", 2.0)).unwrap_or_else(|error| panic!("{error}"));
        // F6: a scaled step's block (texels, mappings of a smaller image) is shared too.
        passes[1] = passes[1].clone().with_scale(0.5);
        let rect = Rect::from_min_size(pos2(40.0, 30.0), vec2(120.0, 80.0));
        let Some(geometry) = layer_geometry(rect, Rect::EVERYTHING, 1.5, [800, 600], 6.0, CornerRadius::same(8), 1.0) else { panic!("visible") };
        let capture = geometry.capture;
        let Some(glow_plan) = plan(&passes, capture.size(), PxRect { x: 0, y: 0, w: capture.w, h: capture.h }, [capture.x, capture.y], [false, true, false], None) else { panic!("planned") };
        let window = ChainWindow::whole(&geometry, [capture.x, capture.y]);
        let Some(wgpu_plan) = plan(&passes, capture.size(), window.rect, window.composite_origin, [false, true, false], None) else { panic!("planned") };
        assert_eq!(glow_plan, wgpu_plan);

        let used = capture.size().map(|side| u32::try_from(side).unwrap_or(0));
        let allocations = Allocations { source: [256, 256], ping_pong: [[320, 192], used], state: [[0, 0]; 2], coverage: [0, 0], region: [0, 0] };
        let time = 1.25;
        let wgpu_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_plan, &passes, &geometry, &allocations, 256);
        let glow_frame = frame_uniforms(&geometry, time);
        let mut glow_blocks: Vec<[u8; UNIFORM_BLOCK_SIZE]> = glow_plan.steps.iter().map(|step| step_uniforms(&glow_frame, &glow_plan, step, &allocations, &passes[step.pass])).collect();
        glow_blocks.push(composite_uniforms(&glow_frame, &glow_plan, &allocations, &geometry));
        assert_eq!(glow_blocks.len(), 4);
        let flags = [offsets::FLIP_Y, offsets::FRAG_MODE];
        let bytes_at = |range: &std::ops::Range<u64>| &wgpu_uniforms.bytes[usize::try_from(range.start).unwrap_or(0)..usize::try_from(range.end).unwrap_or(0)];
        // The named-parameter blocks are byte-identical: both backends upload `plan::params_block`.
        let mut glow_params: Vec<&[u8]> = glow_plan.steps.iter().map(|step| params_block(&passes[step.pass])).collect();
        glow_params.push(composite_params_block());
        for ((index, glow_params), draw) in glow_params.iter().enumerate().zip(&wgpu_uniforms.draws) {
            assert_eq!(*glow_params, bytes_at(&draw.params), "draw {index}: named parameters");
        }
        for ((index, glow_block), draw) in glow_blocks.iter().enumerate().zip(&wgpu_uniforms.draws) {
            let wgpu_block = bytes_at(&draw.fl);
            let differing: Vec<usize> = (0..UNIFORM_BLOCK_SIZE / 4).map(|word| word * 4).filter(|&at| glow_block[at..at + 4] != wgpu_block[at..at + 4]).collect();
            assert_eq!(differing, flags, "block {index}: only the backend flags may differ");
        }
        // F9: a shape-masked run's composite block (coverage mapping, mask flag) is shared too.
        let (glow_masked, wgpu_masked) = (glow_plan.clone().with_coverage(), wgpu_plan.clone().with_coverage());
        assert_eq!(glow_masked, wgpu_masked);
        let allocations = Allocations { coverage: [512, 256], ..allocations };
        let glow_block = composite_uniforms(&glow_frame, &glow_masked, &allocations, &geometry);
        let wgpu_masked_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_masked, &passes, &geometry, &allocations, 256);
        let Some(draw) = wgpu_masked_uniforms.draws.last() else { panic!("the composite draw") };
        let wgpu_block = bytes_at_masked(&wgpu_masked_uniforms.bytes, &draw.fl);
        let differing: Vec<usize> = (0..UNIFORM_BLOCK_SIZE / 4).map(|word| word * 4).filter(|&at| glow_block[at..at + 4] != wgpu_block[at..at + 4]).collect();
        assert_eq!(differing, flags, "masked composite: only the backend flags may differ");
        assert_ne!(glow_block, composite_uniforms(&glow_frame, &glow_plan, &allocations, &geometry), "the mask changes the composite block");
        // R1: a region run's pass blocks (the region-id mapping) are shared too.
        let (glow_regions, wgpu_regions) = (glow_plan.clone().with_regions(3), wgpu_plan.with_regions(3));
        assert_eq!(glow_regions, wgpu_regions);
        let allocations = Allocations { region: [1024, 512], ..allocations };
        let wgpu_region_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_regions, &passes, &geometry, &allocations, 256);
        for ((index, step), draw) in glow_regions.steps.iter().enumerate().zip(&wgpu_region_uniforms.draws) {
            let glow_block = step_uniforms(&glow_frame, &glow_regions, step, &allocations, &passes[step.pass]);
            let wgpu_block = bytes_at_masked(&wgpu_region_uniforms.bytes, &draw.fl);
            let differing: Vec<usize> = (0..UNIFORM_BLOCK_SIZE / 4).map(|word| word * 4).filter(|&at| glow_block[at..at + 4] != wgpu_block[at..at + 4]).collect();
            assert_eq!(differing, flags, "region step {index}: only the backend flags may differ");
            assert_ne!(glow_block[offsets::REGION_SCALE..offsets::REGION_SCALE + 8], [0_u8; 8], "region step {index}: the region mapping is set");
        }
    }

    /// The bytes of `range` in `bytes`.
    #[cfg(feature = "wgpu")]
    fn bytes_at_masked<'a>(bytes: &'a [u8], range: &std::ops::Range<u64>) -> &'a [u8] {
        &bytes[usize::try_from(range.start).unwrap_or(0)..usize::try_from(range.end).unwrap_or(0)]
    }

    #[test]
    fn gl_rects_are_bottom_up() {
        let rect = PxRect { x: 10, y: 20, w: 30, h: 40 };
        assert_eq!(gl_rect(rect, 100), [10, 40, 40, 80]);
        let top_row = PxRect { x: 0, y: 0, w: 5, h: 1 };
        assert_eq!(gl_rect(top_row, 100), [0, 99, 5, 100]);
    }
}
