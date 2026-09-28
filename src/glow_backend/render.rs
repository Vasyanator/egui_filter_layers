/*
File: glow_backend/render.rs

Purpose:
One filter layer, executed inside an `egui_glow` paint callback: capture the backdrop from the
bound framebuffer, run the passes ping-pong in offscreen targets, composite the result back.

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
  those freely. It does NOT restore the framebuffer, unit-1 textures, renderbuffer or
  uniform-buffer bindings, so those are reset here.
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
- GL errors after the passes are attributed to the effects: with one distinct effect in the
  chain it is disabled directly; otherwise the chain's effects become "suspects" and later runs
  check after every pass until the culprit is found. Errors in capture / composite (the crate's
  own work) latch the whole backend.
*/

use std::collections::HashSet;

use glow::HasContext as _;

use super::GlowState;
use super::glsl::GlslTarget;
use super::resources::{GlError, ReadFormat, ReadFormatError, ResolveTarget, Resources, UNIFORM_BINDING_POINT};
use crate::backend::LayerJob;
use crate::geometry::{LayerGeometry, PxRect, layer_geometry, transform_scale};
use crate::pass::{FrameUniforms, Params, pack_uniforms};

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

/// The per-layer inputs every GL step below needs.
struct LayerRun<'a> {
    /// Pixel geometry of the layer.
    geometry: LayerGeometry,
    /// The job being painted.
    job: &'a LayerJob,
    /// Screen size in physical pixels.
    screen: [i32; 2],
    /// The caller's bindings (the capture source and composite destination is `caller.draw`).
    caller: CallerState,
    /// Whether the context is GLES / WebGL (affects how the default read buffer is named).
    embedded: bool,
}

/// How one layer run ended; applied to `GlowState` by [`paint_layer`].
#[derive(Debug)]
enum LayerOutcome {
    /// Composited successfully.
    Done,
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
        log::debug!("egui_filter_layers (glow): discarding a GL error raised before the filter callback: 0x{error:04x}");
    }
}

/// Runs one filter layer. Never panics; failures are latched in `state`. The caller's framebuffer
/// bindings are saved before anything else and restored on every path.
///
/// # Safety
/// `gl` must be the current context on this thread, called from inside an `egui_glow` paint
/// callback (so egui restores its own state afterwards).
pub(super) unsafe fn paint_layer(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob) {
    if state.failure.is_some() || state.destroyed || job.passes.is_empty() {
        return;
    }
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract), inside an egui_glow paint callback (this fn's contract).
    unsafe {
        // FIRST, before resource creation / the RGBA16F probe, which bind framebuffer 0.
        let caller = CallerState::save(gl);
        paint_saved(gl, state, target, info, job, caller);
        caller.restore(gl);
    }
}

/// [`paint_layer`] after the caller's state was saved; may return from anywhere.
///
/// # Safety
/// As [`paint_layer`].
unsafe fn paint_saved(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob, caller: CallerState) {
    // The callback rect is the layer rect after any layer transform; the transform scale joins
    // pixels-per-point in `LayerGeometry::pixels_per_layer_point` (padding, radii, `fl.pixels_per_point`).
    let transform_scale = transform_scale(job.rect, info.viewport);
    let Some(geometry) = layer_geometry(info.viewport, info.clip_rect, info.pixels_per_point, info.screen_size_px, job.padding_points, job.corner_radius, transform_scale) else {
        return;
    };

    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the objects used belong to `state`, were created on this same context, and are alive until `destroy`.
    unsafe {
        state.evict_dropped(gl);
        if state.resources.is_none() {
            match Resources::create(gl, target) {
                Ok(resources) => {
                    log::info!("egui_filter_layers (glow): resources created ({target:?}, float intermediates: {})", resources.float_intermediates());
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
        let diagnose = job.passes.iter().any(|pass| state.suspects.contains(&pass.effect().module().id));
        let Some(mut resources) = state.resources.take() else {
            return;
        };
        let screen = [i32::try_from(info.screen_size_px[0]).unwrap_or(i32::MAX), i32::try_from(info.screen_size_px[1]).unwrap_or(i32::MAX)];
        let run = LayerRun { geometry, job, screen, caller, embedded: target != GlslTarget::Desktop330 };

        drain_errors(gl);
        let outcome = run_layer(gl, &mut resources, &run, &programs, diagnose);
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
        LayerOutcome::Done => {
            // A clean run of the whole chain clears its suspects.
            for module in modules() {
                state.suspects.remove(&module.id);
            }
        }
        LayerOutcome::Latch { what, error } => state.latch(what, &error),
        LayerOutcome::Rejected { format, reason } => {
            if resources.rejected_formats.insert(format) {
                log::error!(
                    "egui_filter_layers (glow): filter layers cannot read this framebuffer and are skipped on it.\n\
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
                log::warn!("egui_filter_layers (glow): a filter chain raised GL error 0x{error:04x}; checking its passes one by one on the next frames to find the effect responsible");
                state.suspects.extend(distinct);
            }
        }
    }
}

/// Capture, passes and composite of one layer. Leaves bindings for `paint_layer` to restore.
///
/// # Safety
/// `gl` must be current on this thread; `programs` matches `run.job.passes`.
unsafe fn run_layer(gl: &glow::Context, resources: &mut Resources, run: &LayerRun<'_>, programs: &[glow::Program], diagnose: bool) -> LayerOutcome {
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

        let size = run.geometry.capture.size();
        let max = resources.max_size;
        for target in &mut resources.ping_pong {
            if let Err(error) = target.ensure_size(gl, size, max) {
                return LayerOutcome::Latch { what: "allocating an intermediate target", error: error.to_string() };
            }
        }

        if let Some((index, error)) = run_passes(gl, resources, run, programs, diagnose) {
            return LayerOutcome::PassFailed { index, error };
        }
        let pass_error = gl.get_error();
        if pass_error != glow::NO_ERROR {
            // Nothing was written to the caller's framebuffer yet: skipping the composite leaves
            // the backdrop unfiltered for this layer.
            return LayerOutcome::PassesFailed { error: pass_error };
        }

        composite(gl, resources, run, programs.len());
        let error = gl.get_error();
        if error != glow::NO_ERROR {
            return LayerOutcome::Latch { what: "compositing the filtered region", error: format!("GL error 0x{error:04x}") };
        }
    }
    LayerOutcome::Done
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
                log::debug!("egui_filter_layers (glow): default framebuffer read format {format:?} (samples: {})", run.caller.samples);
                resources.default_read_format = Some(format);
            }
            Ok(Some(format))
        }
        Err(ReadFormatError::NoColorBuffer(reason)) => {
            Err(CaptureError::Rejected { format: UNKNOWN_FORMAT, reason })
        }
        Err(ReadFormatError::Query(error)) => {
            log::debug!("egui_filter_layers (glow): the driver could not report the framebuffer format (GL error 0x{error:04x}); assuming RGBA8");
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
                log::debug!("egui_filter_layers (glow): framebuffer is multisampled ({} samples); resolving through a 0x{internal_format:04x} renderbuffer", run.caller.samples);
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

/// Uploads one pass's uniforms and binds the buffer to the `fl` binding point.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn upload_uniforms(gl: &glow::Context, buffer: glow::Buffer, frame: &FrameUniforms, params: &Params) {
    let bytes = pack_uniforms(frame, params);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `buffer` is our live uniform buffer, allocated with `UNIFORM_BLOCK_SIZE` bytes, exactly the length of `bytes`.
    unsafe {
        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(buffer));
        gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &bytes);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, Some(buffer));
    }
}

/// The per-frame uniforms shared by every pass of a layer (scales filled per pass).
fn frame_uniforms(run: &LayerRun<'_>) -> FrameUniforms {
    // GL texture storage is bottom-up; the prelude flips so `uv` stays top-down.
    FrameUniforms::for_layer(&run.geometry, run.job.time, true)
}

/// Runs every pass into the ping-pong targets; pass `i` writes `ping_pong[i % 2]`.
///
/// With `diagnose`, checks `glGetError` after every pass and stops at the first failing one,
/// returning `(pass index, error)`; otherwise the caller checks once after all passes.
///
/// # Safety
/// `gl` must be current on this thread; targets must be at least the capture size; no GL error
/// may be pending on entry.
unsafe fn run_passes(gl: &glow::Context, resources: &Resources, run: &LayerRun<'_>, programs: &[glow::Program], diagnose: bool) -> Option<(usize, u32)> {
    let size = run.geometry.capture.size();
    let mut frame = frame_uniforms(run);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); targets are at least the capture size and `programs` matches `run.job.passes` (caller contract); all objects are alive on this context.
    unsafe {
        gl.bind_vertex_array(Some(resources.vertex_array));
        gl.active_texture(glow::TEXTURE0);
        for (index, (pass, program)) in run.job.passes.iter().zip(programs).enumerate() {
            let input = if index == 0 { &resources.capture } else { &resources.ping_pong[(index - 1) % 2] };
            let output = &resources.ping_pong[index % 2];
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(output.framebuffer));
            gl.viewport(0, 0, size[0], size[1]);
            gl.use_program(Some(*program));
            gl.bind_texture(glow::TEXTURE_2D, Some(input.texture));
            frame.src_scale = input.scale(size);
            upload_uniforms(gl, resources.uniform_buffer, &frame, pass.params());
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            if diagnose {
                let error = gl.get_error();
                if error != glow::NO_ERROR {
                    return Some((index, error));
                }
            }
        }
    }
    None
}

/// Writes the last pass output back over the layer region of the caller's framebuffer, masked
/// by the rounded rect (outside the mask the original capture is written, i.e. unchanged).
///
/// # Safety
/// `gl` must be current on this thread; `pass_count >= 1`.
unsafe fn composite(gl: &glow::Context, resources: &Resources, run: &LayerRun<'_>, pass_count: usize) {
    let geometry = &run.geometry;
    let size = geometry.capture.size();
    let filtered = &resources.ping_pong[(pass_count.saturating_sub(1)) % 2];
    let mut frame = frame_uniforms(run);
    frame.src_scale = filtered.scale(size);
    frame.aux_scale = resources.capture.scale(size);
    let params = geometry.composite_params();
    let region = gl_rect(geometry.region, run.screen[1]);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `pass_count >= 1` (caller contract), so `filtered` holds pass output; all objects are alive on this context.
    unsafe {
        run.caller.restore(gl);
        gl.viewport(region[0], region[1], geometry.region.w, geometry.region.h);
        gl.enable(glow::SCISSOR_TEST);
        gl.use_program(Some(resources.composite));
        gl.active_texture(glow::TEXTURE1);
        gl.bind_texture(glow::TEXTURE_2D, Some(resources.capture.texture));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(filtered.texture));
        upload_uniforms(gl, resources.uniform_buffer, &frame, &params);
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
    }
}

/// Unbinds every object of ours that egui's `prepare_painting` does not rebind itself.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn cleanup(gl: &glow::Context) {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); only unbinding (binding name 0) happens here.
    unsafe {
        gl.active_texture(glow::TEXTURE1);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, None);
        gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_vertex_array(None);
        gl.use_program(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_rects_are_bottom_up() {
        let rect = PxRect { x: 10, y: 20, w: 30, h: 40 };
        assert_eq!(gl_rect(rect, 100), [10, 40, 40, 80]);
        let top_row = PxRect { x: 0, y: 0, w: 5, h: 1 };
        assert_eq!(gl_rect(top_row, 100), [0, 99, 5, 100]);
    }
}
