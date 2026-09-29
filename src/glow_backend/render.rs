/*
File: glow_backend/render.rs

Purpose:
One shader layer (all of its entries, R3), executed inside an `egui_glow` paint callback: the ops
of `plan::schedule` (framebuffer source) in order — capture an entry's backdrop from the bound
framebuffer, run the draws of its backend-neutral `plan::ChainPlan` (passes ping-pong in
offscreen targets), composite it back into the caller's framebuffer. Which slot a draw reads
and writes, its viewport and its uniform blocks (`fl` at binding point 0, the named-parameter
block at binding point 1) come from `plan.rs` (shared with wgpu); this file maps slots to GL
objects, binds them on the texture units of `glsl.rs`, uploads the blocks and draws.

Key structures:
- `CallerState`: the framebuffer bindings (and sample count) egui left bound, saved first.
- `EntryTextures`: `egui_glow`'s textures of one entry (resolved by `mod.rs`).
- `LayerRun` / `EntryRun`: the layer-wide and per-visible-entry inputs; `PendingRun`: a run
  waiting for its composite.
- `LayerOutcome`: what a layer paint ended with, applied to `GlowState` by `paint_layer`.

Key functions:
- `paint_layer()`: saves the caller's bindings, runs the layer, applies the outcome, restores.
- `prepare()`: every refusal decided for all visible entries before the first capture.
- `execute()`: maps each `plan::Op` to GL (`capture_entry`, `run_entry`, `composite_entry`).

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
- Entries (R3): `plan.rs` owns the order (`Framebuffer` + `Stacked`: capture, run, composite per
  entry, so a capture sees the earlier composites; `Independent`: every capture first). Entry k
  captures into `Resources::captures[k]` (the pool); the ping-pong, mask and region targets are
  shared, which is sound because a framebuffer schedule composites each run before the next one
  (`execute` latches a schedule that does not, as a crate bug). A multi-entry layer composites
  with the BLENDED composite (erase + add programs, `BlendedComposite`), a single entry with the
  replace composite (unchanged). A failing op stops the layer; earlier composites stay.
- Shape masks (F9): the plan's coverage step runs right after the capture (`coverage.rs`: the
  mask's meshes into the coverage target, an additive union); only the composite binds it (unit 3).
  Region masks (R1) plan `with_regions` (the region count; 0 for a shape mask): the same step then
  runs the id passes into the region-id target, which every PASS step binds on unit 4 (the
  composite gets the white dummy). Their GL errors are the crate's own work (latch).
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
use super::resources::{BlendedComposite, GlError, PARAMS_BINDING_POINT, ParamsBuffer, ReadFormat, ReadFormatError, RenderTarget, ResolveTarget, Resources, TargetFormat, UNIFORM_BINDING_POINT};
use crate::backend::{EntryJob, LayerJob};
use crate::effect::{BINDING_COVERAGE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_REGION_TEXTURE, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, TEXTURE_SLOTS};
use crate::geometry::{LayerGeometry, MaskTransform, PxRect, entry_rect, layer_geometry, mask_transform, transform_scale};
use crate::mask::PackedRegions;
use crate::pass::{FrameUniforms, InputUniforms, UNIFORM_BLOCK_SIZE};
use crate::plan::{Allocations, Bindings, CaptureSource, ChainPlan, Op, Slot, StatePlanInput, composite_params_block, composite_uniforms, params_block, plan, schedule, step_uniforms};
use crate::state::{StateFormat, StateKey, StateRegistry, StateRun};

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

/// `egui_glow`'s textures one ENTRY of a layer samples, resolved by the paint callback
/// (`mod.rs`, one per entry of the job, in entry order).
#[derive(Clone, Copy, Debug)]
pub(super) struct EntryTextures<'a> {
    /// The texture of every attached user texture slot (`None` where none is attached).
    pub user: [Option<glow::Texture>; TEXTURE_SLOTS],
    /// A shape or region mask (F9 / R1): its packed meshes (with each region's draw range; none
    /// for a shape mask) and the texture of each of its draws.
    pub coverage: Option<(&'a PackedRegions, &'a [glow::Texture])>,
}

/// The layer-wide inputs of one paint.
struct LayerRun<'a> {
    /// The job being painted.
    job: &'a LayerJob,
    /// Screen size in physical pixels.
    screen: [i32; 2],
    /// The caller's bindings (the capture source and composite destination is `caller.draw`).
    caller: CallerState,
    /// Whether the context is GLES / WebGL (affects how the default read buffer is named).
    embedded: bool,
    /// A layer of several VISIBLE entries: every entry composites with the BLENDED composite (D4),
    /// so a later entry lands over an earlier one; a single visible entry keeps the replace composite.
    blended: bool,
    /// The layer's `reset_state` request, claimed once for this paint (every stateful entry).
    reset: bool,
}

/// The inputs of one VISIBLE entry (`plan::schedule` gives invisible entries no op).
struct EntryRun<'a> {
    /// The entry's position in the job (its capture target and its ops; its state key goes
    /// through `LayerJob::state_key`, which keys by the user index `EntryJob::index`).
    index: usize,
    /// The entry.
    entry: &'a EntryJob,
    /// Its pixel geometry (`geometry::entry_rect` under the layer transform).
    geometry: LayerGeometry,
    /// The job's per-frame input resolved for this entry's capture (`fl.pointer` in capture uv).
    input: InputUniforms,
    /// `egui_glow`'s texture of every attached user texture slot.
    user_textures: [Option<glow::Texture>; TEXTURE_SLOTS],
    /// A shape or region mask (F9 / R1): its packed meshes and region ranges, `egui_glow`'s
    /// texture of each draw, and the layer-points -> capture-pixels transform.
    coverage: Option<(&'a PackedRegions, &'a [glow::Texture], MaskTransform)>,
    /// One program per pass of `entry.passes`.
    programs: Vec<glow::Program>,
    /// Whether a pass effect is a suspect: check `glGetError` after every pass.
    diagnose: bool,
    /// A stateful entry's state key (`LayerJob::state_key`) and format.
    state: Option<(StateKey, StateFormat)>,
}

/// An entry whose `Op::Run` finished, waiting for its `Op::Composite` (the schedule of a
/// framebuffer source puts it right after: the ping-pong and mask targets are shared).
#[derive(Debug)]
struct PendingRun {
    /// The entry index.
    index: usize,
    /// Its plan (the composite step and where the result is).
    plan: ChainPlan,
    /// A stateful entry's key and what its registry `begin` said.
    state: Option<(StateKey, StateRun)>,
}

/// How one layer paint ended; applied to `GlowState` by [`paint_layer`].
#[derive(Debug)]
enum LayerOutcome {
    /// Every scheduled op ran.
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
    /// Pass `index` of entry `entry` raised `error`: disable its effect.
    PassFailed {
        /// Position in the job's entries (`LayerJob::entries`).
        entry: usize,
        /// Index into that entry's passes.
        index: usize,
        /// The GL error.
        error: u32,
    },
    /// The passes of entry `entry` raised `error` but several effects could have caused it.
    PassesFailed {
        /// Position in the job's entries (`LayerJob::entries`).
        entry: usize,
        /// The GL error.
        error: u32,
    },
}

/// Whether a layer composites with the BLENDED composite: more than one VISIBLE entry (a lone
/// visible entry of a multi-entry layer draws exactly like a single-entry layer, replace composite).
fn blended_layer(geometries: &[Option<LayerGeometry>]) -> bool {
    geometries.iter().flatten().nth(1).is_some()
}

/// A latch for a schedule / bookkeeping inconsistency (a crate bug, never a driver's).
fn schedule_bug(error: &str) -> LayerOutcome {
    LayerOutcome::Latch { what: "executing the layer schedule", error: format!("{error} (crate bug)") }
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

/// Runs one shader layer (every entry, R3). Never panics; failures are latched in `state`. The
/// caller's framebuffer bindings are saved before anything else and restored on every path.
///
/// `textures` are `egui_glow`'s textures of every entry's attached user textures and mask draws
/// (resolved by the caller), one per entry of `job`, in order.
///
/// # Safety
/// `gl` must be the current context on this thread, called from inside an `egui_glow` paint
/// callback (so egui restores its own state afterwards); every texture of `textures` is alive on it.
pub(super) unsafe fn paint_layer(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob, textures: &[EntryTextures<'_>]) {
    if state.failure.is_some() || state.destroyed || job.entries.is_empty() {
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
/// Everything that can refuse the layer (a failing effect, a missing rasterizer or blended
/// program, an unsupported state format) is decided for EVERY visible entry before the first
/// capture, so such a layer draws nothing rather than a part of its entries.
///
/// # Safety
/// As [`paint_layer`].
unsafe fn paint_saved(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, info: &egui::PaintCallbackInfo, job: &LayerJob, caller: CallerState, textures: &[EntryTextures<'_>]) {
    // The callback rect is the layer rect (the union of the entries) after any layer transform;
    // each entry lands at `entry_rect` under the same transform (a single entry: the callback
    // rect itself). The transform scale joins pixels-per-point in
    // `LayerGeometry::pixels_per_layer_point` (padding, radii, `fl.pixels_per_point`).
    let transform_scale = transform_scale(job.rect, info.viewport);
    let geometries: Vec<Option<LayerGeometry>> = job
        .entries
        .iter()
        .map(|entry| layer_geometry(entry_rect(job.rect, info.viewport, entry.rect), info.clip_rect, info.pixels_per_point, info.screen_size_px, entry.padding_points, entry.mask.corner_radius(), transform_scale))
        .collect();
    // `plan.rs` owns the order; a framebuffer capture sees earlier composites (`Stacked`) or is
    // taken before any of them (`Independent`).
    let ops = schedule(&geometries, job.overlap, CaptureSource::Framebuffer);
    if ops.is_empty() {
        return;
    }

    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the objects used belong to `state`, were created on this same context, and are alive until `destroy`.
    unsafe {
        if textures.len() != job.entries.len() {
            state.latch("resolving the layer's textures", &format!("{} texture sets for {} entries (crate bug)", textures.len(), job.entries.len()));
            return;
        }
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

        let Some(programs) = prepare(gl, state, target, job, &geometries, textures) else {
            return;
        };
        let blended = blended_layer(&geometries);
        let Some(mut resources) = state.resources.take() else {
            return;
        };
        let screen = [i32::try_from(info.screen_size_px[0]).unwrap_or(i32::MAX), i32::try_from(info.screen_size_px[1]).unwrap_or(i32::MAX)];
        let entries: Vec<Option<EntryRun<'_>>> = job
            .entries
            .iter()
            .zip(&geometries)
            .zip(textures)
            .zip(programs)
            .enumerate()
            .map(|(index, (((entry, geometry), textures), programs))| {
                let (geometry, programs) = ((*geometry)?, programs?);
                Some(EntryRun {
                    index,
                    entry,
                    geometry,
                    input: job.frame_input.resolve(info.pixels_per_point, geometry.capture),
                    user_textures: textures.user,
                    coverage: textures.coverage.map(|(packed, textures)| (packed, textures, mask_transform(job.rect, info.viewport, info.pixels_per_point, geometry.capture))),
                    programs,
                    diagnose: entry.passes.iter().any(|pass| state.suspects.contains(&pass.effect().module().id)),
                    state: job.state_key(index).map(|key| (key, entry.state.format())),
                })
            })
            .collect();

        // Unpainted states are freed at every paint (GL is current here); the layer's reset
        // ticket is claimed once, for every stateful entry, where the registry starts their runs.
        let (frame, time) = (job.frame_input.frame, f64::from(job.time));
        GlowState::free_state_targets(gl, state.states.evict(job.viewport, frame, time));
        let stateful = entries.iter().flatten().any(|run| run.state.is_some());
        let reset = stateful && job.claim_reset();
        let layer = LayerRun { job, screen, caller, embedded: target != GlslTarget::Desktop330, blended, reset };
        drain_errors(gl);
        let outcome = execute(gl, &mut resources, &mut state.states, &layer, &entries, &ops);
        cleanup(gl);
        apply_outcome(gl, state, &mut resources, job, outcome);
        state.resources = Some(resources);
    }
}

/// Everything that can refuse the layer, decided for EVERY visible entry before the first
/// capture: each entry's pass programs (`None` for invisible entries; a failing effect skips the
/// layer), the blended composite of a multi-entry layer and the mask rasterizer (created on first
/// need; a failure latches), and each stateful entry's state format. `None`: skip the layer.
///
/// # Safety
/// `gl` must be current on this thread; `state.resources` exist.
unsafe fn prepare(gl: &glow::Context, state: &mut GlowState, target: GlslTarget, job: &LayerJob, geometries: &[Option<LayerGeometry>], textures: &[EntryTextures<'_>]) -> Option<Vec<Option<Vec<glow::Program>>>> {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object created belongs to `state`.
    unsafe {
        // Build (or fetch) every visible entry's pass programs BEFORE touching the framebuffer,
        // so a failing effect skips the layer without leaving a half-processed region behind.
        let mut programs: Vec<Option<Vec<glow::Program>>> = Vec::with_capacity(job.entries.len());
        for (entry, geometry) in job.entries.iter().zip(geometries) {
            if geometry.is_none() {
                programs.push(None);
                continue;
            }
            let mut entry_programs = Vec::with_capacity(entry.passes.len());
            for pass in &entry.passes {
                entry_programs.push(state.program_for(gl, target, pass.effect().module())?);
            }
            programs.push(Some(entry_programs));
        }
        // The blended composite is created with the first multi-entry layer (a failure is a
        // crate bug in the composite shader or a driver refusal: latched like the composite's).
        if blended_layer(geometries)
            && let Some(resources) = state.resources.as_mut()
            && resources.blended.is_none()
        {
            match BlendedComposite::create(gl, target) {
                Ok(created) => resources.blended = Some(created),
                Err(error) => {
                    state.latch("creating the blended composite", &error);
                    return None;
                }
            }
        }
        // The mask rasterizer is created with the first masked layer (a failure is a crate
        // bug in the mesh shader or a driver refusal: latched like the composite's).
        let masked = textures.iter().zip(geometries).any(|(textures, geometry)| geometry.is_some() && textures.coverage.is_some());
        if masked
            && let Some(resources) = state.resources.as_mut()
            && resources.coverage.is_none()
        {
            match CoverageResources::create(gl, target) {
                Ok(created) => resources.coverage = Some(created),
                Err(error) => {
                    state.latch("creating the shape-mask rasterizer", &error);
                    return None;
                }
            }
        }
        let float_targets = state.resources.as_ref().is_some_and(Resources::float_intermediates);
        for (index, (entry, geometry)) in job.entries.iter().zip(geometries).enumerate() {
            let format = entry.state.format();
            if geometry.is_some() && job.state_key(index).is_some() && format == StateFormat::Rgba16Float && !float_targets {
                state.fail_state(format, "RGBA16F is not renderable on this context (GLES / WebGL2 without EXT_color_buffer_float)".to_owned());
                return None;
            }
        }
        Some(programs)
    }
}

/// Executes `ops` (`plan::schedule` with a framebuffer source) in order: `Capture(k)` copies the
/// framebuffer into entry `k`'s capture, `Run(k)` runs its chain, `Composite(k)` writes it back.
/// `plan.rs` owns the order; this only maps each op to GL. Stops at the first failing op (earlier
/// composites stay on screen: the framebuffer cannot be un-drawn).
///
/// # Safety
/// `gl` must be current on this thread; `resources` alive on it; `entries[k]` is `Some` for every
/// entry `ops` names.
unsafe fn execute(gl: &glow::Context, resources: &mut Resources, states: &mut StateRegistry<StateTargets>, layer: &LayerRun<'_>, entries: &[Option<EntryRun<'_>>], ops: &[Op]) -> LayerOutcome {
    let mut captured = vec![false; entries.len()];
    let mut pending: Option<PendingRun> = None;
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object used belongs to `resources` / `states`, alive on this context.
    unsafe {
        // Offscreen work: scissor would clip the blits and pass draws, blending would mix them.
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::BLEND);
        for op in ops {
            let index = match *op {
                Op::Capture(index) | Op::Run(index) | Op::Composite(index) => index,
                // Only a pristine source needs hand-offs; a framebuffer capture already sees
                // the earlier composites.
                Op::HandOff { from, into, viewport, origin } => return schedule_bug(&format!("a hand-off {from} -> {into} (viewport {viewport:?}, origin {origin:?}) in a framebuffer schedule")),
            };
            let Some(Some(run)) = entries.get(index) else {
                return schedule_bug(&format!("an op for entry {index}, which is invisible or missing"));
            };
            match *op {
                Op::Capture(_) => {
                    if let Err(outcome) = capture_entry(gl, resources, layer, run) {
                        return outcome;
                    }
                    captured[index] = true;
                }
                Op::Run(_) => {
                    // The ping-pong and mask targets are shared: a run must be composited before
                    // the next one (the framebuffer schedules always do).
                    if let Some(previous) = &pending {
                        return schedule_bug(&format!("entry {index} ran before entry {} was composited", previous.index));
                    }
                    if !captured[index] {
                        return schedule_bug(&format!("entry {index} ran before its capture"));
                    }
                    match run_entry(gl, resources, states, layer, run) {
                        Ok(done) => pending = done,
                        Err(outcome) => return outcome,
                    }
                }
                Op::Composite(_) => {
                    // No pending run: the entry planned nothing (no draw), so it composites nothing.
                    let Some(done) = pending.take() else {
                        continue;
                    };
                    if done.index != index {
                        return schedule_bug(&format!("entry {index} composited while entry {} is pending", done.index));
                    }
                    if let Err(outcome) = composite_entry(gl, resources, states, layer, run, &done) {
                        return outcome;
                    }
                }
                Op::HandOff { .. } => {}
            }
        }
    }
    match pending {
        Some(done) => schedule_bug(&format!("entry {} ran but was never composited", done.index)),
        None => LayerOutcome::Done,
    }
}

/// `Op::Capture`: copies entry `run`'s padded capture rect of the caller's framebuffer into its
/// own capture target (grown into the pool on first use), checking the copy's GL error.
///
/// # Errors
/// The layer's outcome: `Rejected` for an uncapturable framebuffer, `Latch` for our own failures.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn capture_entry(gl: &glow::Context, resources: &mut Resources, layer: &LayerRun<'_>, run: &EntryRun<'_>) -> Result<(), LayerOutcome> {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `resources` were created on it.
    unsafe {
        if let Err(error) = resources.ensure_captures(gl, run.index.saturating_add(1)) {
            return Err(LayerOutcome::Latch { what: "allocating the capture target", error: error.to_string() });
        }
        match capture(gl, resources, layer, run) {
            Ok(()) => {}
            Err(CaptureError::Rejected { format, reason }) => return Err(LayerOutcome::Rejected { format, reason }),
            Err(CaptureError::Gl(error)) => return Err(LayerOutcome::Latch { what: "allocating the capture target", error: error.to_string() }),
        }
        let capture_error = gl.get_error();
        if capture_error != glow::NO_ERROR {
            return Err(LayerOutcome::Latch {
                what: "copying the backdrop out of the framebuffer",
                error: format!("GL error 0x{capture_error:04x} (framebuffer samples: {})", layer.caller.samples),
            });
        }
    }
    Ok(())
}

/// `Op::Run`: starts entry `run`'s state (if stateful), then runs its chain over its capture.
/// Returns the run to composite, or `None` when the entry plans nothing. A failed or empty run
/// forgets the state's content (the next run starts from zero).
///
/// # Errors
/// The layer's outcome for a failing step.
///
/// # Safety
/// `gl` must be current on this thread; the entry's capture was taken this paint.
unsafe fn run_entry(gl: &glow::Context, resources: &mut Resources, states: &mut StateRegistry<StateTargets>, layer: &LayerRun<'_>, run: &EntryRun<'_>) -> Result<Option<PendingRun>, LayerOutcome> {
    let (frame, time) = (layer.job.frame_input.frame, f64::from(layer.job.time));
    let mut state = None;
    if let Some((key, format)) = run.state {
        let (_, begun, replaced) = states.begin(key, format, frame, time, layer.reset, || [None, None]);
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the replaced targets were created on it by this backend.
        unsafe { GlowState::free_state_targets(gl, replaced) };
        state = Some((key, format, begun));
    }
    let planned = {
        let mut layer_state = state.and_then(|(key, format, begun)| {
            states.get_mut(key).map(|state_entry| LayerState {
                targets: &mut state_entry.targets,
                format: if format == StateFormat::Rgba16Float { TargetFormat::Rgba16F } else { TargetFormat::Rgba8 },
                input: StatePlanInput { scale: run.entry.state.scale(), current: begun.current, previous_px: begun.previous, step: begun.step },
            })
        });
        // SAFETY: forwarded contract.
        unsafe { run_passes_of(gl, resources, layer, run, layer_state.as_mut()) }
    };
    let state = state.map(|(key, _, begun)| (key, begun));
    match planned {
        Ok(Some(plan)) => Ok(Some(PendingRun { index: run.index, plan, state })),
        Ok(None) => {
            invalidate(states, state);
            Ok(None)
        }
        Err(outcome) => {
            invalidate(states, state);
            Err(outcome)
        }
    }
}

/// Forgets the content of a stateful entry's state (a run that did not finish cleanly).
fn invalidate(states: &mut StateRegistry<StateTargets>, state: Option<(StateKey, StateRun)>) {
    if let Some(state_entry) = state.and_then(|(key, _)| states.get_mut(key)) {
        state_entry.invalidate();
    }
}

/// `Op::Composite`: writes the pending run of entry `run` back into the caller's framebuffer
/// and records where a stateful entry's latest state now is.
///
/// # Errors
/// `Latch` (the composite is the crate's own work).
///
/// # Safety
/// `gl` must be current on this thread; `done` is `run`'s pending run.
unsafe fn composite_entry(gl: &glow::Context, resources: &mut Resources, states: &mut StateRegistry<StateTargets>, layer: &LayerRun<'_>, run: &EntryRun<'_>, done: &PendingRun) -> Result<(), LayerOutcome> {
    let result = {
        let targets = done.state.and_then(|(key, _)| states.get_mut(key)).map(|state_entry| &state_entry.targets);
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `done` ran this paint, so its plan's composite source holds the result.
        unsafe { composite(gl, resources, targets, layer, run, &done.plan) }
    };
    // SAFETY: as above; only an error query.
    let error = unsafe { gl.get_error() };
    let outcome = match result {
        Err(error) => Err(LayerOutcome::Latch { what: "uploading the composite's parameter block", error: error.to_string() }),
        Ok(()) if error != glow::NO_ERROR => Err(LayerOutcome::Latch { what: "compositing the filtered region", error: format!("GL error 0x{error:04x}") }),
        Ok(()) => Ok(()),
    };
    match (&outcome, done.state, done.plan.state) {
        (Ok(()), Some((key, begun)), Some(state_plan)) => {
            if let Some(state_entry) = states.get_mut(key) {
                state_entry.finish(begun, layer.job.frame_input.frame, state_plan.final_index, state_plan.final_px);
            }
        }
        // A stateful run always plans its state; without a plan, or not composited cleanly, the
        // content is unknown: start over from zero.
        (_, state, _) => invalidate(states, state),
    }
    outcome
}

/// Applies a [`LayerOutcome`] to the backend state (latches, effect failures, suspects).
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn apply_outcome(gl: &glow::Context, state: &mut GlowState, resources: &mut Resources, job: &LayerJob, outcome: LayerOutcome) {
    let modules = |entry: &'_ EntryJob| entry.passes.iter().map(|pass| pass.effect().module()).cloned().collect::<Vec<_>>();
    match outcome {
        LayerOutcome::Done => {
            // A clean run of every chain clears their suspects.
            for module in job.entries.iter().flat_map(modules) {
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
        LayerOutcome::PassFailed { entry, index, error } => {
            if let Some(pass) = job.entries.get(entry).and_then(|entry| entry.passes.get(index)) {
                // SAFETY: forwarded contract.
                unsafe { state.fail_effect(gl, pass.effect().module(), "running on the GPU", &format!("GL error 0x{error:04x}")) };
            }
        }
        LayerOutcome::PassesFailed { entry, error } => {
            let modules = job.entries.get(entry).map(modules).unwrap_or_default();
            let distinct: HashSet<u64> = modules.iter().map(|module| module.id).collect();
            if let (1, Some(module)) = (distinct.len(), modules.first()) {
                // SAFETY: forwarded contract.
                unsafe { state.fail_effect(gl, module, "running on the GPU", &format!("GL error 0x{error:04x}")) };
            } else {
                log::warn!("egui_shader_layers (glow): a filter chain raised GL error 0x{error:04x}; checking its passes one by one on the next frames to find the effect responsible");
                state.suspects.extend(distinct);
            }
        }
    }
}

/// A stateful entry's state for one run: its two targets (kept in `GlowState::states`), their
/// format and what the registry knows of them.
struct LayerState<'a> {
    /// `Slot::StateA` / `Slot::StateB`.
    targets: &'a mut StateTargets,
    /// Storage format.
    format: TargetFormat,
    /// The plan's input.
    input: StatePlanInput,
}

/// Plans and runs entry `run`'s passes over its capture (mask coverage first). Returns the plan
/// (`None`: nothing to draw). A stateful entry needs `layer_state`; its targets are grown
/// (keeping content) and cleared here. Leaves bindings for `paint_layer` to restore.
///
/// # Errors
/// The layer's outcome for a failing step (nothing of this entry was written to the caller's
/// framebuffer yet).
///
/// # Safety
/// `gl` must be current on this thread; `run.programs` matches `run.entry.passes`; the entry's
/// capture was taken this paint.
unsafe fn run_passes_of(gl: &glow::Context, resources: &mut Resources, layer: &LayerRun<'_>, run: &EntryRun<'_>, layer_state: Option<&mut LayerState<'_>>) -> Result<Option<ChainPlan>, LayerOutcome> {
    let capture = run.geometry.capture;
    // glow always computes the whole capture (fragment mode 0 interpolates over the viewport);
    // the composite's origin is the capture's screen position, as on the wgpu UI.
    let attached = run.user_textures.map(|texture| texture.is_some());
    let state_input = layer_state.as_ref().map(|state| state.input);
    let Some(plan) = plan(&run.entry.passes, capture.size(), PxRect { x: 0, y: 0, w: capture.w, h: capture.h }, [capture.x, capture.y], attached, state_input) else {
        return Ok(None);
    };
    // A masked run plans the coverage step; `with_regions(0)` is exactly `with_coverage`.
    let plan = match run.coverage {
        Some((packed, _, _)) => match u32::try_from(packed.regions.len()) {
            Ok(count) => plan.with_regions(count),
            Err(_) => return Err(LayerOutcome::Latch { what: "planning the region passes", error: format!("{} regions (crate bug: `Mask::check` allows at most 255)", packed.regions.len()) }),
        },
        None => plan,
    };
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `resources` and `run.programs` were created on this context and are alive, and `run.programs` matches `run.entry.passes` (caller contract).
    unsafe {
        // Both ping-pong targets hold every step's viewport (the plan's pool size).
        let needed = plan.target_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
        let max = resources.max_size;
        for target in &mut resources.ping_pong {
            if let Err(error) = target.ensure_size(gl, needed, max) {
                return Err(LayerOutcome::Latch { what: "allocating an intermediate target", error: error.to_string() });
            }
        }
        // State targets: grown keeping the latest content, the empty one cleared.
        let states: Option<&StateTargets> = match (plan.state, layer_state) {
            (Some(state_plan), Some(layer_state)) => {
                let needed = state_plan.target_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
                for slot in layer_state.targets.iter_mut() {
                    if let Err(error) = RenderTarget::ensure_kept(slot, gl, layer_state.format, needed, max) {
                        return Err(LayerOutcome::Latch { what: "allocating a state target", error: error.to_string() });
                    }
                }
                if let Some(clear) = state_plan.clear
                    && let Some(target) = layer_state.targets.get(usize::from(clear == Slot::StateB)).and_then(Option::as_ref)
                {
                    target.clear(gl);
                }
                Some(&*layer_state.targets)
            }
            (Some(_), None) => return Ok(None),
            (None, _) => None,
        };
        // The mask's coverage (F9) and region ids (R1), before the passes (they depend only on
        // the mask).
        if let (Some(step), Some((packed, textures, transform))) = (plan.coverage, run.coverage) {
            let Some(coverage) = resources.coverage.as_mut() else {
                return Err(LayerOutcome::Latch { what: "rasterizing the mask", error: "the mask rasterizer is missing (crate bug)".to_owned() });
            };
            if !CoverageResources::fits(packed, textures, &step) {
                return Err(LayerOutcome::Latch { what: "rasterizing the mask", error: "the packed mask does not match its plan (crate bug)".to_owned() });
            }
            if let Err(error) = coverage.draw(gl, packed, textures, &transform, &step, max) {
                return Err(LayerOutcome::Latch { what: "allocating the mask coverage target", error: error.to_string() });
            }
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return Err(LayerOutcome::Latch { what: "rasterizing the mask", error: format!("GL error 0x{error:04x}") });
            }
        }
        match run_passes(gl, resources, states, layer, run, &plan) {
            Ok(Some((index, error))) => return Err(LayerOutcome::PassFailed { entry: run.index, index, error }),
            Ok(None) => {}
            Err(error) => return Err(LayerOutcome::Latch { what: "uploading a named-parameter block", error: error.to_string() }),
        }
        let pass_error = gl.get_error();
        if pass_error != glow::NO_ERROR {
            // Nothing of this entry was written to the caller's framebuffer yet: skipping its
            // composite leaves the backdrop unfiltered there.
            return Err(LayerOutcome::PassesFailed { entry: run.index, error: pass_error });
        }
    }
    Ok(Some(plan))
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
/// `gl` must be current on this thread with `layer.caller.draw` bound as READ framebuffer.
unsafe fn read_format(gl: &glow::Context, resources: &mut Resources, layer: &LayerRun<'_>) -> Result<Option<ReadFormat>, CaptureError> {
    let default_framebuffer = layer.caller.draw.is_none();
    if default_framebuffer && let Some(format) = resources.default_read_format {
        return Ok(Some(format));
    }
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract), with `layer.caller.draw` bound as READ framebuffer (caller contract).
    match unsafe { ReadFormat::query(gl, default_framebuffer, layer.embedded) } {
        Ok(format) => {
            if default_framebuffer {
                log::debug!("egui_shader_layers (glow): default framebuffer read format {format:?} (samples: {})", layer.caller.samples);
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

/// Copies entry `run`'s capture rect of the caller's framebuffer into `resources.captures[run.index]`
/// at `(0, 0)` (the pool holds it: `Resources::ensure_captures` ran).
///
/// # Errors
/// [`CaptureError::Rejected`] for a read buffer that cannot be captured (checked BEFORE any
/// blit, so no GL error is raised); [`CaptureError::Gl`] when a target cannot be allocated.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn capture(gl: &glow::Context, resources: &mut Resources, layer: &LayerRun<'_>, run: &EntryRun<'_>) -> Result<(), CaptureError> {
    let rect = run.geometry.capture;
    let screen = layer.screen;
    let src = gl_rect(rect, screen[1]);
    let max = resources.max_size;
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `resources` were created on this context and are alive, and `layer.caller` holds framebuffer names saved from this same context.
    unsafe {
        let Some(target) = resources.captures.get_mut(run.index) else {
            return Err(CaptureError::Gl(GlError::CreateResource { resource: "capture target", detail: format!("no pool slot for entry {} (crate bug)", run.index) }));
        };
        target.ensure_size(gl, rect.size(), max).map_err(CaptureError::Gl)?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, layer.caller.draw);
        let format = read_format(gl, resources, layer)?;
        if let Some(format) = format {
            if resources.rejected_formats.contains(&format) {
                return Err(CaptureError::Rejected { format, reason: "previously rejected".to_owned() });
            }
            if !format.is_normalized() {
                return Err(CaptureError::Rejected { format, reason: format!("the framebuffer is not normalized fixed-point ({format:?}), so it cannot be copied into an 8-bit capture") });
            }
        }
        if layer.caller.samples > 0 {
            // GLES 3 / WebGL2: a multisample resolve needs identical rects and an identical
            // internal format on the draw side.
            let internal_format = match format.map(|format| format.resolve_format()) {
                Some(Ok(internal_format)) => internal_format,
                Some(Err(reason)) => return Err(CaptureError::Rejected { format: format.unwrap_or(UNKNOWN_FORMAT), reason }),
                None => glow::RGBA8,
            };
            if resources.resolve.is_none() {
                resources.resolve = Some(ResolveTarget::create(gl).map_err(CaptureError::Gl)?);
                log::debug!("egui_shader_layers (glow): framebuffer is multisampled ({} samples); resolving through a 0x{internal_format:04x} renderbuffer", layer.caller.samples);
            }
            let Some(resolve) = resources.resolve.as_mut() else {
                return Ok(());
            };
            if let Err(error) = resolve.ensure(gl, internal_format, screen, max) {
                return Err(CaptureError::Rejected { format: format.unwrap_or(UNKNOWN_FORMAT), reason: error.to_string() });
            }
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, layer.caller.draw);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(resolve.framebuffer));
            gl.blit_framebuffer(src[0], src[1], src[2], src[3], src[0], src[1], src[2], src[3], glow::COLOR_BUFFER_BIT, glow::NEAREST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(resolve.framebuffer));
        }
        let Some(target) = resources.captures.get(run.index) else {
            return Ok(());
        };
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(target.framebuffer));
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

/// The GL target a plan slot names for the entry whose capture is pool slot `capture`, or `None`
/// for a state slot without state targets (no plan binds one: a state slot exists only with
/// `plan.state`, whose targets `run_passes_of` created) and for a mask slot without the mask
/// rasterizer (no plan binds one: `paint_saved` creates it for every masked layer).
fn target_of<'a>(resources: &'a Resources, capture: usize, states: Option<&'a StateTargets>, slot: Slot) -> Option<&'a RenderTarget> {
    match slot {
        Slot::Source => resources.captures.get(capture),
        Slot::Ping => Some(&resources.ping_pong[0]),
        Slot::Pong => Some(&resources.ping_pong[1]),
        Slot::StateA => states.and_then(|states| states[0].as_ref()),
        Slot::StateB => states.and_then(|states| states[1].as_ref()),
        Slot::Coverage => resources.coverage.as_ref().map(|coverage| &coverage.target),
        Slot::RegionId => resources.coverage.as_ref().map(|coverage| &coverage.region_target),
    }
}

/// The allocated sizes of the capture (pool slot `capture`), the ping-pong, the state, the
/// coverage and the region-id targets (the plan maps each step's content, held at its target's
/// `(0, 0)` corner, through them).
fn allocations(resources: &Resources, capture: usize, states: Option<&StateTargets>) -> Allocations {
    let state = |index: usize| states.and_then(|states| states[index].as_ref()).map_or([0, 0], RenderTarget::allocated_px);
    let (coverage, region) = resources.coverage.as_ref().map_or(([0, 0], [0, 0]), |coverage| (coverage.target.allocated_px(), coverage.region_target.allocated_px()));
    let source = resources.captures.get(capture).map_or([0, 0], RenderTarget::allocated_px);
    Allocations { source, ping_pong: [resources.ping_pong[0].allocated_px(), resources.ping_pong[1].allocated_px()], state: [state(0), state(1)], coverage, region }
}

/// Binds every texture of one draw on its texture units (`glsl.rs`): the `@group(0)` textures on
/// units `0..5` (the plan's slots where it names one, the dummies elsewhere; `Slot::Source` is
/// pool slot `capture`), and user texture slot `i` on unit `USER_UNIT_BASE + i` (the attached
/// texture where the plan binds it, the transparent dummy elsewhere). Leaves `TEXTURE0` active.
///
/// # Safety
/// `gl` must be current on this thread; `resources` and `user_textures` alive on it.
unsafe fn bind_textures(gl: &glow::Context, resources: &Resources, capture: usize, states: Option<&StateTargets>, bindings: Bindings, user_textures: &[Option<glow::Texture>; TEXTURE_SLOTS]) {
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
        let texture_of = |slot: Option<Slot>| slot.and_then(|slot| target_of(resources, capture, states, slot)).map_or(resources.transparent, |target| target.texture);
        for (unit, binding) in (0_u32..).zip(GROUP0_TEXTURE_BINDINGS) {
            let texture = match binding {
                BINDING_SOURCE_TEXTURE => texture_of(Some(bindings.src)),
                BINDING_ORIGINAL_TEXTURE => texture_of(bindings.original),
                BINDING_STATE_TEXTURE => texture_of(bindings.state),
                // Full coverage (white) unless the plan binds a shape mask's coverage.
                BINDING_COVERAGE_TEXTURE => bindings.coverage.and_then(|slot| target_of(resources, capture, states, slot)).map_or(resources.white, |target| target.texture),
                // White = id 255 = `REGION_NONE` unless the plan binds a region-id texture.
                BINDING_REGION_TEXTURE => bindings.region.and_then(|slot| target_of(resources, capture, states, slot)).map_or(resources.white, |target| target.texture),
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
/// With `run.diagnose`, checks `glGetError` after every pass and stops at the first failing one,
/// returning `(pass index, error)`; otherwise the caller checks once after all passes.
///
/// # Safety
/// `gl` must be current on this thread; targets must be at least `plan.target_px`;
/// `run.programs` matches `run.entry.passes`; no GL error may be pending on entry.
unsafe fn run_passes(gl: &glow::Context, resources: &mut Resources, states: Option<&StateTargets>, layer: &LayerRun<'_>, run: &EntryRun<'_>, plan: &ChainPlan) -> Result<Option<(usize, u32)>, GlError> {
    let allocations = allocations(resources, run.index, states);
    let frame = frame_uniforms(&run.geometry, layer.job.time).with_input(&run.input);
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); targets are at least the plan's pool size and `run.programs` matches `run.entry.passes` (caller contract); all objects are alive on this context.
    unsafe {
        gl.bind_vertex_array(Some(resources.vertex_array));
        for step in &plan.steps {
            // `plan` was built from `run.entry.passes`, which `run.programs` matches one to one.
            let (Some(pass), Some(program), Some(output)) = (run.entry.passes.get(step.pass), run.programs.get(step.pass), target_of(resources, run.index, states, step.output)) else {
                continue;
            };
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(output.framebuffer));
            gl.viewport(0, 0, step.used_px[0], step.used_px[1]);
            gl.use_program(Some(*program));
            bind_textures(gl, resources, run.index, states, step.bindings, &run.user_textures);
            upload_uniforms(gl, resources.uniform_buffer, &step_uniforms(&frame, plan, step, &allocations, pass));
            upload_params(gl, &mut resources.params, params_block(pass))?;
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            if run.diagnose {
                let error = gl.get_error();
                if error != glow::NO_ERROR {
                    return Ok(Some((step.pass, error)));
                }
            }
        }
    }
    Ok(None)
}

/// Writes entry `run`'s last pass output back over its region of the caller's framebuffer,
/// masked by its coverage. A single-entry layer uses the REPLACE composite (outside the mask the
/// original capture is written, i.e. unchanged); a multi-entry layer the BLENDED one (D4: the
/// erase draw scales the framebuffer by `1 - coverage`, the add draw adds `coverage * filtered`),
/// which leaves what earlier entries composited outside this entry's mask. Scissor (egui's clip
/// rect) is on for the composite only; blending and scissor are off again on return, ready for
/// the next entry's offscreen work.
///
/// # Errors
/// As [`upload_params`]; [`GlError::CreateResource`] when the blended programs are missing
/// (crate bug: `paint_saved` creates them for every multi-entry layer).
///
/// # Safety
/// `gl` must be current on this thread; the passes of `plan` have run.
unsafe fn composite(gl: &glow::Context, resources: &mut Resources, states: Option<&StateTargets>, layer: &LayerRun<'_>, run: &EntryRun<'_>, plan: &ChainPlan) -> Result<(), GlError> {
    let geometry = &run.geometry;
    let bytes = composite_uniforms(&frame_uniforms(geometry, layer.job.time).with_input(&run.input), plan, &allocations(resources, run.index, states), geometry);
    let region = gl_rect(geometry.region, layer.screen[1]);
    let blended = match (layer.blended, &resources.blended) {
        (false, _) => None,
        (true, Some(blended)) => Some((blended.erase, blended.add)),
        (true, None) => return Err(GlError::CreateResource { resource: "blended composite", detail: "not created before a multi-entry layer (crate bug)".to_owned() }),
    };
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the passes ran (caller contract), so the plan's composite source holds their output; all objects are alive on this context.
    unsafe {
        layer.caller.restore(gl);
        gl.viewport(region[0], region[1], geometry.region.w, geometry.region.h);
        gl.enable(glow::SCISSOR_TEST);
        bind_textures(gl, resources, run.index, states, plan.composite.bindings, &run.user_textures);
        upload_uniforms(gl, resources.uniform_buffer, &bytes);
        upload_params(gl, &mut resources.params, composite_params_block())?;
        match blended {
            None => {
                gl.use_program(Some(resources.composite));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
            Some((erase, add)) => {
                // `mix(dst, filtered, c)` as two blended draws (the wgpu UI's `ERASE_BLEND` /
                // `ADD_BLEND`): dst * (1 - c), then + c * filtered, on all four channels.
                gl.enable(glow::BLEND);
                gl.blend_equation(glow::FUNC_ADD);
                gl.blend_func_separate(glow::ZERO, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE_MINUS_SRC_ALPHA);
                gl.use_program(Some(erase));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                gl.blend_func_separate(glow::ONE, glow::ONE, glow::ONE, glow::ONE);
                gl.use_program(Some(add));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                gl.disable(glow::BLEND);
            }
        }
        gl.disable(glow::SCISSOR_TEST);
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
        let wgpu_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_plan, &passes, &geometry, &allocations, 256, &[]);
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
        let wgpu_masked_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_masked, &passes, &geometry, &allocations, 256, &[]);
        let Some(draw) = wgpu_masked_uniforms.draws.last() else { panic!("the composite draw") };
        let wgpu_block = bytes_at_masked(&wgpu_masked_uniforms.bytes, &draw.fl);
        let differing: Vec<usize> = (0..UNIFORM_BLOCK_SIZE / 4).map(|word| word * 4).filter(|&at| glow_block[at..at + 4] != wgpu_block[at..at + 4]).collect();
        assert_eq!(differing, flags, "masked composite: only the backend flags may differ");
        assert_ne!(glow_block, composite_uniforms(&glow_frame, &glow_plan, &allocations, &geometry), "the mask changes the composite block");
        // R1: a region run's pass blocks (the region-id mapping) are shared too.
        let (glow_regions, wgpu_regions) = (glow_plan.clone().with_regions(3), wgpu_plan.with_regions(3));
        assert_eq!(glow_regions, wgpu_regions);
        let allocations = Allocations { region: [1024, 512], ..allocations };
        let wgpu_region_uniforms = chain_uniforms(&wgpu_frame_uniforms(&geometry, time), &wgpu_regions, &passes, &geometry, &allocations, 256, &[]);
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
