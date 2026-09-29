/*
File: glow_backend/mod.rs

Purpose:
The `egui_glow` backend: public `install_glow` / `destroy_glow`, the `Backend` implementation,
and the latched state shared between the GUI thread and the paint callbacks.

Key structures:
- `GlowBackend`: the registered backend (`Arc<Mutex<GlowState>>` + chosen GLSL dialect).
- `GlowState`: lazily created `Resources`, per-effect program cache, failure latches, effects
  under per-pass GL error diagnosis (`suspects`), attached user textures that could not be
  resolved (`failed_textures`), persistent layer states (`states`, F7) and refused state formats
  (`failed_states`).

Key functions:
- `install_glow()`, `destroy_glow()` (public).
- `GlowState::program_for()`: cached effect program, built on first use.

Notes:
Submodules: `glsl` (pure WGSL->GLSL translation), `resources` (GL objects), `render` (one layer
per paint callback), `coverage` (shape-mask rasterizer, F9; a shape mask is packed on the GUI
thread in `paint_layer` — a region mask with each region's draw range, R1 — its textures resolved
in the callback like user textures). Failure policy: a resource failure or an unexplained capture / composite GL
error latches the WHOLE backend (logged once at error level, reported by `status()`); a
per-effect failure (GLSL translation, driver compile/link, or a GL error raised by that effect's
draw) latches only that effect (layers using it are skipped, reported by `failed_effects()`); a
framebuffer whose colour format cannot be captured skips only the layers painted into it (logged
once per format). An attached user texture `egui_glow::Painter::texture` does not know (freed,
foreign) skips the layers attaching it (logged once per id, reported by `failed_textures()`);
egui_glow applies a frame's texture deltas before painting and frees after, so a live id always
resolves to its current texture. Nothing is retried per frame. Re-installing on the same context frees the
previous glow backend's GL objects first.
*/

mod coverage;
mod glsl;
mod render;
mod resources;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use glow::HasContext as _;

use crate::backend::{Backend, BackendKind, BackendStatus, EffectFailure, InstallError, LayerJob, TextureFailure, installed, register, uninstall};
use crate::effect::{CompiledModule, TEXTURE_SLOTS};
use crate::state::{StateError, StateFailure, StateFormat, StateRegistry};
use glsl::{GlslTarget, glsl_target, translate};
use resources::{GlError, RenderTarget, Resources, build_program};

/// The two state textures of one stateful layer (`Slot::StateA` / `StateB`), created and grown
/// by `render.rs`.
type StateTargets = [Option<RenderTarget>; 2];

/// A compiled effect program and the module it was built from (to evict it once dropped).
#[derive(Debug)]
struct CachedProgram {
    /// Dies with the last `Effect` clone; the entry is then evicted and the program deleted.
    module: Weak<CompiledModule>,
    /// The linked program.
    program: glow::Program,
}

/// An effect that failed to build on this context.
#[derive(Debug)]
struct FailedEffect {
    /// For eviction, like `CachedProgram::module`.
    module: Weak<CompiledModule>,
    /// What is reported by `failed_effects()`.
    failure: EffectFailure,
}

/// Mutable backend state, locked by the paint callbacks (GUI thread) and by status queries.
#[derive(Debug, Default)]
pub(crate) struct GlowState {
    /// `None` until the first painted layer creates them.
    resources: Option<Resources>,
    /// The latched backend failure, if any. Once set nothing is drawn again.
    failure: Option<String>,
    /// Effect programs by `CompiledModule::id`.
    programs: HashMap<u64, CachedProgram>,
    /// Effects that failed to build or run, by `CompiledModule::id`.
    failed_effects: HashMap<u64, FailedEffect>,
    /// Effects of a chain whose passes raised a GL error that could not be attributed to one of
    /// them; chains containing one run with a `glGetError` after every pass until attributed.
    suspects: HashSet<u64>,
    /// Attached user textures `egui_glow::Painter::texture` could not resolve, by id (egui never
    /// reuses an id, so an entry is permanent).
    failed_textures: HashMap<egui::TextureId, TextureFailure>,
    /// Persistent layer states (F7), by viewport and layer id; freed by eviction and `destroy`.
    states: StateRegistry<StateTargets>,
    /// State formats this context cannot render / filter, latched per format.
    failed_states: HashMap<StateFormat, StateFailure>,
    /// Set by `destroy_glow`; a late callback must not recreate resources.
    destroyed: bool,
}

impl GlowState {
    /// Latches a backend-wide failure and logs it once.
    fn latch(&mut self, what: &str, error: &dyn std::fmt::Display) {
        if self.failure.is_some() {
            return;
        }
        let message = format!("{what}: {error}");
        log::error!(
            "egui_shader_layers (glow): the filter backend failed and is disabled for this session.\n\
             Error: {message}\n\
             Effect: shader layers draw nothing extra; the UI underneath is unaffected.\n\
             Possible cause: the GPU driver rejected a shader or a render-target format, or the \
             framebuffer cannot be read back (e.g. an unusual multisampled format)."
        );
        self.failure = Some(message);
    }

    /// Deletes programs (and forgets failures) of effects whose last clone was dropped.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn evict_dropped(&mut self, gl: &glow::Context) {
        let dead: Vec<u64> = self.programs.iter().filter(|(_, entry)| entry.module.strong_count() == 0).map(|(id, _)| *id).collect();
        for id in dead {
            if let Some(entry) = self.programs.remove(&id) {
                // SAFETY: forwarded contract; the program was created by this state.
                unsafe { gl.delete_program(entry.program) };
            }
        }
        self.failed_effects.retain(|_, entry| entry.module.strong_count() > 0);
        let programs = &self.programs;
        self.suspects.retain(|id| programs.contains_key(id));
    }

    /// Disables one effect on this backend: logs `error` once at error level, frees its cached
    /// program, and reports it through `failed_effects()`. Layers using it are skipped from now on.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn fail_effect(&mut self, gl: &glow::Context, module: &Arc<CompiledModule>, what: &str, error: &dyn std::fmt::Display) {
        log::error!(
            "egui_shader_layers (glow): effect `{}` failed while {what} and is disabled; layers using it are skipped.\n\
             Error: {error}\n\
             Possible cause: the driver rejected or could not run naga's GLSL translation of this effect.",
            module.label
        );
        if let Some(entry) = self.programs.remove(&module.id) {
            // SAFETY: forwarded contract; the program was created by this state.
            unsafe { gl.delete_program(entry.program) };
        }
        self.suspects.remove(&module.id);
        let failure = EffectFailure { label: module.label.to_string(), error: format!("{what}: {error}") };
        self.failed_effects.insert(module.id, FailedEffect { module: Arc::downgrade(module), failure });
    }

    /// Records that the attached user texture `id` could not be resolved (logged once per id).
    fn fail_texture(&mut self, id: egui::TextureId) {
        if self.failed_textures.contains_key(&id) {
            return;
        }
        let error = "egui_glow's painter has no texture with this id (freed, or never allocated on this painter)".to_owned();
        log::error!(
            "egui_shader_layers (glow): a shader layer attaches texture {id:?}, which cannot be resolved; layers attaching it draw nothing extra.\n\
             Error: {error}\n\
             Possible cause: the TextureHandle was dropped (egui freed the texture) while the layer still attaches its id."
        );
        self.failed_textures.insert(id, TextureFailure { id, error });
    }

    /// Records that stateful layers asking for `format` are refused and logs it once per format.
    fn fail_state(&mut self, format: StateFormat, reason: String) {
        if self.failed_states.contains_key(&format) {
            return;
        }
        let error = StateError::UnsupportedFormat { format, reason };
        log::error!("egui_shader_layers (glow): stateful shader layers are skipped (they draw nothing extra).\nError: {error}");
        self.failed_states.insert(format, StateFailure { format, error });
    }

    /// Frees the GL objects of `targets` (evicted or replaced state targets).
    ///
    /// # Safety
    /// `gl` must be current on this thread and own the targets.
    unsafe fn free_state_targets(gl: &glow::Context, targets: impl IntoIterator<Item = StateTargets>) {
        for target in targets.into_iter().flatten().flatten() {
            // SAFETY: forwarded contract.
            unsafe { target.destroy(gl) };
        }
    }

    /// The program for `module`, building and caching it on first use.
    ///
    /// Returns `None` when this effect failed (now or earlier); the failure is logged once.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn program_for(&mut self, gl: &glow::Context, target: GlslTarget, module: &Arc<CompiledModule>) -> Option<glow::Program> {
        if let Some(entry) = self.programs.get(&module.id) {
            return Some(entry.program);
        }
        if self.failed_effects.contains_key(&module.id) {
            return None;
        }
        let built = translate(module, target).map_err(GlError::from).and_then(|glsl| {
            // SAFETY: forwarded contract.
            unsafe { build_program(gl, &glsl) }
        });
        match built {
            Ok(program) => {
                log::debug!("egui_shader_layers (glow): built program for effect `{}` ({target:?})", module.label);
                self.programs.insert(module.id, CachedProgram { module: Arc::downgrade(module), program });
                Some(program)
            }
            Err(error) => {
                // SAFETY: forwarded contract.
                unsafe { self.fail_effect(gl, module, &format!("being built for {target:?}"), &error) };
                None
            }
        }
    }

    /// Frees every GL object and marks the state destroyed.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn destroy(&mut self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object freed was created on this context by this state, and `take`/`drain` prevent a double free.
        unsafe {
            if let Some(resources) = self.resources.take() {
                resources.destroy(gl);
            }
            for (_, entry) in self.programs.drain() {
                gl.delete_program(entry.program);
            }
            Self::free_state_targets(gl, self.states.drain());
        }
        self.failed_states.clear();
        self.failed_effects.clear();
        self.failed_textures.clear();
        self.suspects.clear();
        self.destroyed = true;
    }
}

/// The glow backend, registered with the context by the crate's registry plugin.
#[derive(Debug)]
pub(crate) struct GlowBackend {
    /// GLSL dialect for this context, or why the context is unsupported.
    target: Result<GlslTarget, String>,
    /// Shared with every paint callback closure.
    state: Arc<Mutex<GlowState>>,
    /// Address of the `glow::Context` passed to `install_glow`: re-installing frees this
    /// backend's objects only when handed the very same context object.
    context_address: usize,
}

/// Identity of a `glow::Context` object, for [`GlowBackend::context_address`].
fn context_address(gl: &glow::Context) -> usize {
    std::ptr::from_ref(gl).addr()
}

impl GlowBackend {
    /// Locks the state, tolerating poisoning (a panic elsewhere must not disable filtering; the
    /// state holds no invariant a half-finished operation could break beyond GL names, which are
    /// only ever added after successful creation).
    fn lock(&self) -> std::sync::MutexGuard<'_, GlowState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Backend for GlowBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Glow
    }

    fn status(&self) -> BackendStatus {
        if let Err(reason) = &self.target {
            return BackendStatus::Failed { backend: BackendKind::Glow, error: reason.clone() };
        }
        let state = self.lock();
        if let Some(error) = &state.failure {
            return BackendStatus::Failed { backend: BackendKind::Glow, error: error.clone() };
        }
        if state.destroyed {
            return BackendStatus::Failed { backend: BackendKind::Glow, error: "destroyed by destroy_glow".to_owned() };
        }
        match &state.resources {
            Some(resources) => BackendStatus::Active { backend: BackendKind::Glow, float_intermediates: resources.float_intermediates() },
            None => BackendStatus::Installed { backend: BackendKind::Glow },
        }
    }

    fn failed_effects(&self) -> Vec<EffectFailure> {
        self.lock().failed_effects.values().map(|entry| entry.failure.clone()).collect()
    }

    fn failed_textures(&self) -> Vec<TextureFailure> {
        self.lock().failed_textures.values().cloned().collect()
    }

    fn failed_states(&self) -> Vec<StateFailure> {
        self.lock().failed_states.values().cloned().collect()
    }

    fn state_count(&self) -> usize {
        self.lock().states.len()
    }

    fn paint_layer(&self, painter: &egui::Painter, job: LayerJob) {
        let Ok(target) = self.target.clone() else {
            return;
        };
        let state = Arc::clone(&self.state);
        let rect = job.rect;
        // A shape or region mask (F9 / R1) is packed once here, on the GUI thread; the callback
        // only resolves the textures its draws sample.
        let packed = job.mask.pack();
        let callback = egui_glow::CallbackFn::new(move |info, glow_painter| {
            let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
            let mut mask_textures = Vec::with_capacity(packed.as_ref().map_or(0, |packed| packed.packed.draws.len()));
            for draw in packed.iter().flat_map(|packed| &packed.packed.draws) {
                let Some(texture) = glow_painter.texture(draw.texture) else {
                    state.fail_texture(draw.texture);
                    return;
                };
                mask_textures.push(texture);
            }
            let coverage = packed.as_ref().map(|packed| (packed, mask_textures.as_slice()));
            // egui_glow applied this frame's texture deltas before painting, so a live id maps to
            // its current texture; an unknown one skips the layer (no dummy stands in for it).
            let mut user_textures = [None; TEXTURE_SLOTS];
            for (resolved, id) in user_textures.iter_mut().zip(job.textures) {
                let Some(id) = id else {
                    continue;
                };
                let Some(texture) = glow_painter.texture(id) else {
                    state.fail_texture(id);
                    return;
                };
                *resolved = Some(texture);
            }
            // SAFETY: egui_glow invokes paint callbacks on the thread that owns the current
            // context, between its own draws; `render::paint_layer` restores the framebuffer
            // bindings and leaves none of its objects bound (egui re-runs prepare_painting). The
            // user and mask textures belong to `glow_painter`, alive for this callback.
            unsafe { render::paint_layer(glow_painter.gl(), &mut state, target, &info, &job, render::LayerTextures { user: user_textures, coverage }) };
        });
        painter.add(egui::PaintCallback { rect, callback: Arc::new(callback) });
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Installs the glow filter backend on `ctx`. Call once at startup, e.g. from the eframe app
/// creator with `cc.gl`.
///
/// `gl` must be the context `egui_glow` paints with, current on the calling thread (always the case
/// in an eframe app creator). GPU objects are created lazily on the first painted layer.
///
/// On an unsupported context the backend is still registered in a failed state, so
/// [`crate::status`] explains why layers are not drawn.
///
/// Calling it again on the same `ctx` with the same `glow::Context` object frees the previous
/// glow backend's GL objects before installing a fresh one. With a different `glow::Context` the
/// old objects are left alone (they belong to the other context) and a warning is logged:
/// before replacing a GL context, call [`destroy_glow`] with the old one while it is current.
/// The backend lives in the egui `Context` itself (not in egui `Memory`), so resetting egui
/// memory does not uninstall it.
///
/// # Errors
/// [`InstallError::Unsupported`] for `WebGL1`, OpenGL ES 2 and desktop OpenGL below 3.3.
pub fn install_glow(ctx: &egui::Context, gl: &glow::Context) -> Result<(), InstallError> {
    if let Some(previous) = installed(ctx)
        && let Some(previous) = previous.as_any().downcast_ref::<GlowBackend>()
    {
        if previous.context_address == context_address(gl) {
            // SAFETY: documented precondition — `gl` is current, and it is the context object the
            // previous backend was installed with, so it owns those objects.
            unsafe { previous.lock().destroy(gl) };
            log::debug!("egui_shader_layers (glow): freed the previously installed glow backend before re-installing");
        } else {
            log::warn!(
                "egui_shader_layers (glow): install_glow was given a different GL context than the installed glow backend; \
                 its GL objects are not freed (call destroy_glow with the old context before replacing it)"
            );
        }
    }
    let version = gl.version();
    let target = glsl_target(version.major, version.minor, version.is_embedded, cfg!(target_arch = "wasm32"));
    log::debug!("egui_shader_layers (glow): context {}.{} (embedded: {}) -> {target:?}", version.major, version.minor, version.is_embedded);
    let result = match &target {
        Ok(_) => Ok(()),
        Err(reason) => {
            log::error!("egui_shader_layers (glow): unsupported context, shader layers are disabled: {reason}");
            Err(InstallError::Unsupported { backend: BackendKind::Glow, reason: reason.clone() })
        }
    };
    register(ctx, Arc::new(GlowBackend { target, state: Arc::new(Mutex::new(GlowState::default())), context_address: context_address(gl) }));
    result
}

/// Frees every OpenGL object of the glow backend installed on `ctx` and uninstalls it.
///
/// Call from `eframe::App::on_exit` (which receives the `glow::Context`) while the context is still
/// current. Idempotent; a no-op when no glow backend is installed.
pub fn destroy_glow(ctx: &egui::Context, gl: &glow::Context) {
    let Some(backend) = installed(ctx) else {
        return;
    };
    let Some(glow_backend) = backend.as_any().downcast_ref::<GlowBackend>() else {
        log::warn!("egui_shader_layers: destroy_glow called but the installed backend is {}; left installed and untouched", backend.kind());
        return;
    };
    // SAFETY: documented precondition — `gl` is the current context on this thread.
    unsafe { glow_backend.lock().destroy(gl) };
    uninstall(ctx);
    log::debug!("egui_shader_layers (glow): GPU resources freed, backend uninstalled");
}
