/*
File: glow_backend/resources.rs

Purpose:
Every OpenGL object the glow backend owns, and the fallible code that creates, grows and frees
them. No per-frame policy lives here — see `render.rs`.

Key structures:
- `GlError`: typed failure of any GL object operation.
- `TargetFormat`, `RenderTarget`: a texture + framebuffer pair, grown on demand (never shrunk,
  rounded up to `GROW_STEP` so a growing layer does not reallocate every frame);
  `RenderTarget::ensure_kept` grows one KEEPING its content (the persistent state targets, F7).
- `ReadFormat`: the colour format of the framebuffer being captured, and the renderbuffer format
  a multisample resolve must use to match it (GLES 3 / WebGL2 require identical formats).
- `ResolveTarget`: renderbuffer + framebuffer for that resolve, reallocated on format change.
- `ParamsBuffer`: the grow-only named-parameter uniform buffer and its allocated size.
- `Resources`: the shared set (VAO, `fl` and named-parameter uniform buffers, capture / resolve / ping-pong targets,
  composite program, dummy textures for the texture units a draw leaves unused, cached
  default-framebuffer format).

Key functions:
- `build_program()`: compile + link a translated `GlslProgram` and bind its block / samplers.
- `Resources::create()` / `Resources::destroy()`.
- `grown_size()`: pure growth policy of every target.

Notes:
All functions are `unsafe` with one contract: the `glow::Context` passed in must be current on the
calling thread (true inside an `egui_glow` paint callback and in `eframe::App::on_exit`).
Every creation path cleans up what it already created before returning an error.
*/

use std::collections::HashSet;

use glow::HasContext as _;

use super::coverage::CoverageResources;
use super::glsl::{GlslProgram, GlslTarget, TranslateError, translate};
use crate::effect::composite_module;
use crate::pass::UNIFORM_BLOCK_SIZE;

/// Converts a GL enum (`u32` in glow) to the `i32` the `*_i32` entry points take.
///
/// Const-evaluated at every use below, so a value that does not fit is a compile error, which
/// makes the single `as` a proven-lossless conversion.
const fn gl_i32(value: u32) -> i32 {
    assert!(value <= 0x7fff_ffff, "GL enum does not fit in an i32");
    #[allow(clippy::cast_possible_wrap)] // Guarded by the assert above (const-evaluated).
    let converted = value as i32;
    converted
}

/// [`UNIFORM_BLOCK_SIZE`] as the `i32` `glBufferData` takes (a small constant; checked at compile time).
const UNIFORM_BLOCK_SIZE_I32: i32 = {
    assert!(UNIFORM_BLOCK_SIZE <= 0x7fff_ffff);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // Guarded by the const assert above.
    let size = UNIFORM_BLOCK_SIZE as i32;
    size
};

/// Uniform-buffer binding point used for the `fl` block. `egui_glow` uses no uniform buffers.
pub(super) const UNIFORM_BINDING_POINT: u32 = 0;
/// Uniform-buffer binding point of the named-parameter block (`p: Params`, WGSL binding 4).
pub(super) const PARAMS_BINDING_POINT: u32 = UNIFORM_BINDING_POINT + 1;

/// A failure of a GL object operation. Resource-level failures latch the whole backend; program
/// failures latch only the effect that caused them.
#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum GlError {
    /// A `glCreate*` call refused to hand out an object.
    #[error("could not create a GL {resource}: {detail}")]
    CreateResource {
        /// Which kind of object.
        resource: &'static str,
        /// The driver's message.
        detail: String,
    },
    /// A shader stage failed to compile.
    #[error("the driver rejected the {stage} shader: {log}")]
    CompileShader {
        /// `vertex` or `fragment`.
        stage: &'static str,
        /// The driver's compile log.
        log: String,
    },
    /// The program failed to link.
    #[error("the driver could not link the program: {log}")]
    LinkProgram {
        /// The driver's link log.
        log: String,
    },
    /// A render target's framebuffer is not complete.
    #[error("the {format:?} render target framebuffer is incomplete (status 0x{status:04x})")]
    IncompleteFramebuffer {
        /// Format of the target.
        format: TargetFormat,
        /// `glCheckFramebufferStatus` result.
        status: u32,
    },
    /// The multisample-resolve renderbuffer cannot be used with this internal format.
    #[error("the resolve renderbuffer (internal format 0x{internal_format:04x}) is not usable (GL error 0x{error:04x}, framebuffer status 0x{status:04x})")]
    IncompleteResolve {
        /// The renderbuffer internal format.
        internal_format: u32,
        /// `glGetError` after allocating it.
        error: u32,
        /// `glCheckFramebufferStatus` result.
        status: u32,
    },
    /// naga could not translate the module to GLSL.
    #[error(transparent)]
    Translate(#[from] TranslateError),
    /// The crate's own composite shader failed to compile (crate bug).
    #[error("internal composite shader is invalid: {0}")]
    Composite(String),
}

/// Storage format of a render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TargetFormat {
    /// 8-bit normalized RGBA (always renderable).
    Rgba8,
    /// Half-float RGBA (renderable on GL 3.3; on GLES/WebGL2 only with a colour-buffer-float extension).
    Rgba16F,
}

impl TargetFormat {
    /// `(internal format, format, type)` for `glTexImage2D`.
    fn tex_image_params(self) -> (i32, u32, u32) {
        match self {
            Self::Rgba8 => (gl_i32(glow::RGBA8), glow::RGBA, glow::UNSIGNED_BYTE),
            Self::Rgba16F => (gl_i32(glow::RGBA16F), glow::RGBA, glow::HALF_FLOAT),
        }
    }
}

/// Granularity (pixels) targets grow by: a layer being enlarged interactively reallocates once
/// per `GROW_STEP` pixels instead of every frame. GL size limits are powers of two >= 1024, so a
/// rounded size never exceeds one unless the needed size already does.
const GROW_STEP: i32 = 64;

/// The storage size for a target that has `allocated` and must hold `needed`, per axis: unchanged
/// when it already fits, else `needed` rounded up to [`GROW_STEP`] and clamped to `max` (but never
/// below `needed`, so an oversized request still fails loudly in GL instead of silently cropping).
/// Never shrinks and never returns a zero axis.
pub(super) fn grown_size(allocated: [i32; 2], needed: [i32; 2], max: i32) -> [i32; 2] {
    let axis = |allocated: i32, needed: i32| {
        if needed <= allocated {
            return allocated.max(1);
        }
        let rounded = needed.checked_add(GROW_STEP - 1).map_or(needed, |n| n / GROW_STEP * GROW_STEP);
        rounded.min(max).max(needed).max(allocated).max(1)
    };
    [axis(allocated[0], needed[0]), axis(allocated[1], needed[1])]
}

/// A colour texture attached to its own framebuffer. The allocation only grows; a frame uses the
/// `(0, 0, w, h)` corner of it (mapped by `plan::TextureMapping::of`).
#[derive(Debug)]
pub(super) struct RenderTarget {
    /// Framebuffer with `texture` as `COLOR_ATTACHMENT0`.
    pub framebuffer: glow::Framebuffer,
    /// The colour texture (`LINEAR`, `CLAMP_TO_EDGE`).
    pub texture: glow::Texture,
    /// Allocated size in pixels; `[0, 0]` before the first `ensure_size`.
    pub allocated: [i32; 2],
    /// Storage format.
    pub format: TargetFormat,
}

impl RenderTarget {
    /// Creates the texture and framebuffer objects; storage is allocated by `ensure_size`.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn create(gl: &glow::Context, format: TargetFormat) -> Result<Self, GlError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); objects are created here, and freed again on the error path.
        unsafe {
            let texture = gl.create_texture().map_err(|detail| GlError::CreateResource { resource: "texture", detail })?;
            let framebuffer = match gl.create_framebuffer() {
                Ok(framebuffer) => framebuffer,
                Err(detail) => {
                    gl.delete_texture(texture);
                    return Err(GlError::CreateResource { resource: "framebuffer", detail });
                }
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            // LINEAR: effects sample at arbitrary positions (blur steps, refraction). No mipmaps,
            // so MIN_FILTER must not be a mipmap filter or the texture would be incomplete.
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_i32(glow::LINEAR));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_i32(glow::LINEAR));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_i32(glow::CLAMP_TO_EDGE));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_i32(glow::CLAMP_TO_EDGE));
            gl.bind_texture(glow::TEXTURE_2D, None);
            Ok(Self { framebuffer, texture, allocated: [0, 0], format })
        }
    }

    /// Grows the storage to at least `needed` (per axis, see [`grown_size`]; `max` is the GL size
    /// limit) and verifies framebuffer completeness.
    ///
    /// Leaves `GL_DRAW_FRAMEBUFFER` and the `TEXTURE_2D` binding of the active unit changed; the
    /// caller restores them.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn ensure_size(&mut self, gl: &glow::Context, needed: [i32; 2], max: i32) -> Result<(), GlError> {
        if self.allocated[0] >= needed[0] && self.allocated[1] >= needed[1] {
            return Ok(());
        }
        let size = grown_size(self.allocated, needed, max);
        let (internal, format, ty) = self.format.tex_image_params();
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `self.texture` / `self.framebuffer` were created on this context and are alive.
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, internal, size[0], size[1], 0, format, ty, glow::PixelUnpackData::Slice(None));
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.framebuffer));
            gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.texture), 0);
            let status = gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(GlError::IncompleteFramebuffer { format: self.format, status });
            }
        }
        self.allocated = size;
        Ok(())
    }

    /// Makes `slot` hold a target of `format` at least `needed` pixels large KEEPING its content
    /// (F7 state textures): a target that must grow is replaced by a new one into which the old
    /// one's whole allocation is blitted (at the origin, `NEAREST`, same format), then freed; a
    /// target of another format is freed without a copy. Leaves the READ / DRAW framebuffer
    /// bindings changed (the caller restores them).
    ///
    /// # Errors
    /// As [`RenderTarget::create`] / [`RenderTarget::ensure_size`]; the slot is then empty.
    ///
    /// # Safety
    /// `gl` must be current on this thread; `slot`'s target was created on it.
    pub unsafe fn ensure_kept(slot: &mut Option<Self>, gl: &glow::Context, format: TargetFormat, needed: [i32; 2], max: i32) -> Result<(), GlError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every target is created on it here or was (caller contract), and each freed one is taken out of `slot` first.
        unsafe {
            if let Some(old) = slot.take_if(|target| target.format != format) {
                old.destroy(gl);
            }
            if slot.as_ref().is_some_and(|target| target.allocated[0] >= needed[0] && target.allocated[1] >= needed[1]) {
                return Ok(());
            }
            let mut grown = Self::create(gl, format)?;
            let previous = slot.take();
            let base = previous.as_ref().map_or([0, 0], |target| target.allocated);
            if let Err(error) = grown.ensure_size(gl, grown_size(base, needed, max), max) {
                grown.destroy(gl);
                if let Some(previous) = previous {
                    previous.destroy(gl);
                }
                return Err(error);
            }
            if let Some(previous) = previous {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(previous.framebuffer));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(grown.framebuffer));
                gl.blit_framebuffer(0, 0, base[0], base[1], 0, 0, base[0], base[1], glow::COLOR_BUFFER_BIT, glow::NEAREST);
                previous.destroy(gl);
            }
            *slot = Some(grown);
        }
        Ok(())
    }

    /// Clears the whole target to transparent zero. Leaves `GL_DRAW_FRAMEBUFFER` bound to it and
    /// the clear colour changed (`egui_glow` sets its own before clearing).
    ///
    /// # Safety
    /// `gl` must be current on this thread with scissoring disabled; the target was created on it.
    pub unsafe fn clear(&self, gl: &glow::Context) {
        // SAFETY: forwarded contract.
        unsafe { self.clear_to(gl, [0.0; 4]) };
    }

    /// Clears the whole target to `rgba` (e.g. the region-id target to id 255 = `REGION_NONE`).
    /// Same effects and contract as [`RenderTarget::clear`].
    ///
    /// # Safety
    /// As [`RenderTarget::clear`].
    pub unsafe fn clear_to(&self, gl: &glow::Context, rgba: [f32; 4]) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the framebuffer is this target's own.
        unsafe {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.framebuffer));
            gl.clear_color(rgba[0], rgba[1], rgba[2], rgba[3]);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
    }

    /// The allocated size as the unsigned pixel counts `plan::TextureMapping::of` takes
    /// (`0` before the first allocation, which that mapping treats as neutral).
    #[must_use]
    pub fn allocated_px(&self) -> [u32; 2] {
        self.allocated.map(|side| u32::try_from(side).unwrap_or(0))
    }

    /// Frees both objects.
    ///
    /// # Safety
    /// `gl` must be current on this thread and own these objects.
    pub unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract) and owns both objects (caller contract); `self` is consumed, so they are not used again.
        unsafe {
            gl.delete_framebuffer(self.framebuffer);
            gl.delete_texture(self.texture);
        }
    }
}

/// The colour format of the buffer a framebuffer capture reads from, as GL reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ReadFormat {
    /// Bits per channel.
    pub bits: [i32; 4],
    /// `FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE` (`UNSIGNED_NORMALIZED`, `FLOAT`, `INT`, ...).
    pub component_type: u32,
    /// `FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING` (`LINEAR` or `SRGB`).
    pub encoding: u32,
}

/// Why [`ReadFormat::query`] produced no format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ReadFormatError {
    /// The framebuffer has no readable colour buffer; nothing can be captured.
    NoColorBuffer(String),
    /// The driver raised this GL error while answering the queries.
    Query(u32),
}

impl ReadFormat {
    /// Queries the colour buffer selected by `GL_READ_BUFFER` of the bound READ framebuffer.
    ///
    /// `default_framebuffer` is whether framebuffer 0 is bound (its buffers are named `BACK` etc.
    /// instead of `COLOR_ATTACHMENTi`); `embedded` is GLES / WebGL, where the default back
    /// buffer is queried as `BACK` (desktop GL requires `BACK_LEFT`).
    ///
    /// # Errors
    /// [`ReadFormatError::NoColorBuffer`] when the read buffer is `NONE` or has nothing attached;
    /// [`ReadFormatError::Query`] when a query raised a GL error (drained here).
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn query(gl: &glow::Context, default_framebuffer: bool, embedded: bool) -> Result<Self, ReadFormatError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); only queries of the bound READ framebuffer follow.
        unsafe {
            let read_buffer = u32::try_from(gl.get_parameter_i32(glow::READ_BUFFER)).unwrap_or(glow::NONE);
            if read_buffer == glow::NONE {
                return Err(ReadFormatError::NoColorBuffer("the framebuffer's read buffer is NONE".to_owned()));
            }
            let attachment = match (default_framebuffer && !embedded, read_buffer) {
                (true, glow::BACK) => glow::BACK_LEFT,
                (true, glow::FRONT) => glow::FRONT_LEFT,
                (_, other) => other,
            };
            let query = |pname: u32| gl.get_framebuffer_attachment_parameter_i32(glow::READ_FRAMEBUFFER, attachment, pname);
            let object_type = u32::try_from(query(glow::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)).unwrap_or(glow::NONE);
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return Err(ReadFormatError::Query(error));
            }
            if object_type == glow::NONE {
                return Err(ReadFormatError::NoColorBuffer(format!("nothing is attached to read buffer 0x{read_buffer:04x}")));
            }
            let bits = [
                query(glow::FRAMEBUFFER_ATTACHMENT_RED_SIZE),
                query(glow::FRAMEBUFFER_ATTACHMENT_GREEN_SIZE),
                query(glow::FRAMEBUFFER_ATTACHMENT_BLUE_SIZE),
                query(glow::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE),
            ];
            let component_type = u32::try_from(query(glow::FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE)).unwrap_or(glow::NONE);
            let encoding = u32::try_from(query(glow::FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)).unwrap_or(glow::NONE);
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return Err(ReadFormatError::Query(error));
            }
            Ok(Self { bits, component_type, encoding })
        }
    }

    /// Whether the buffer holds normalized fixed-point values, the only kind a blit may convert
    /// into the RGBA8 capture texture (float / integer sources are an `INVALID_OPERATION`).
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.component_type == glow::UNSIGNED_NORMALIZED
    }

    /// The renderbuffer internal format a multisample resolve of this buffer must use: GLES 3.0 /
    /// WebGL2 (and the letter of the desktop GL spec) require the draw buffer of a multisample
    /// resolve blit to have the SAME internal format as the read buffer.
    ///
    /// # Errors
    /// A human-readable description when no colour-renderable format matches (float, integer,
    /// unusual bit layouts, sRGB without alpha, which GLES 3 cannot render to).
    pub fn resolve_format(&self) -> Result<u32, String> {
        let describe = || format!("{}-bit RGBA {:?}, component type 0x{:04x}, encoding 0x{:04x}", self.bits.iter().sum::<i32>(), self.bits, self.component_type, self.encoding);
        if !self.is_normalized() {
            return Err(format!("{} is not a normalized fixed-point format", describe()));
        }
        let format = match (self.encoding, self.bits) {
            (glow::LINEAR, [8, 8, 8, 8]) => glow::RGBA8,
            (glow::LINEAR, [8, 8, 8, 0]) => glow::RGB8,
            (glow::LINEAR, [5, 6, 5, 0]) => glow::RGB565,
            (glow::LINEAR, [4, 4, 4, 4]) => glow::RGBA4,
            (glow::LINEAR, [5, 5, 5, 1]) => glow::RGB5_A1,
            (glow::LINEAR, [10, 10, 10, 2]) => glow::RGB10_A2,
            (glow::SRGB, [8, 8, 8, 8]) => glow::SRGB8_ALPHA8,
            _ => return Err(format!("{} has no colour-renderable resolve format", describe())),
        };
        Ok(format)
    }
}

/// A single-sample renderbuffer attached to its own framebuffer: the destination of a
/// multisample resolve blit. Grows like [`RenderTarget`]; a new internal format reallocates it.
#[derive(Debug)]
pub(super) struct ResolveTarget {
    /// Framebuffer with `renderbuffer` as `COLOR_ATTACHMENT0`.
    pub framebuffer: glow::Framebuffer,
    /// The colour renderbuffer.
    renderbuffer: glow::Renderbuffer,
    /// Allocated size; `[0, 0]` until the first successful `ensure`.
    allocated: [i32; 2],
    /// Internal format of the current storage (`NONE` before the first allocation).
    internal_format: u32,
}

impl ResolveTarget {
    /// Creates the renderbuffer and framebuffer objects; storage is allocated by `ensure`.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn create(gl: &glow::Context) -> Result<Self, GlError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); objects are created here, and freed again on the error path.
        unsafe {
            let renderbuffer = gl.create_renderbuffer().map_err(|detail| GlError::CreateResource { resource: "renderbuffer", detail })?;
            match gl.create_framebuffer() {
                Ok(framebuffer) => Ok(Self { framebuffer, renderbuffer, allocated: [0, 0], internal_format: glow::NONE }),
                Err(detail) => {
                    gl.delete_renderbuffer(renderbuffer);
                    Err(GlError::CreateResource { resource: "framebuffer", detail })
                }
            }
        }
    }

    /// Makes the storage `internal_format` and at least `needed` pixels (see [`grown_size`]).
    ///
    /// Leaves `GL_DRAW_FRAMEBUFFER` bound to this target's framebuffer. On failure the storage is
    /// marked unallocated, so the next call retries from scratch.
    ///
    /// # Errors
    /// [`GlError::IncompleteResolve`] when the driver rejects the format or size.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn ensure(&mut self, gl: &glow::Context, internal_format: u32, needed: [i32; 2], max: i32) -> Result<(), GlError> {
        let fits = self.allocated[0] >= needed[0] && self.allocated[1] >= needed[1];
        if fits && self.internal_format == internal_format {
            return Ok(());
        }
        let base = if self.internal_format == internal_format { self.allocated } else { [0, 0] };
        let size = grown_size(base, needed, max);
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `self.renderbuffer` / `self.framebuffer` were created on this context and are alive.
        unsafe {
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.renderbuffer));
            gl.renderbuffer_storage(glow::RENDERBUFFER, internal_format, size[0], size[1]);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            let error = gl.get_error();
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.framebuffer));
            gl.framebuffer_renderbuffer(glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(self.renderbuffer));
            let status = gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER);
            if error != glow::NO_ERROR || status != glow::FRAMEBUFFER_COMPLETE {
                self.allocated = [0, 0];
                self.internal_format = glow::NONE;
                return Err(GlError::IncompleteResolve { internal_format, error, status });
            }
        }
        self.allocated = size;
        self.internal_format = internal_format;
        Ok(())
    }

    /// Frees both objects.
    ///
    /// # Safety
    /// `gl` must be current on this thread and own these objects.
    pub unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract) and owns both objects (caller contract); `self` is consumed, so they are not used again.
        unsafe {
            gl.delete_framebuffer(self.framebuffer);
            gl.delete_renderbuffer(self.renderbuffer);
        }
    }
}

/// Creates a 1x1 RGBA8 texture holding `rgba` (linear, clamp-to-edge), bound on units a draw
/// leaves unused. Leaves no texture bound on the active unit.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn dummy_texture(gl: &glow::Context, rgba: [u8; 4]) -> Result<glow::Texture, GlError> {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the texture is created here and 4 bytes are exactly one RGBA8 texel.
    unsafe {
        let texture = gl.create_texture().map_err(|detail| GlError::CreateResource { resource: "dummy texture", detail })?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        for (name, value) in [(glow::TEXTURE_MIN_FILTER, glow::LINEAR), (glow::TEXTURE_MAG_FILTER, glow::LINEAR), (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE), (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE)] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, name, gl_i32(value));
        }
        gl.tex_image_2d(glow::TEXTURE_2D, 0, gl_i32(glow::RGBA8), 1, 1, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&rgba)));
        gl.bind_texture(glow::TEXTURE_2D, None);
        Ok(texture)
    }
}

/// The named-parameter uniform buffer (WGSL binding 4, GL binding point
/// [`PARAMS_BINDING_POINT`]): grow-only, so a draw re-allocates it only when its block is larger
/// than anything uploaded before (`render.rs` writes blocks with `glBufferSubData`).
#[derive(Debug)]
pub(super) struct ParamsBuffer {
    /// The GL buffer.
    pub buffer: glow::Buffer,
    /// Its allocated size in bytes (0 until the first upload).
    pub capacity: usize,
}

impl ParamsBuffer {
    /// Smallest allocation: every block up to 256 bytes fits the first allocation.
    const MIN_CAPACITY: usize = 256;

    /// The capacity to re-allocate to for a `needed`-byte block, or `None` when `capacity`
    /// already holds it. Grows to the next power of two (at least [`Self::MIN_CAPACITY`]), so a
    /// chain alternating block sizes settles after one allocation; never shrinks.
    #[must_use]
    pub fn grown_capacity(capacity: usize, needed: usize) -> Option<usize> {
        (needed > capacity).then(|| needed.checked_next_power_of_two().unwrap_or(needed).max(Self::MIN_CAPACITY))
    }
}

/// The objects shared by every layer painted through one backend.
#[derive(Debug)]
pub(super) struct Resources {
    /// Empty VAO: the vertex stage is attribute-less (`gl_VertexID`), but core profiles still
    /// require a bound VAO to draw.
    pub vertex_array: glow::VertexArray,
    /// The `fl` uniform buffer (re-filled per pass).
    pub uniform_buffer: glow::Buffer,
    /// The named-parameter uniform buffer (grow-only; each draw writes its block at offset 0).
    pub params: ParamsBuffer,
    /// 8-bit copy of the backdrop.
    pub capture: RenderTarget,
    /// Screen-sized renderbuffer for resolving a multisampled framebuffer (format matched to it);
    /// created on first need.
    pub resolve: Option<ResolveTarget>,
    /// Largest texture / renderbuffer side the context allows (`i32::MAX` when unknown).
    pub max_size: i32,
    /// Format of the default framebuffer's read buffer, queried once (it cannot change for the
    /// lifetime of the context); formats of application FBOs are queried per layer.
    pub default_read_format: Option<ReadFormat>,
    /// Read formats that were rejected (logged once each, never retried).
    pub rejected_formats: HashSet<ReadFormat>,
    /// Ping-pong pass outputs.
    pub ping_pong: [RenderTarget; 2],
    /// The composite program writing the result back.
    pub composite: glow::Program,
    /// 1x1 transparent texture for texture units a draw leaves unused (original, state, user
    /// texture slots with nothing attached).
    pub transparent: glow::Texture,
    /// 1x1 opaque white texture for the coverage units when there is no coverage texture.
    pub white: glow::Texture,
    /// The shape-mask coverage rasterizer (F9), created with the first shape-masked layer.
    pub coverage: Option<CoverageResources>,
}

impl Resources {
    /// Creates every shared object; intermediates are RGBA16F when the context can render to
    /// it, else RGBA8.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub unsafe fn create(gl: &glow::Context, target: GlslTarget) -> Result<Self, GlError> {
        let composite_glsl = composite_module().map_err(|error| GlError::Composite(error.to_string()))?;
        let composite_glsl = translate(&composite_glsl, target)?;
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object is created here, and `Partial` frees the created ones on the error path.
        unsafe {
            let composite = build_program(gl, &composite_glsl)?;
            let mut created = Partial { composite: Some(composite), ..Partial::default() };
            let result = created.fill(gl);
            match result {
                Ok(resources) => Ok(resources),
                Err(error) => {
                    created.destroy(gl);
                    Err(error)
                }
            }
        }
    }

    /// Whether the ping-pong targets are half-float.
    #[must_use]
    pub fn float_intermediates(&self) -> bool {
        self.ping_pong[0].format == TargetFormat::Rgba16F
    }

    /// Frees every object.
    ///
    /// # Safety
    /// `gl` must be current on this thread and own these objects.
    pub unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract) and owns every object (caller contract); `self` is consumed, so they are not used again.
        unsafe {
            gl.delete_vertex_array(self.vertex_array);
            gl.delete_buffer(self.uniform_buffer);
            gl.delete_buffer(self.params.buffer);
            self.capture.destroy(gl);
            if let Some(resolve) = self.resolve {
                resolve.destroy(gl);
            }
            let [a, b] = self.ping_pong;
            a.destroy(gl);
            b.destroy(gl);
            gl.delete_program(self.composite);
            gl.delete_texture(self.transparent);
            gl.delete_texture(self.white);
            if let Some(coverage) = self.coverage {
                coverage.destroy(gl);
            }
        }
    }
}

/// The objects created so far during `Resources::create`, so a failure can free exactly those.
#[derive(Default)]
struct Partial {
    composite: Option<glow::Program>,
    vertex_array: Option<glow::VertexArray>,
    uniform_buffer: Option<glow::Buffer>,
    params_buffer: Option<glow::Buffer>,
    capture: Option<RenderTarget>,
    ping: Option<RenderTarget>,
    pong: Option<RenderTarget>,
    transparent: Option<glow::Texture>,
    white: Option<glow::Texture>,
}

impl Partial {
    /// Creates the remaining objects, moving them out of `self` on success.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn fill(&mut self, gl: &glow::Context) -> Result<Resources, GlError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the uniform buffer is allocated with `UNIFORM_BLOCK_SIZE` bytes, the size every upload writes.
        unsafe {
            self.vertex_array = Some(gl.create_vertex_array().map_err(|detail| GlError::CreateResource { resource: "vertex array", detail })?);
            let buffer = gl.create_buffer().map_err(|detail| GlError::CreateResource { resource: "uniform buffer", detail })?;
            self.uniform_buffer = Some(buffer);
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(buffer));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, UNIFORM_BLOCK_SIZE_I32, glow::DYNAMIC_DRAW);
            // Allocated on first use and grown by `render.rs` (`ParamsBuffer::grown_capacity`).
            self.params_buffer = Some(gl.create_buffer().map_err(|detail| GlError::CreateResource { resource: "uniform buffer", detail })?);
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);

            let mut capture = RenderTarget::create(gl, TargetFormat::Rgba8)?;
            let capture_ok = capture.ensure_size(gl, [1, 1], i32::MAX);
            self.capture = Some(capture);
            capture_ok?;

            // Probe half-float renderability with a 1x1 allocation; an incomplete framebuffer
            // (GLES / WebGL2 without EXT_color_buffer_float) falls back to RGBA8 — a documented
            // precision difference, reported through `BackendStatus::Active`.
            let format = if Self::probe(gl, TargetFormat::Rgba16F) { TargetFormat::Rgba16F } else { TargetFormat::Rgba8 };
            for slot in [&mut self.ping, &mut self.pong] {
                let mut target = RenderTarget::create(gl, format)?;
                let ok = target.ensure_size(gl, [1, 1], i32::MAX);
                *slot = Some(target);
                ok?;
            }
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);

            self.transparent = Some(dummy_texture(gl, [0; 4])?);
            self.white = Some(dummy_texture(gl, [u8::MAX; 4])?);

            match (self.composite.take(), self.vertex_array.take(), self.uniform_buffer.take(), self.params_buffer.take(), self.capture.take(), self.ping.take(), self.pong.take(), self.transparent.take(), self.white.take()) {
                (Some(composite), Some(vertex_array), Some(uniform_buffer), Some(params_buffer), Some(capture), Some(first), Some(second), Some(transparent), Some(white)) => {
                    let max_size = [glow::MAX_TEXTURE_SIZE, glow::MAX_RENDERBUFFER_SIZE].into_iter().map(|limit| gl.get_parameter_i32(limit)).filter(|limit| *limit > 0).min().unwrap_or(i32::MAX);
                    Ok(Resources {
                        vertex_array,
                        uniform_buffer,
                        params: ParamsBuffer { buffer: params_buffer, capacity: 0 },
                        capture,
                        resolve: None,
                        max_size,
                        default_read_format: None,
                        rejected_formats: HashSet::new(),
                        ping_pong: [first, second],
                        composite,
                        transparent,
                        white,
                        coverage: None,
                    })
                }
                // Unreachable by construction (every field was set above); handled without a
                // panic anyway, reporting it as a creation failure.
                _ => Err(GlError::CreateResource { resource: "resource set", detail: "internal: partially created".to_owned() }),
            }
        }
    }

    /// Whether a 1x1 target of `format` is framebuffer-complete (the probe objects are freed).
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn probe(gl: &glow::Context, format: TargetFormat) -> bool {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the probe target is created and freed within this block.
        unsafe {
            let Ok(mut target) = RenderTarget::create(gl, format) else {
                return false;
            };
            let complete = target.ensure_size(gl, [1, 1], i32::MAX).is_ok();
            // Some drivers raise GL_INVALID_ENUM/OPERATION for an unsupported internal format
            // instead of (or before) reporting incompleteness; consume it so it is not blamed on
            // a later operation.
            let error = gl.get_error();
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            target.destroy(gl);
            if !complete || error != glow::NO_ERROR {
                log::info!("egui_shader_layers: {format:?} is not renderable on this context (GL error 0x{error:04x}); using RGBA8 intermediates");
            }
            complete && error == glow::NO_ERROR
        }
    }

    /// Frees whatever was created.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn destroy(&mut self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); each object is `take`n before it is freed, so none is freed twice.
        unsafe {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            if let Some(program) = self.composite.take() {
                gl.delete_program(program);
            }
            if let Some(vertex_array) = self.vertex_array.take() {
                gl.delete_vertex_array(vertex_array);
            }
            for buffer in [self.uniform_buffer.take(), self.params_buffer.take()].into_iter().flatten() {
                gl.delete_buffer(buffer);
            }
            for target in [self.capture.take(), self.ping.take(), self.pong.take()].into_iter().flatten() {
                target.destroy(gl);
            }
            for texture in [self.transparent.take(), self.white.take()].into_iter().flatten() {
                gl.delete_texture(texture);
            }
        }
    }
}

/// Compiles and links a translated program, binds its `fl` block to [`UNIFORM_BINDING_POINT`],
/// its named-parameter block (if any) to [`PARAMS_BINDING_POINT`] and its samplers to their
/// texture units.
///
/// Leaves no program bound.
///
/// # Safety
/// `gl` must be current on this thread.
pub(super) unsafe fn build_program(gl: &glow::Context, glsl: &GlslProgram) -> Result<glow::Program, GlError> {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the program and shaders are created here and freed again on every error path.
    unsafe {
        let program = gl.create_program().map_err(|detail| GlError::CreateResource { resource: "program", detail })?;
        let mut shaders: Vec<glow::Shader> = Vec::with_capacity(2);
        for (kind, stage, source) in [(glow::VERTEX_SHADER, "vertex", &glsl.vertex), (glow::FRAGMENT_SHADER, "fragment", &glsl.fragment)] {
            let shader = match gl.create_shader(kind) {
                Ok(shader) => shader,
                Err(detail) => {
                    delete_program(gl, program, &shaders);
                    return Err(GlError::CreateResource { resource: "shader", detail });
                }
            };
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                delete_program(gl, program, &shaders);
                return Err(GlError::CompileShader { stage, log });
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            delete_program(gl, program, &shaders);
            return Err(GlError::LinkProgram { log });
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        // GLSL 330 / ES 300 have no `layout(binding = N)`, so block and sampler bindings are
        // program state set once here (they persist for the program's lifetime).
        for (block, point) in [(&glsl.uniform_block, UNIFORM_BINDING_POINT), (&glsl.params_block, PARAMS_BINDING_POINT)] {
            if let Some(block) = block
                && let Some(index) = gl.get_uniform_block_index(program, block)
            {
                gl.uniform_block_binding(program, index, point);
            }
        }
        gl.use_program(Some(program));
        for (name, unit) in &glsl.samplers {
            // Uniform locations are looked up here and never stored: on wasm32 they are
            // `!Send` `WebGlUniformLocation`s and the backend state must be `Send`.
            let location = gl.get_uniform_location(program, name);
            gl.uniform_1_i32(location.as_ref(), i32::try_from(*unit).unwrap_or(0));
        }
        gl.use_program(None);
        Ok(program)
    }
}

/// Deletes a half-built program and the shaders already attached to it.
///
/// # Safety
/// `gl` must be current; `program` and every shader must still exist.
unsafe fn delete_program(gl: &glow::Context, program: glow::Program, shaders: &[glow::Shader]) {
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); `program` and `shaders` still exist (caller contract) and are not used after this.
    unsafe {
        for shader in shaders {
            gl.detach_shader(program, *shader);
            gl.delete_shader(*shader);
        }
        gl.delete_program(program);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_formats_map_to_sized_internal_formats() {
        assert_eq!(TargetFormat::Rgba8.tex_image_params(), (gl_i32(glow::RGBA8), glow::RGBA, glow::UNSIGNED_BYTE));
        assert_eq!(TargetFormat::Rgba16F.tex_image_params(), (gl_i32(glow::RGBA16F), glow::RGBA, glow::HALF_FLOAT));
    }

    #[test]
    fn targets_grow_in_steps_and_never_shrink() {
        assert_eq!(grown_size([0, 0], [1, 1], i32::MAX), [64, 64]);
        assert_eq!(grown_size([64, 64], [65, 10], i32::MAX), [128, 64], "only the axis that outgrew changes");
        assert_eq!(grown_size([128, 64], [100, 64], i32::MAX), [128, 64], "fits: unchanged");
        assert_eq!(grown_size([0, 0], [1000, 1], 1010), [1010, 64], "rounded up to 1024 but clamped to the limit");
        assert_eq!(grown_size([0, 0], [2000, 1], 1024), [2000, 64], "never below what is needed");
        assert_eq!(grown_size([0, 0], [i32::MAX, 0], i32::MAX), [i32::MAX, 1], "no overflow, no zero axis");
    }

    #[test]
    fn the_params_buffer_only_grows_in_powers_of_two() {
        assert_eq!(ParamsBuffer::grown_capacity(0, 16), Some(256));
        assert_eq!(ParamsBuffer::grown_capacity(256, 16), None, "a smaller block reuses the allocation");
        assert_eq!(ParamsBuffer::grown_capacity(256, 256), None);
        assert_eq!(ParamsBuffer::grown_capacity(256, 272), Some(512));
        assert_eq!(ParamsBuffer::grown_capacity(512, 16 * 1024), Some(16 * 1024), "the 16 KiB cap is a power of two");
    }

    #[test]
    fn resolve_formats_match_the_read_buffer() {
        let format = |bits: [i32; 4], component_type: u32, encoding: u32| ReadFormat { bits, component_type, encoding };
        let unorm = glow::UNSIGNED_NORMALIZED;
        assert_eq!(format([8, 8, 8, 8], unorm, glow::LINEAR).resolve_format(), Ok(glow::RGBA8));
        assert_eq!(format([8, 8, 8, 0], unorm, glow::LINEAR).resolve_format(), Ok(glow::RGB8), "WebGL2 alpha:false / RGB888 EGL configs");
        assert_eq!(format([5, 6, 5, 0], unorm, glow::LINEAR).resolve_format(), Ok(glow::RGB565));
        assert_eq!(format([10, 10, 10, 2], unorm, glow::LINEAR).resolve_format(), Ok(glow::RGB10_A2));
        assert_eq!(format([8, 8, 8, 8], unorm, glow::SRGB).resolve_format(), Ok(glow::SRGB8_ALPHA8));
        assert!(format([8, 8, 8, 0], unorm, glow::SRGB).resolve_format().is_err(), "SRGB8 is not colour-renderable on GLES 3");
        let half = format([16, 16, 16, 16], glow::FLOAT, glow::LINEAR);
        assert!(!half.is_normalized());
        assert!(half.resolve_format().is_err_and(|reason| reason.contains("not a normalized")));
        assert!(format([6, 6, 6, 6], unorm, glow::LINEAR).resolve_format().is_err());
    }
}
