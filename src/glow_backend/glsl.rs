/*
File: glow_backend/glsl.rs

Purpose:
Pure (GL-free) translation of a validated effect module into the GLSL program the glow backend
compiles: GLSL 330 core on desktop GL, GLSL ES 300 on GLES 3 / WebGL2.

Key structures:
- `GlslTarget`: which GLSL dialect to emit.
- `GlslProgram`: vertex + fragment source plus the reflected names the GL side must bind.

Key functions:
- `glsl_target()`: picks the dialect for a GL version, or explains why the context is unsupported.
- `translate()`: naga `glsl-out` for both entry points + reflection lookup.

Notes:
Writer flags are EMPTY on purpose: `ADJUST_COORDINATE_SPACE` (naga's default) would flip clip-space
y, but the crate's coordinate convention is identical on every backend and the GL storage flip is
handled in the prelude's `src()` (`internal_flip_y`). Uniform-block and sampler names are taken
from naga's `ReflectionInfo`, never guessed.
Texture units (`texture_unit`): GLSL combines texture and sampler, so a unit is chosen per
`(texture binding, sampler binding)` pair of the prelude's `@group(0)` layout — the texture's
position in `GROUP0_TEXTURE_BINDINGS` through the linear sampler (or with no sampler), that plus
`NEAREST_UNIT_BASE` through `fl_nearest_sampler`. Units 4..=6 stay free for user textures. At most
unit 10 is used: GLES 3.0 / WebGL2 guarantee 16 fragment texture units. naga's writer rejects one
texture read through two samplers in one module ("A image was used with multiple samplers").
*/

use naga::back::glsl;

use crate::effect::{BINDING_COVERAGE_TEXTURE, BINDING_NEAREST_SAMPLER, BINDING_ORIGINAL_TEXTURE, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, CompiledModule, FRAGMENT_ENTRY, VERTEX_ENTRY};

/// The prelude's `@group(0)` texture bindings; a texture's index here is its texture unit when
/// read through the linear sampler (`render.rs` binds units in this same order).
pub(super) const GROUP0_TEXTURE_BINDINGS: [u32; 4] = [BINDING_SOURCE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_STATE_TEXTURE, BINDING_COVERAGE_TEXTURE];

/// First texture unit of the nearest-sampler combinations: unit `NEAREST_UNIT_BASE + i` holds
/// `GROUP0_TEXTURE_BINDINGS[i]` with the backend's nearest GL sampler object bound.
pub(super) const NEAREST_UNIT_BASE: u32 = 7;

/// Which GLSL dialect to emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlslTarget {
    /// `#version 330 core` (desktop OpenGL 3.3+).
    Desktop330,
    /// `#version 300 es` on native OpenGL ES 3.0+.
    Es300,
    /// `#version 300 es` for WebGL2.
    WebGl2,
}

impl GlslTarget {
    /// The naga version value for this dialect.
    fn naga_version(self) -> glsl::Version {
        match self {
            Self::Desktop330 => glsl::Version::Desktop(330),
            Self::Es300 => glsl::Version::Embedded { version: 300, is_webgl: false },
            Self::WebGl2 => glsl::Version::Embedded { version: 300, is_webgl: true },
        }
    }
}

/// Chooses the GLSL dialect for a GL context.
///
/// `major`/`minor`/`embedded` come from `glow::Context::version()`, which reports every WebGL
/// version as embedded with major + 1: `WebGL2` is `3.0`, `WebGL1` is `2.0`. `web` is whether this
/// is a browser context (every GL context on wasm32 is WebGL).
///
/// # Errors
/// A human-readable reason when the context is too old: `WebGL1` / OpenGL ES 2 (no uniform
/// buffers, no framebuffer blits, no guaranteed VAOs) or desktop OpenGL below 3.3.
pub(crate) fn glsl_target(major: u32, minor: u32, embedded: bool, web: bool) -> Result<GlslTarget, String> {
    if embedded {
        if major >= 3 {
            Ok(if web { GlslTarget::WebGl2 } else { GlslTarget::Es300 })
        } else if web {
            // glow's numbers are shifted for WebGL (see above), so they are not echoed here.
            Err("WebGL1 context; WebGL2 is required".to_owned())
        } else {
            Err(format!("OpenGL ES {major}.{minor} context; OpenGL ES 3.0 or newer is required"))
        }
    } else if (major, minor) >= (3, 3) {
        Ok(GlslTarget::Desktop330)
    } else {
        Err(format!("OpenGL {major}.{minor} context; OpenGL 3.3 or newer is required"))
    }
}

/// A translated program and the reflected names the GL side binds by.
#[derive(Clone, Debug)]
pub(crate) struct GlslProgram {
    /// Vertex shader source.
    pub vertex: String,
    /// Fragment shader source.
    pub fragment: String,
    /// Name of the `fl` uniform block in the fragment shader, if the fragment uses it.
    pub uniform_block: Option<String>,
    /// `(sampler uniform name, texture unit)` pairs to set once after linking.
    pub samplers: Vec<(String, u32)>,
}

/// Why naga could not translate a module to GLSL.
#[derive(Clone, Debug, thiserror::Error)]
#[error("GLSL translation of the {stage} stage failed: {message}")]
pub(crate) struct TranslateError {
    /// `vertex` or `fragment`.
    pub stage: &'static str,
    /// naga's error.
    pub message: String,
}

/// Texture unit of a combined sampler reading texture `texture` through sampler `sampler` (both
/// `@binding`s; `None` = no sampler, e.g. `textureLoad`), or `None` for an unknown pair.
fn texture_unit(texture: u32, sampler: Option<u32>) -> Option<u32> {
    let index = u32::try_from(GROUP0_TEXTURE_BINDINGS.iter().position(|&binding| binding == texture)?).ok()?;
    match sampler {
        None | Some(BINDING_SAMPLER) => Some(index),
        Some(BINDING_NEAREST_SAMPLER) => Some(NEAREST_UNIT_BASE + index),
        Some(_) => None,
    }
}

/// Translates one entry point, returning the source and naga's reflection.
fn write_stage(compiled: &CompiledModule, target: GlslTarget, stage: naga::ShaderStage, entry: &str) -> Result<(String, glsl::ReflectionInfo), TranslateError> {
    let stage_name = if stage == naga::ShaderStage::Vertex { "vertex" } else { "fragment" };
    let options = glsl::Options { version: target.naga_version(), writer_flags: glsl::WriterFlags::empty(), ..glsl::Options::default() };
    let pipeline = glsl::PipelineOptions { shader_stage: stage, entry_point: entry.to_owned(), multiview: None };
    let mut source = String::new();
    let reflection = glsl::Writer::new(&mut source, &compiled.module, &compiled.info, &options, &pipeline, naga::proc::BoundsCheckPolicies::default())
        .and_then(|mut writer| writer.write())
        .map_err(|error| TranslateError { stage: stage_name, message: error.to_string() })?;
    Ok((source, reflection))
}

/// Translates both entry points of `compiled` to `target` GLSL.
///
/// # Errors
/// [`TranslateError`] when naga's GLSL back end rejects the module (e.g. a construct the target
/// dialect cannot express) or when a sampler maps to an unknown binding.
pub(crate) fn translate(compiled: &CompiledModule, target: GlslTarget) -> Result<GlslProgram, TranslateError> {
    let (vertex, _) = write_stage(compiled, target, naga::ShaderStage::Vertex, VERTEX_ENTRY)?;
    let (fragment, reflection) = write_stage(compiled, target, naga::ShaderStage::Fragment, FRAGMENT_ENTRY)?;

    let binding_of = |handle: naga::Handle<naga::GlobalVariable>| compiled.module.global_variables[handle].binding.as_ref().map(|b| b.binding);

    let uniform_block = reflection.uniforms.iter().find(|(handle, _)| binding_of(**handle) == Some(BINDING_UNIFORMS)).map(|(_, name)| name.clone());

    let mut samplers = Vec::with_capacity(reflection.texture_mapping.len());
    for (name, mapping) in &reflection.texture_mapping {
        let unit = binding_of(mapping.texture).and_then(|texture| texture_unit(texture, mapping.sampler.and_then(binding_of))).ok_or_else(|| TranslateError { stage: "fragment", message: format!("sampler `{name}` is bound to no known texture unit") })?;
        samplers.push((name.clone(), unit));
    }
    samplers.sort_by_key(|(_, unit)| *unit);
    Ok(GlslProgram { vertex, fragment, uniform_block, samplers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{Effect, composite_module};
    use crate::presets::BUILTIN_EFFECTS;

    const TARGETS: [GlslTarget; 3] = [GlslTarget::Desktop330, GlslTarget::Es300, GlslTarget::WebGl2];

    fn check_program(label: &str, program: &GlslProgram, target: GlslTarget, texture_units: &[u32]) {
        let header = if target == GlslTarget::Desktop330 { "#version 330 core" } else { "#version 300 es" };
        assert!(program.vertex.starts_with(header), "{label} {target:?}: vertex header\n{}", program.vertex);
        assert!(program.fragment.starts_with(header), "{label} {target:?}: fragment header\n{}", program.fragment);
        assert!(program.uniform_block.is_some(), "{label} {target:?}: the fragment must use the fl block\n{}", program.fragment);
        let units: Vec<u32> = program.samplers.iter().map(|(_, unit)| *unit).collect();
        assert_eq!(units, texture_units, "{label} {target:?}: samplers {:?}", program.samplers);
        for (name, _) in &program.samplers {
            assert!(program.fragment.contains(name.as_str()), "{label}: sampler {name} not in source");
        }
        if let Some(block) = &program.uniform_block {
            assert!(program.fragment.contains(block.as_str()), "{label}: block {block} not in source");
        }
        assert!(program.vertex.contains("gl_VertexID"), "{label}: the vertex stage is attribute-less");
        assert!(!program.vertex.contains("gl_Position.yz"), "{label}: coordinate-space adjustment must be off");
    }

    #[test]
    fn every_builtin_effect_translates_for_every_target() {
        for (label, source) in BUILTIN_EFFECTS {
            let Ok(effect) = Effect::new(label, source) else { panic!("{label} must validate") };
            for target in TARGETS {
                match translate(effect.module(), target) {
                    Ok(program) => check_program(label, &program, target, &[0]),
                    Err(error) => panic!("{label} {target:?}: {error}"),
                }
            }
        }
    }

    #[test]
    fn the_composite_pass_translates_with_two_texture_units() {
        let Ok(composite) = composite_module() else { panic!("composite must validate") };
        for target in TARGETS {
            match translate(&composite, target) {
                Ok(program) => check_program("composite", &program, target, &[0, 1]),
                Err(error) => panic!("composite {target:?}: {error}"),
            }
        }
    }

    #[test]
    fn texture_units_follow_the_group0_layout() {
        assert_eq!(texture_unit(BINDING_SOURCE_TEXTURE, Some(BINDING_SAMPLER)), Some(0));
        assert_eq!(texture_unit(BINDING_ORIGINAL_TEXTURE, Some(BINDING_SAMPLER)), Some(1));
        assert_eq!(texture_unit(BINDING_STATE_TEXTURE, None), Some(2));
        assert_eq!(texture_unit(BINDING_COVERAGE_TEXTURE, Some(BINDING_SAMPLER)), Some(3));
        assert_eq!(texture_unit(BINDING_SOURCE_TEXTURE, Some(BINDING_NEAREST_SAMPLER)), Some(7));
        assert_eq!(texture_unit(BINDING_COVERAGE_TEXTURE, Some(BINDING_NEAREST_SAMPLER)), Some(10));
        assert_eq!(texture_unit(BINDING_UNIFORMS, None), None);
        assert_eq!(texture_unit(BINDING_SOURCE_TEXTURE, Some(BINDING_ORIGINAL_TEXTURE)), None);
    }

    #[test]
    fn a_nearest_read_gets_its_own_unit() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) + textureSampleLevel(fl_state_texture, fl_nearest_sampler, uv, 0.0); }";
        let Ok(effect) = Effect::new("nearest", snippet) else { panic!("must validate") };
        for target in TARGETS {
            match translate(effect.module(), target) {
                Ok(program) => check_program("nearest", &program, target, &[0, NEAREST_UNIT_BASE + 2]),
                Err(error) => panic!("nearest {target:?}: {error}"),
            }
        }
    }

    /// naga's GLSL back end refuses one texture read through two samplers in one program
    /// ("A image was used with multiple samplers"), so on glow a texture is read EITHER linearly
    /// or nearest within one module; pinned so a naga change is noticed.
    #[test]
    fn one_texture_through_two_samplers_does_not_translate() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) + textureSampleLevel(fl_src_texture, fl_nearest_sampler, uv, 0.0); }";
        let Ok(effect) = Effect::new("mixed", snippet) else { panic!("must validate") };
        assert!(translate(effect.module(), GlslTarget::Desktop330).is_err_and(|error| error.message.contains("multiple samplers")));
    }

    #[test]
    fn old_contexts_are_rejected_with_a_reason() {
        assert_eq!(glsl_target(3, 3, false, false), Ok(GlslTarget::Desktop330));
        assert_eq!(glsl_target(4, 6, false, false), Ok(GlslTarget::Desktop330));
        assert_eq!(glsl_target(3, 0, true, false), Ok(GlslTarget::Es300));
        assert_eq!(glsl_target(3, 0, true, true), Ok(GlslTarget::WebGl2));
        // What glow 0.17 reports for a WebGL1 context: "2.0", embedded.
        assert_eq!(glsl_target(2, 0, true, true), Err("WebGL1 context; WebGL2 is required".to_owned()));
        assert!(glsl_target(2, 0, true, false).is_err_and(|reason| reason.contains("OpenGL ES 2.0")));
        assert!(glsl_target(2, 0, true, false).is_err());
        assert!(glsl_target(3, 2, false, false).is_err());
        assert!(glsl_target(2, 1, false, false).is_err());
    }
}
