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
- `translate_mesh()`: the shape-mask coverage rasterizer's program (`shaders/mesh.wgsl`, F9).

Notes:
Writer flags are EMPTY on purpose: `ADJUST_COORDINATE_SPACE` (naga's default) would flip clip-space
y, but the crate's coordinate convention is identical on every backend and the GL storage flip is
handled in the prelude's `src()` (`internal_flip_y`). Uniform-block and sampler names are taken
from naga's `ReflectionInfo`, never guessed: the `fl` block (binding 0) and, for effects with
`struct Params`, the named-parameter block (binding 4) are found by their `@binding`.
Texture units (`texture_unit`): GLSL combines texture and sampler, so a unit is chosen per
`(texture, sampler)` pair of the prelude's layout — a `@group(0)` texture's position in
`GROUP0_TEXTURE_BINDINGS` (through the linear `fl_sampler` or with no sampler, `textureLoad`), and
user texture slot `i` (`@group(1 + i)`, through its own sampler or none) on unit
`USER_UNIT_BASE + i` (5..=7). At most unit 7 is used: GLES 3.0 / WebGL2 guarantee 16 fragment
texture units. naga's writer rejects one texture read through two samplers in one module ("A image
was used with multiple samplers"), so the prelude reads every texture through one sampler and
exact reads through `textureLoad`, which naga combines with that same sampler.
*/

use naga::back::glsl;

use crate::effect::{
    BINDING_COVERAGE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_PARAMS, BINDING_REGION_TEXTURE, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, CompiledModule, FRAGMENT_ENTRY, TEXTURE_SLOTS, USER_SAMPLER_BINDING, USER_TEXTURE_BINDING, USER_TEXTURE_GROUP_BASE,
    VERTEX_ENTRY,
};

/// The prelude's `@group(0)` texture bindings; a texture's index here is its texture unit
/// (`render.rs` binds units in this same order): 0 src, 1 original, 2 state, 3 coverage,
/// 4 region ids (R1).
pub(super) const GROUP0_TEXTURE_BINDINGS: [u32; 5] = [BINDING_SOURCE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_STATE_TEXTURE, BINDING_COVERAGE_TEXTURE, BINDING_REGION_TEXTURE];

/// Texture unit of user texture slot 0; slot `i` is unit `USER_UNIT_BASE + i` (read through the
/// texture object's own sampling parameters: no sampler object is bound there).
pub(super) const USER_UNIT_BASE: u32 = 5;

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
    /// Name of the named-parameter block (`p: Params`, binding 4), if the fragment uses it.
    pub params_block: Option<String>,
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

/// Texture unit of a combined sampler reading the texture at `texture` through the sampler at
/// `sampler` (both `(@group, @binding)`; `None` = no sampler, e.g. `textureLoad`), or `None` for a
/// pair the layout does not have (a texture read through another texture's sampler).
fn texture_unit(texture: (u32, u32), sampler: Option<(u32, u32)>) -> Option<u32> {
    match texture {
        (0, binding) => {
            let index = u32::try_from(GROUP0_TEXTURE_BINDINGS.iter().position(|&candidate| candidate == binding)?).ok()?;
            matches!(sampler, None | Some((0, BINDING_SAMPLER))).then_some(index)
        }
        (group, USER_TEXTURE_BINDING) => {
            let slot = group.checked_sub(USER_TEXTURE_GROUP_BASE).filter(|&slot| usize::try_from(slot).is_ok_and(|slot| slot < TEXTURE_SLOTS))?;
            // A user texture is read only through the sampler of its own group.
            (sampler.is_none() || sampler == Some((group, USER_SAMPLER_BINDING))).then_some(USER_UNIT_BASE + slot)
        }
        _ => None,
    }
}

/// Translates one entry point of `module` (validated as `info`), returning the source and naga's
/// reflection.
fn write_stage(module: &naga::Module, info: &naga::valid::ModuleInfo, target: GlslTarget, stage: naga::ShaderStage, entry: &str) -> Result<(String, glsl::ReflectionInfo), TranslateError> {
    let stage_name = if stage == naga::ShaderStage::Vertex { "vertex" } else { "fragment" };
    let options = glsl::Options { version: target.naga_version(), writer_flags: glsl::WriterFlags::empty(), ..glsl::Options::default() };
    let pipeline = glsl::PipelineOptions { shader_stage: stage, entry_point: entry.to_owned(), multiview: None };
    let mut source = String::new();
    let reflection = glsl::Writer::new(&mut source, module, info, &options, &pipeline, naga::proc::BoundsCheckPolicies::default())
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
    let (vertex, _) = write_stage(&compiled.module, &compiled.info, target, naga::ShaderStage::Vertex, VERTEX_ENTRY)?;
    let (fragment, reflection) = write_stage(&compiled.module, &compiled.info, target, naga::ShaderStage::Fragment, FRAGMENT_ENTRY)?;

    let binding_of = |handle: naga::Handle<naga::GlobalVariable>| compiled.module.global_variables[handle].binding.as_ref().map(|b| (b.group, b.binding));

    let block_at = |binding: u32| reflection.uniforms.iter().find(|(handle, _)| binding_of(**handle) == Some((0, binding))).map(|(_, name)| name.clone());
    let uniform_block = block_at(BINDING_UNIFORMS);
    let params_block = block_at(BINDING_PARAMS);

    let mut samplers = Vec::with_capacity(reflection.texture_mapping.len());
    for (name, mapping) in &reflection.texture_mapping {
        let unit = binding_of(mapping.texture).and_then(|texture| texture_unit(texture, mapping.sampler.and_then(binding_of))).ok_or_else(|| TranslateError { stage: "fragment", message: format!("sampler `{name}` is bound to no known texture unit") })?;
        samplers.push((name.clone(), unit));
    }
    samplers.sort_by_key(|(_, unit)| *unit);
    Ok(GlslProgram { vertex, fragment, uniform_block, params_block, samplers })
}

/// The mesh program (`shaders/mesh.wgsl`, `vs_main` + `fs_main`) that rasterizes shape-mask
/// coverage (F9), translated like an effect.
#[derive(Clone, Debug)]
pub(crate) struct MeshProgram {
    /// Vertex shader source (attributes at locations 0 = pos, 1 = uv, 2 = packed colour `uint`).
    pub vertex: String,
    /// Fragment shader source.
    pub fragment: String,
    /// The `MeshLocals` uniform block's name in each stage that uses it (naga names blocks per
    /// stage); every one is bound to the same binding point.
    pub locals_blocks: Vec<String>,
    /// `(sampler uniform name, texture unit)`: the mesh texture on unit 0.
    pub samplers: Vec<(String, u32)>,
}

/// Translates `shaders/mesh.wgsl` (the vertex stage and the gamma fragment entry `fs_main`) to
/// `target` GLSL: the coverage rasterizer's program on glow. The module is parsed and validated
/// here (once per `Resources` creation).
///
/// # Errors
/// [`TranslateError`] when the crate's mesh shader does not parse, validate or translate (a
/// crate bug, pinned by `the_mesh_shader_translates_for_every_target`).
pub(crate) fn translate_mesh(target: GlslTarget) -> Result<MeshProgram, TranslateError> {
    let source = crate::mask::MESH_WGSL;
    let module = naga::front::wgsl::parse_str(source).map_err(|error| TranslateError { stage: "vertex", message: error.emit_to_string(source) })?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).map_err(|error| TranslateError { stage: "vertex", message: error.to_string() })?;
    let (vertex, vertex_reflection) = write_stage(&module, &info, target, naga::ShaderStage::Vertex, "vs_main")?;
    let (fragment, fragment_reflection) = write_stage(&module, &info, target, naga::ShaderStage::Fragment, "fs_main")?;
    let locals_blocks = vertex_reflection.uniforms.values().chain(fragment_reflection.uniforms.values()).cloned().collect();
    // The mesh module has exactly one texture (group 1, binding 0) read through one sampler.
    let samplers = fragment_reflection.texture_mapping.keys().map(|name| (name.clone(), 0)).collect();
    Ok(MeshProgram { vertex, fragment, locals_blocks, samplers })
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
                    // Unit 0 = the pass input, 1 = `original()` (bloom), 2 = the state (F7).
                    Ok(program) => {
                        let units: Vec<u32> = [("src", 0), ("original(", 1), ("state", 2)].iter().filter(|(name, _)| source.contains(name)).map(|&(_, unit)| unit).collect();
                        check_program(label, &program, target, &units);
                    }
                    Err(error) => panic!("{label} {target:?}: {error}"),
                }
            }
        }
    }

    /// Units 0 (the last pass output), 1 (the original) and 3 (the shape-mask coverage, F9; the
    /// white dummy for a rounded rect).
    #[test]
    fn the_composite_pass_translates_with_three_texture_units() {
        let Ok(composite) = composite_module() else { panic!("composite must validate") };
        for target in TARGETS {
            match translate(&composite, target) {
                Ok(program) => check_program("composite", &program, target, &[0, 1, 3]),
                Err(error) => panic!("composite {target:?}: {error}"),
            }
        }
    }

    #[test]
    fn an_effect_reading_the_original_translates_with_two_texture_units() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) - original(uv); }";
        let Ok(effect) = Effect::new("original", snippet) else { panic!("must validate") };
        for target in TARGETS {
            match translate(effect.module(), target) {
                Ok(program) => check_program("original", &program, target, &[0, 1]),
                Err(error) => panic!("original {target:?}: {error}"),
            }
        }
    }

    /// F4: an effect with `struct Params` translates to a program with TWO uniform blocks, each
    /// found by its binding; the texture units are unchanged.
    #[test]
    fn a_named_params_effect_translates_with_two_blocks() {
        let snippet = "struct Params { color: vec4<f32>, lights: array<vec4<f32>, 4>, gain: f32, count: u32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.color * p.gain + p.lights[p.count]; }";
        let effect = Effect::new("named", snippet);
        let Ok(effect) = effect else { panic!("{effect:?}") };
        for target in TARGETS {
            match translate(effect.module(), target) {
                Ok(program) => {
                    check_program("named", &program, target, &[0]);
                    let Some(block) = &program.params_block else { panic!("{target:?}: the params block must be reflected\n{}", program.fragment) };
                    assert_ne!(Some(block), program.uniform_block.as_ref(), "two distinct blocks");
                    assert!(program.fragment.contains(block.as_str()) && program.fragment.contains("std140"), "{target:?}\n{}", program.fragment);
                }
                Err(error) => panic!("named {target:?}: {error}"),
            }
        }
        // Without `struct Params` there is no second block.
        let Ok(plain) = Effect::new("plain", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("must validate") };
        assert!(translate(plain.module(), GlslTarget::Es300).is_ok_and(|program| program.params_block.is_none()));
    }

    #[test]
    fn texture_units_follow_the_prelude_layout() {
        assert_eq!(texture_unit((0, BINDING_SOURCE_TEXTURE), Some((0, BINDING_SAMPLER))), Some(0));
        assert_eq!(texture_unit((0, BINDING_ORIGINAL_TEXTURE), Some((0, BINDING_SAMPLER))), Some(1));
        assert_eq!(texture_unit((0, BINDING_STATE_TEXTURE), None), Some(2));
        assert_eq!(texture_unit((0, BINDING_COVERAGE_TEXTURE), Some((0, BINDING_SAMPLER))), Some(3));
        assert_eq!(texture_unit((0, BINDING_REGION_TEXTURE), None), Some(4));
        assert_eq!(USER_UNIT_BASE, 5);
        assert_eq!(texture_unit((1, USER_TEXTURE_BINDING), Some((1, USER_SAMPLER_BINDING))), Some(USER_UNIT_BASE));
        assert_eq!(texture_unit((3, USER_TEXTURE_BINDING), None), Some(USER_UNIT_BASE + 2));
        assert_eq!(texture_unit((0, BINDING_UNIFORMS), None), None);
        assert_eq!(texture_unit((0, BINDING_SOURCE_TEXTURE), Some((0, BINDING_ORIGINAL_TEXTURE))), None);
        assert_eq!(texture_unit((0, BINDING_SOURCE_TEXTURE), Some((1, USER_SAMPLER_BINDING))), None, "a group-0 texture through a user sampler");
        assert_eq!(texture_unit((2, USER_TEXTURE_BINDING), Some((1, USER_SAMPLER_BINDING))), None, "a user texture through another slot's sampler");
        assert_eq!(texture_unit((4, USER_TEXTURE_BINDING), None), None, "only three slots");
    }

    /// F8: the texture helpers translate for every dialect. Sampled (`tex`) and exact
    /// (`tex_load`, `texelFetch`) reads of one texture share ONE combined sampler (naga's writer
    /// allows a texture with a sampler plus sampler-less loads), so each slot is one unit.
    #[test]
    fn user_texture_helpers_translate_one_unit_per_slot() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let exact = tex_load(0u, vec2<i32>(uv * tex_size(0u)));
    return tex(0u, uv) + exact + tex(1u, uv) * tex_load(2u, vec2<i32>(1, 0)) + src(uv);
}";
        let effect = Effect::new("textures", snippet);
        let Ok(effect) = effect else { panic!("{effect:?}") };
        for target in TARGETS {
            match translate(effect.module(), target) {
                Ok(program) => {
                    check_program("textures", &program, target, &[0, USER_UNIT_BASE, USER_UNIT_BASE + 1, USER_UNIT_BASE + 2]);
                    assert!(program.fragment.contains("texelFetch") && program.fragment.contains("textureLod") && program.fragment.contains("textureSize"), "{target:?}\n{}", program.fragment);
                }
                Err(error) => panic!("textures {target:?}: {error}"),
            }
        }
    }

    /// F7: sampled (`state`) and exact (`state_load` / `state_px`, `texelFetch`) reads of the
    /// state share ONE combined sampler on unit 2, beside the pass input and a user texture read
    /// both ways — no texture is read through two samplers, so every dialect translates it.
    #[test]
    fn state_helpers_translate_with_src_and_a_user_texture() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let exact = state_load(vec2<i32>(2, 3)) + state_px(vec2<i32>(-1, 0)) * state_size().x * state_texel().y;
    return state(uv) + exact + src(uv) + tex(0u, uv) + tex_load(0u, vec2<i32>(0));
}";
        for build in [Effect::new_state, Effect::new] {
            let effect = build("state", snippet);
            let Ok(effect) = effect else { panic!("{effect:?}") };
            for target in TARGETS {
                match translate(effect.module(), target) {
                    Ok(program) => {
                        // `tex(0u, ..)` switches over every slot, so all three user units are referenced.
                        check_program("state", &program, target, &[0, 2, USER_UNIT_BASE, USER_UNIT_BASE + 1, USER_UNIT_BASE + 2]);
                        assert!(program.fragment.contains("texelFetch") && program.fragment.contains("textureLod"), "{target:?}\n{}", program.fragment);
                    }
                    Err(error) => panic!("state {target:?}: {error}"),
                }
            }
        }
    }

    /// R1: `region()` (a sampler-less `textureLoad` of the region texture, unit 4) translates with
    /// `src()` and a user texture for every dialect: units `[0, 4, 5..]`.
    #[test]
    fn region_helpers_translate_with_src_and_a_user_texture() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let same = region_at(uv + fl.texel) == region();
    return select(src(uv), tex(0u, uv), same && region() != REGION_NONE);
}";
        let effect = Effect::new("regions", snippet);
        let Ok(effect) = effect else { panic!("{effect:?}") };
        for target in TARGETS {
            match translate(effect.module(), target) {
                Ok(program) => {
                    check_program("regions", &program, target, &[0, 4, USER_UNIT_BASE, USER_UNIT_BASE + 1, USER_UNIT_BASE + 2]);
                    assert!(program.fragment.contains("texelFetch"), "{target:?}\n{}", program.fragment);
                }
                Err(error) => panic!("regions {target:?}: {error}"),
            }
        }
    }

    /// naga's GLSL back end refuses one texture read through two samplers in one program
    /// ("A image was used with multiple samplers"): the reason the prelude gives every texture
    /// exactly one sampler; pinned so a naga change is noticed. Also the trigger of the glow
    /// per-effect failure scene (`tests/glow_headless.rs`): valid WGSL that glow cannot translate.
    #[test]
    fn one_texture_through_two_samplers_does_not_translate() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) + textureSampleLevel(fl_src_texture, fl_user0_sampler, uv, 0.0); }";
        let Ok(effect) = Effect::new("mixed", snippet) else { panic!("must validate") };
        assert!(translate(effect.module(), GlslTarget::Desktop330).is_err_and(|error| error.message.contains("multiple samplers")));
    }

    /// F9 (roadmap risk 3): naga's GLSL writer translates the mesh shader for every dialect —
    /// three vertex attributes with explicit locations (the `u32` colour as an integer
    /// attribute), one uniform block per stage, one combined sampler.
    #[test]
    fn the_mesh_shader_translates_for_every_target() {
        for target in TARGETS {
            let program = translate_mesh(target);
            let Ok(program) = program else { panic!("{target:?}: {program:?}") };
            for location in 0..3 {
                assert!(program.vertex.contains(&format!("layout(location = {location}) in")), "{target:?}: attribute {location}\n{}", program.vertex);
            }
            assert!(program.vertex.contains(") in uint "), "{target:?}: the colour is an integer attribute\n{}", program.vertex);
            assert!(!program.vertex.contains("gl_Position.yz"), "coordinate-space adjustment must be off");
            assert_eq!(program.locals_blocks.len(), 2, "{target:?}: {:?}", program.locals_blocks);
            assert!(program.locals_blocks.iter().all(|block| program.vertex.contains(block.as_str()) || program.fragment.contains(block.as_str())));
            assert_eq!(program.samplers.len(), 1, "{target:?}: {:?}", program.samplers);
            assert!(program.fragment.contains(program.samplers[0].0.as_str()));
        }
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
