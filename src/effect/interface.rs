/*
File: effect/interface.rs

Purpose:
Checks on the validated naga IR of an assembled effect: which resources, entry points and
constants a module may declare (the one `@group(0)` layout of the prelude), and whether it reads
`fl.time` (animated effects).

Key functions:
- `check_interface()`: every bound global must be one the prelude declares (`PRELUDE_RESOURCES`)
  or the crate-appended named-parameter block.
- `reads_time()`: conservative IR scan deciding `Effect::is_animated`.
*/

use super::params::PARAMS_GLOBAL;
use super::{BINDING_COVERAGE_TEXTURE, BINDING_NEAREST_SAMPLER, BINDING_ORIGINAL_TEXTURE, BINDING_PARAMS, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, EffectError, FRAGMENT_ENTRY, VERTEX_ENTRY};

/// Every resource the prelude declares, as `(name, @binding)` in `@group(0)`. The interface check
/// accepts exactly these: a snippet cannot redeclare a prelude name (naga rejects the
/// redefinition), so a bound global with one of these names IS the prelude's.
pub(super) const PRELUDE_RESOURCES: [(&str, u32); 7] = [
    ("fl", BINDING_UNIFORMS),
    ("fl_src_texture", BINDING_SOURCE_TEXTURE),
    ("fl_sampler", BINDING_SAMPLER),
    ("fl_original_texture", BINDING_ORIGINAL_TEXTURE),
    ("fl_state_texture", BINDING_STATE_TEXTURE),
    ("fl_coverage_texture", BINDING_COVERAGE_TEXTURE),
    ("fl_nearest_sampler", BINDING_NEAREST_SAMPLER),
];

/// Rejects entry points, `override`s, and every resource binding that is not one of the
/// prelude's ([`PRELUDE_RESOURCES`]) or, when `named_params` (the crate appended the declaration
/// for the snippet's `struct Params`), the named-parameter block `p` at `BINDING_PARAMS`; the same
/// rule holds for user effects and the composite.
pub(super) fn check_interface(label: &str, module: &naga::Module, named_params: bool) -> Result<(), EffectError> {
    let interface_error = |reason: String| EffectError::Interface { label: label.to_owned(), reason, location: None };

    for entry in &module.entry_points {
        let expected = matches!((entry.stage, entry.name.as_str()), (naga::ShaderStage::Vertex, VERTEX_ENTRY) | (naga::ShaderStage::Fragment, FRAGMENT_ENTRY));
        if !expected {
            return Err(interface_error(format!("declares its own entry point `{}`; an effect is a plain `fn effect(uv: vec2<f32>) -> vec4<f32>`", entry.name)));
        }
    }

    // Pipeline-overridable constants would need per-backend specialization (naga's GLSL writer
    // refuses them outright), so they are rejected here for every backend alike.
    if let Some((_, constant)) = module.overrides.iter().next() {
        let name = constant.name.as_deref().unwrap_or("<unnamed>");
        return Err(interface_error(format!("declares the pipeline-overridable constant `{name}`; effects cannot use `override` — use `const` or `params(i)`")));
    }

    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        let name = global.name.as_deref().unwrap_or("<unnamed>");
        let declared_by_prelude = binding.group == 0 && PRELUDE_RESOURCES.iter().any(|&(prelude_name, prelude_binding)| prelude_name == name && prelude_binding == binding.binding);
        // A snippet cannot declare `p` itself when the crate appends it (naga rejects the
        // redefinition), so a `p` at the params binding IS the crate's.
        let appended_params = named_params && binding.group == 0 && binding.binding == BINDING_PARAMS && name == PARAMS_GLOBAL;
        if !declared_by_prelude && !appended_params {
            return Err(interface_error(format!(
                "declares resource `{name}` at @group({}) @binding({}); effects cannot bind their own resources — use `src()` for the input, and `struct Params` or `params(i)` for values",
                binding.group, binding.binding
            )));
        }
    }
    Ok(())
}

/// Whether any function of `module` may read the `time` member of the `fl` uniform block.
///
/// Walks naga's IR: an `AccessIndex` of `time` on the `fl` global counts, and so does any `Load`
/// of the WHOLE `fl` global (`let u = fl; ... u.time`), because the copy may be read anywhere —
/// erring towards "animated" only costs repaints, erring the other way freezes the effect.
/// The prelude does neither (pinned by a preset test), so a hit always comes from the snippet.
pub(super) fn reads_time(module: &naga::Module) -> bool {
    let Some((fl, global)) = module.global_variables.iter().find(|(_, global)| global.name.as_deref() == Some("fl")) else {
        return false;
    };
    let Ok(naga::Type { inner: naga::TypeInner::Struct { members, .. }, .. }) = module.types.get_handle(global.ty) else {
        return false;
    };
    let Some(time_index) = members.iter().position(|member| member.name.as_deref() == Some("time")).and_then(|index| u32::try_from(index).ok()) else {
        return false;
    };
    let is_fl = |function: &naga::Function, handle: naga::Handle<naga::Expression>| matches!(function.expressions.try_get(handle), Ok(naga::Expression::GlobalVariable(global)) if *global == fl);
    let touches = |function: &naga::Function| {
        function.expressions.iter().any(|(_, expression)| match expression {
            naga::Expression::AccessIndex { base, index } if *index == time_index => is_fl(function, *base),
            naga::Expression::Load { pointer } => is_fl(function, *pointer),
            _ => false,
        })
    };
    module.functions.iter().any(|(_, function)| touches(function)) || module.entry_points.iter().any(|entry| touches(&entry.function))
}

