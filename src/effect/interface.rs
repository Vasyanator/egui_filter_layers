/*
File: effect/interface.rs

Purpose:
Checks on the validated naga IR of an assembled effect: which resources, entry points and
constants a module may declare (the prelude's `@group(0)` layout and the user texture groups
1..=3), whether it reads `fl.time` (animated effects), and which user texture slots it reads.

Key functions:
- `check_interface()`: every bound global must be one the prelude declares (`PRELUDE_RESOURCES`)
  or the crate-appended named-parameter block.
- `reads_time()`: conservative IR scan deciding `Effect::is_animated`.
- `texture_slots()`: conservative IR scan deciding `Effect::reads_texture`.
- `reads_regions()`: IR scan deciding `Effect::reads_regions`.
*/

use super::params::PARAMS_GLOBAL;
use super::{
    BINDING_COVERAGE_TEXTURE, BINDING_ORIGINAL_TEXTURE, BINDING_PARAMS, BINDING_REGION_TEXTURE, BINDING_SAMPLER, BINDING_SOURCE_TEXTURE, BINDING_STATE_TEXTURE, BINDING_UNIFORMS, EffectError, FRAGMENT_ENTRY, TEXTURE_SLOTS, USER_SAMPLER_BINDING, USER_TEXTURE_BINDING, USER_TEXTURE_GROUP_BASE,
    VERTEX_ENTRY,
};

/// Every resource the prelude declares, as `(name, @group, @binding)`. The interface check
/// accepts exactly these: a snippet cannot redeclare a prelude name (naga rejects the
/// redefinition), so a bound global with one of these names IS the prelude's.
pub(super) const PRELUDE_RESOURCES: [(&str, u32, u32); 13] = [
    ("fl", 0, BINDING_UNIFORMS),
    ("fl_src_texture", 0, BINDING_SOURCE_TEXTURE),
    ("fl_sampler", 0, BINDING_SAMPLER),
    ("fl_original_texture", 0, BINDING_ORIGINAL_TEXTURE),
    ("fl_state_texture", 0, BINDING_STATE_TEXTURE),
    ("fl_coverage_texture", 0, BINDING_COVERAGE_TEXTURE),
    (REGION_TEXTURE_GLOBAL, 0, BINDING_REGION_TEXTURE),
    (USER_TEXTURE_GLOBALS[0][0], USER_TEXTURE_GROUP_BASE, USER_TEXTURE_BINDING),
    (USER_TEXTURE_GLOBALS[0][1], USER_TEXTURE_GROUP_BASE, USER_SAMPLER_BINDING),
    (USER_TEXTURE_GLOBALS[1][0], USER_TEXTURE_GROUP_BASE + 1, USER_TEXTURE_BINDING),
    (USER_TEXTURE_GLOBALS[1][1], USER_TEXTURE_GROUP_BASE + 1, USER_SAMPLER_BINDING),
    (USER_TEXTURE_GLOBALS[2][0], USER_TEXTURE_GROUP_BASE + 2, USER_TEXTURE_BINDING),
    (USER_TEXTURE_GLOBALS[2][1], USER_TEXTURE_GROUP_BASE + 2, USER_SAMPLER_BINDING),
];

/// The prelude's globals of user texture slot `i`: `[texture, its sampler]`.
const USER_TEXTURE_GLOBALS: [[&str; 2]; TEXTURE_SLOTS] = [["fl_user0", "fl_user0_sampler"], ["fl_user1", "fl_user1_sampler"], ["fl_user2", "fl_user2_sampler"]];

/// The prelude's region-id texture global (R1).
const REGION_TEXTURE_GLOBAL: &str = "fl_region_texture";

/// The prelude helpers that read the region-id texture (`region` calls `region_at`).
const REGION_HELPERS: [&str; 2] = ["region", "region_at"];

/// The prelude helpers whose FIRST argument is a user texture slot.
const SLOT_HELPERS: [&str; 3] = ["tex", "tex_size", "tex_load"];

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
        let declared_by_prelude = PRELUDE_RESOURCES.iter().any(|&(prelude_name, group, prelude_binding)| prelude_name == name && group == binding.group && prelude_binding == binding.binding);
        // A snippet cannot declare `p` itself when the crate appends it (naga rejects the
        // redefinition), so a `p` at the params binding IS the crate's.
        let appended_params = named_params && binding.group == 0 && binding.binding == BINDING_PARAMS && name == PARAMS_GLOBAL;
        if !declared_by_prelude && !appended_params {
            return Err(interface_error(format!(
                "declares resource `{name}` at @group({}) @binding({}); effects cannot bind their own resources — use `src()` for the input, `tex(i, uv)` for textures attached to the layer, and `struct Params` or `params(i)` for values",
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


/// Which user texture slots `module` may read: a slot counts when a function other than the
/// prelude's slot helpers ([`SLOT_HELPERS`]) names one of its globals, or calls a helper with
/// that slot as a constant first argument (a literal, `const`, or zero value). A helper call whose
/// slot is not a constant counts as reading EVERY slot — conservative, like [`reads_time`]: a
/// false "reads" only makes a layer demand a texture, a false "does not read" would let it sample
/// a dummy. Constant slots `>= TEXTURE_SLOTS` read nothing (the helpers return zero).
pub(super) fn texture_slots(module: &naga::Module) -> [bool; TEXTURE_SLOTS] {
    let helpers: Vec<naga::Handle<naga::Function>> = module.functions.iter().filter(|(_, function)| function.name.as_deref().is_some_and(|name| SLOT_HELPERS.contains(&name))).map(|(handle, _)| handle).collect();
    let slot_of_global = |global: naga::Handle<naga::GlobalVariable>| {
        let name = module.global_variables.try_get(global).ok()?.name.as_deref()?;
        USER_TEXTURE_GLOBALS.iter().position(|names| names.contains(&name))
    };
    let mut slots = [false; TEXTURE_SLOTS];
    let mut visit = |function: &naga::Function| {
        for (_, expression) in function.expressions.iter() {
            if let naga::Expression::GlobalVariable(global) = expression
                && let Some(slot) = slot_of_global(*global)
            {
                slots[slot] = true;
            }
        }
        let mut calls = Vec::new();
        helper_calls(&function.body, &helpers, &mut calls);
        for argument in calls {
            match constant_u32(module, function, argument) {
                Some(slot) => {
                    if let Some(read) = usize::try_from(slot).ok().and_then(|slot| slots.get_mut(slot)) {
                        *read = true;
                    }
                }
                None => slots = [true; TEXTURE_SLOTS],
            }
        }
    };
    for (handle, function) in module.functions.iter() {
        if !helpers.contains(&handle) {
            visit(function);
        }
    }
    for entry in &module.entry_points {
        visit(&entry.function);
    }
    slots
}

/// Collects the first argument of every call to one of `helpers` in `block`, nested blocks
/// included.
fn helper_calls(block: &naga::Block, helpers: &[naga::Handle<naga::Function>], out: &mut Vec<naga::Handle<naga::Expression>>) {
    for statement in block {
        match statement {
            naga::Statement::Call { function, arguments, .. } => {
                if helpers.contains(function)
                    && let Some(first) = arguments.first()
                {
                    out.push(*first);
                }
            }
            naga::Statement::Block(inner) => helper_calls(inner, helpers, out),
            naga::Statement::If { accept, reject, .. } => {
                helper_calls(accept, helpers, out);
                helper_calls(reject, helpers, out);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    helper_calls(&case.body, helpers, out);
                }
            }
            naga::Statement::Loop { body, continuing, .. } => {
                helper_calls(body, helpers, out);
                helper_calls(continuing, helpers, out);
            }
            _ => {}
        }
    }
}

/// The value of `expression` (of `function`) when it is a constant non-negative integer: a
/// literal, a zero value, or a module `const` initialised with one. `None` for anything computed.
fn constant_u32(module: &naga::Module, function: &naga::Function, expression: naga::Handle<naga::Expression>) -> Option<u32> {
    let literal = |literal: &naga::Literal| match *literal {
        naga::Literal::U32(value) => Some(value),
        naga::Literal::I32(value) => u32::try_from(value).ok(),
        naga::Literal::AbstractInt(value) => u32::try_from(value).ok(),
        _ => None,
    };
    match function.expressions.try_get(expression).ok()? {
        naga::Expression::Literal(value) => literal(value),
        naga::Expression::ZeroValue(_) => Some(0),
        naga::Expression::Constant(constant) => match module.global_expressions.try_get(module.constants.try_get(*constant).ok()?.init).ok()? {
            naga::Expression::Literal(value) => literal(value),
            naga::Expression::ZeroValue(_) => Some(0),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `module` may read region ids: a function other than the prelude's region helpers
/// ([`REGION_HELPERS`]) — or an entry point — names the region texture or calls a helper. The
/// helpers themselves always exist in the assembled module (the prelude declares them), so they
/// alone do not count. Informational (`Effect::reads_regions`): nothing is refused on it.
pub(super) fn reads_regions(module: &naga::Module) -> bool {
    let helpers: Vec<naga::Handle<naga::Function>> = module.functions.iter().filter(|(_, function)| function.name.as_deref().is_some_and(|name| REGION_HELPERS.contains(&name))).map(|(handle, _)| handle).collect();
    let is_region_texture = |global: naga::Handle<naga::GlobalVariable>| module.global_variables.try_get(global).ok().and_then(|global| global.name.as_deref()) == Some(REGION_TEXTURE_GLOBAL);
    let reads = |function: &naga::Function| {
        let names_texture = function.expressions.iter().any(|(_, expression)| matches!(expression, naga::Expression::GlobalVariable(global) if is_region_texture(*global)));
        names_texture || calls_any(&function.body, &helpers)
    };
    module.functions.iter().any(|(handle, function)| !helpers.contains(&handle) && reads(function)) || module.entry_points.iter().any(|entry| reads(&entry.function))
}

/// Whether `block` (nested blocks included) calls one of `helpers`, with or without arguments.
fn calls_any(block: &naga::Block, helpers: &[naga::Handle<naga::Function>]) -> bool {
    block.iter().any(|statement| match statement {
        naga::Statement::Call { function, .. } => helpers.contains(function),
        naga::Statement::Block(inner) => calls_any(inner, helpers),
        naga::Statement::If { accept, reject, .. } => calls_any(accept, helpers) || calls_any(reject, helpers),
        naga::Statement::Switch { cases, .. } => cases.iter().any(|case| calls_any(&case.body, helpers)),
        naga::Statement::Loop { body, continuing, .. } => calls_any(body, helpers) || calls_any(continuing, helpers),
        _ => false,
    })
}
