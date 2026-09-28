# Module: src/effect

## Purpose
The shader model: turning a user's WGSL snippet into a validated `Effect` (prelude + snippet +
optional named-parameter declaration, parsed and validated by naga once), with errors located in
the snippet, and the reflection of its named parameters.

## Architecture
```text
Effect::new(label, snippet)
  └─ compile():
       params::find_params_struct(snippet)      token scan (comments skipped) for `struct Params`
       diagnostics::Assembled::new              prelude + snippet + [PARAMS_TAIL]
       naga::front::wgsl::parse_str             -> EffectError::Parse (located via Assembled)
       params::ParamLayout::from_module         member types / strides / 16 KiB cap -> Interface
       naga Validator                           -> EffectError::Validation (located)
       interface::check_interface               bindings / entry points / `override`
       interface::texture_slots                 user texture slots the module may read
  -> CompiledModule { id, label, source, module, info, params: ParamLayout, texture_slots }
```
Detection runs BEFORE the parse because a snippet reading `p.x` does not parse without the
appended `var<uniform> p: Params;`; naga's IR then confirms it (the struct is found by name).

## Files and submodules
- `mod.rs`: `Effect` (incl. `reads_texture`; `new_state` / `is_state`: a state pass, F7 — the
  same module, flagged on the `Effect`), `EffectError`, `CompiledModule`, binding constants
  (`@group(0)` bindings, user texture groups, public `TEXTURE_SLOTS`), `compile()`, the internal
  composite modules.
- `diagnostics.rs`: `SnippetLocation`, `Assembled` (where prelude / snippet / tail lie; naga
  location -> snippet location + hint; `codespan-reporting` rendering against the snippet).
- `interface.rs`: `PRELUDE_RESOURCES` (`(name, group, binding)`: `@group(0)` and the user
  texture groups 1..=3), `check_interface` (only prelude bindings, plus the crate's `p` at binding
  4 when `struct Params` was detected), `reads_time` (animated detection), `texture_slots` (which
  user texture slots a module reads).
- `params.rs`: public `ParamType` / `ParamInfo` / `ParamValue` / `ParamError`; crate-internal
  `ParamLayout` (members + padded block size), the scan, element ranges, value (de)serialization.

## Contracts and invariants
- The prelude comes first and the tail last: naga line `n` is snippet line `n - prelude_lines`;
  errors located in the tail point at the `struct Params` declaration with `TAIL_HINT`.
- `ParamLayout` offsets ARE naga's member offsets, and `from_module` rejects the struct when any
  offset, array stride or the span differs from GLSL std140 (naga's GLSL writer drops WGSL
  offsets, so `@size` / `@align` that move a member would desync glow from wgpu). Pinned by
  `named_params_are_reflected_in_the_std140_layout` and
  `size_and_align_attributes_that_break_std140_are_rejected`. `block_size` = struct span rounded up to 16 (0 = no named parameters),
  at most `MAX_PARAMS_BLOCK_SIZE` (16 KiB, the WebGL2 / GLES 3.0 uniform block minimum).
- Accepted member types: `f32`, `i32`, `u32`, `vec2/3/4<f32>`, `mat4x4<f32>`, constant-size arrays
  of `vec3/vec4<f32>` / `mat4x4<f32>` (stride a multiple of 16). Everything else is an
  `EffectError::Interface` naming the member, located at the struct.
- Values are native-endian, like `pass::pack_uniforms`. No implicit type conversions.
- Colour values are premultiplied gamma space (`From<Color32>`, matching `src()`); linear only via
  `ParamValue::linear_rgba`. `ParamValue` is `#[non_exhaustive]`.
- `texture_slots` is conservative like `reads_time`: a helper call (`tex`, `tex_size`,
  `tex_load`) with a constant slot counts that slot, any other slot expression counts every slot,
  a direct `fl_user*` reference counts its slot. A false "reads" only makes a layer demand a
  texture; a false "does not read" would let an effect sample a dummy.
- Public API never panics on user input.

## Editing map
- New accepted member type: `params.rs` (`ParamType`, `element_type`, `ParamValue`), then the
  std140 test in `mod.rs`.
- New prelude resource: `interface.rs` `PRELUDE_RESOURCES` and the binding constants in `mod.rs`.
- Error location / rendering: `diagnostics.rs`.
- Which bytes a draw binds at binding 4: `src/plan.rs` (`params_block`), not here.
