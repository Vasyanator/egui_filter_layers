/*
File: effect/params.rs

Purpose:
Named effect parameters (F4). A snippet may declare `struct Params { ... }`; the crate then
appends `@group(0) @binding(4) var<uniform> p: Params;` and the effect reads `p.<name>`. This file
detects the declaration, derives the byte layout from naga's IR, rejects member types the crate
cannot upload, and defines the typed values a `Pass` stores into that block.

Key structures:
- `ParamType`, `ParamInfo`, `ParamValue`, `ParamError` (public): reflection and typed access.
- `ParamLayout` (crate-internal): every member's `ParamInfo` plus the padded block size.

Key functions:
- `find_params_struct()`: comment-aware token scan of the SNIPPET for `struct Params`, run before
  the parse (a snippet reading `p.x` does not parse without the appended declaration, so the
  declaration cannot be discovered from naga's IR first).
- `ParamLayout::from_module()`: the layout of the parsed `Params` struct (naga's member offsets,
  checked to equal GLSL std140, so `@size` / `@align` that move a member are rejected), with the
  WebGL2-safe size cap.
- `ParamValue::write()` / `ParamValue::read()`: native-endian (de)serialization of one element.
- `From<egui::Color32>` (premultiplied gamma space, like `src()`) / `ParamValue::linear_rgba()`.

Notes:
Accepted member types: `f32`, `i32`, `u32`, `vec2/3/4<f32>`, `mat4x4<f32>` and fixed-size arrays
of `vec3<f32>` / `vec4<f32>` / `mat4x4<f32>` (uniform arrays need a 16-byte-multiple stride, so
arrays of scalars or `vec2` are rejected with a hint to use `vec4`). naga validation stays the
backstop for anything this file lets through.
*/

use std::fmt;

use super::{EffectError, SnippetLocation};

/// Name of the struct a snippet declares to get named parameters.
pub(super) const PARAMS_STRUCT: &str = "Params";
/// Name of the uniform global the crate appends (`p.<member>` in WGSL).
pub(super) const PARAMS_GLOBAL: &str = "p";
/// The declaration appended after the snippet when it declares `struct Params`. Appended AFTER
/// the snippet so snippet line numbers are unaffected.
pub(super) const PARAMS_TAIL: &str = "@group(0) @binding(4) var<uniform> p: Params;\n";

/// Largest accepted `Params` block, in bytes: WebGL2's (and GL 3.3's / GLES 3.0's) minimum
/// `MAX_UNIFORM_BLOCK_SIZE`, also wgpu's `downlevel_webgl2_defaults` uniform binding limit.
pub(crate) const MAX_PARAMS_BLOCK_SIZE: usize = 16 * 1024;

/// Every uploaded parameter block is padded to a multiple of this (a GL std140 block's size is a
/// multiple of 16, and the buffer range bound must cover it).
const PARAMS_BLOCK_ALIGN: usize = 16;

/// The type of one `struct Params` member (of one element, for arrays).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParamType {
    /// `f32`.
    F32,
    /// `i32`.
    I32,
    /// `u32`.
    U32,
    /// `vec2<f32>`.
    Vec2,
    /// `vec3<f32>`.
    Vec3,
    /// `vec4<f32>`.
    Vec4,
    /// `mat4x4<f32>` (four columns).
    Mat4,
}

impl ParamType {
    /// Bytes one value of this type occupies (a `vec3` is 12; its 16-byte alignment is in the
    /// member offsets and array strides).
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Vec2 => 8,
            Self::Vec3 => 12,
            Self::Vec4 => 16,
            Self::Mat4 => 64,
        }
    }

    /// The WGSL spelling of the type.
    #[must_use]
    pub fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat4 => "mat4x4<f32>",
        }
    }
}

impl ParamType {
    /// Base alignment of a plain member of this type under GLSL std140 (which equals WGSL's
    /// uniform-address-space alignment for every accepted type).
    fn std140_align(self) -> u32 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Vec2 => 8,
            Self::Vec3 | Self::Vec4 | Self::Mat4 => 16,
        }
    }

    /// Stride of an array of this type under GLSL std140: the element size rounded up to 16.
    fn std140_array_stride(self) -> u32 {
        match self {
            Self::F32 | Self::I32 | Self::U32 | Self::Vec2 | Self::Vec3 | Self::Vec4 => 16,
            Self::Mat4 => 64,
        }
    }
}

impl fmt::Display for ParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wgsl())
    }
}

/// One member of an effect's `struct Params`, as reflected from naga's layout: enough for a UI to
/// build a control per parameter ([`crate::Effect::params`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamInfo {
    /// The member name (`p.<name>` in WGSL, the key of [`crate::Pass::set`]).
    pub name: String,
    /// The type of one element.
    pub ty: ParamType,
    /// Byte offset of the member inside the block.
    pub offset: u32,
    /// Number of elements: the array length, or 1 for a plain member.
    pub count: u32,
    /// Whether the member is an array (set it with [`crate::Pass::set_slice`]).
    pub is_array: bool,
    /// Bytes between consecutive array elements (the type's size for a plain member).
    pub stride: u32,
}

impl ParamInfo {
    /// Byte range of element `index` inside the block, or `None` past the end.
    fn element(&self, index: u32) -> Option<std::ops::Range<usize>> {
        if index >= self.count {
            return None;
        }
        let start = usize::try_from(self.offset.checked_add(self.stride.checked_mul(index)?)?).ok()?;
        Some(start..start.checked_add(self.ty.size())?)
    }
}

/// A typed value for one named parameter (one element of an array member).
///
/// Colours: effects run on premultiplied-alpha, GAMMA-space values (what `src()` returns), so
/// `ParamValue::from(egui::Color32)` yields premultiplied gamma-space floats in `0..=1`. A linear
/// colour must be asked for by name ([`ParamValue::linear_rgba`]); there is deliberately no
/// `From<egui::Rgba>`, whose linear values would silently shift a gamma-space effect's colours.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ParamValue {
    /// For an `f32` member.
    F32(f32),
    /// For an `i32` member.
    I32(i32),
    /// For a `u32` member.
    U32(u32),
    /// For a `vec2<f32>` member.
    Vec2([f32; 2]),
    /// For a `vec3<f32>` member.
    Vec3([f32; 3]),
    /// For a `vec4<f32>` member.
    Vec4([f32; 4]),
    /// For a `mat4x4<f32>` member: four COLUMNS (WGSL `m[i]` is `value[i]`).
    Mat4([[f32; 4]; 4]),
}

impl ParamValue {
    /// A `vec4<f32>` holding `color` in LINEAR premultiplied RGBA — for effects that convert
    /// their samples to linear light themselves. Effects work in gamma space by default; a colour
    /// meant to match `src()` values comes from `ParamValue::from(Color32)` instead.
    #[must_use]
    pub fn linear_rgba(color: egui::Rgba) -> Self {
        Self::Vec4(color.to_array())
    }

    /// The member type this value fits.
    #[must_use]
    pub fn ty(&self) -> ParamType {
        match self {
            Self::F32(_) => ParamType::F32,
            Self::I32(_) => ParamType::I32,
            Self::U32(_) => ParamType::U32,
            Self::Vec2(_) => ParamType::Vec2,
            Self::Vec3(_) => ParamType::Vec3,
            Self::Vec4(_) => ParamType::Vec4,
            Self::Mat4(_) => ParamType::Mat4,
        }
    }

    /// The all-zero value of `ty`.
    #[must_use]
    pub fn zero(ty: ParamType) -> Self {
        match ty {
            ParamType::F32 => Self::F32(0.0),
            ParamType::I32 => Self::I32(0),
            ParamType::U32 => Self::U32(0),
            ParamType::Vec2 => Self::Vec2([0.0; 2]),
            ParamType::Vec3 => Self::Vec3([0.0; 3]),
            ParamType::Vec4 => Self::Vec4([0.0; 4]),
            ParamType::Mat4 => Self::Mat4([[0.0; 4]; 4]),
        }
    }

    /// Writes the value native-endian into the first `self.ty().size()` bytes of `out` (words
    /// past the end of a shorter slice are skipped).
    pub(crate) fn write(&self, out: &mut [u8]) {
        let mut put = |index: usize, word: [u8; 4]| {
            if let Some(chunk) = out.get_mut(index * 4..index * 4 + 4) {
                chunk.copy_from_slice(&word);
            }
        };
        match self {
            Self::F32(value) => put(0, value.to_ne_bytes()),
            Self::I32(value) => put(0, value.to_ne_bytes()),
            Self::U32(value) => put(0, value.to_ne_bytes()),
            Self::Vec2(values) => values.iter().enumerate().for_each(|(index, value)| put(index, value.to_ne_bytes())),
            Self::Vec3(values) => values.iter().enumerate().for_each(|(index, value)| put(index, value.to_ne_bytes())),
            Self::Vec4(values) => values.iter().enumerate().for_each(|(index, value)| put(index, value.to_ne_bytes())),
            Self::Mat4(columns) => columns.iter().flatten().enumerate().for_each(|(index, value)| put(index, value.to_ne_bytes())),
        }
    }

    /// Reads one value of `ty` from `bytes` (native-endian), or `None` when too short.
    pub(crate) fn read(ty: ParamType, bytes: &[u8]) -> Option<Self> {
        let word = |index: usize| -> Option<[u8; 4]> { bytes.get(index * 4..index * 4 + 4)?.try_into().ok() };
        let float = |index: usize| word(index).map(f32::from_ne_bytes);
        let floats = |count: usize| -> Option<Vec<f32>> { (0..count).map(float).collect() };
        Some(match ty {
            ParamType::F32 => Self::F32(float(0)?),
            ParamType::I32 => Self::I32(i32::from_ne_bytes(word(0)?)),
            ParamType::U32 => Self::U32(u32::from_ne_bytes(word(0)?)),
            ParamType::Vec2 => Self::Vec2(floats(2)?.try_into().ok()?),
            ParamType::Vec3 => Self::Vec3(floats(3)?.try_into().ok()?),
            ParamType::Vec4 => Self::Vec4(floats(4)?.try_into().ok()?),
            ParamType::Mat4 => {
                let all = floats(16)?;
                let column = |c: usize| -> Option<[f32; 4]> { all.get(c * 4..c * 4 + 4)?.try_into().ok() };
                Self::Mat4([column(0)?, column(1)?, column(2)?, column(3)?])
            }
        })
    }
}

impl From<f32> for ParamValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<i32> for ParamValue {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<u32> for ParamValue {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<[f32; 2]> for ParamValue {
    fn from(value: [f32; 2]) -> Self {
        Self::Vec2(value)
    }
}

impl From<[f32; 3]> for ParamValue {
    fn from(value: [f32; 3]) -> Self {
        Self::Vec3(value)
    }
}

impl From<[f32; 4]> for ParamValue {
    fn from(value: [f32; 4]) -> Self {
        Self::Vec4(value)
    }
}

impl From<[[f32; 4]; 4]> for ParamValue {
    fn from(value: [[f32; 4]; 4]) -> Self {
        Self::Mat4(value)
    }
}

impl From<egui::Vec2> for ParamValue {
    fn from(value: egui::Vec2) -> Self {
        Self::Vec2([value.x, value.y])
    }
}

impl From<egui::Color32> for ParamValue {
    /// Premultiplied GAMMA-space RGBA in `0..=1` as a `vec4<f32>` — the space `src()` returns and
    /// effects work in, so `p.tint` compares and mixes directly with sampled pixels.
    fn from(value: egui::Color32) -> Self {
        Self::Vec4(value.to_normalized_gamma_f32())
    }
}

/// Why setting a named parameter failed. The pass is left unchanged.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ParamError {
    /// The effect's `struct Params` has no member of that name (or the effect declares none).
    #[error("effect `{effect}` has no named parameter `{name}`")]
    Unknown {
        /// The effect's label.
        effect: String,
        /// The requested name.
        name: String,
    },
    /// The value's type is not the member's (no implicit conversions: `1` is an `i32`, write
    /// `1.0` for an `f32` or `1u32` for a `u32`).
    #[error("named parameter `{name}` is `{expected}`, but a `{given}` was given")]
    Type {
        /// The member name.
        name: String,
        /// The member's element type.
        expected: ParamType,
        /// The given value's type.
        given: ParamType,
    },
    /// More values than the member has elements.
    #[error("named parameter `{name}` has {expected} element(s), but {given} were given")]
    Length {
        /// The member name.
        name: String,
        /// The member's element count.
        expected: u32,
        /// How many values were given.
        given: usize,
    },
    /// A single value was set on an array member.
    #[error("named parameter `{name}` is an array of {count} elements; set it with `Pass::set_slice`")]
    Array {
        /// The member name.
        name: String,
        /// The array length.
        count: u32,
    },
}

/// The named-parameter block of one effect: members in declaration order and the padded size
/// every pass of the effect stores and uploads (0 when the effect declares no `struct Params`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParamLayout {
    /// Every member of `struct Params`.
    pub fields: Vec<ParamInfo>,
    /// The struct span rounded up to 16 bytes; 0 = no named parameters.
    pub block_size: usize,
}

impl ParamLayout {
    /// The member called `name`.
    #[must_use]
    pub(crate) fn field(&self, name: &str) -> Option<&ParamInfo> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// The layout of the `Params` struct of a parsed module (found by name among its types).
    /// `location` points at the declaration in the snippet (for errors).
    ///
    /// # Errors
    /// [`EffectError::Interface`] naming the member when a member type is not accepted (see the
    /// file notes), an array has a stride uniform blocks cannot hold, a member's offset or the
    /// struct's span differs from GLSL std140 (`@size` / `@align`), or the block exceeds
    /// [`MAX_PARAMS_BLOCK_SIZE`]; also when the module has no `Params` struct at all.
    pub(super) fn from_module(module: &naga::Module, label: &str, location: SnippetLocation) -> Result<Self, EffectError> {
        let error = |reason: String| EffectError::Interface { label: label.to_owned(), reason, location: Some(location) };
        let Some(naga::TypeInner::Struct { members, span }) = module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some(PARAMS_STRUCT)).map(|(_, ty)| &ty.inner) else {
            return Err(error(format!("`{PARAMS_STRUCT}` must be a `struct` declared at module scope")));
        };
        let mut fields = Vec::with_capacity(members.len());
        // End (in bytes) of the previous member under GLSL std140, the layout the glow backend's
        // GL driver gives the block naga emits (naga's GLSL writer prints the members without the
        // WGSL offsets, so a WGSL-only offset would silently desync glow from wgpu).
        let mut std140_end: u32 = 0;
        for member in members {
            let name = member.name.clone().unwrap_or_default();
            let inner = |handle: naga::Handle<naga::Type>| module.types.get_handle(handle).ok().map(|ty| &ty.inner);
            let (ty, count, is_array, stride) = match inner(member.ty) {
                Some(naga::TypeInner::Array { base, size, stride }) => {
                    let Some(ty) = inner(*base).and_then(element_type) else {
                        return Err(error(format!("`{PARAMS_STRUCT}.{name}` is an array of an unsupported type; arrays may hold `vec3<f32>`, `vec4<f32>` or `mat4x4<f32>`")));
                    };
                    let naga::ArraySize::Constant(count) = size else {
                        return Err(error(format!("`{PARAMS_STRUCT}.{name}` must be an array of a constant size")));
                    };
                    if stride % 16 != 0 {
                        return Err(error(format!(
                            "`{PARAMS_STRUCT}.{name}`: an `array<{ty}, N>` has a {stride}-byte stride, but arrays in a uniform block need a multiple of 16; use `array<vec4<f32>, N>`"
                        )));
                    }
                    (ty, count.get(), true, *stride)
                }
                other => {
                    let Some(ty) = other.and_then(element_type) else {
                        return Err(error(format!(
                            "`{PARAMS_STRUCT}.{name}` has an unsupported type; named parameters are `f32`, `i32`, `u32`, `vec2<f32>`, `vec3<f32>`, `vec4<f32>`, `mat4x4<f32>` or arrays of the last three (no `bool`, `f16`, nested structs or arrays of arrays)"
                        )));
                    };
                    (ty, 1, false, u32::try_from(ty.size()).unwrap_or(u32::MAX))
                }
            };
            let too_big = || error(format!("`{PARAMS_STRUCT}` is larger than {MAX_PARAMS_BLOCK_SIZE} bytes (the WebGL2 / GLES 3.0 uniform block limit)"));
            let std140_offset = std140_end.checked_next_multiple_of(if is_array { 16 } else { ty.std140_align() }).ok_or_else(too_big)?;
            let std140_stride = if is_array { ty.std140_array_stride() } else { stride };
            if member.offset != std140_offset || stride != std140_stride {
                return Err(error(format!(
                    "`{PARAMS_STRUCT}.{name}` lies at byte {} (stride {stride}), but the GL backend lays the block out by std140, which puts it at byte {std140_offset} (stride {std140_stride}); `@size` and `@align` are not supported on `{PARAMS_STRUCT}` members",
                    member.offset
                )));
            }
            std140_end = stride.checked_mul(count).and_then(|bytes| std140_offset.checked_add(bytes)).ok_or_else(too_big)?;
            fields.push(ParamInfo { name, ty, offset: member.offset, count, is_array, stride });
        }
        // std140 pads the block to a multiple of 16; WGSL's span exceeds that only through a
        // `@size` on the last member, which GL would not honour either.
        let std140_size = std140_end.checked_next_multiple_of(16).unwrap_or(u32::MAX);
        if *span > std140_size {
            let last = fields.last().map_or("", |field| field.name.as_str());
            return Err(error(format!(
                "`{PARAMS_STRUCT}` spans {span} bytes, but the GL backend lays the block out by std140, which ends at byte {std140_size}; `@size` and `@align` are not supported on `{PARAMS_STRUCT}` members (check `{PARAMS_STRUCT}.{last}`)"
            )));
        }
        let span = usize::try_from(*span).unwrap_or(usize::MAX);
        let block_size = span.div_ceil(PARAMS_BLOCK_ALIGN).saturating_mul(PARAMS_BLOCK_ALIGN);
        if block_size > MAX_PARAMS_BLOCK_SIZE {
            return Err(error(format!("`{PARAMS_STRUCT}` is {block_size} bytes; at most {MAX_PARAMS_BLOCK_SIZE} are allowed (the WebGL2 / GLES 3.0 uniform block limit)")));
        }
        Ok(Self { fields, block_size })
    }
}

/// The [`ParamType`] of a non-array naga type, or `None` when it is not accepted.
fn element_type(inner: &naga::TypeInner) -> Option<ParamType> {
    match *inner {
        naga::TypeInner::Scalar(naga::Scalar::F32) => Some(ParamType::F32),
        naga::TypeInner::Scalar(naga::Scalar::I32) => Some(ParamType::I32),
        naga::TypeInner::Scalar(naga::Scalar::U32) => Some(ParamType::U32),
        naga::TypeInner::Vector { size, scalar: naga::Scalar::F32 } => Some(match size {
            naga::VectorSize::Bi => ParamType::Vec2,
            naga::VectorSize::Tri => ParamType::Vec3,
            naga::VectorSize::Quad => ParamType::Vec4,
        }),
        naga::TypeInner::Matrix { columns: naga::VectorSize::Quad, rows: naga::VectorSize::Quad, scalar: naga::Scalar::F32 } => Some(ParamType::Mat4),
        _ => None,
    }
}

/// Byte range of element `index` of `field`, checked against a block of `len` bytes.
pub(crate) fn element_range(field: &ParamInfo, index: u32, len: usize) -> Option<std::ops::Range<usize>> {
    field.element(index).filter(|range| range.end <= len)
}

/// Byte offset (in `snippet`) of the `struct` keyword of a module-scope `struct Params`
/// declaration, or `None` when there is none.
///
/// A token scan that skips `//` line comments and (nested) `/* */` block comments, so a
/// commented-out declaration does not count and `struct /* c */ Params` does. WGSL has no string
/// literals and allows `struct` only at module scope, so no other context needs handling.
/// Identifier characters are ASCII alphanumerics, `_` and every non-ASCII byte (conservative:
/// `Params` followed by a Unicode letter is a different identifier).
pub(super) fn find_params_struct(snippet: &str) -> Option<usize> {
    let bytes = snippet.as_bytes();
    let is_ident = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_' || !byte.is_ascii();
    let mut index = 0;
    // The previous identifier token: (start, text) — `struct` waiting for its name.
    let mut pending_struct: Option<usize> = None;
    while index < bytes.len() {
        let rest = &bytes[index..];
        if rest.starts_with(b"//") {
            index = rest.iter().position(|&byte| byte == b'\n').map_or(bytes.len(), |end| index + end);
        } else if rest.starts_with(b"/*") {
            // WGSL block comments nest.
            let mut depth = 0_usize;
            while index < bytes.len() {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    index += 1;
                }
            }
        } else if is_ident(bytes[index]) {
            let start = index;
            while index < bytes.len() && is_ident(bytes[index]) {
                index += 1;
            }
            let token = &bytes[start..index];
            if let Some(keyword) = pending_struct.take()
                && token == PARAMS_STRUCT.as_bytes()
            {
                return Some(keyword);
            }
            if token == b"struct" {
                pending_struct = Some(start);
            }
        } else {
            if !bytes[index].is_ascii_whitespace() {
                pending_struct = None;
            }
            index += 1;
        }
    }
    None
}

/// The snippet location of byte `offset` of `snippet` (1-based line and byte column).
pub(super) fn location_of(snippet: &str, offset: usize) -> SnippetLocation {
    let before = snippet.get(..offset).unwrap_or(snippet);
    let line = before.bytes().filter(|&byte| byte == b'\n').count() + 1;
    let column = before.len() - before.rfind('\n').map_or(0, |newline| newline + 1) + 1;
    SnippetLocation { line: u32::try_from(line).unwrap_or(u32::MAX), column: u32::try_from(column).unwrap_or(u32::MAX) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scan_finds_only_real_declarations() {
        assert_eq!(find_params_struct("struct Params { a: f32 }"), Some(0));
        assert_eq!(find_params_struct("fn f() {}\n  struct\n\tParams{a:f32}"), Some(12));
        assert_eq!(find_params_struct("struct /* x */ Params { a: f32 }"), Some(0));
        assert_eq!(find_params_struct("struct // x\nParams { a: f32 }"), Some(0));
        for absent in ["// struct Params { a: f32 }", "/* struct Params {} */", "/* /* */ struct Params {} */", "struct ParamsX { a: f32 }", "struct MyParams { a: f32 }", "mystruct Params", "struct Paramsé {}", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }", "struct Other { a: f32 } Params"] {
            assert_eq!(find_params_struct(absent), None, "{absent}");
        }
        // After a nested comment closes, scanning resumes.
        assert_eq!(find_params_struct("/* /* */ */ struct Params {}"), Some(12));
        // An unterminated comment ends the scan without panicking.
        assert_eq!(find_params_struct("/* struct Params"), None);
    }

    #[test]
    fn locations_are_one_based() {
        let snippet = "// a\n  struct Params {}";
        assert_eq!(location_of(snippet, 7), SnippetLocation { line: 2, column: 3 });
        assert_eq!(location_of(snippet, 0), SnippetLocation { line: 1, column: 1 });
    }

    #[test]
    fn values_round_trip_through_bytes() {
        let values = [
            ParamValue::F32(1.5),
            ParamValue::I32(-7),
            ParamValue::U32(u32::MAX),
            ParamValue::Vec2([1.0, 2.0]),
            ParamValue::Vec3([1.0, 2.0, 3.0]),
            ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]),
            ParamValue::Mat4([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]),
        ];
        for value in values {
            let mut bytes = vec![0_u8; value.ty().size()];
            value.write(&mut bytes);
            assert_eq!(ParamValue::read(value.ty(), &bytes), Some(value));
            assert_eq!(ParamValue::zero(value.ty()).ty(), value.ty());
        }
        assert_eq!(ParamValue::read(ParamType::Vec4, &[0; 12]), None, "too short");
        assert_eq!(ParamValue::from(egui::vec2(1.0, 2.0)), ParamValue::Vec2([1.0, 2.0]));
        assert_eq!(ParamValue::from(3_u32).ty(), ParamType::U32);
    }

    /// `Color32` becomes premultiplied GAMMA-space floats (the space of `src()`), a linear colour
    /// only through the explicitly named constructor.
    #[test]
    fn colours_convert_in_an_explicit_space() {
        let half_red = egui::Color32::from_rgba_premultiplied(128, 0, 0, 128);
        assert_eq!(ParamValue::from(half_red), ParamValue::Vec4([128.0 / 255.0, 0.0, 0.0, 128.0 / 255.0]));
        // Opaque mid-grey red: gamma 128/255 is linear ~0.216.
        let ParamValue::Vec4([red, _, _, alpha]) = ParamValue::linear_rgba(egui::Rgba::from(egui::Color32::from_rgb(128, 0, 0))) else { panic!("a vec4") };
        assert!((red - 0.216).abs() < 0.01 && (alpha - 1.0).abs() < 1e-6, "linear red {red}, alpha {alpha}");
    }
}
