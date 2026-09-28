/*
File: effect/mod.rs

Purpose:
The shader model of the crate. An `Effect` is one user-authored WGSL function
`fn effect(uv: vec2<f32>) -> vec4<f32>` wrapped with the fixed prelude
(`shaders/prelude.wgsl`), parsed and validated by naga once, at construction.

Key structures:
- `Effect`: a validated, cheaply clonable effect handle (label + compiled module + defaults).
- `EffectError`: typed construction failure (parse / validation / interface), with the naga
  diagnostic and a location, both relative to the user's snippet.
- `CompiledModule` (crate-internal): the validated naga module every backend translates.

Key functions:
- `Effect::new()` / `compile()`: detect `struct Params`, assemble (prelude + snippet + the named
  parameter tail), parse, lay out the named parameters, validate, check the resource interface.
- `Effect::params()` / `Effect::param()`: reflection of the named parameters.
- `composite_module()` / `composite_linear_module()` (crate-internal): the internal replace pass
  that writes the filtered region back (gamma output / linear output for `*Srgb` targets; glow,
  offline). `composite_erase_module()` / `composite_add_module()` / `composite_add_linear_module()`:
  the two draws of the blended composite the wgpu UI draws over egui's pass.

Submodules: `diagnostics.rs` (snippet-relative errors), `interface.rs` (IR checks: bindings,
entry points, `fl.time`), `params.rs` (named parameters: detection, layout, typed values).

Notes:
The prelude is placed BEFORE the user snippet so a naga line number maps to a snippet line by a
plain subtraction. WGSL allows use-before-declaration at module scope, which is what lets the
prelude's fragment entry point call the user's `effect` defined after it.
*/

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use naga::valid::{Capabilities, ValidationFlags, Validator};

mod diagnostics;
mod interface;
mod params;

pub use diagnostics::SnippetLocation;
use diagnostics::{Assembled, error_chain, error_sources, location_suffix};
use interface::{check_interface, reads_time};
pub(crate) use params::{ParamLayout, element_range};
pub use params::{ParamError, ParamInfo, ParamType, ParamValue};

/// The prelude prepended to every effect. See the crate docs, "Writing an effect".
pub(crate) const PRELUDE_WGSL: &str = include_str!("../shaders/prelude.wgsl");

/// The body of the internal composite pass (appended after the prelude like a user effect).
const COMPOSITE_WGSL: &str = include_str!("../shaders/composite.wgsl");

/// Name of the vertex entry point every assembled module exposes.
pub(crate) const VERTEX_ENTRY: &str = "fl_vertex";
/// Name of the fragment entry point every assembled module exposes.
pub(crate) const FRAGMENT_ENTRY: &str = "fl_fragment";

/// `@binding` of the `fl` uniform block inside `@group(0)`.
pub(crate) const BINDING_UNIFORMS: u32 = 0;
/// `@binding` of the pass input texture (`src()`).
pub(crate) const BINDING_SOURCE_TEXTURE: u32 = 1;
/// `@binding` of the shared linear clamp-to-edge sampler.
pub(crate) const BINDING_SAMPLER: u32 = 2;
/// `@binding` of the original (unfiltered) capture (`fl_original_texture`).
pub(crate) const BINDING_ORIGINAL_TEXTURE: u32 = 3;
/// `@binding` of the named-parameter block `p: Params`, which the crate appends to a module whose
/// snippet declares `struct Params` (never declared by the prelude or a snippet). Every draw binds
/// the bytes `plan.rs` gives it there (16 zero bytes for a module without named parameters).
pub(crate) const BINDING_PARAMS: u32 = 4;
/// `@binding` of the state texture (`fl_state_texture`; a transparent dummy without state).
pub(crate) const BINDING_STATE_TEXTURE: u32 = 5;
/// `@binding` of the mask coverage texture (`fl_coverage_texture`; a white dummy without one).
pub(crate) const BINDING_COVERAGE_TEXTURE: u32 = 6;
/// `@binding` of the shared nearest clamp-to-edge sampler (`fl_nearest_sampler`).
pub(crate) const BINDING_NEAREST_SAMPLER: u32 = 7;

/// Monotonic source of `CompiledModule::id`. Ids are never reused within a process, so a backend
/// cache keyed by id can never confuse a dropped module with a new one.
///
/// An accepted exception to "no global mutable state": modules are built independently of any
/// install or context (presets are process-wide `OnceLock`s shared by every backend), so only a
/// process-wide counter gives ids unique across all of them. It is a monotonic id source, never
/// reset or read back as state, and `u64` cannot wrap in practice.
static NEXT_MODULE_ID: AtomicU64 = AtomicU64::new(1);

/// A parsed and validated WGSL module, shared by every clone of the `Effect` that owns it.
///
/// Backends cache their per-module GPU objects keyed by [`CompiledModule::id`] and hold a
/// `Weak` to it so they can evict the cache entry once the last `Effect` clone is dropped.
pub(crate) struct CompiledModule {
    /// Process-unique identity of this module.
    pub(crate) id: u64,
    /// Human-readable label (from `Effect::new`), used in logs and errors.
    pub(crate) label: Arc<str>,
    /// The full assembled WGSL source (prelude + snippet).
    pub(crate) source: String,
    /// The naga IR of `source`.
    pub(crate) module: naga::Module,
    /// Validation info for `module` (needed by every naga back end).
    #[cfg_attr(not(feature = "glow"), allow(dead_code))] // Read by naga's GLSL back end only; wgpu takes the WGSL source.
    pub(crate) info: naga::valid::ModuleInfo,
    /// The named-parameter block (`struct Params`); empty with `block_size` 0 when undeclared.
    pub(crate) params: ParamLayout,
}

impl fmt::Debug for CompiledModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledModule").field("id", &self.id).field("label", &self.label).finish_non_exhaustive()
    }
}

/// Why an [`Effect`] could not be built.
///
/// Every variant carries the effect's label and a one-line `message`/`reason`; parse and
/// validation failures also carry naga's full multi-line `diagnostic` — print it when the one-liner
/// is not enough. The accessors ([`EffectError::label`], [`EffectError::location`],
/// [`EffectError::diagnostic`]) read these without matching on the (non-exhaustive) variants.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EffectError {
    /// The WGSL did not parse.
    #[error("effect `{label}`: WGSL parse error{}: {message}", location_suffix(.location.as_ref()))]
    Parse {
        /// The effect's label.
        label: String,
        /// naga's one-line message, plus a hint when the error is outside the snippet.
        message: String,
        /// Where in the snippet, or `None` when naga pointed into the generated wrapper.
        location: Option<SnippetLocation>,
        /// naga's full diagnostic. Rendered against the snippet (its line numbers match
        /// `location`) when naga points into it, else against the assembled source.
        diagnostic: String,
    },
    /// The WGSL parsed but failed naga validation (types, uniformity, limits, ...).
    #[error("effect `{label}`: WGSL validation error{}: {message}", location_suffix(.location.as_ref()))]
    Validation {
        /// The effect's label.
        label: String,
        /// naga's error chain joined into one line.
        message: String,
        /// Where in the snippet, or `None` when naga pointed into the generated wrapper.
        location: Option<SnippetLocation>,
        /// naga's full diagnostic. Rendered against the snippet (its line numbers match
        /// `location`) when naga points into it, else against the assembled source.
        diagnostic: String,
    },
    /// The snippet declared something the crate cannot bind: extra resources, entry points,
    /// pipeline-overridable constants (`override`), or a `struct Params` the crate cannot upload
    /// (unsupported member type, too large).
    #[error("effect `{label}`{}: {reason}", location_suffix(.location.as_ref()))]
    Interface {
        /// The effect's label.
        label: String,
        /// What was declared and why it is rejected.
        reason: String,
        /// Where in the snippet (the `struct Params` declaration for named-parameter errors),
        /// `None` when the declaration is not located.
        location: Option<SnippetLocation>,
    },
}

impl EffectError {
    /// The label of the effect that failed.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Parse { label, .. } | Self::Validation { label, .. } | Self::Interface { label, .. } => label,
        }
    }

    /// Where in the snippet the error is, when it was located inside the snippet (errors naga
    /// reports at the crate-appended named-parameter declaration point at `struct Params`).
    #[must_use]
    pub fn location(&self) -> Option<SnippetLocation> {
        match self {
            Self::Parse { location, .. } | Self::Validation { location, .. } | Self::Interface { location, .. } => *location,
        }
    }

    /// naga's full multi-line diagnostic, for parse and validation errors (`None` for
    /// [`EffectError::Interface`], whose `Display` already says everything).
    #[must_use]
    pub fn diagnostic(&self) -> Option<&str> {
        match self {
            Self::Parse { diagnostic, .. } | Self::Validation { diagnostic, .. } => Some(diagnostic),
            Self::Interface { .. } => None,
        }
    }
}

/// A validated filter effect: one WGSL function run once per pixel of the filtered region.
///
/// Build it once (e.g. in a `OnceLock` or your app state) and reuse it: construction parses and
/// validates WGSL, and backends cache their compiled GPU program per effect. Cloning is cheap
/// (an `Arc` bump) and clones share the cached program.
///
/// The snippet must define exactly:
///
/// ```wgsl
/// fn effect(uv: vec2<f32>) -> vec4<f32> { ... }
/// ```
///
/// and may use the prelude helpers documented at the crate root.
#[derive(Clone)]
pub struct Effect {
    /// The validated module (shared with every clone).
    module: Arc<CompiledModule>,
    /// How far (in points) this effect samples away from the pixel it writes.
    sample_radius_points: f32,
    /// Whether painting a layer with this effect requests a repaint every frame (detected from
    /// the snippet reading `fl.time`, overridable).
    animated: bool,
}

impl fmt::Debug for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Effect")
            .field("label", &self.module.label)
            .field("id", &self.module.id)
            .field("sample_radius_points", &self.sample_radius_points)
            .field("animated", &self.animated)
            .finish()
    }
}

impl Effect {
    /// Assembles `wgsl` with the prelude, parses and validates it.
    ///
    /// `label` names the effect in logs and errors. Validation happens here, on every backend, so a
    /// broken custom effect is reported to the caller instead of failing later on the GPU thread.
    ///
    /// An effect whose code reads `fl.time` is detected as animated (see [`Effect::is_animated`]).
    ///
    /// # Errors
    /// [`EffectError::Parse`] / [`EffectError::Validation`] carry naga's diagnostic with a
    /// location relative to `wgsl`; [`EffectError::Interface`] reports resources or entry points
    /// the snippet must not declare.
    pub fn new(label: &str, wgsl: &str) -> Result<Self, EffectError> {
        let module = compile(label, wgsl)?;
        let animated = reads_time(&module.module);
        Ok(Self { module: Arc::new(module), sample_radius_points: 0.0, animated })
    }

    /// Declares how far, in layer points, the effect samples away from the pixel it writes (it
    /// scales with a layer transform's zoom, like `fl.pixels_per_point`).
    ///
    /// The capture is padded by this much so neighbourhood effects (blur, refraction) read real
    /// backdrop instead of smeared edges. Negative or non-finite values are treated as `0`.
    #[must_use]
    pub fn with_sample_radius(mut self, points: f32) -> Self {
        self.sample_radius_points = sanitize_radius(points);
        self
    }

    /// Marks the effect as animated: painting a layer that uses it requests a repaint every frame.
    /// Shorthand for `with_animated(true)`; rarely needed since reading `fl.time` is detected.
    #[must_use]
    pub fn animated(self) -> Self {
        self.with_animated(true)
    }

    /// Overrides whether the effect is animated. [`Effect::new`] sets it when the code reads
    /// `fl.time`; pass `false` to stop continuous repaints (e.g. a time-based effect you only
    /// want updated when egui repaints anyway), `true` for an effect animated some other way.
    #[must_use]
    pub fn with_animated(mut self, animated: bool) -> Self {
        self.animated = animated;
        self
    }

    /// The label given to [`Effect::new`].
    #[must_use]
    pub fn label(&self) -> &str {
        &self.module.label
    }

    /// The default sample radius in points (see [`Effect::with_sample_radius`]).
    #[must_use]
    pub fn sample_radius(&self) -> f32 {
        self.sample_radius_points
    }

    /// Whether painting a layer with this effect requests a repaint every frame: `true` when its
    /// code reads `fl.time` (found in naga's IR, so comments and unused names do not count;
    /// copying the whole block, e.g. `let u = fl;`, conservatively counts as reading it),
    /// unless overridden with [`Effect::with_animated`].
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.animated
    }

    /// The complete WGSL module (prelude + snippet) that was validated. Useful for debugging.
    #[must_use]
    pub fn assembled_source(&self) -> &str {
        &self.module.source
    }

    /// The named parameters the snippet declared in `struct Params`, in declaration order (empty
    /// when it declares none). A UI can build one control per entry; values are set per pass with
    /// [`crate::Pass::set`].
    #[must_use]
    pub fn params(&self) -> &[ParamInfo] {
        &self.module.params.fields
    }

    /// The named parameter `name`, if `struct Params` has such a member.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&ParamInfo> {
        self.module.params.field(name)
    }

    /// The shared compiled module (crate-internal: backends key their caches on it).
    pub(crate) fn module(&self) -> &Arc<CompiledModule> {
        &self.module
    }

    /// The named-parameter layout (crate-internal: `Pass` sizes and fills its block from it).
    pub(crate) fn param_layout(&self) -> &ParamLayout {
        &self.module.params
    }
}

/// Clamps a user radius to a finite, non-negative value.
pub(crate) fn sanitize_radius(points: f32) -> f32 {
    if points.is_finite() && points > 0.0 { points } else { 0.0 }
}

/// Assembles, parses, validates and interface-checks one module.
///
/// A snippet declaring `struct Params` (found by a token scan BEFORE the parse: a snippet reading
/// `p.x` only parses with the declaration of `p` present) gets [`params::PARAMS_TAIL`] appended,
/// and the struct's layout is checked after the parse, before validation, so an unsupported member
/// is reported by name instead of as naga's generic "global variable is invalid".
fn compile(label: &str, snippet: &str) -> Result<CompiledModule, EffectError> {
    let declaration = params::find_params_struct(snippet).map(|offset| params::location_of(snippet, offset));
    let assembled = Assembled::new(snippet, if declaration.is_some() { params::PARAMS_TAIL } else { "" }, declaration);
    let source = &assembled.source;

    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        let (location, hint) = assembled.locate(error.location(source));
        let message = format!("{}{hint}", error.message());
        let labels: Vec<(Range<usize>, String)> = error.labels().filter_map(|(span, text)| span.to_range().map(|range| (range, text.to_owned()))).collect();
        let notes = error.notes().map(str::to_owned).collect();
        let diagnostic = assembled.render(label, error.message(), &labels, notes);
        EffectError::Parse { label: label.to_owned(), message, location, diagnostic }
    })?;

    let params = match declaration {
        Some(location) => ParamLayout::from_module(&module, label, location)?,
        None => ParamLayout::default(),
    };

    let info = Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module).map_err(|error| {
        let (location, hint) = assembled.locate(error.location(source));
        let message = format!("{}{hint}", error_chain(error.as_inner()));
        let labels: Vec<(Range<usize>, String)> = error.spans().filter_map(|(span, text)| span.to_range().map(|range| (range, text.clone()))).collect();
        let diagnostic = assembled.render(label, &error.as_inner().to_string(), &labels, error_sources(error.as_inner()));
        EffectError::Validation { label: label.to_owned(), message, location, diagnostic }
    })?;

    check_interface(label, &module, declaration.is_some())?;

    let id = NEXT_MODULE_ID.fetch_add(1, Ordering::Relaxed);
    log::debug!("egui_filter_layers: compiled effect `{label}` (module id {id})");
    Ok(CompiledModule { id, label: Arc::from(label), source: assembled.source, module, info, params })
}

/// `effect` wrapper appended to the composite body for targets that store gamma values as-is.
const COMPOSITE_GAMMA_TAIL: &str = "fn effect(frag_uv: vec2<f32>) -> vec4<f32> {\n    return fl_composite(frag_uv);\n}\n";

/// `effect` wrapper for `*Srgb` targets: the hardware encodes linear -> sRGB on write, so the
/// premultiplied gamma colour is decoded per channel first (what egui-wgpu does for such targets).
const COMPOSITE_LINEAR_TAIL: &str = "fn effect(frag_uv: vec2<f32>) -> vec4<f32> {\n    return to_linear(fl_composite(frag_uv));\n}\n";

/// `effect` wrapper of the blended composite's ERASE draw (wgpu UI only): outputs the mask
/// coverage as alpha under a `{Zero, OneMinusSrcAlpha}` blend, scaling the pass's pixels by
/// `1 - coverage` on all four channels; uncovered fragments are discarded (left untouched). Zero
/// colour needs no sRGB variant.
const COMPOSITE_ERASE_TAIL: &str = "fn effect(frag_uv: vec2<f32>) -> vec4<f32> {\n    return vec4<f32>(0.0, 0.0, 0.0, fl_coverage_or_discard(fl_composite_uv(frag_uv)));\n}\n";

/// `effect` wrapper of the blended composite's ADD draw on gamma targets: `coverage * filtered`
/// under a `{One, One}` blend, completing `mix(dst, filtered, coverage)` after the erase draw.
const COMPOSITE_ADD_GAMMA_TAIL: &str =
    "fn effect(frag_uv: vec2<f32>) -> vec4<f32> {\n    let uv = fl_composite_uv(frag_uv);\n    let coverage = fl_coverage_or_discard(uv);\n    return coverage * src(uv);\n}\n";

/// The ADD draw for `*Srgb` targets: the filtered premultiplied gamma colour is decoded first (the
/// hardware re-encodes after blending in linear, as egui-wgpu's own draws blend there).
const COMPOSITE_ADD_LINEAR_TAIL: &str =
    "fn effect(frag_uv: vec2<f32>) -> vec4<f32> {\n    let uv = fl_composite_uv(frag_uv);\n    let coverage = fl_coverage_or_discard(uv);\n    return coverage * to_linear(src(uv));\n}\n";

/// Compiles the composite body with one of the `effect` wrappers.
fn compile_composite(label: &str, tail: &str) -> Result<Arc<CompiledModule>, EffectError> {
    let mut snippet = String::with_capacity(COMPOSITE_WGSL.len() + tail.len() + 1);
    snippet.push_str(COMPOSITE_WGSL);
    if !COMPOSITE_WGSL.ends_with('\n') {
        snippet.push('\n');
    }
    snippet.push_str(tail);
    compile(label, &snippet).map(Arc::new)
}

/// The internal composite pass writing gamma-space output, compiled once per process.
///
/// # Errors
/// Only if the crate's own composite WGSL is broken — a crate bug that unit tests pin; the
/// backends latch it as a backend failure rather than panicking.
pub(crate) fn composite_module() -> Result<Arc<CompiledModule>, EffectError> {
    static COMPOSITE: OnceLock<Result<Arc<CompiledModule>, EffectError>> = OnceLock::new();
    COMPOSITE.get_or_init(|| compile_composite("egui_filter_layers::composite", COMPOSITE_GAMMA_TAIL)).clone()
}

/// The composite pass for `*Srgb` render targets (linear output), compiled once per process.
///
/// # Errors
/// As [`composite_module`].
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // Only the wgpu backend can face an sRGB target.
pub(crate) fn composite_linear_module() -> Result<Arc<CompiledModule>, EffectError> {
    static COMPOSITE: OnceLock<Result<Arc<CompiledModule>, EffectError>> = OnceLock::new();
    COMPOSITE.get_or_init(|| compile_composite("egui_filter_layers::composite_linear", COMPOSITE_LINEAR_TAIL)).clone()
}

/// The ERASE draw of the blended composite (see [`COMPOSITE_ERASE_TAIL`]), compiled once per
/// process. Output format independent.
///
/// # Errors
/// As [`composite_module`].
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // Only the wgpu UI backend blends its composite.
pub(crate) fn composite_erase_module() -> Result<Arc<CompiledModule>, EffectError> {
    static COMPOSITE: OnceLock<Result<Arc<CompiledModule>, EffectError>> = OnceLock::new();
    COMPOSITE.get_or_init(|| compile_composite("egui_filter_layers::composite_erase", COMPOSITE_ERASE_TAIL)).clone()
}

/// The ADD draw of the blended composite for gamma targets, compiled once per process.
///
/// # Errors
/// As [`composite_module`].
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // Only the wgpu UI backend blends its composite.
pub(crate) fn composite_add_module() -> Result<Arc<CompiledModule>, EffectError> {
    static COMPOSITE: OnceLock<Result<Arc<CompiledModule>, EffectError>> = OnceLock::new();
    COMPOSITE.get_or_init(|| compile_composite("egui_filter_layers::composite_add", COMPOSITE_ADD_GAMMA_TAIL)).clone()
}

/// The ADD draw of the blended composite for `*Srgb` targets (linear output), compiled once per
/// process.
///
/// # Errors
/// As [`composite_module`].
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // Only the wgpu UI backend blends its composite.
pub(crate) fn composite_add_linear_module() -> Result<Arc<CompiledModule>, EffectError> {
    static COMPOSITE: OnceLock<Result<Arc<CompiledModule>, EffectError>> = OnceLock::new();
    COMPOSITE.get_or_init(|| compile_composite("egui_filter_layers::composite_add_linear", COMPOSITE_ADD_LINEAR_TAIL)).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_effect_validates() {
        let effect = Effect::new("identity", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }");
        assert!(effect.is_ok(), "{effect:?}");
    }

    #[test]
    fn the_prelude_is_followed_by_the_snippet_verbatim() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return src_px(vec2<f32>(1.0, 0.0));\n}";
        let effect = Effect::new("snippet", snippet).map_err(|e| e.to_string());
        let Ok(effect) = effect else { panic!("{effect:?}") };
        let source = effect.assembled_source();
        assert!(source.starts_with(PRELUDE_WGSL));
        assert_eq!(&source[PRELUDE_WGSL.len()..], format!("{snippet}\n"));
    }

    #[test]
    fn every_prelude_helper_is_callable_from_a_snippet() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let a = src(uv) + src_px(vec2<f32>(-1.0, 2.0));
    let b = to_gamma(to_linear(a));
    let c = premultiply(unpremultiply(b));
    let l = luma(c.rgb);
    let t = fl.texel + fl.size_px + fl.rect_uv.xy + vec2<f32>(fl.time, fl.pixels_per_point);
    return vec4<f32>(vec3<f32>(l), 1.0) * params(3u).x + vec4<f32>(t, 0.0, 0.0);
}";
        let effect = Effect::new("helpers", snippet);
        assert!(effect.is_ok(), "{effect:?}");
    }

    #[test]
    fn a_parse_error_points_into_the_snippet() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let x = ;\n    return src(uv);\n}";
        match Effect::new("broken", snippet) {
            Err(EffectError::Parse { label, location: Some(location), diagnostic, .. }) => {
                assert_eq!(label, "broken");
                assert_eq!(location.line, 2, "{diagnostic}");
                assert!(!diagnostic.is_empty());
            }
            other => panic!("expected a located parse error, got {other:?}"),
        }
    }

    #[test]
    fn a_type_error_is_a_validation_or_parse_error_in_the_snippet() {
        // naga's WGSL front end already type-checks most expressions, so a type mismatch may be
        // reported by either stage; both must locate it inside the snippet.
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return uv;\n}";
        let location = match Effect::new("mistyped", snippet) {
            Err(EffectError::Parse { location, .. } | EffectError::Validation { location, .. }) => location,
            other => panic!("expected an error, got {other:?}"),
        };
        // naga's first span may be the enclosing function (line 1) or the return (line 2).
        assert!(matches!(location, Some(l) if (1..=2).contains(&l.line)), "{location:?}");
    }

    #[test]
    fn a_missing_effect_function_gets_the_wrapper_hint() {
        match Effect::new("empty", "fn not_it() {}") {
            Err(error @ (EffectError::Parse { location: None, .. } | EffectError::Validation { location: None, .. })) => {
                assert!(error.to_string().contains("must define"), "{error}");
            }
            other => panic!("expected an unlocated error with a hint, got {other:?}"),
        }
    }

    #[test]
    fn extra_resources_and_entry_points_are_rejected() {
        // Any binding a snippet declares is rejected, also at a slot of the prelude's layout or
        // in another group; only the prelude's own declarations are bound.
        for declaration in ["@group(0) @binding(8) var other: texture_2d<f32>;", "@group(0) @binding(4) var<uniform> other: vec4<f32>;", "@group(0) @binding(6) var other: texture_2d<f32>;", "@group(1) @binding(0) var other: texture_2d<f32>;"] {
            let snippet = format!("{declaration}\nfn effect(uv: vec2<f32>) -> vec4<f32> {{ return src(uv); }}");
            assert!(matches!(Effect::new("binding", &snippet), Err(EffectError::Interface { .. })), "{declaration}");
        }

        let redeclared = "@group(0) @binding(3) var fl_original_texture: texture_2d<f32>;\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        assert!(Effect::new("redeclared", redeclared).is_err(), "a prelude resource cannot be redeclared");

        let entry = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }\n@compute @workgroup_size(1) fn extra() {}";
        assert!(matches!(Effect::new("entry", entry), Err(EffectError::Interface { .. })));
    }

    #[test]
    fn redefining_a_prelude_name_is_an_error() {
        let snippet = "fn src(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        assert!(Effect::new("clash", snippet).is_err());
    }

    #[test]
    fn the_composite_modules_validate() {
        let composite = composite_module();
        assert!(composite.is_ok(), "{composite:?}");
        let linear = composite_linear_module();
        assert!(linear.is_ok(), "{linear:?}");
        for (name, module) in [("erase", composite_erase_module()), ("add", composite_add_module()), ("add linear", composite_add_linear_module())] {
            assert!(module.is_ok(), "{name}: {module:?}");
        }
    }

    #[test]
    fn reading_fl_time_marks_an_effect_animated_and_the_override_wins() {
        let timed = Effect::new("timed", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * sin(fl.time); }");
        let Ok(timed) = timed else { panic!("{timed:?}") };
        assert!(timed.is_animated());
        assert!(!timed.clone().with_animated(false).is_animated(), "the override wins");

        // Other `fl` members, and `time` only in a comment, do not count.
        let still = Effect::new("still", "// fl.time\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * fl.pixels_per_point; }");
        let Ok(still) = still else { panic!("{still:?}") };
        assert!(!still.is_animated());
        assert!(still.with_animated(true).is_animated(), "the override wins");
    }

    #[test]
    fn copying_the_whole_block_counts_as_reading_time() {
        let aliased = Effect::new("aliased", "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let u = fl;\n    return src(uv) * u.time;\n}");
        let Ok(aliased) = aliased else { panic!("{aliased:?}") };
        assert!(aliased.is_animated(), "a whole-`fl` copy may read `time`");
    }

    #[test]
    fn override_declarations_are_rejected_on_every_backend() {
        let snippet = "override k: f32 = 1.0;\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * k; }";
        match Effect::new("overridden", snippet) {
            Err(error @ EffectError::Interface { .. }) => {
                assert!(error.to_string().contains("`k`"), "{error}");
                assert_eq!(error.label(), "overridden");
                assert_eq!(error.location(), None);
                assert_eq!(error.diagnostic(), None);
            }
            other => panic!("expected an interface error, got {other:?}"),
        }
    }

    #[test]
    fn diagnostics_use_snippet_line_numbers() {
        let snippet = "// line 1\n// line 2\nfn effect(uv: vec2<f32>) -> vec4<f32> {\n    let x = ;\n    return src(uv);\n}";
        let error = match Effect::new("numbered", snippet) {
            Err(error) => error,
            Ok(effect) => panic!("must not compile: {effect:?}"),
        };
        assert_eq!(error.label(), "numbered");
        assert_eq!(error.location().map(|l| l.line), Some(4));
        let Some(diagnostic) = error.diagnostic() else { panic!("a parse error has a diagnostic") };
        assert!(diagnostic.contains("numbered:4:"), "{diagnostic}");
        assert!(diagnostic.contains("let x = ;"), "{diagnostic}");
        assert!(!diagnostic.contains("fl_fragment"), "the prelude must not be rendered: {diagnostic}");
    }

    #[test]
    fn wrapper_diagnostics_fall_back_to_the_assembled_source() {
        let error = match Effect::new("empty", "fn not_it() {}") {
            Err(error) => error,
            Ok(effect) => panic!("must not compile: {effect:?}"),
        };
        let Some(diagnostic) = error.diagnostic() else { panic!("{error:?}") };
        assert!(diagnostic.contains("empty (assembled)"), "{diagnostic}");
    }

    #[test]
    fn radii_are_sanitized() {
        assert!((sanitize_radius(3.5) - 3.5).abs() < f32::EPSILON);
        assert!(sanitize_radius(-1.0).abs() < f32::EPSILON);
        assert!(sanitize_radius(f32::NAN).abs() < f32::EPSILON);
        assert!(sanitize_radius(f32::INFINITY).abs() < f32::EPSILON);
    }

    /// A `struct Params` with every accepted type: the reflected table equals naga's layout, which
    /// is the GLSL std140 layout the glow backend's block uses (hand-computed below).
    #[test]
    fn named_params_are_reflected_in_the_std140_layout() {
        let snippet = "struct Params {
    a: f32,
    b: vec3<f32>,
    c: f32,
    d: vec2<f32>,
    e: i32,
    f: u32,
    g: mat4x4<f32>,
    h: array<vec4<f32>, 3>,
    k: vec3<f32>,
    m: array<mat4x4<f32>, 2>,
    n: array<vec3<f32>, 2>,
}
fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.a; }";
        let effect = Effect::new("all types", snippet);
        let Ok(effect) = effect else { panic!("{effect:?}") };
        let table: Vec<(&str, ParamType, u32, u32, bool, u32)> = effect.params().iter().map(|info| (info.name.as_str(), info.ty, info.offset, info.count, info.is_array, info.stride)).collect();
        assert_eq!(
            table,
            [
                ("a", ParamType::F32, 0, 1, false, 4),
                ("b", ParamType::Vec3, 16, 1, false, 12),
                ("c", ParamType::F32, 28, 1, false, 4),
                ("d", ParamType::Vec2, 32, 1, false, 8),
                ("e", ParamType::I32, 40, 1, false, 4),
                ("f", ParamType::U32, 44, 1, false, 4),
                ("g", ParamType::Mat4, 48, 1, false, 64),
                ("h", ParamType::Vec4, 112, 3, true, 16),
                ("k", ParamType::Vec3, 160, 1, false, 12),
                ("m", ParamType::Mat4, 176, 2, true, 64),
                ("n", ParamType::Vec3, 304, 2, true, 16),
            ]
        );
        assert_eq!(effect.param_layout().block_size, 336);
        // The table IS naga's computed layout.
        let module = &effect.module().module;
        let Some(naga::TypeInner::Struct { members, span }) = module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some("Params")).map(|(_, ty)| &ty.inner) else { panic!("Params in the IR") };
        assert!(members.iter().zip(effect.params()).all(|(member, info)| member.offset == info.offset && member.name.as_deref() == Some(info.name.as_str())));
        assert_eq!(*span, 336);
        assert_eq!(effect.param("h").map(|info| info.count), Some(3));
        assert!(effect.param("missing").is_none());
        assert!(effect.assembled_source().ends_with("var<uniform> p: Params;\n"), "the tail is appended after the snippet");
    }

    #[test]
    fn a_small_params_block_is_padded_to_16_bytes() {
        let Ok(effect) = Effect::new("one", "struct Params { gain: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.gain; }") else { panic!("must compile") };
        assert_eq!(effect.param_layout().block_size, 16);
        let Ok(plain) = Effect::new("plain", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("must compile") };
        assert!(plain.params().is_empty());
        assert_eq!(plain.param_layout().block_size, 0);
        assert!(!plain.assembled_source().ends_with("var<uniform> p: Params;\n"), "no tail without struct Params");
    }

    #[test]
    fn unsupported_param_members_are_rejected_by_name_at_the_struct() {
        let cases = [
            ("flag: bool", "flag"),
            ("inner: Inner", "inner"),
            ("list: array<f32, 4>", "list"),
            ("pairs: array<vec2<f32>, 4>", "pairs"),
            ("grid: array<array<vec4<f32>, 2>, 2>", "grid"),
            ("ints: vec2<i32>", "ints"),
            ("rot: mat3x3<f32>", "rot"),
        ];
        for (member, name) in cases {
            let snippet = format!("struct Inner {{ x: vec4<f32> }}\n// the struct\nstruct Params {{ ok: f32, {member} }}\nfn effect(uv: vec2<f32>) -> vec4<f32> {{ return src(uv) * p.ok; }}");
            match Effect::new("bad", &snippet) {
                Err(error @ EffectError::Interface { .. }) => {
                    assert!(error.to_string().contains(&format!("Params.{name}")), "{member}: {error}");
                    assert_eq!(error.location(), Some(SnippetLocation { line: 3, column: 1 }), "{member}");
                }
                other => panic!("{member}: expected an interface error, got {other:?}"),
            }
        }
    }

    /// `@size` / `@align` move WGSL member offsets, but naga's GLSL writer drops them and GL lays
    /// the block out by std140: any layout differing from std140 is rejected, naming the member.
    #[test]
    fn size_and_align_attributes_that_break_std140_are_rejected() {
        let cases = [
            ("@size(16) a: f32, b: f32", "Params.b"),
            ("a: f32, @align(16) b: f32", "Params.b"),
            ("a: vec4<f32>, @align(32) c: vec4<f32>", "Params.c"),
            ("a: f32, @size(32) last: f32", "Params.last"),
            ("@size(16) a: f32, @align(32) c: vec4<f32>", "Params.c"),
        ];
        for (members, name) in cases {
            let snippet = format!("// params\nstruct Params {{ {members} }}\nfn effect(uv: vec2<f32>) -> vec4<f32> {{ return src(uv); }}");
            match Effect::new("layout", &snippet) {
                Err(error @ EffectError::Interface { .. }) => {
                    let text = error.to_string();
                    assert!(text.contains(name) && text.contains("std140") && text.contains("@align"), "{members}: {text}");
                    assert_eq!(error.location(), Some(SnippetLocation { line: 2, column: 1 }), "{members}");
                }
                other => panic!("{members}: expected an interface error, got {other:?}"),
            }
        }
        // Attributes that restate the std140 layout change nothing and are accepted.
        for members in ["@size(4) a: f32, b: f32", "a: f32, @align(8) b: vec2<f32>", "a: f32, @align(16) b: vec4<f32>", "@size(12) a: vec3<f32>, b: f32"] {
            let snippet = format!("struct Params {{ {members} }}\nfn effect(uv: vec2<f32>) -> vec4<f32> {{ return src(uv); }}");
            assert!(Effect::new("layout", &snippet).is_ok(), "{members}");
        }
    }

    #[test]
    fn params_blocks_are_capped_at_the_webgl2_limit() {
        let fits = Effect::new("fits", "struct Params { v: array<vec4<f32>, 1024> }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return p.v[0]; }");
        assert!(fits.is_ok_and(|effect| effect.param_layout().block_size == 16 * 1024));
        match Effect::new("big", "struct Params { v: array<vec4<f32>, 1024>, w: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return p.v[0]; }") {
            Err(error @ EffectError::Interface { .. }) => assert!(error.to_string().contains("16384"), "{error}"),
            other => panic!("expected an interface error, got {other:?}"),
        }
    }

    /// Errors naga reports at the crate-appended declaration point at `struct Params` with a hint;
    /// errors inside the snippet keep their own line numbers (the tail follows the snippet).
    #[test]
    fn tail_errors_point_at_the_struct_and_snippet_lines_are_kept() {
        let clash = "fn p() {}\nstruct Params { a: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        match Effect::new("clash", clash) {
            Err(error @ (EffectError::Parse { .. } | EffectError::Validation { .. })) => {
                assert_eq!(error.location(), Some(SnippetLocation { line: 2, column: 1 }), "{error}");
                assert!(error.to_string().contains("crate-appended"), "{error}");
            }
            other => panic!("expected a located error, got {other:?}"),
        }
        let broken = "struct Params { a: f32 }\n\nfn effect(uv: vec2<f32>) -> vec4<f32> {\n    let x = ;\n    return src(uv) * p.a;\n}";
        match Effect::new("broken", broken) {
            Err(error @ EffectError::Parse { .. }) => {
                assert_eq!(error.location().map(|l| l.line), Some(4));
                let Some(diagnostic) = error.diagnostic() else { panic!("a parse error has a diagnostic") };
                assert!(diagnostic.contains("broken:4:") && !diagnostic.contains("var<uniform> p"), "{diagnostic}");
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    fn a_commented_declaration_is_not_named_params() {
        // Without a real `struct Params`, `p` is undefined and reported in the snippet.
        let snippet = "// struct Params { a: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.a; }";
        match Effect::new("commented", snippet) {
            Err(error @ EffectError::Parse { .. }) => assert_eq!(error.location().map(|l| l.line), Some(2), "{error}"),
            other => panic!("expected a parse error, got {other:?}"),
        }
        // A snippet may not bind `p` itself either.
        let own = "@group(0) @binding(4) var<uniform> p: vec4<f32>;\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p; }";
        assert!(matches!(Effect::new("own", own), Err(EffectError::Interface { .. })));
    }

    #[test]
    fn clones_share_one_module_and_new_effects_get_new_ids() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        let (Ok(a), Ok(b)) = (Effect::new("a", snippet), Effect::new("b", snippet)) else { panic!("identity must compile") };
        let a2 = a.clone().with_sample_radius(4.0);
        assert_eq!(a.module().id, a2.module().id);
        assert_ne!(a.module().id, b.module().id);
    }
}
