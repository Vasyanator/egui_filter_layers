/*
File: effect.rs

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
- `Effect::new()`: assemble, parse, validate, check the resource interface.
- `render_diagnostic()`: renders naga's labels with `codespan-reporting` against the snippet
  (line numbers match `SnippetLocation`), or the assembled source when naga points at the wrapper.
- `reads_time()`: conservative IR scan deciding `Effect::is_animated`.
- `composite_module()` / `composite_linear_module()` (crate-internal): the internal replace pass
  that writes the filtered region back (gamma output / linear output for `*Srgb` targets; glow,
  offline). `composite_erase_module()` / `composite_add_module()` / `composite_add_linear_module()`:
  the two draws of the blended composite the wgpu UI draws over egui's pass.

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

/// The prelude prepended to every effect. See the crate docs, "Writing an effect".
pub(crate) const PRELUDE_WGSL: &str = include_str!("shaders/prelude.wgsl");

/// The body of the internal composite pass (appended after the prelude like a user effect).
const COMPOSITE_WGSL: &str = include_str!("shaders/composite.wgsl");

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
/// `@binding` of the original (unfiltered) capture, visible to the composite pass only.
pub(crate) const BINDING_ORIGINAL_TEXTURE: u32 = 3;

/// Monotonic source of `CompiledModule::id`. Ids are never reused within a process, so a backend
/// cache keyed by id can never confuse a dropped module with a new one.
///
/// An accepted exception to "no global mutable state": modules are built independently of any
/// install or context (presets are process-wide `OnceLock`s shared by every backend), so only a
/// process-wide counter gives ids unique across all of them. It is a monotonic id source, never
/// reset or read back as state, and `u64` cannot wrap in practice.
static NEXT_MODULE_ID: AtomicU64 = AtomicU64::new(1);

/// Which wrapper a module was assembled with; decides the allowed resource interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModuleKind {
    /// A user or preset effect: bindings 0..=2.
    Effect,
    /// The crate's composite pass: bindings 0..=3.
    Composite,
}

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
    /// Which wrapper was used.
    pub(crate) kind: ModuleKind,
}

impl fmt::Debug for CompiledModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledModule").field("id", &self.id).field("label", &self.label).field("kind", &self.kind).finish_non_exhaustive()
    }
}

/// A location inside the user's WGSL snippet (not the assembled module).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnippetLocation {
    /// 1-based line number within the snippet passed to [`Effect::new`].
    pub line: u32,
    /// 1-based column (in bytes) within that line.
    pub column: u32,
}

impl fmt::Display for SnippetLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Formats an optional snippet location for the `EffectError` messages.
fn location_suffix(location: Option<&SnippetLocation>) -> String {
    match location {
        Some(location) => format!(" at snippet line {location}"),
        None => String::new(),
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
    /// The snippet declared something the crate cannot bind: extra resources, entry points, or
    /// pipeline-overridable constants (`override`).
    #[error("effect `{label}`: {reason}")]
    Interface {
        /// The effect's label.
        label: String,
        /// What was declared and why it is rejected.
        reason: String,
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

    /// Where in the snippet the error is, when naga located it inside the snippet.
    /// Always `None` for [`EffectError::Interface`].
    #[must_use]
    pub fn location(&self) -> Option<SnippetLocation> {
        match self {
            Self::Parse { location, .. } | Self::Validation { location, .. } => *location,
            Self::Interface { .. } => None,
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
        let module = compile(label, wgsl, ModuleKind::Effect)?;
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

    /// The shared compiled module (crate-internal: backends key their caches on it).
    pub(crate) fn module(&self) -> &Arc<CompiledModule> {
        &self.module
    }
}

/// Whether any function of `module` may read the `time` member of the `fl` uniform block.
///
/// Walks naga's IR: an `AccessIndex` of `time` on the `fl` global counts, and so does any `Load`
/// of the WHOLE `fl` global (`let u = fl; ... u.time`), because the copy may be read anywhere —
/// erring towards "animated" only costs repaints, erring the other way freezes the effect.
/// The prelude does neither (pinned by a preset test), so a hit always comes from the snippet.
fn reads_time(module: &naga::Module) -> bool {
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

/// Clamps a user radius to a finite, non-negative value.
pub(crate) fn sanitize_radius(points: f32) -> f32 {
    if points.is_finite() && points > 0.0 { points } else { 0.0 }
}

/// Number of lines the prelude occupies; a naga line `n > PRELUDE_LINES` is snippet line
/// `n - PRELUDE_LINES`.
fn prelude_line_count() -> u32 {
    let lines = PRELUDE_WGSL.bytes().filter(|&b| b == b'\n').count();
    // The prelude is a compile-time constant of a few dozen lines; it cannot exceed u32.
    u32::try_from(lines).unwrap_or(u32::MAX)
}

/// Maps a naga location in the assembled source to the snippet, or `None` for the wrapper.
fn snippet_location(location: Option<naga::SourceLocation>) -> Option<SnippetLocation> {
    let location = location?;
    let prelude = prelude_line_count();
    (location.line_number > prelude).then(|| SnippetLocation { line: location.line_number - prelude, column: location.line_position })
}

/// Hint appended when naga reports an error outside the user's snippet — almost always a missing
/// or mistyped `effect` function, or a name clashing with the prelude.
const WRAPPER_HINT: &str = " (reported inside the generated wrapper: the snippet must define \
    `fn effect(uv: vec2<f32>) -> vec4<f32>` and must not redefine prelude names such as `src`, \
    `params`, `fl` or anything starting with `fl_`)";

/// Assembles, parses, validates and interface-checks one module.
fn compile(label: &str, snippet: &str, kind: ModuleKind) -> Result<CompiledModule, EffectError> {
    let mut source = String::with_capacity(PRELUDE_WGSL.len() + snippet.len() + 1);
    source.push_str(PRELUDE_WGSL);
    source.push_str(snippet);
    if !snippet.ends_with('\n') {
        source.push('\n');
    }

    let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
        let location = snippet_location(error.location(&source));
        let mut message = error.message().to_owned();
        if location.is_none() {
            message.push_str(WRAPPER_HINT);
        }
        let labels: Vec<(Range<usize>, String)> = error.labels().filter_map(|(span, text)| span.to_range().map(|range| (range, text.to_owned()))).collect();
        let notes = error.notes().map(str::to_owned).collect();
        let diagnostic = render_diagnostic(label, error.message(), &labels, notes, &source);
        EffectError::Parse { label: label.to_owned(), message, location, diagnostic }
    })?;

    let info = Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module).map_err(|error| {
        let location = snippet_location(error.location(&source));
        let mut message = error_chain(error.as_inner());
        if location.is_none() {
            message.push_str(WRAPPER_HINT);
        }
        let labels: Vec<(Range<usize>, String)> = error.spans().filter_map(|(span, text)| span.to_range().map(|range| (range, text.clone()))).collect();
        let diagnostic = render_diagnostic(label, &error.as_inner().to_string(), &labels, error_sources(error.as_inner()), &source);
        EffectError::Validation { label: label.to_owned(), message, location, diagnostic }
    })?;

    check_interface(label, &module, kind)?;

    let id = NEXT_MODULE_ID.fetch_add(1, Ordering::Relaxed);
    log::debug!("egui_filter_layers: compiled effect `{label}` (module id {id})");
    Ok(CompiledModule { id, label: Arc::from(label), source, module, info, kind })
}

/// The `source()` chain of `error` (without `error` itself), one entry per cause.
fn error_sources(error: &dyn std::error::Error) -> Vec<String> {
    std::iter::successors(error.source(), |cause| cause.source()).map(ToString::to_string).collect()
}

/// Renders a naga error as a multi-line `codespan-reporting` diagnostic.
///
/// `labels` are byte ranges into the assembled `source`. When the first (primary) label lies in
/// the user's snippet, the snippet alone is rendered — named after the effect `label`, with only
/// the labels inside it, shifted — so its line numbers equal [`SnippetLocation::line`]. Otherwise
/// (naga pointed into the prelude, e.g. a missing `effect` function) the assembled source is
/// rendered under the name `<label> (assembled)`. Never fails: a rendering error degrades to
/// `message` plus the notes.
fn render_diagnostic(label: &str, message: &str, labels: &[(Range<usize>, String)], notes: Vec<String>, source: &str) -> String {
    use codespan_reporting::diagnostic::{Diagnostic, Label};
    use codespan_reporting::files::SimpleFile;
    use codespan_reporting::term::{Config, emit_into_string};

    let prelude = PRELUDE_WGSL.len();
    let in_snippet = labels.first().is_some_and(|(range, _)| range.start >= prelude) && source.len() >= prelude;
    let (name, text, shown): (String, &str, Vec<Label<()>>) = if in_snippet {
        // Only labels fully inside the snippet can be drawn on it; a prelude label (e.g. the
        // wrapper's call site) is dropped rather than shown on the wrong line.
        let shifted = labels.iter().filter(|(range, _)| range.start >= prelude).map(|(range, text)| Label::primary((), (range.start - prelude)..(range.end - prelude)).with_message(text.clone())).collect();
        (label.to_owned(), &source[prelude..], shifted)
    } else {
        (format!("{label} (assembled)"), source, labels.iter().map(|(range, text)| Label::primary((), range.clone()).with_message(text.clone())).collect())
    };
    let file = SimpleFile::new(name, text);
    let diagnostic = Diagnostic::error().with_message(message).with_labels(shown).with_notes(notes.clone());
    emit_into_string(&Config::default(), &file, &diagnostic).unwrap_or_else(|error| {
        log::debug!("egui_filter_layers: could not render a WGSL diagnostic ({error}); using the plain message");
        std::iter::once(message.to_owned()).chain(notes).collect::<Vec<_>>().join("\n")
    })
}

/// Joins an error and its `source()` chain into one line.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut next = error.source();
    while let Some(cause) = next {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        next = cause.source();
    }
    message
}

/// Rejects entry points and resource bindings the backends do not provide.
fn check_interface(label: &str, module: &naga::Module, kind: ModuleKind) -> Result<(), EffectError> {
    let interface_error = |reason: String| EffectError::Interface { label: label.to_owned(), reason };

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

    let max_binding = match kind {
        ModuleKind::Effect => BINDING_SAMPLER,
        ModuleKind::Composite => BINDING_ORIGINAL_TEXTURE,
    };
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        if binding.group != 0 || binding.binding > max_binding {
            let name = global.name.as_deref().unwrap_or("<unnamed>");
            return Err(interface_error(format!(
                "declares resource `{name}` at @group({}) @binding({}); effects cannot bind their own resources — use `src()` for the input and `params(i)` for values",
                binding.group, binding.binding
            )));
        }
    }
    Ok(())
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
    compile(label, &snippet, ModuleKind::Composite).map(Arc::new)
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
        let extra_binding = "@group(0) @binding(7) var other: texture_2d<f32>;\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        assert!(matches!(Effect::new("binding", extra_binding), Err(EffectError::Interface { .. })));

        let original = "fn effect(uv: vec2<f32>) -> vec4<f32> { return textureSampleLevel(fl_original_texture, fl_sampler, uv, 0.0); }";
        assert!(Effect::new("private", original).is_err(), "the composite-only texture must not be reachable from effects");

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

    #[test]
    fn clones_share_one_module_and_new_effects_get_new_ids() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }";
        let (Ok(a), Ok(b)) = (Effect::new("a", snippet), Effect::new("b", snippet)) else { panic!("identity must compile") };
        let a2 = a.clone().with_sample_radius(4.0);
        assert_eq!(a.module().id, a2.module().id);
        assert_ne!(a.module().id, b.module().id);
    }
}
