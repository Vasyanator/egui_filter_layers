/*
File: effect/diagnostics.rs

Purpose:
Maps naga's parse / validation errors from the assembled module (prelude + snippet + crate tail)
back to the user's snippet: `SnippetLocation`, line mapping, and the multi-line
`codespan-reporting` diagnostic rendered against the snippet alone.

Key structures:
- `SnippetLocation` (public): 1-based line / column inside the snippet.
- `Assembled` (crate-internal): the assembled source and the byte / line ranges of its parts.

Key functions:
- `Assembled::locate()`: naga location -> snippet location plus a hint (`None` inside the
  prelude; the `struct Params` declaration for errors inside the crate tail).
- `Assembled::render()`: renders naga's labels against the snippet or the assembled source.
- `error_chain()` / `error_sources()`: flatten an error's `source()` chain.
*/

use std::fmt;
use std::ops::Range;

use super::PRELUDE_WGSL;

/// A location inside the user's WGSL snippet (not the assembled module).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnippetLocation {
    /// 1-based line number within the snippet passed to [`Effect::new`](crate::Effect::new).
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
pub(super) fn location_suffix(location: Option<&SnippetLocation>) -> String {
    match location {
        Some(location) => format!(" at snippet line {location}"),
        None => String::new(),
    }
}

/// Number of `\n`-terminated lines in `text`, saturating at `u32::MAX`.
fn line_count(text: &str) -> u32 {
    u32::try_from(text.bytes().filter(|&b| b == b'\n').count()).unwrap_or(u32::MAX)
}

/// Hint appended when naga reports an error outside the user's snippet — almost always a missing
/// or mistyped `effect` function, or a name clashing with the prelude.
const WRAPPER_HINT: &str = " (reported inside the generated wrapper: the snippet must define \
    `fn effect(uv: vec2<f32>) -> vec4<f32>` and must not redefine prelude names such as `src`, \
    `params`, `fl` or anything starting with `fl_`)";

/// Hint appended when naga reports an error at the declaration the crate appends for
/// `struct Params` (the location then points at the struct).
const TAIL_HINT: &str = " (reported at the crate-appended `@group(0) @binding(4) var<uniform> p: Params;`: \
    `struct Params` must be valid in a uniform block, and the snippet must not declare its own \
    module-scope `p`)";

/// An assembled module (prelude + snippet + optional crate tail) and where its parts lie, so
/// naga's locations and labels map back to the snippet.
#[derive(Debug)]
pub(super) struct Assembled {
    /// The full WGSL source handed to naga.
    pub(super) source: String,
    /// Byte range of the snippet (with the newline the crate ensures at its end).
    snippet: Range<usize>,
    /// Lines of the prelude: naga line `n` is snippet line `n - prelude_lines`.
    prelude_lines: u32,
    /// Lines of the snippet (with its ensured final newline).
    snippet_lines: u32,
    /// Where errors reported inside the tail are shown: the `struct Params` declaration.
    tail_location: Option<SnippetLocation>,
}

impl Assembled {
    /// `PRELUDE_WGSL`, then `snippet` (newline-terminated), then `tail` (empty, or the named
    /// parameter declaration whose errors are attributed to `tail_location`).
    pub(super) fn new(snippet: &str, tail: &str, tail_location: Option<SnippetLocation>) -> Self {
        let mut source = String::with_capacity(PRELUDE_WGSL.len() + snippet.len() + 1 + tail.len());
        source.push_str(PRELUDE_WGSL);
        source.push_str(snippet);
        if !snippet.ends_with('\n') {
            source.push('\n');
        }
        let snippet_range = PRELUDE_WGSL.len()..source.len();
        let snippet_lines = line_count(&source[snippet_range.clone()]);
        source.push_str(tail);
        Self { source, snippet: snippet_range, prelude_lines: line_count(PRELUDE_WGSL), snippet_lines, tail_location }
    }

    /// Maps a naga location to the snippet, with the hint to append to the error message:
    /// inside the snippet -> its location and no hint; inside the prelude (or unknown) -> `None`
    /// and the wrapper hint; inside the tail -> the `struct Params` location and the tail hint.
    pub(super) fn locate(&self, location: Option<naga::SourceLocation>) -> (Option<SnippetLocation>, &'static str) {
        let Some(location) = location else {
            return (None, WRAPPER_HINT);
        };
        let line = location.line_number;
        if line <= self.prelude_lines {
            (None, WRAPPER_HINT)
        } else if line - self.prelude_lines > self.snippet_lines {
            (self.tail_location, TAIL_HINT)
        } else {
            (Some(SnippetLocation { line: line - self.prelude_lines, column: location.line_position }), "")
        }
    }

    /// Renders a naga error as a multi-line `codespan-reporting` diagnostic.
    ///
    /// `labels` are byte ranges into `self.source`. When the first (primary) label lies in the
    /// user's snippet, the snippet alone is rendered — named after the effect `label`, with only
    /// the labels inside it, shifted — so its line numbers equal [`SnippetLocation::line`].
    /// Otherwise (naga pointed into the prelude, e.g. a missing `effect` function, or into the
    /// crate tail) the assembled source is rendered under the name `<label> (assembled)`. Never
    /// fails: a rendering error degrades to `message` plus the notes.
    pub(super) fn render(&self, label: &str, message: &str, labels: &[(Range<usize>, String)], notes: Vec<String>) -> String {
        use codespan_reporting::diagnostic::{Diagnostic, Label};
        use codespan_reporting::files::SimpleFile;
        use codespan_reporting::term::{Config, emit_into_string};

        let snippet = &self.snippet;
        let inside = |range: &Range<usize>| range.start >= snippet.start && range.end <= snippet.end;
        let in_snippet = labels.first().is_some_and(|(range, _)| inside(range));
        let (name, text, shown): (String, &str, Vec<Label<()>>) = if in_snippet {
            // Only labels fully inside the snippet can be drawn on it; a prelude or tail label
            // (e.g. the wrapper's call site) is dropped rather than shown on the wrong line.
            let shifted = labels.iter().filter(|(range, _)| inside(range)).map(|(range, text)| Label::primary((), (range.start - snippet.start)..(range.end - snippet.start)).with_message(text.clone())).collect();
            (label.to_owned(), &self.source[snippet.clone()], shifted)
        } else {
            (format!("{label} (assembled)"), self.source.as_str(), labels.iter().map(|(range, text)| Label::primary((), range.clone()).with_message(text.clone())).collect())
        };
        let file = SimpleFile::new(name, text);
        let diagnostic = Diagnostic::error().with_message(message).with_labels(shown).with_notes(notes.clone());
        emit_into_string(&Config::default(), &file, &diagnostic).unwrap_or_else(|error| {
            log::debug!("egui_shader_layers: could not render a WGSL diagnostic ({error}); using the plain message");
            std::iter::once(message.to_owned()).chain(notes).collect::<Vec<_>>().join("\n")
        })
    }
}

/// The `source()` chain of `error` (without `error` itself), one entry per cause.
pub(super) fn error_sources(error: &dyn std::error::Error) -> Vec<String> {
    std::iter::successors(error.source(), |cause| cause.source()).map(ToString::to_string).collect()
}

/// Joins an error and its `source()` chain into one line.
pub(super) fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut next = error.source();
    while let Some(cause) = next {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        next = cause.source();
    }
    message
}

