/*
File: hot.rs

Purpose:
F5 (feature `hot-reload`, native only): hot reload of a custom effect from a WGSL file while the
app runs. A background `std::thread` polls the file's metadata, re-reads and recompiles it
(`Effect::new`, naga) when it changes, and hands the result to the GUI thread, which only swaps
an `Effect` (an `Arc` bump) in `HotEffect::poll`.

Key structures:
- `HotEffect` (public): the GUI-side handle — the last good effect, the current error, `poll`.
- `HotOptions` (public): poll interval, the `egui::Context` to repaint, an effect set-up closure,
  whether the file is a state pass (`with_state`, compiled with `Effect::new_state`, F7).
- `HotReloadEvent`, `HotReloadError` (public): what one `poll` observed; why the file is broken.
- `Rebound` + `rebind()` (public): rebuild a `Pass` on a reloaded effect, keeping the named and
  anonymous parameter values that still fit.

Notes:
No new dependency (owner decision §16 (d)): change detection is `(mtime, length)` polling, plus a
content comparison so a touch without an edit does not reload. The worker and the GUI share only
channels (no lock at all); the GUI never blocks on the worker except for a bounded wait in `Drop`.
On wasm32 there is no file system or thread: the module is not compiled there (see `lib.rs`).
*/

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

use crate::effect::{Effect, EffectError};
use crate::pass::Pass;

/// The poll interval [`HotOptions::default`] uses: fast enough to feel instant after a save,
/// slow enough to cost nothing (one `metadata` call per tick).
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The shortest poll interval accepted ([`HotOptions::with_interval`] clamps to it), so a zero
/// interval cannot turn the watcher into a busy loop.
pub const MIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// How long dropping a [`HotEffect`] waits for its watcher thread to exit. The worker wakes at
/// once on the stop signal; only a compile or a read already in flight can outlast this, and
/// then the thread is detached (it exits right after that step, see [`HotEffect`]'s `Drop`).
const SHUTDOWN_WAIT: Duration = Duration::from_millis(100);

/// Turns every freshly compiled effect into the one published (sample radius, animation flag).
type Setup = Box<dyn Fn(Effect) -> Effect + Send + 'static>;

/// Options of [`HotEffect::watch_with`].
pub struct HotOptions {
    /// Time between two metadata polls.
    interval: Duration,
    /// Repainted after each published change, so an idle app shows it without input.
    repaint: Option<egui::Context>,
    /// Applied to each compiled effect before it is published.
    setup: Option<Setup>,
    /// Compile the file as a state pass (`Effect::new_state`).
    state: bool,
}

impl Default for HotOptions {
    /// Poll every [`DEFAULT_POLL_INTERVAL`], no repaint, effects published as compiled.
    fn default() -> Self {
        Self { interval: DEFAULT_POLL_INTERVAL, repaint: None, setup: None, state: false }
    }
}

impl fmt::Debug for HotOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotOptions").field("interval", &self.interval).field("repaint", &self.repaint.is_some()).field("setup", &self.setup.is_some()).field("state", &self.state).finish()
    }
}

impl HotOptions {
    /// Polls the file every `interval`, clamped to at least [`MIN_POLL_INTERVAL`].
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval.max(MIN_POLL_INTERVAL);
        self
    }

    /// Requests a repaint of `ctx` whenever a new effect or a new error is published, so the
    /// change shows up (and [`HotEffect::poll`] runs) even when the app is idle.
    #[must_use]
    pub fn with_repaint(mut self, ctx: &egui::Context) -> Self {
        self.repaint = Some(ctx.clone());
        self
    }

    /// Runs `setup` on every freshly compiled effect (on the watcher thread) before it is
    /// published, e.g. `|e| e.with_sample_radius(12.0)`: the file holds only the WGSL, so what
    /// the code would configure on a hand-built [`Effect`] goes here.
    #[must_use]
    pub fn with_setup(mut self, setup: impl Fn(Effect) -> Effect + Send + 'static) -> Self {
        self.setup = Some(Box::new(setup));
        self
    }

    /// Compiles the file as a STATE pass ([`Effect::new_state`], F7) instead of a display pass.
    #[must_use]
    pub fn with_state(mut self, state: bool) -> Self {
        self.state = state;
        self
    }

    /// The poll interval in use.
    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }
}

/// Why the watched file currently yields no new effect.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HotReloadError {
    /// The file does not exist (yet, or any more). The watcher keeps polling for it.
    #[error("hot reload: `{}` not found", .path.display())]
    Missing {
        /// The watched path.
        path: PathBuf,
    },
    /// Reading the file or its metadata failed (permissions, not UTF-8, ...).
    #[error("hot reload: cannot read `{}`: {message}", .path.display())]
    Io {
        /// The watched path.
        path: PathBuf,
        /// The OS error kind.
        kind: io::ErrorKind,
        /// The OS error message.
        message: String,
    },
    /// The file was read but the WGSL does not compile; the last good effect stays in use.
    #[error("hot reload: {0}")]
    Compile(EffectError),
    /// The watcher thread could not be started ([`HotEffect::watch`] returns it).
    #[error("hot reload: cannot start the watcher thread: {message}")]
    Spawn {
        /// The OS error message.
        message: String,
    },
    /// The watcher thread ended unexpectedly; the effect no longer follows the file.
    #[error("hot reload: the watcher of `{}` stopped", .path.display())]
    Stopped {
        /// The watched path.
        path: PathBuf,
    },
}

impl HotReloadError {
    /// The error of an I/O operation on `path`: [`HotReloadError::Missing`] for "not found".
    fn io(path: &Path, error: &io::Error) -> Self {
        if error.kind() == io::ErrorKind::NotFound {
            Self::Missing { path: path.to_owned() }
        } else {
            Self::Io { path: path.to_owned(), kind: error.kind(), message: error.to_string() }
        }
    }

    /// naga's full diagnostic, rendered against the file's own lines (see
    /// [`EffectError::diagnostic`]), for a compile error that has one.
    #[must_use]
    pub fn diagnostic(&self) -> Option<&str> {
        match self {
            Self::Compile(error) => error.diagnostic(),
            Self::Missing { .. } | Self::Io { .. } | Self::Spawn { .. } | Self::Stopped { .. } => None,
        }
    }

    /// The compile error, when the file was read but did not compile.
    #[must_use]
    pub fn effect_error(&self) -> Option<&EffectError> {
        match self {
            Self::Compile(error) => Some(error),
            Self::Missing { .. } | Self::Io { .. } | Self::Spawn { .. } | Self::Stopped { .. } => None,
        }
    }
}

/// What one [`HotEffect::poll`] observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotReloadEvent {
    /// Nothing new since the previous poll.
    Unchanged,
    /// [`HotEffect::current`] is a new effect: rebuild the passes that use it ([`rebind`]).
    /// A failure published in the same poll is visible through [`HotEffect::error`].
    Reloaded,
    /// A new failure: [`HotEffect::error`] says why; [`HotEffect::current`] is unchanged.
    Failed,
}

/// One message from the watcher thread.
enum Update {
    /// A compiled (and set-up) effect.
    Reloaded(Effect),
    /// A new, distinct failure.
    Failed(HotReloadError),
}

/// An effect that follows a WGSL file on disk (feature `hot-reload`, native only).
///
/// A background thread polls the file and recompiles it when it changes; the GUI thread calls
/// [`HotEffect::poll`] once per frame, which never blocks (it drains a channel). While the file
/// is broken, [`HotEffect::current`] keeps the last good effect, so layers keep rendering, and
/// [`HotEffect::error`] says what is wrong (with naga's diagnostic for compile errors).
///
/// ```no_run
/// # fn demo(ctx: &egui::Context) -> Result<(), egui_filter_layers::HotReloadError> {
/// use egui_filter_layers::{HotEffect, HotOptions, HotReloadEvent, Pass, rebind};
/// let mut hot = HotEffect::watch_with("water", "shaders/water.wgsl", HotOptions::default().with_repaint(ctx))?;
/// let mut pass: Option<Pass> = None;
/// // every frame:
/// if hot.poll() == HotReloadEvent::Reloaded {
///     if let Some(effect) = hot.current() {
///         pass = Some(match &pass { Some(old) => rebind(old, effect).pass, None => Pass::new(effect) });
///     }
/// }
/// # Ok(()) }
/// ```
///
/// Dropping it stops the watcher: the thread is woken at once and joined within a short bound
/// (100 ms); a thread still busy compiling after that is detached and exits right after.
pub struct HotEffect {
    /// The effect label (also in every error).
    label: String,
    /// The watched file.
    path: PathBuf,
    /// Updates published by the watcher.
    updates: Receiver<Update>,
    /// Dropping it (disconnecting) tells the watcher to stop; it never carries a message.
    stop: Option<Sender<()>>,
    /// Disconnects when the watcher thread has exited (its sender is dropped with the thread).
    done: Receiver<()>,
    /// The watcher thread, `None` once joined or detached.
    thread: Option<JoinHandle<()>>,
    /// The last effect that compiled.
    current: Option<Effect>,
    /// The current failure, cleared by the next successful reload.
    error: Option<HotReloadError>,
}

impl fmt::Debug for HotEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotEffect").field("label", &self.label).field("path", &self.path).field("current", &self.current).field("error", &self.error).finish_non_exhaustive()
    }
}

impl HotEffect {
    /// Watches `path` with [`HotOptions::default`]; see [`HotEffect::watch_with`].
    ///
    /// # Errors
    /// [`HotReloadError::Spawn`] when the watcher thread cannot be started.
    pub fn watch(label: &str, path: impl Into<PathBuf>) -> Result<Self, HotReloadError> {
        Self::watch_with(label, path, HotOptions::default())
    }

    /// Starts watching `path`: the file is read and compiled on the watcher thread, so
    /// [`HotEffect::current`] is `None` until the first compile succeeds (a missing or broken
    /// file shows up in [`HotEffect::error`] after a [`HotEffect::poll`]). `label` names the
    /// effect in errors and logs.
    ///
    /// # Errors
    /// [`HotReloadError::Spawn`] when the watcher thread cannot be started. Everything about the
    /// file itself is reported asynchronously.
    pub fn watch_with(label: &str, path: impl Into<PathBuf>, options: HotOptions) -> Result<Self, HotReloadError> {
        let path = path.into();
        let (update_tx, updates) = mpsc::channel();
        let (stop, stop_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel::<()>();
        let watcher = Watcher { label: label.to_owned(), path: path.clone(), options, updates: update_tx, last_key: None, last_source: None, last_error: None };
        let thread = std::thread::Builder::new()
            .name(format!("filter-hot-reload:{label}"))
            .spawn(move || {
                // Held for the thread's lifetime: its drop disconnects `done`, telling `Drop`
                // that the thread has exited without a timed join.
                let _done_guard = done_tx;
                watcher.run(&stop_rx);
            })
            .map_err(|error| HotReloadError::Spawn { message: error.to_string() })?;
        log::debug!("hot reload `{label}`: watching `{}`", path.display());
        Ok(Self { label: label.to_owned(), path, updates, stop: Some(stop), done, thread: Some(thread), current: None, error: None })
    }

    /// Takes what the watcher published since the previous call (GUI thread, once per frame;
    /// never blocks). Returns [`HotReloadEvent::Reloaded`] when [`HotEffect::current`] changed,
    /// else [`HotReloadEvent::Failed`] when a new failure arrived, else `Unchanged`.
    pub fn poll(&mut self) -> HotReloadEvent {
        let mut reloaded = false;
        let mut failed = false;
        loop {
            match self.updates.try_recv() {
                Ok(Update::Reloaded(effect)) => {
                    self.current = Some(effect);
                    self.error = None;
                    reloaded = true;
                }
                Ok(Update::Failed(error)) => {
                    self.error = Some(error);
                    failed = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // The watcher ended while we still hold `stop`: it can only have died
                    // (a panic in a user `setup` closure). Report it once.
                    if self.stop.is_some() && !matches!(self.error, Some(HotReloadError::Stopped { .. })) {
                        let error = HotReloadError::Stopped { path: self.path.clone() };
                        log::error!("hot reload `{}`: {error}", self.label);
                        self.error = Some(error);
                        failed = true;
                    }
                    break;
                }
            }
        }
        if reloaded {
            HotReloadEvent::Reloaded
        } else if failed {
            HotReloadEvent::Failed
        } else {
            HotReloadEvent::Unchanged
        }
    }

    /// The last effect that compiled, `None` before the first success.
    #[must_use]
    pub fn current(&self) -> Option<&Effect> {
        self.current.as_ref()
    }

    /// Why the file currently yields no new effect, `None` while it is fine.
    #[must_use]
    pub fn error(&self) -> Option<&HotReloadError> {
        self.error.as_ref()
    }

    /// The watched path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The effect label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// A human-readable status for a label or an overlay: the error with naga's diagnostic when
    /// the file is broken (and whether an older version is still shown), else a one-line "ok".
    #[must_use]
    pub fn status_text(&self) -> String {
        match (&self.error, &self.current) {
            (Some(error), current) => {
                let mut text = error.to_string();
                if let Some(diagnostic) = error.diagnostic() {
                    text.push('\n');
                    text.push_str(diagnostic);
                }
                if current.is_some() {
                    text.push_str("\n(showing the last version that compiled)");
                }
                text
            }
            (None, Some(_)) => format!("`{}`: ok ({})", self.label, self.path.display()),
            (None, None) => format!("`{}`: loading {}", self.label, self.path.display()),
        }
    }

    /// Stops the watcher: wakes it, then waits up to [`SHUTDOWN_WAIT`] for it to exit. Returns
    /// whether the thread was joined (`false`: still busy, detached — it exits after its current
    /// step because every channel it could use is closed).
    fn shutdown(&mut self) -> bool {
        // Disconnecting `stop` wakes the watcher's `recv_timeout` immediately.
        self.stop = None;
        let Some(thread) = self.thread.take() else { return true };
        match self.done.recv_timeout(SHUTDOWN_WAIT) {
            // `done` never carries a message; a disconnect means the thread body returned, so
            // this join does not block.
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                if thread.join().is_err() {
                    log::error!("hot reload `{}`: the watcher thread panicked", self.label);
                }
                true
            }
            Err(RecvTimeoutError::Timeout) => {
                log::debug!("hot reload `{}`: watcher still busy after {SHUTDOWN_WAIT:?}, detached", self.label);
                false
            }
        }
    }
}

impl Drop for HotEffect {
    fn drop(&mut self) {
        // Detaching on timeout is sound: the thread owns everything it touches and exits once it
        // sees `stop` disconnected or fails to publish into the dropped `updates` receiver.
        let joined = self.shutdown();
        log::debug!("hot reload `{}`: stopped (joined: {joined})", self.label);
    }
}

/// Identity of one version of the file, as far as metadata can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileKey {
    /// Modification time; `None` where the platform does not report it (then only `len` counts).
    modified: Option<SystemTime>,
    /// File length in bytes.
    len: u64,
}

/// The watcher thread's state.
struct Watcher {
    /// The effect label.
    label: String,
    /// The watched file.
    path: PathBuf,
    /// Interval, repaint context, set-up closure.
    options: HotOptions,
    /// Where updates are published.
    updates: Sender<Update>,
    /// The metadata of the last version read, `None` before the first read or after an error.
    last_key: Option<FileKey>,
    /// The content of the last version compiled (good or bad).
    last_source: Option<String>,
    /// The `Display` of the last failure published, `None` after a success: failures are
    /// published and logged once per distinct error.
    last_error: Option<String>,
}

impl Watcher {
    /// The poll loop: one [`Watcher::tick`], then sleep on `stop` until the interval passes.
    /// Returns when `stop` disconnects or the GUI side is gone.
    fn run(mut self, stop: &Receiver<()>) {
        loop {
            if !self.tick() {
                return;
            }
            match stop.recv_timeout(self.options.interval) {
                Err(RecvTimeoutError::Timeout) => {}
                Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }

    /// Checks the file once and publishes what changed. Returns `false` when the receiver is gone.
    fn tick(&mut self) -> bool {
        let key = match std::fs::metadata(&self.path) {
            Ok(metadata) => FileKey { modified: metadata.modified().ok(), len: metadata.len() },
            Err(error) => {
                self.last_key = None;
                return self.fail(HotReloadError::io(&self.path, &error));
            }
        };
        if self.last_key == Some(key) {
            return true;
        }
        let source = match std::fs::read_to_string(&self.path) {
            Ok(source) => source,
            Err(error) => {
                self.last_key = None;
                return self.fail(HotReloadError::io(&self.path, &error));
            }
        };
        // The key read before the content: an edit landing in between changes the key again and
        // is read on the next tick.
        self.last_key = Some(key);
        // A touch or a save without edits: nothing to do, unless a failure is showing (a file
        // that comes back unchanged after being missing must clear the error).
        if self.last_error.is_none() && self.last_source.as_deref() == Some(source.as_str()) {
            return true;
        }
        let result = if self.options.state { Effect::new_state(&self.label, &source) } else { Effect::new(&self.label, &source) };
        self.last_source = Some(source);
        match result {
            Ok(effect) => {
                let effect = match &self.options.setup {
                    Some(setup) => setup(effect),
                    None => effect,
                };
                self.last_error = None;
                log::info!("hot reload `{}`: reloaded `{}`", self.label, self.path.display());
                self.publish(Update::Reloaded(effect))
            }
            Err(error) => self.fail(HotReloadError::Compile(error)),
        }
    }

    /// Publishes and logs `error` unless it equals the last one published.
    fn fail(&mut self, error: HotReloadError) -> bool {
        let text = error.to_string();
        if self.last_error.as_deref() == Some(text.as_str()) {
            return true;
        }
        match error.diagnostic() {
            Some(diagnostic) => log::warn!("hot reload `{}`: {text}\n{diagnostic}", self.label),
            None => log::warn!("hot reload `{}`: {text}", self.label),
        }
        self.last_error = Some(text);
        self.publish(Update::Failed(error))
    }

    /// Sends `update` and requests a repaint. Returns `false` when the GUI side is gone.
    fn publish(&self, update: Update) -> bool {
        if self.updates.send(update).is_err() {
            return false;
        }
        if let Some(ctx) = &self.options.repaint {
            ctx.request_repaint();
        }
        true
    }
}

/// A [`Pass`] rebuilt on another effect by [`rebind`].
#[derive(Clone, Debug)]
pub struct Rebound {
    /// The new pass.
    pub pass: Pass,
    /// Named parameters of the old pass whose values were NOT carried over: the new effect has
    /// no member of that name, or one of another type or array-ness. In the old declaration order.
    pub dropped: Vec<String>,
}

/// Rebuilds `pass` on `effect` (typically a hot-reloaded version of its effect), keeping what
/// still fits: every named parameter whose name, element type and array-ness match (an array
/// keeps its first `min(old, new)` elements; the rest start at zero), the anonymous `params(i)`
/// block and the output scale. A sample-radius override is kept when the old pass had one (its
/// radius differed from its effect's); otherwise the pass follows the new effect's radius.
#[must_use]
pub fn rebind(pass: &Pass, effect: &Effect) -> Rebound {
    let mut rebound = Pass::new(effect).with_params(*pass.params()).with_scale(pass.scale());
    let old = pass.effect();
    // `Pass` exposes only the effective radius: a value different from the old effect's is an
    // explicit override (an override equal to the old default is indistinguishable and follows
    // the new effect, which is what it meant).
    if pass.sample_radius().to_bits() != old.sample_radius().to_bits() {
        rebound = rebound.with_sample_radius(pass.sample_radius());
    }
    let mut dropped = Vec::new();
    for info in old.params() {
        let fits = effect.param(&info.name).is_some_and(|new| new.ty == info.ty && new.is_array == info.is_array);
        let carried = fits && carry(pass, &mut rebound, &info.name, effect);
        if !carried {
            dropped.push(info.name.clone());
        }
    }
    Rebound { pass: rebound, dropped }
}

/// Copies named parameter `name` from `old` into `new` (whose effect is `effect`); `false` when
/// the values could not be read or written (then `new` keeps zeros for it).
fn carry(old: &Pass, new: &mut Pass, name: &str, effect: &Effect) -> bool {
    let Some(values) = old.get_slice(name) else { return false };
    let capacity = effect.param(name).map_or(0, |info| usize::try_from(info.count).unwrap_or(usize::MAX));
    let kept = values.get(..values.len().min(capacity)).unwrap_or_default();
    new.set_slice(name, kept).is_ok()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use super::*;
    use crate::effect::ParamValue;

    /// Generous: the watcher polls every few milliseconds in tests; this only bounds a hang.
    const WAIT: Duration = Duration::from_secs(10);
    const FAST: Duration = Duration::from_millis(5);

    /// A unique file under the system temp dir, removed on drop.
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let unique = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!("egui_filter_layers_hot_{}_{unique}_{name}.wgsl", std::process::id())))
        }

        fn write(&self, content: &str) {
            if let Err(error) = std::fs::write(&self.0, content) {
                panic!("cannot write {}: {error}", self.0.display());
            }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_file(&self.0) {
                eprintln!("cannot remove {}: {error}", self.0.display());
            }
        }
    }

    /// Polls `hot` until `done` holds, failing after [`WAIT`].
    fn wait_until(hot: &mut HotEffect, what: &str, done: impl Fn(&HotEffect) -> bool) {
        let start = Instant::now();
        while start.elapsed() < WAIT {
            hot.poll();
            if done(hot) {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for {what}: {}", hot.status_text());
    }

    fn source_contains(hot: &HotEffect, marker: &str) -> bool {
        hot.error().is_none() && hot.current().is_some_and(|effect| effect.assembled_source().contains(marker))
    }

    fn watch(label: &str, file: &TempFile) -> HotEffect {
        match HotEffect::watch_with(label, &file.0, HotOptions::default().with_interval(FAST).with_repaint(&egui::Context::default())) {
            Ok(hot) => hot,
            Err(error) => panic!("{error}"),
        }
    }

    const GOOD_A: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return src(uv); // version_a\n}\n";
    const GOOD_B: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return src(uv) * 0.5; // version_b, longer\n}\n";
    const BROKEN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let x = ;\n    return src(uv);\n}\n";
    const GOOD_C: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return vec4<f32>(1.0) - src(uv); // version_c, fixed again\n}\n";

    #[test]
    fn hot_reload_follows_write_modify_break_fix() {
        let file = TempFile::new("cycle");
        file.write(GOOD_A);
        let mut hot = watch("cycle", &file);
        wait_until(&mut hot, "first compile", |hot| source_contains(hot, "version_a"));

        file.write(GOOD_B);
        wait_until(&mut hot, "reload", |hot| source_contains(hot, "version_b"));

        file.write(BROKEN);
        wait_until(&mut hot, "compile error", |hot| hot.error().is_some());
        let Some(error) = hot.error() else { unreachable!("waited for it") };
        assert_eq!(error.effect_error().and_then(EffectError::location).map(|l| l.line), Some(2), "{error}");
        assert!(error.diagnostic().is_some_and(|d| d.contains("let x = ;")), "{error:?}");
        assert!(hot.current().is_some_and(|effect| effect.assembled_source().contains("version_b")), "the last good effect stays");
        assert!(hot.status_text().contains("let x = ;"), "{}", hot.status_text());

        file.write(GOOD_C);
        wait_until(&mut hot, "fixed", |hot| source_contains(hot, "version_c"));
        assert_eq!(hot.poll(), HotReloadEvent::Unchanged);
    }

    #[test]
    fn hot_reload_reports_a_missing_file_and_picks_it_up() {
        let file = TempFile::new("missing");
        let mut hot = watch("missing", &file);
        wait_until(&mut hot, "missing error", |hot| matches!(hot.error(), Some(HotReloadError::Missing { .. })));
        assert!(hot.current().is_none());
        file.write(GOOD_A);
        wait_until(&mut hot, "appeared", |hot| source_contains(hot, "version_a"));
    }

    #[test]
    fn hot_reload_applies_the_setup_closure() {
        let file = TempFile::new("setup");
        file.write(GOOD_A);
        let options = HotOptions::default().with_interval(FAST).with_setup(|effect| effect.with_sample_radius(7.0));
        let mut hot = match HotEffect::watch_with("setup", &file.0, options) {
            Ok(hot) => hot,
            Err(error) => panic!("{error}"),
        };
        wait_until(&mut hot, "first compile", |hot| hot.current().is_some());
        assert!(hot.current().is_some_and(|effect| (effect.sample_radius() - 7.0).abs() < f32::EPSILON));
    }

    #[test]
    fn hot_reload_shutdown_joins_the_watcher_promptly() {
        let file = TempFile::new("shutdown");
        file.write(GOOD_A);
        // A long interval: shutdown must wake the sleeping watcher, not wait the interval out.
        let mut hot = match HotEffect::watch_with("shutdown", &file.0, HotOptions::default().with_interval(Duration::from_mins(1))) {
            Ok(hot) => hot,
            Err(error) => panic!("{error}"),
        };
        wait_until(&mut hot, "first compile", |hot| hot.current().is_some());
        let start = Instant::now();
        assert!(hot.shutdown(), "the watcher must be joined, not detached");
        assert!(start.elapsed() < Duration::from_secs(1), "{:?}", start.elapsed());
        assert!(hot.thread.is_none());
        drop(hot);
    }

    #[test]
    fn hot_reload_rebind_keeps_matching_named_params() {
        let old_effect = match Effect::new("old", "struct Params { amount: f32, color: vec4<f32>, stops: array<vec4<f32>, 4> }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.amount + p.color + p.stops[0]; }") {
            Ok(effect) => effect,
            Err(error) => panic!("{error}"),
        };
        let new_effect = match Effect::new("new", "struct Params { color: vec3<f32>, amount: f32, stops: array<vec4<f32>, 2>, extra: u32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.amount + vec4<f32>(p.color, 0.0) + p.stops[1] * f32(p.extra); }") {
            Ok(effect) => effect.with_sample_radius(3.0),
            Err(error) => panic!("{error}"),
        };
        let mut pass = Pass::new(&old_effect).with_scale(0.5).with_sample_radius(9.0);
        let filled = pass.set("amount", 0.25).and_then(|()| pass.set("color", [1.0, 0.0, 0.0, 1.0])).and_then(|()| pass.set_slice("stops", &[[1.0_f32, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]));
        assert!(filled.is_ok(), "{filled:?}");
        assert!(pass.set_param(3, [4.0, 5.0, 6.0, 7.0]).is_ok());

        let Rebound { pass: rebound, dropped } = rebind(&pass, &new_effect);
        assert_eq!(dropped, vec!["color".to_owned()], "vec4 -> vec3 is a type change");
        assert_eq!(rebound.get("amount"), Some(ParamValue::F32(0.25)));
        assert_eq!(rebound.get("color"), Some(ParamValue::Vec3([0.0; 3])));
        assert_eq!(rebound.get_slice("stops"), Some(vec![ParamValue::Vec4([1.0; 4]), ParamValue::Vec4([2.0; 4])]));
        assert_eq!(rebound.get("extra"), Some(ParamValue::U32(0)));
        assert_eq!(rebound.param(3), Some([4.0, 5.0, 6.0, 7.0]));
        assert!((rebound.scale() - 0.5).abs() < f32::EPSILON);
        assert!((rebound.sample_radius() - 9.0).abs() < f32::EPSILON, "the override is kept");

        let plain = rebind(&Pass::new(&old_effect), &new_effect).pass;
        assert!((plain.sample_radius() - 3.0).abs() < f32::EPSILON, "no override: follow the new effect");
    }
}
