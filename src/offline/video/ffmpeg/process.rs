/*
File: offline/video/ffmpeg/process.rs

Purpose:
A child process of the ffmpeg adapters (`ffmpeg` / `ffprobe`) with the plumbing that keeps pipes
from deadlocking and never leaks a process: stderr is drained on its own thread (keeping only its
tail for error messages), and a process that was not waited for is killed and reaped on drop.

Key structures:
- `ToolProcess`: the child, its stderr collector, its program path.

Key functions:
- `ToolProcess::spawn()`: typed spawn errors (`ToolNotFound` for a missing program).
- `ToolProcess::wait()`: exit status -> `Ok` or `ProcessFailed` with the stderr tail.
- `ToolProcess::kill()`: kill + wait (used by `Drop`, sinks' `abort`).
- `run_to_string()`: run a program to completion and return its stdout (ffprobe).

Notes:
On Windows the child is created without a console window (`CREATE_NO_WINDOW`), so a GUI app
spawning ffmpeg does not flash a console.
*/

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;

use super::super::error::VideoError;

/// Bytes of stderr kept for error messages (the end of the output).
const STDERR_TAIL_BYTES: usize = 4096;
/// Lines of stderr kept for error messages.
const STDERR_TAIL_LINES: usize = 20;

/// A running `ffmpeg` / `ffprobe`.
#[derive(Debug)]
pub(crate) struct ToolProcess {
    /// The program (for messages).
    program: PathBuf,
    /// The child.
    child: Child,
    /// Drains stderr; returns its tail.
    stderr: Option<JoinHandle<String>>,
    /// Whether the child was waited for (then `Drop` has nothing to do).
    reaped: bool,
}

impl ToolProcess {
    /// Starts `program` with `args`; stdin / stdout are piped when asked, else null. Stderr is
    /// always piped and drained on a thread.
    ///
    /// # Errors
    /// [`VideoError::ToolNotFound`] when the program does not exist, [`VideoError::Spawn`]
    /// otherwise (also when the stderr thread cannot be created; the child is then killed).
    pub(crate) fn spawn(program: &Path, args: &[OsString], pipe_stdin: bool, pipe_stdout: bool) -> Result<Self, VideoError> {
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(if pipe_stdin { Stdio::piped() } else { Stdio::null() })
            .stdout(if pipe_stdout { Stdio::piped() } else { Stdio::null() })
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            /// `CREATE_NO_WINDOW`: no console window for the child of a GUI process.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        log::debug!("egui_shader_layers (video): running {} {}", program.display(), args.iter().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>().join(" "));
        let mut child = command.spawn().map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound { VideoError::ToolNotFound { program: program.to_owned() } } else { VideoError::Spawn { program: program.to_owned(), source } }
        })?;
        let stderr = match child.stderr.take() {
            Some(stream) => Some(spawn_drain(stream).map_err(|source| {
                // Without a drain the child could block on a full stderr pipe: stop it.
                if let Err(error) = child.kill().and_then(|()| child.wait().map(drop)) {
                    log::warn!("egui_shader_layers (video): could not stop `{}` after a failed thread spawn: {error}", program.display());
                }
                VideoError::Spawn { program: program.to_owned(), source }
            })?),
            None => None,
        };
        Ok(Self { program: program.to_owned(), child, stderr, reaped: false })
    }

    /// The program.
    pub(crate) fn program(&self) -> &Path {
        &self.program
    }

    /// Takes the child's stdin pipe (once; `None` if not piped).
    pub(crate) fn take_stdin(&mut self) -> Option<ChildStdin> {
        self.child.stdin.take()
    }

    /// Takes the child's stdout pipe (once; `None` if not piped).
    pub(crate) fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    /// Waits for the child to exit and returns its status and stderr tail.
    ///
    /// # Errors
    /// [`VideoError::Io`] when waiting fails.
    pub(crate) fn wait_status(&mut self) -> Result<(ExitStatus, String), VideoError> {
        let status = self.child.wait().map_err(|source| VideoError::Io { context: format!("waiting for `{}`", self.program.display()), source })?;
        self.reaped = true;
        Ok((status, self.stderr_tail()))
    }

    /// Waits for the child to exit.
    ///
    /// # Errors
    /// [`VideoError::ProcessFailed`] with the stderr tail on a non-zero exit; [`VideoError::Io`]
    /// when waiting fails.
    pub(crate) fn wait(&mut self) -> Result<(), VideoError> {
        let (status, stderr_tail) = self.wait_status()?;
        if status.success() {
            if !stderr_tail.trim().is_empty() {
                log::debug!("egui_shader_layers (video): `{}` reported: {}", self.program.display(), stderr_tail.trim_end());
            }
            Ok(())
        } else {
            Err(VideoError::ProcessFailed { program: self.program.clone(), status: status.to_string(), stderr_tail })
        }
    }

    /// Kills the child (if still running) and reaps it. Idempotent.
    pub(crate) fn kill(&mut self) {
        if self.reaped {
            return;
        }
        // Killing fails only when the child already exited; `wait` reaps it either way.
        if let Err(error) = self.child.kill() {
            log::debug!("egui_shader_layers (video): killing `{}`: {error}", self.program.display());
        }
        if let Err(error) = self.child.wait() {
            log::warn!("egui_shader_layers (video): could not reap `{}`: {error}", self.program.display());
        }
        self.reaped = true;
        // Joins the drain thread (it ends because the child's stderr closed); the tail only
        // matters for error reports, which a killed process does not produce.
        self.stderr_tail();
    }

    /// Joins the stderr thread (once) and returns the tail it collected.
    fn stderr_tail(&mut self) -> String {
        self.stderr.take().map_or_else(String::new, |thread| thread.join().unwrap_or_else(|_| "(the stderr reader panicked)".to_owned()))
    }
}

impl Drop for ToolProcess {
    fn drop(&mut self) {
        if !self.reaped {
            log::debug!("egui_shader_layers (video): stopping `{}` early", self.program.display());
        }
        self.kill();
    }
}

/// Runs `program` to completion and returns its stdout as text (stderr drained separately).
///
/// # Errors
/// As [`ToolProcess::spawn`] and [`ToolProcess::wait`]; [`VideoError::Io`] when reading fails.
pub(crate) fn run_to_string(program: &Path, args: &[OsString]) -> Result<String, VideoError> {
    let mut process = ToolProcess::spawn(program, args, false, true)?;
    let mut output = String::new();
    if let Some(mut stdout) = process.take_stdout() {
        stdout.read_to_string(&mut output).map_err(|source| VideoError::Io { context: format!("reading the output of `{}`", program.display()), source })?;
    }
    process.wait()?;
    Ok(output)
}

/// Starts the thread draining `stream` (a child's stderr); it returns the last lines read.
///
/// # Errors
/// The operating system's error when the thread cannot be created.
fn spawn_drain(mut stream: std::process::ChildStderr) -> std::io::Result<JoinHandle<String>> {
    std::thread::Builder::new().name("ffmpeg-stderr".to_owned()).spawn(move || {
        let mut tail = Vec::new();
        let mut chunk = [0_u8; 1024];
        // Read until the pipe closes (the child exited); errors end the drain too, since
        // the tail only decorates error messages.
        while let Ok(read) = stream.read(&mut chunk) {
            let Some(bytes) = chunk.get(..read).filter(|bytes| !bytes.is_empty()) else { break };
            tail.extend_from_slice(bytes);
            if tail.len() > 2 * STDERR_TAIL_BYTES {
                tail.drain(..tail.len() - STDERR_TAIL_BYTES);
            }
        }
        last_lines(&String::from_utf8_lossy(&tail), STDERR_TAIL_LINES)
    })
}

/// The last `count` lines of `text`.
fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stderr_tails_keep_the_last_lines() {
        let text = (0..30).map(|line| format!("line {line}")).collect::<Vec<_>>().join("\n");
        let tail = last_lines(&text, 3);
        assert_eq!(tail, "line 27\nline 28\nline 29");
        assert_eq!(last_lines("one", 5), "one");
        assert_eq!(last_lines("", 5), "");
    }

    #[test]
    fn a_missing_program_is_a_typed_error() {
        let program = Path::new("egui-shader-layers-no-such-program");
        assert!(matches!(ToolProcess::spawn(program, &[], false, false), Err(VideoError::ToolNotFound { .. })));
        let message = run_to_string(program, &[]).err().map(|error| error.to_string()).unwrap_or_default();
        assert!(message.contains("install FFmpeg"), "{message}");
    }
}
