/*
File: offline/video/ffmpeg/paths.rs

Purpose:
File paths as the ffmpeg adapters hand them to the programs and manage them on disk: the `file:`
protocol prefix (so no path is read as an option, a URL or stdin/stdout), same-file detection
(an output must never be its input), and the reserved temporary file an encode writes before it
is renamed over the output.

Key functions:
- `url()`: `file:` + path, as an `OsString`.
- `same_file()`: whether two paths name one existing file.
- `TemporaryFile::reserve()`: creates a unique empty file beside the output, keeping its
  extension; `persist()` renames it over the output, dropping it deletes it.

Notes:
Hard links are detected on Unix (device + inode); on Windows only through `canonicalize`.
*/

use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use super::super::error::VideoError;

/// Attempts at a free temporary name before giving up.
const MAX_TEMPORARY_ATTEMPTS: u32 = 1000;

/// `path` as an ffmpeg URL of the `file` protocol: `a:b.mp4`, `-` or `pipe:x` stay local paths.
pub(crate) fn url(path: &Path) -> OsString {
    let mut url = OsString::from("file:");
    url.push(path.as_os_str());
    url
}

/// Whether `a` and `b` name the same existing file (through symbolic links, and hard links on
/// Unix). A path that does not exist is never the same file.
pub(crate) fn same_file(a: &Path, b: &Path) -> bool {
    let (Ok(canonical_a), Ok(canonical_b)) = (a.canonicalize(), b.canonicalize()) else {
        return false;
    };
    if canonical_a == canonical_b {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(meta_a), Ok(meta_b)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return meta_a.dev() == meta_b.dev() && meta_a.ino() == meta_b.ino();
        }
    }
    false
}

/// A file an encoder writes before it becomes the output: renamed over the output by
/// [`TemporaryFile::persist`], deleted when dropped otherwise.
#[derive(Debug)]
pub(crate) struct TemporaryFile {
    /// The file (`None` once persisted).
    path: Option<PathBuf>,
}

impl TemporaryFile {
    /// Reserves a temporary file beside `output` (see [`reserve_temporary`]).
    ///
    /// # Errors
    /// As [`reserve_temporary`].
    pub(crate) fn reserve(output: &Path) -> Result<Self, VideoError> {
        reserve_temporary(output).map(|path| Self { path: Some(path) })
    }

    /// The temporary path (empty once persisted).
    pub(crate) fn path(&self) -> &Path {
        self.path.as_deref().unwrap_or_else(|| Path::new(""))
    }

    /// Atomically renames the file to `output`, replacing an existing file (same directory, so
    /// the same filesystem).
    ///
    /// # Errors
    /// [`VideoError::Io`] when the rename fails; the temporary file is then deleted on drop.
    pub(crate) fn persist(mut self, output: &Path) -> Result<(), VideoError> {
        let Some(path) = self.path.take() else { return Ok(()) };
        std::fs::rename(&path, output).map_err(|source| {
            // Keep the path so `Drop` deletes the file.
            let context = format!("moving the finished `{}` to `{}`", path.display(), output.display());
            self.path = Some(path);
            VideoError::Io { context, source }
        })
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if let Some(path) = self.path.take()
            && let Err(error) = std::fs::remove_file(&path)
            && error.kind() != ErrorKind::NotFound
        {
            log::warn!("egui_filter_layers (video): could not delete the partial file `{}`: {error}", path.display());
        }
    }
}

/// Creates a new empty file beside `output`, named `<stem>.partial-<pid>-<n>.<extension>` (the
/// extension kept so ffmpeg picks the same container), and returns its path.
///
/// # Errors
/// [`VideoError::Io`] when the directory is not writable or no free name was found.
fn reserve_temporary(output: &Path) -> Result<PathBuf, VideoError> {
    let stem = output.file_stem().map_or_else(|| OsString::from("output"), ToOwned::to_owned);
    let extension = output.extension().map(ToOwned::to_owned);
    let mut last_error = None;
    for attempt in 0..MAX_TEMPORARY_ATTEMPTS {
        let mut name = stem.clone();
        name.push(format!(".partial-{}-{attempt}", std::process::id()));
        if let Some(extension) = &extension {
            name.push(".");
            name.push(extension);
        }
        let candidate = output.with_file_name(name);
        // `create_new` reserves the name atomically; ffmpeg then overwrites the empty file.
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(_) => return Ok(candidate),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => last_error = Some(error),
            Err(source) => return Err(VideoError::Io { context: format!("creating a temporary file beside `{}`", output.display()), source }),
        }
    }
    let source = last_error.unwrap_or_else(|| std::io::Error::from(ErrorKind::AlreadyExists));
    Err(VideoError::Io { context: format!("finding a free temporary name beside `{}`", output.display()), source })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_use_the_file_protocol() {
        assert_eq!(url(Path::new("a:b.mp4")), OsString::from("file:a:b.mp4"));
        assert_eq!(url(Path::new("-")), OsString::from("file:-"));
    }

    #[test]
    fn temporary_files_keep_the_extension_and_detect_same_files() {
        let dir = std::env::temp_dir().join(format!("egui_filter_layers_paths_{}", std::process::id()));
        assert!(std::fs::create_dir_all(&dir).is_ok());
        let output = dir.join("clip.final.mp4");
        let first = reserve_temporary(&output).unwrap_or_else(|error| panic!("{error}"));
        let second = reserve_temporary(&output).unwrap_or_else(|error| panic!("{error}"));
        assert_ne!(first, second);
        assert_eq!(first.extension().and_then(|ext| ext.to_str()), Some("mp4"));
        assert_eq!(first.parent(), Some(dir.as_path()));
        assert!(same_file(&first, &dir.join(".").join(first.file_name().unwrap_or_default())));
        assert!(!same_file(&first, &second));
        assert!(!same_file(&output, &output), "a missing file is not the same file");
        #[cfg(unix)]
        {
            let link = dir.join("hard.mp4");
            assert!(std::fs::hard_link(&first, &link).is_ok());
            assert!(same_file(&first, &link));
        }
        let kept = TemporaryFile::reserve(&output).unwrap_or_else(|error| panic!("{error}"));
        let dropped = TemporaryFile::reserve(&output).unwrap_or_else(|error| panic!("{error}"));
        let dropped_path = dropped.path().to_owned();
        drop(dropped);
        assert!(!dropped_path.exists(), "a dropped temporary file is deleted");
        assert!(std::fs::write(kept.path(), b"done").is_ok());
        assert!(kept.persist(&output).is_ok());
        assert_eq!(std::fs::read(&output).ok().as_deref(), Some(&b"done"[..]));
        assert!(std::fs::remove_dir_all(&dir).is_ok());
    }
}
