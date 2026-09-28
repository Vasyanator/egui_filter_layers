/*
File: offline/video/ffmpeg/version.rs

Purpose:
Checks what the installed `ffmpeg` can do before a command line relies on it: the release (5.1
or newer, the command lines use `-fps_mode`), so an old installation fails with a clear typed
error instead of "Unrecognized option"; and whether an optional decoder is built in.

Key functions:
- `check_ffmpeg()`: runs `ffmpeg -version` and rejects releases older than 5.1.
- `parse_release()`: the pure parser of the first line (unit-tested).
- `require_decoder()`: `ffmpeg -decoders` must list a decoder (e.g. `libvpx-vp9` for VP9 alpha).

Notes:
Builds whose version is not a release number (git snapshots such as `N-113000-g...` or
`2024-03-04-git-...`) are accepted: they are newer than any release that lacks the options.
*/

use std::ffi::OsString;
use std::path::Path;

use super::super::error::VideoError;
use super::process::run_to_string;

/// Oldest supported release (major, minor).
const MINIMUM: (u32, u32) = (5, 1);

/// Runs `ffmpeg -version` and checks the release.
///
/// # Errors
/// [`VideoError::UnsupportedFfmpeg`] for releases before 5.1; the errors of running the program
/// ([`VideoError::ToolNotFound`], ...).
pub(crate) fn check_ffmpeg(program: &Path) -> Result<(), VideoError> {
    let text = run_to_string(program, &[OsString::from("-hide_banner"), OsString::from("-version")])?;
    let first_line = text.lines().next().unwrap_or_default();
    match parse_release(first_line) {
        Some(release) if release < MINIMUM => {
            let version = first_line.split_whitespace().nth(2).unwrap_or(first_line).to_owned();
            Err(VideoError::UnsupportedFfmpeg { program: program.to_owned(), version })
        }
        Some(_) => Ok(()),
        None => {
            log::debug!("egui_filter_layers (video): `{}` reports a non-release version ({first_line}); assuming it is recent", program.display());
            Ok(())
        }
    }
}

/// Checks that `program` has the decoder `decoder`, needed to `purpose`.
///
/// # Errors
/// [`VideoError::DecoderNotAvailable`] when it is missing; the errors of running the program.
pub(crate) fn require_decoder(program: &Path, decoder: &str, purpose: &'static str) -> Result<(), VideoError> {
    let text = run_to_string(program, &[OsString::from("-hide_banner"), OsString::from("-decoders")])?;
    if lists_decoder(&text, decoder) {
        Ok(())
    } else {
        Err(VideoError::DecoderNotAvailable { program: program.to_owned(), decoder: decoder.to_owned(), purpose })
    }
}

/// Whether `ffmpeg -decoders` output lists `decoder` (lines are ` FLAGS name description`).
fn lists_decoder(text: &str, decoder: &str) -> bool {
    text.lines().any(|line| line.split_whitespace().nth(1) == Some(decoder))
}

/// `(major, minor)` from `ffmpeg version 8.0.1-3ubuntu2 ...` or `ffmpeg version n5.1.2 ...`;
/// `None` when the version is not a release number.
pub(crate) fn parse_release(first_line: &str) -> Option<(u32, u32)> {
    let version = first_line.strip_prefix("ffmpeg version ")?.split_whitespace().next()?;
    let version = version.strip_prefix('n').unwrap_or(version);
    let mut parts = version.split(['.', '-', '+', '~']);
    let major = parts.next()?.parse().ok()?;
    // A git date (`2024-03-04-git`) parses as major 2024 with a missing minor: not a release.
    if major > 99 {
        return None;
    }
    let minor = parts.next().and_then(|minor| minor.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_are_parsed_and_snapshots_are_not() {
        assert_eq!(parse_release("ffmpeg version 8.0.1-3ubuntu2 Copyright (c) 2000-2025"), Some((8, 0)));
        assert_eq!(parse_release("ffmpeg version n5.1.2 Copyright"), Some((5, 1)));
        assert_eq!(parse_release("ffmpeg version 4.4.2-0ubuntu0.22.04.1"), Some((4, 4)));
        assert_eq!(parse_release("ffmpeg version 5 Copyright"), Some((5, 0)));
        assert_eq!(parse_release("ffmpeg version N-113000-g1234abcd"), None);
        assert_eq!(parse_release("ffmpeg version 2024-03-04-git-e30369bc1c-full_build"), None);
        assert_eq!(parse_release("something else"), None);
        assert!(parse_release("ffmpeg version 5.0.3").is_some_and(|release| release < MINIMUM));
        assert!(parse_release("ffmpeg version 5.1").is_some_and(|release| release >= MINIMUM));
    }

    #[test]
    fn decoder_lists_are_matched_by_name() {
        let text = "Decoders:\n V....D = Video\n ------\n V....D vp9                  Google VP9\n V..... libvpx-vp9           libvpx VP9 (codec vp9)\n";
        assert!(lists_decoder(text, "libvpx-vp9") && lists_decoder(text, "vp9"));
        assert!(!lists_decoder(text, "libvpx") && !lists_decoder(text, "VP9"));
    }
}
