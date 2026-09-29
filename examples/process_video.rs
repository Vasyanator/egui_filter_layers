/*
File: examples/process_video.rs

Purpose:
Offline processing of a video file with the `video` feature: decode it with the `ffmpeg`
program, filter every frame on a headless wgpu device (an animated ripple over a vignette, so
`fl.time` follows the video's timeline), encode it, and pass the audio through.

Usage:
    cargo run --example process_video --features video -- INPUT OUTPUT
        [--codec h264|vp9|av1|prores|ffv1|gif] [--crf N] [--audio copy|drop|opus]
        [--background RRGGBB] [--ppp F]

Requires `ffmpeg` and `ffprobe` (5.1 or newer) on PATH. The default codec is H.264 (crf 18);
WebM output needs `--codec vp9 --audio opus`. Codecs without alpha (H.264, AV1, GIF) reject
translucent frames unless `--background` gives a colour to draw them over.
*/

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::ExitCode;

use egui::Color32;
use egui_shader_layers::offline::video::VideoProcessor;
use egui_shader_layers::offline::video::ffmpeg::{AudioMode, FfmpegTranscode, VideoCodec};
use egui_shader_layers::{FilterChain, OfflineOptions, OfflineRenderer, presets};

/// Parsed command line.
struct Args {
    /// Video to read.
    input: PathBuf,
    /// Video to write (overwritten).
    output: PathBuf,
    /// Output codec.
    codec: VideoCodec,
    /// Audio treatment.
    audio: AudioMode,
    /// Run settings.
    options: OfflineOptions,
}

/// Parses `std::env::args`; `Err` carries the message to print.
fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let (mut codec_name, mut crf, mut audio) = ("h264".to_owned(), None::<u8>, AudioMode::Copy);
    let mut options = OfflineOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} expects a value"));
        match arg.as_str() {
            "--codec" => codec_name = value("--codec")?,
            "--crf" => crf = Some(value("--crf")?.parse().map_err(|error| format!("--crf: {error}"))?),
            "--audio" => {
                audio = match value("--audio")?.as_str() {
                    "copy" => AudioMode::Copy,
                    "drop" => AudioMode::Drop,
                    "opus" => AudioMode::Encode(vec!["-c:a".to_owned(), "libopus".to_owned(), "-b:a".to_owned(), "128k".to_owned()]),
                    other => return Err(format!("--audio: unknown mode {other} (copy, drop, opus)")),
                };
            }
            "--background" => {
                let hex = value("--background")?;
                let rgb = u32::from_str_radix(&hex, 16).ok().filter(|_| hex.len() == 6).ok_or_else(|| format!("--background: expected RRGGBB, got {hex}"))?;
                let [_, r, g, b] = rgb.to_be_bytes();
                options = options.with_background(Some(Color32::from_rgb(r, g, b)));
            }
            "--ppp" => options = options.with_pixels_per_point(value("--ppp")?.parse().map_err(|error| format!("--ppp: {error}"))?),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let codec = match codec_name.as_str() {
        "h264" => VideoCodec::H264 { crf: crf.unwrap_or(18) },
        "vp9" => VideoCodec::Vp9 { crf: crf.unwrap_or(31) },
        "av1" => VideoCodec::Av1 { crf: crf.unwrap_or(30) },
        "prores" => VideoCodec::ProRes4444,
        "ffv1" => VideoCodec::Ffv1,
        "gif" => VideoCodec::Gif,
        other => return Err(format!("--codec: unknown codec {other}")),
    };
    let [input, output]: [PathBuf; 2] = positional.try_into().map_err(|_| "expected INPUT and OUTPUT paths".to_owned())?;
    Ok(Args { input, output, codec, audio, options })
}

/// The filter stack: a vignette and a slow ripple (animated through `fl.time`).
fn stack() -> FilterChain {
    FilterChain::from(presets::vignette(0.5, 0.35, 0.5)).then(presets::ripple(3.0, 48.0, 0.25))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\nusage: process_video INPUT OUTPUT [--codec h264|vp9|av1|prores|ffv1|gif] [--crf N] [--audio copy|drop|opus] [--background RRGGBB] [--ppp F]");
            return ExitCode::FAILURE;
        }
    };
    let renderer = match OfflineRenderer::headless() {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("cannot create a GPU renderer: {error}\nOffline processing needs a wgpu adapter (Vulkan, Metal, DX12 or GL).");
            return ExitCode::FAILURE;
        }
    };
    let mut processor = VideoProcessor::new(renderer, stack(), args.options);
    let job = FfmpegTranscode::new(&args.input, &args.output).with_codec(args.codec).with_audio(args.audio);
    let result = job.run(&mut processor, |progress| {
        match progress.fraction() {
            Some(fraction) => eprint!("\rframe {} ({:.0}%)", progress.frames, fraction * 100.0),
            None => eprint!("\rframe {}", progress.frames),
        }
        ControlFlow::Continue(())
    });
    eprintln!();
    match result {
        Ok(frames) => {
            println!("wrote {} ({frames} frames)", args.output.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("processing {} failed: {error}", args.input.display());
            ExitCode::FAILURE
        }
    }
}
