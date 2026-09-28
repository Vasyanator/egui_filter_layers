//! Video processing on a real GPU (and, for the end-to-end test, the real ffmpeg programs).
//!
//! * In memory: `process_video` over an in-memory source and sink must write every frame, in
//!   order, byte-identical to `OfflineRenderer::process` of the same frame at the same time; the
//!   source's times must reach the shader as `fl.time`; cancelling from the progress callback must
//!   abort the sink.
//! * ffmpeg end to end: a synthetic clip (lavfi `testsrc` + `sine`) is inverted into a lossless
//!   FFV1 file with the audio copied; the output must have every frame, the size, an audio stream,
//!   and frames equal to the offline path's result for the decoded input frames. A translucent
//!   stream into H.264 must be rejected, and an unknown encoder must report ffmpeg's message.
//! * ffmpeg colour and files: colours survive an H.264 round trip (output tagged BT.709, inputs
//!   tagged BT.601 or untagged decoded with the right matrix); VP9 alpha survives a round trip;
//!   the output is never the input and never left half-written; cover art is skipped, the
//!   sample aspect ratio is kept, and file names with `:` or a leading `-` work.
//!
//! Needs a GPU adapter (and `ffmpeg` / `ffprobe` on `PATH` for the end-to-end test), so the tests
//! are ignored by default: `cargo test --all-features --test video -- --ignored --nocapture`.

#![cfg(all(feature = "video", not(target_arch = "wasm32")))]

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::Command;

use egui_filter_layers::offline::video::ffmpeg::{AudioMode, FfmpegSink, FfmpegSource, FfmpegTranscode, VideoCodec};
use egui_filter_layers::offline::video::{FrameRate, VideoError, VideoInfo, VideoProcessor, VideoSink, VideoSource, process_video};
use egui_filter_layers::{AlphaMode, Effect, FilterChain, FrameRef, OfflineOptions, OfflineRenderer, Pass, presets};

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

/// A headless renderer; panics with the reason when there is no GPU.
fn renderer() -> OfflineRenderer {
    OfflineRenderer::headless().unwrap_or_else(|error| panic!("no GPU for the test: {error}"))
}

fn rate(numerator: u32, denominator: u32) -> FrameRate {
    FrameRate::new(numerator, denominator).unwrap_or_else(|error| panic!("{error}"))
}

/// Frames generated in memory, at `rate`, with `alpha` everywhere.
struct MemorySource {
    info: VideoInfo,
    frames: u64,
    next: u64,
    alpha: u8,
}

impl MemorySource {
    fn new(frames: u64, alpha: u8) -> Self {
        Self { info: VideoInfo::new(WIDTH, HEIGHT, rate(30000, 1001)).with_frame_count(Some(frames)).with_alpha(alpha != 255), frames, next: 0, alpha }
    }

    /// Frame `index`: gradients shifted by the index.
    fn frame(index: u64, alpha: u8) -> Vec<u8> {
        let shift = u8::try_from(index % 256).unwrap_or(0);
        (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
            .flat_map(|(x, y)| [u8::try_from(x * 5).unwrap_or(0).wrapping_add(shift), u8::try_from(y * 7).unwrap_or(0), shift.wrapping_mul(3), alpha])
            .collect()
    }
}

impl VideoSource for MemorySource {
    fn info(&self) -> &VideoInfo {
        &self.info
    }

    fn read_frame(&mut self, pixels: &mut Vec<u8>) -> Result<Option<f64>, VideoError> {
        if self.next == self.frames {
            return Ok(None);
        }
        *pixels = Self::frame(self.next, self.alpha);
        let time = self.info.frame_rate().frame_time(self.next);
        self.next += 1;
        Ok(Some(time))
    }
}

/// Collects frames and records the calls it received.
#[derive(Default)]
struct MemorySink {
    frames: Vec<(Vec<u8>, f64)>,
    begun: bool,
    finished: bool,
    aborted: bool,
}

impl VideoSink for MemorySink {
    fn begin(&mut self, info: &VideoInfo) -> Result<(), VideoError> {
        assert_eq!((info.width(), info.height()), (WIDTH, HEIGHT));
        self.begun = true;
        Ok(())
    }

    fn write_frame(&mut self, frame: &FrameRef<'_>, time: f64) -> Result<(), VideoError> {
        assert_eq!(frame.alpha(), AlphaMode::Straight);
        assert_eq!(frame.stride(), usize::try_from(WIDTH * 4).unwrap_or(0), "the processor's frames are tightly packed");
        self.frames.push((frame.pixels().to_vec(), time));
        Ok(())
    }

    fn finish(&mut self) -> Result<(), VideoError> {
        self.finished = true;
        Ok(())
    }

    fn abort(&mut self) {
        self.aborted = true;
    }
}

/// A custom effect writing `fract(fl.time)` into red and the frame's own green/blue/alpha.
fn time_probe() -> FilterChain {
    let effect = Effect::new("time probe", "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let c = src(uv);\n    return vec4<f32>(fract(fl.time) * c.a, c.g, c.b, c.a);\n}\n").unwrap_or_else(|error| panic!("{error}"));
    Pass::new(&effect).into()
}

#[test]
#[ignore = "needs a GPU adapter"]
fn process_video_matches_the_offline_path_frame_by_frame() {
    let chain = presets::gaussian_blur(2.0).then(presets::invert(1.0)).then(presets::ripple(2.0, 12.0, 0.7));
    let options = OfflineOptions::default().with_pixels_per_point(1.5);
    let mut processor = VideoProcessor::new(renderer(), chain.clone(), options.clone());
    let mut source = MemorySource::new(7, 255);
    let mut sink = MemorySink::default();
    let mut reports = Vec::new();
    let written = process_video(&mut processor, &mut source, &mut sink, |progress| {
        reports.push((progress.frames, progress.fraction()));
        ControlFlow::Continue(())
    })
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(written, 7);
    assert!(sink.begun && sink.finished && !sink.aborted);
    assert_eq!(reports.last().copied(), Some((7, Some(1.0))));

    let mut reference = renderer();
    let ntsc = rate(30000, 1001);
    for (index, (pixels, time)) in sink.frames.iter().enumerate() {
        let index = u64::try_from(index).unwrap_or(0);
        assert!((time - ntsc.frame_time(index)).abs() < 1e-12, "frame {index} time");
        let input = MemorySource::frame(index, 255);
        #[allow(clippy::cast_possible_truncation)] // exact for these small times
        let expected = reference.process(&FrameRef::new(&input, WIDTH, HEIGHT, AlphaMode::Straight), &chain, &options.clone().with_time(*time as f32)).unwrap_or_else(|error| panic!("{error}"));
        assert!(expected.pixels() == pixels.as_slice(), "frame {index} differs from the offline path");
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn frame_times_reach_the_shader_and_cancelling_aborts() {
    let mut processor = VideoProcessor::new(renderer(), time_probe(), OfflineOptions::default());
    let mut source = MemorySource::new(40, 255);
    let mut sink = MemorySink::default();
    process_video(&mut processor, &mut source, &mut sink, |_| ControlFlow::Continue(())).unwrap_or_else(|error| panic!("{error}"));
    let ntsc = rate(30000, 1001);
    for (index, (pixels, _)) in sink.frames.iter().enumerate() {
        let time = ntsc.frame_time(u64::try_from(index).unwrap_or(0));
        let expected = time.fract() * 255.0;
        let red = f64::from(pixels[0]);
        assert!((red - expected).abs() <= 1.0, "frame {index}: red {red}, fract(time) * 255 = {expected}");
    }

    // Direct per-frame API: the same time gives the same pixels.
    let frame = MemorySource::frame(3, 255);
    let first = processor.process_frame(&FrameRef::new(&frame, WIDTH, HEIGHT, AlphaMode::Straight), 0.25).map(|out| out.pixels().to_vec()).unwrap_or_else(|error| panic!("{error}"));
    let mut into = Vec::new();
    processor.process_frame_into(&FrameRef::new(&frame, WIDTH, HEIGHT, AlphaMode::Straight), 0.25, &mut into).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(first, into);
    assert_eq!(first[0], 64, "fract(0.25) * 255 rounds to 64");

    let mut source = MemorySource::new(40, 255);
    let mut sink = MemorySink::default();
    let result = process_video(&mut processor, &mut source, &mut sink, |progress| if progress.frames == 3 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) });
    assert!(matches!(result, Err(VideoError::Cancelled { frames: 3 })), "{result:?}");
    assert!(sink.aborted && !sink.finished);
    assert_eq!(sink.frames.len(), 3);
}

// ---- ffmpeg end to end ----

/// A fresh scratch directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("egui_filter_layers_video_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("{error}"));
    dir
}

/// Runs a command and panics with its stderr on failure.
fn run(command: &mut Command) -> String {
    let output = command.output().unwrap_or_else(|error| panic!("cannot run {command:?}: {error} (is ffmpeg installed?)"));
    assert!(output.status.success(), "{command:?} failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Stream types of `path` in order, e.g. `["video", "audio"]`.
fn stream_types(path: &Path) -> Vec<String> {
    run(Command::new("ffprobe").args(["-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0"]).arg(path)).lines().map(str::to_owned).collect()
}

/// The frames stored in `path` (no frame-rate conversion), as straight RGBA8.
fn stored_frames(path: &Path, width: usize, height: usize) -> Vec<Vec<u8>> {
    let output = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]).output().unwrap_or_else(|error| panic!("{error}"));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    output.stdout.chunks(width * height * 4).map(<[u8]>::to_vec).collect()
}

/// Every frame of `path`, decoded through the crate's own source.
fn decode_all(path: &Path) -> (VideoInfo, Vec<(Vec<u8>, f64)>) {
    let mut source = FfmpegSource::open(path).unwrap_or_else(|error| panic!("{error}"));
    let mut frames = Vec::new();
    let mut pixels = Vec::new();
    while let Some(time) = source.read_frame(&mut pixels).unwrap_or_else(|error| panic!("{error}")) {
        frames.push((pixels.clone(), time));
    }
    (source.info().clone(), frames)
}

#[test]
#[ignore = "needs a GPU adapter and ffmpeg / ffprobe on PATH"]
fn ffmpeg_round_trip_is_exact_with_a_lossless_codec() {
    let dir = scratch("roundtrip");
    let input = dir.join("input.mp4");
    // 20 frames at 25 fps, 64x48, with a 440 Hz tone: a lossy H.264 + AAC input.
    run(Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.8", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.8"]).args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]).arg(&input));

    let chain: FilterChain = presets::invert(1.0).into();
    let options = OfflineOptions::default();
    let mut processor = VideoProcessor::new(renderer(), chain.clone(), options.clone());
    let output = dir.join("output.mkv");
    let written = FfmpegTranscode::new(&input, &output).with_codec(VideoCodec::Ffv1).with_audio(AudioMode::Copy).run(&mut processor, |_| ControlFlow::Continue(())).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(written, 20);

    assert_eq!(stream_types(&output), ["video", "audio"], "the audio is passed through");
    let (in_info, in_frames) = decode_all(&input);
    assert_eq!(in_info.frame_rate().to_string(), "25/1");
    assert_eq!(in_frames.len(), 20);
    // The crate's source pads to the container start: the copied AAC's encoder delay moves the
    // video start in the output, so compare the frames actually stored.
    let (out_info, _) = decode_all(&output);
    assert_eq!((out_info.width(), out_info.height(), out_info.frame_rate()), (64, 48, in_info.frame_rate()));
    let out_frames = stored_frames(&output, 64, 48);
    assert_eq!(out_frames.len(), 20, "every frame is written");
    let mut reference = processor.into_renderer();
    for (index, ((input_pixels, time), output_pixels)) in in_frames.iter().zip(&out_frames).enumerate() {
        assert!((time - 0.04 * f64::from(u32::try_from(index).unwrap_or(0))).abs() < 1e-9, "frame {index} time");
        #[allow(clippy::cast_possible_truncation)] // exact for these small times
        let expected = reference.process(&FrameRef::new(input_pixels, 64, 48, AlphaMode::Straight), &chain, &options.clone().with_time(*time as f32)).unwrap_or_else(|error| panic!("{error}"));
        assert!(expected.pixels() == output_pixels.as_slice(), "decoded output frame {index} differs from the offline path");
    }

    // Dropping the audio.
    let silent = dir.join("silent.mkv");
    let mut processor = VideoProcessor::new(reference, chain, options);
    FfmpegTranscode::new(&input, &silent).with_codec(VideoCodec::Ffv1).with_audio(AudioMode::Drop).run(&mut processor, |_| ControlFlow::Continue(())).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stream_types(&silent), ["video"]);
    let (_, silent_frames) = decode_all(&silent);
    assert_eq!(silent_frames.len(), 20, "without audio nothing shifts the video start");
    assert!(silent_frames.iter().zip(&out_frames).all(|((silent, _), stored)| silent == stored));
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}

#[test]
#[ignore = "needs a GPU adapter and ffmpeg / ffprobe on PATH"]
fn ffmpeg_errors_are_typed() {
    let dir = scratch("errors");
    let mut processor = VideoProcessor::new(renderer(), presets::invert(1.0).into(), OfflineOptions::default());

    // Translucent frames into a codec without alpha: rejected, never flattened.
    let mut sink = FfmpegSink::new(dir.join("alpha.mp4"), VideoCodec::H264 { crf: 18 });
    let result = process_video(&mut processor, &mut MemorySource::new(3, 128), &mut sink, |_| ControlFlow::Continue(()));
    assert!(matches!(result, Err(VideoError::AlphaNotSupported { frame: 0, .. })), "{result:?}");
    // ... unless a background makes them opaque.
    processor.set_options(OfflineOptions::default().with_background(Some(egui::Color32::BLACK)));
    let mut sink = FfmpegSink::new(dir.join("alpha.mp4"), VideoCodec::H264 { crf: 18 });
    assert_eq!(process_video(&mut processor, &mut MemorySource::new(3, 128), &mut sink, |_| ControlFlow::Continue(())).ok(), Some(3));
    // ... or the codec keeps alpha.
    processor.set_options(OfflineOptions::default());
    let mut sink = FfmpegSink::new(dir.join("alpha.mov"), VideoCodec::ProRes4444);
    assert_eq!(process_video(&mut processor, &mut MemorySource::new(3, 128), &mut sink, |_| ControlFlow::Continue(())).ok(), Some(3));

    // An encoder ffmpeg does not know: its own message comes back.
    let codec = VideoCodec::Custom { args: vec!["-c:v".to_owned(), "no_such_encoder".to_owned()], alpha: true, yuv: false };
    let mut sink = FfmpegSink::new(dir.join("bad.mkv"), codec);
    let result = process_video(&mut processor, &mut MemorySource::new(30, 255), &mut sink, |_| ControlFlow::Continue(()));
    let message = result.as_ref().err().map(ToString::to_string).unwrap_or_default();
    assert!(matches!(result, Err(VideoError::BrokenPipe { .. } | VideoError::ProcessFailed { .. })), "{result:?}");
    assert!(message.contains("no_such_encoder"), "{message}");

    assert_eq!(partial_files(&dir), Vec::<String>::new(), "a failed encode leaves no partial file");

    // Odd sizes for 4:2:0 codecs are refused before starting ffmpeg.
    let mut sink = FfmpegSink::new(dir.join("odd.mp4"), VideoCodec::default());
    assert!(matches!(sink.begin(&VideoInfo::new(5, 4, rate(25, 1))), Err(VideoError::UnsupportedSize { .. })));

    // Unreadable input.
    assert!(matches!(FfmpegSource::open(dir.join("missing.mp4")), Err(VideoError::ProcessFailed { .. })));
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}

/// Names of the encoders' temporary files left in `dir`.
fn partial_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|entries| entries.filter_map(Result::ok).map(|entry| entry.file_name().to_string_lossy().into_owned()).filter(|name| name.contains(".partial-")).collect())
        .unwrap_or_default()
}

/// Colours of the colour tests: red, green, blue, grey (straight RGBA, opaque).
const SWATCHES: [[u8; 4]; 4] = [[220, 40, 40, 255], [40, 200, 60, 255], [40, 60, 220, 255], [128, 128, 128, 255]];

/// A `width x height` frame of four horizontal bands, one per swatch.
fn swatch_frame(width: u32, height: u32, alpha: u8) -> Vec<u8> {
    let band = height / 4;
    (0..height).flat_map(|y| (0..width).map(move |_| y)).flat_map(|y| {
        let [r, g, b, _] = SWATCHES[usize::try_from((y / band).min(3)).unwrap_or(3)];
        [r, g, b, alpha]
    }).collect()
}

/// Writes `frames` swatch frames through an [`FfmpegSink`] (no GPU needed).
fn encode_swatches(path: &Path, codec: VideoCodec, width: u32, height: u32, alpha: u8, frames: u32) {
    let mut sink = FfmpegSink::new(path, codec);
    sink.begin(&VideoInfo::new(width, height, rate(25, 1))).unwrap_or_else(|error| panic!("{error}"));
    let pixels = swatch_frame(width, height, alpha);
    for index in 0..frames {
        sink.write_frame(&FrameRef::new(&pixels, width, height, AlphaMode::Straight), f64::from(index) / 25.0).unwrap_or_else(|error| panic!("{error}"));
    }
    sink.finish().unwrap_or_else(|error| panic!("{error}"));
}

/// Asserts that the band centres of `pixels` match the swatches within `tolerance` per channel
/// (alpha compared with `alpha`).
fn assert_swatches(pixels: &[u8], width: u32, height: u32, alpha: u8, tolerance: u8, what: &str) {
    let band = height / 4;
    for (index, swatch) in SWATCHES.iter().enumerate() {
        let y = band * u32::try_from(index).unwrap_or(0) + band / 2;
        let offset = usize::try_from((y * width + width / 2) * 4).unwrap_or(0);
        let got = &pixels[offset..offset + 4];
        let expected = [swatch[0], swatch[1], swatch[2], alpha];
        assert!(got.iter().zip(expected).all(|(&got, expected)| got.abs_diff(expected) <= tolerance), "{what}: band {index} is {got:?}, expected {expected:?} (±{tolerance})");
    }
}

/// `color_range,color_space,color_transfer,color_primaries` of the first video stream.
fn color_tags(path: &Path) -> String {
    run(Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_range,color_space,color_transfer,color_primaries", "-of", "csv=p=0"]).arg(path)).trim().to_owned()
}

#[test]
#[ignore = "needs ffmpeg / ffprobe on PATH"]
fn ffmpeg_colours_survive_a_yuv_round_trip() {
    let dir = scratch("colour");
    let (width, height) = (64, 64);
    let output = dir.join("swatches.mp4");
    encode_swatches(&output, VideoCodec::H264 { crf: 0 }, width, height, 255, 5);
    assert_eq!(color_tags(&output), "tv,bt709,bt709,bt709", "the output is tagged BT.709, limited range");
    let (info, frames) = decode_all(&output);
    assert_eq!((info.width(), info.height(), frames.len()), (width, height, 5));
    assert_swatches(&frames[0].0, width, height, 255, 3, "H.264 round trip");

    // Other encodings of the same colours: tagged BT.601, and untagged (SD -> BT.601, HD ->
    // BT.709). The tags are stripped from the bitstream afterwards: with ffmpeg 8, frames tagged
    // "unknown" in the filter chain would also make `scale` fall back to BT.601 when encoding.
    let raw = dir.join("swatches.rgba");
    for (name, (w, h), matrix, strip) in [("bt601.mp4", (64, 64), "smpte170m", false), ("untagged_sd.mp4", (64, 64), "smpte170m", true), ("untagged_hd.mp4", (1280, 720), "bt709", true)] {
        assert!(std::fs::write(&raw, swatch_frame(w, h, 255)).is_ok());
        let tagged = dir.join(format!("tagged_{name}"));
        let scale_matrix = if matrix == "bt709" { "bt709" } else { "bt601" };
        let filter = format!("scale=out_color_matrix={scale_matrix}:out_range=tv,setparams=colorspace={matrix}:range=tv");
        run(Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{w}x{h}"), "-framerate", "25", "-i"]).arg(&raw).args(["-vf", &filter, "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p"]).arg(&tagged));
        let input = if strip {
            let stripped = dir.join(name);
            run(Command::new("ffmpeg").args(["-v", "error", "-y", "-i"]).arg(&tagged).args(["-c", "copy", "-bsf:v", "h264_metadata=matrix_coefficients=2:colour_primaries=2:transfer_characteristics=2"]).arg(&stripped));
            assert!(color_tags(&stripped).contains("unknown,unknown,unknown"), "{name} is untagged");
            stripped
        } else {
            tagged
        };
        let (_, frames) = decode_all(&input);
        assert_swatches(&frames[0].0, w, h, 255, 3, name);
    }
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}

#[test]
#[ignore = "needs ffmpeg / ffprobe on PATH"]
fn ffmpeg_vp9_alpha_survives_a_round_trip() {
    let dir = scratch("vp9_alpha");
    let output = dir.join("alpha.webm");
    encode_swatches(&output, VideoCodec::Vp9 { crf: 10 }, 64, 64, 128, 5);
    assert_eq!(color_tags(&output), "tv,bt709,bt709,bt709");
    let (info, frames) = decode_all(&output);
    assert!(info.has_alpha(), "the alpha side channel is detected");
    assert_eq!(frames.len(), 5);
    assert_swatches(&frames[0].0, 64, 64, 128, 4, "VP9 with alpha");
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}

#[test]
#[ignore = "needs a GPU adapter and ffmpeg / ffprobe on PATH"]
fn ffmpeg_outputs_never_overwrite_the_input_or_stay_half_written() {
    let dir = scratch("files");
    let input = dir.join("input.mkv");
    encode_swatches(&input, VideoCodec::Ffv1, 16, 16, 255, 3);
    let original = std::fs::read(&input).unwrap_or_else(|error| panic!("{error}"));
    let mut processor = VideoProcessor::new(renderer(), presets::invert(1.0).into(), OfflineOptions::default());
    let same = [input.clone(), dir.join(".").join("input.mkv")];
    for output in &same {
        let result = FfmpegTranscode::new(&input, output).with_codec(VideoCodec::Ffv1).run(&mut processor, |_| ControlFlow::Continue(()));
        assert!(matches!(result, Err(VideoError::OutputIsInput { .. })), "{result:?}");
    }
    #[cfg(unix)]
    {
        let link = dir.join("link.mkv");
        assert!(std::os::unix::fs::symlink(&input, &link).is_ok());
        let result = FfmpegTranscode::new(&input, &link).with_codec(VideoCodec::Ffv1).run(&mut processor, |_| ControlFlow::Continue(()));
        assert!(matches!(result, Err(VideoError::OutputIsInput { .. })), "{result:?}");
    }
    // The sink alone: audio taken from its own output.
    let mut sink = FfmpegSink::new(&input, VideoCodec::Ffv1).with_audio_from(&input, AudioMode::Copy);
    assert!(matches!(sink.begin(&VideoInfo::new(16, 16, rate(25, 1))), Err(VideoError::OutputIsInput { .. })));
    assert_eq!(std::fs::read(&input).ok(), Some(original), "the input is untouched");

    // An aborted encode leaves an existing output as it was, and no partial file.
    let output = dir.join("existing.mkv");
    assert!(std::fs::write(&output, b"previous").is_ok());
    let mut sink = FfmpegSink::new(&output, VideoCodec::Ffv1);
    sink.begin(&VideoInfo::new(16, 16, rate(25, 1))).unwrap_or_else(|error| panic!("{error}"));
    let pixels = swatch_frame(16, 16, 255);
    sink.write_frame(&FrameRef::new(&pixels, 16, 16, AlphaMode::Straight), 0.0).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(partial_files(&dir).len(), 1, "the encoder writes a temporary file");
    sink.abort();
    assert_eq!(std::fs::read(&output).ok().as_deref(), Some(&b"previous"[..]));
    assert_eq!(partial_files(&dir), Vec::<String>::new());
    // A finished one replaces it.
    encode_swatches(&output, VideoCodec::Ffv1, 16, 16, 255, 2);
    assert_eq!(stored_frames(&output, 16, 16).len(), 2);
    assert_eq!(partial_files(&dir), Vec::<String>::new());
    // Non-straight frames are a typed error.
    let mut sink = FfmpegSink::new(dir.join("premultiplied.mkv"), VideoCodec::Ffv1);
    sink.begin(&VideoInfo::new(16, 16, rate(25, 1))).unwrap_or_else(|error| panic!("{error}"));
    let result = sink.write_frame(&FrameRef::new(&pixels, 16, 16, AlphaMode::Premultiplied), 0.0);
    assert!(matches!(result, Err(VideoError::NotStraightAlpha { alpha: AlphaMode::Premultiplied })), "{result:?}");
    sink.abort();
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}

#[test]
#[ignore = "needs a GPU adapter and ffmpeg / ffprobe on PATH"]
fn ffmpeg_skips_cover_art_keeps_the_aspect_and_takes_any_file_name() {
    let dir = scratch("streams");
    // Cover art (an attached picture) first, then a 4:3-anamorphic video, in a file whose name
    // starts with `-` and contains `:`.
    let cover = dir.join("cover.png");
    run(Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:size=16x16", "-frames:v", "1"]).arg(&cover));
    let input = dir.join("-in:put.mp4");
    let mut input_arg = std::ffi::OsString::from("file:");
    input_arg.push(&input);
    run(Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&cover)
        .args(["-f", "lavfi", "-i", "testsrc=size=32x24:rate=25:duration=0.4", "-map", "0", "-map", "1", "-c:v:0", "mjpeg", "-disposition:v:0", "attached_pic", "-c:v:1", "libx264", "-pix_fmt", "yuv420p", "-filter:v:1", "setsar=4/3"])
        .arg(&input_arg));
    let (info, frames) = decode_all(&input);
    assert_eq!((info.width(), info.height(), info.sample_aspect_ratio()), (32, 24, (4, 3)), "the real video stream, not the cover");
    assert_eq!(frames.len(), 10);

    let output = dir.join("-out:put.mkv");
    let mut processor = VideoProcessor::new(renderer(), presets::invert(1.0).into(), OfflineOptions::default());
    let written = FfmpegTranscode::new(&input, &output).with_codec(VideoCodec::Ffv1).run(&mut processor, |_| ControlFlow::Continue(())).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(written, 10);
    let mut output_arg = std::ffi::OsString::from("file:");
    output_arg.push(&output);
    let sar = run(Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=sample_aspect_ratio", "-of", "csv=p=0"]).arg(&output_arg));
    assert_eq!(sar.trim(), "4:3", "the sample aspect ratio is kept");
    assert!(std::fs::remove_dir_all(&dir).is_ok());
}
