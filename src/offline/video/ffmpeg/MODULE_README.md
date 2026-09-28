# Module: src/offline/video/ffmpeg

## Purpose
Native adapters to the `ffmpeg` / `ffprobe` command-line programs (5.1+): a `VideoSource` that
decodes a file, a `VideoSink` that encodes one, codec presets, and `FfmpegTranscode` for the
whole file-to-file job. Only `std::process`; nothing of ffmpeg is linked, so codec availability
and licensing are those of the user's ffmpeg build.

## Architecture
```text
FfmpegSource::open_with
   version.rs: ffmpeg -version (>= 5.1), ffmpeg -decoders (libvpx for VP8/VP9 alpha)
   probe.rs:   ffprobe -show_entries ... -show_pixel_formats -of flat  -> Probe
               (first non-attached-picture video stream: index, codec, rotation-swapped size,
                SAR, avg / plausible r frame rate, duration, colour tags, alpha incl. alpha_mode,
                RGB coding, audio presence)
   color.rs:   decode_color -> scale=in_color_matrix=M:in_range=R,format=rgba (or reject)
   args.rs:    decode_args -> ffmpeg [-c:v libvpx-vp9] -i file:IN -map 0:<index> -fps_mode cfr
               -r N/D -filter:v <colour> -f rawvideo -pix_fmt rgba pipe:1
   process.rs: ToolProcess (stdout piped, stderr drained on a named thread)
   read_frame: exactly width*height*4 bytes; time = FrameRate::frame_time(i); errors latch
FfmpegSink
   begin:  size / same-file / version checks, paths.rs TemporaryFile beside the output,
           encode_args -> ffmpeg -f rawvideo ... -i pipe:0 [-i file:AUDIO] -map 0:v:0
           -filter:v setsar,<preset/custom filter>,<colour> <tags> <codec args>
           [-map 1:a? <audio args> -shortest | -an] file:TEMP
   write_frame: size, straight-alpha and opacity checks, write to stdin; BrokenPipe -> stderr
   finish: close stdin, wait, rename TEMP over the output   abort / drop / error: kill, delete TEMP
```

## Files and submodules
- `mod.rs`: `FfmpegTools` (program locations), `FfmpegTranscode` (refuses output == input),
  re-exports; module docs (installation, licensing, CFR, colour, output files).
- `probe.rs`: `probe()` and the pure `parse_flat()` parser (tested on captured output).
- `color.rs`: decode matrix / range choice and rejection, output colour filters and tags.
- `args.rs`: `VideoCodec`, `AudioMode`, `decode_args()`, `encode_args()` (pure, tested).
- `paths.rs`: `file:` URLs, `same_file()`, `TemporaryFile` (reserve, persist, delete on drop).
- `version.rs`: ffmpeg release check and decoder lookup.
- `process.rs`: `ToolProcess` (spawn with typed errors, stderr tail thread, kill-on-drop,
  `CREATE_NO_WINDOW` on Windows), `run_to_string()`.
- `source.rs`: `FfmpegSource`. `sink.rs`: `FfmpegSink`.

## Contracts and invariants
- Export = preview colours: decoding honours the input's matrix / range tags (untagged: BT.709
  if HD, BT.601 if SD); HDR transfers and wide-gamut primaries are `UnsupportedColor`. YCbCr
  outputs are converted with an explicit `scale=out_color_matrix=bt709:out_range=tv` and tagged
  on the frames (`setparams`, which ffmpeg 8 needs: frame properties override `-color_*`) and on
  the output; RGB outputs are tagged sRGB full range; GIF untagged. Exactly one `-filter:v`.
- Every path is a `file:` URL; no file name is parsed as an option, protocol or pipe.
- The output path only ever receives a complete file (temporary file + rename in `finish`); an
  output that is the input (canonical path, or inode on Unix) is `OutputIsInput`.
- No deadlocks: stderr is always drained on its own thread; the source's stdin and the sink's
  stdout are null; a sink's `Running` drops its process (kill) before its stdin and temp file.
- No leaked children: every `ToolProcess` is waited for or killed + reaped (`Drop`), also when
  its stderr thread cannot be created.
- Error mapping: spawn `NotFound` -> `ToolNotFound`; release < 5.1 -> `UnsupportedFfmpeg`;
  non-zero exit -> `ProcessFailed` with the last stderr lines; no usable video stream ->
  `NoVideoStream` / `MalformedProbe`; missing libvpx for alpha -> `DecoderNotAvailable`; a pipe
  ending mid-frame -> `FrameBytesMismatch`; the encoder closing stdin -> `BrokenPipe`.
- Timing: decode is constant-frame-rate at the probed (or given) rate, padded from the container
  start (`-fps_mode cfr`), so audio copied from the same file stays in sync and frame `i` has time
  `i / rate`. Autorotation is ffmpeg's default; `probe` swaps sides for quarter turns.
- Presets declare `supports_alpha` / `supports_audio` / `needs_even_size` and their colour
  encoding; the sink enforces them (non-opaque frames for alpha-less codecs are an error).

## Editing map
- New codec preset: `VideoCodec` in `args.rs` (args, the three capability methods,
  `output_color`, tests).
- Probe fields: `probe.rs` (`-show_entries` list and `parse_flat`).
- Colour conversion rules: `color.rs`.
- Process handling / stderr tails: `process.rs`. Output file handling: `paths.rs` + `sink.rs`.
- End-to-end checks: `tests/video.rs` (ignored; needs a GPU and ffmpeg on `PATH`).
