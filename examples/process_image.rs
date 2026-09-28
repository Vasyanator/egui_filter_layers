/*
File: examples/process_image.rs

Purpose:
Offline processing of an image file with the `image` feature: load, apply a filter stack on a
headless wgpu device, save. The same chain painted as a `FilterLayer` over the image shown 1:1
in the UI gives the same pixels (see the README, "Offline: images and video").

Usage:
    cargo run --example process_image --features image -- INPUT OUTPUT [--ppp F] [--time S]
        [--max-tile N]

`--ppp` = image pixels per layer point (default 1), `--time` = `fl.time` in seconds,
`--max-tile` caps the texture side (larger images are tiled; the result is identical).
The stack: a 3-point Gaussian blur, saturation 1.4, a vignette and a custom effect that shifts
the red channel by 2 points. Formats: PNG and JPEG (enable more through the `image` crate's
features). A JPEG output of a translucent image needs a background: the example uses white.
*/

use std::path::PathBuf;
use std::process::ExitCode;

use egui::Color32;
use egui_filter_layers::{Effect, FilterChain, OfflineOptions, OfflineRenderer, Pass, presets};

/// The custom effect: red sampled 2 points to the left.
const RED_SHIFT: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = src(uv);
    let shifted = src_px(vec2<f32>(-params(0u).x * fl.pixels_per_point, 0.0));
    return vec4<f32>(min(shifted.r, c.a), c.g, c.b, c.a);
}
";

/// Parsed command line.
struct Args {
    /// Image to read.
    input: PathBuf,
    /// Image to write (format from the extension).
    output: PathBuf,
    /// Run settings.
    options: OfflineOptions,
}

/// Parses `std::env::args`; `Err` carries the message to print.
fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut options = OfflineOptions::default().with_background(Some(Color32::WHITE));
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} expects a value"));
        match arg.as_str() {
            "--ppp" => options = options.with_pixels_per_point(value("--ppp")?.parse().map_err(|error| format!("--ppp: {error}"))?),
            "--time" => options = options.with_time(value("--time")?.parse().map_err(|error| format!("--time: {error}"))?),
            "--max-tile" => options = options.with_max_tile_size(Some(value("--max-tile")?.parse().map_err(|error| format!("--max-tile: {error}"))?)),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [input, output]: [PathBuf; 2] = positional.try_into().map_err(|_| "expected INPUT and OUTPUT paths".to_owned())?;
    Ok(Args { input, output, options })
}

/// The filter stack applied by the example.
fn stack() -> Result<FilterChain, String> {
    let shift = Effect::new("red shift", RED_SHIFT).map_err(|error| error.diagnostic().map_or_else(|| error.to_string(), str::to_owned))?.with_sample_radius(2.0);
    let shift = Pass::new(&shift).with_param(0, [2.0, 0.0, 0.0, 0.0]).map_err(|error| error.to_string())?;
    Ok(presets::gaussian_blur(3.0).then(presets::saturation(1.4)).then(presets::vignette(0.5, 0.4, 0.6)).then(shift))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\nusage: process_image INPUT OUTPUT [--ppp F] [--time S] [--max-tile N]");
            return ExitCode::FAILURE;
        }
    };
    let chain = match stack() {
        Ok(chain) => chain,
        Err(message) => {
            eprintln!("the example's effect does not compile: {message}");
            return ExitCode::FAILURE;
        }
    };
    let mut renderer = match OfflineRenderer::headless() {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("cannot create a GPU renderer: {error}\nOffline processing needs a wgpu adapter (Vulkan, Metal, DX12 or GL).");
            return ExitCode::FAILURE;
        }
    };
    match renderer.process_file(&args.input, &args.output, &chain, &args.options) {
        Ok(()) => {
            println!("wrote {}", args.output.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("processing {} failed: {error}", args.input.display());
            ExitCode::FAILURE
        }
    }
}
