/*
File: examples/demo/stack.rs

Purpose:
The preset stack editor: a list of presets (`Kind`) the user adds, reorders, toggles and tunes,
plus a live-compiled custom WGSL effect. Produces one `FilterChain` for a shader layer. Reusable
by any scene; the "stack" scene is its main user.

Key structures:
- Kind: the presets offered (command-line key, label, slider table).
- Entry: one stack item (slider values, colour, and for named-parameter kinds the built passes).
- StackEditor: the stack + the custom effect source / compile result; `ui` and `chain`.

Key functions:
- parse_stack(): the `--stack name:v1:...,name` command-line syntax.

Notes:
Two ways parameters reach a pass. Anonymous-slot presets (`params(i)`) are rebuilt every frame
from the sliders. Named-parameter kinds (kawase blur, bloom, and the custom effect when its source
declares `struct Params`) keep their built passes in the entry; the sliders are the preset's
CONSTRUCTOR arguments (a change rebuilds the passes, resetting named edits), and the generic
`params_ui` editor tunes each built pass's `struct Params` members in place.
*/

use eframe::egui::{self, Color32};
use egui_shader_layers::{Effect, EffectError, FilterChain, Pass, presets};

use crate::params_ui::pass_params_ui;

/// The custom effect shown on start: an animated chromatic wave. Edit it live in the demo.
const CUSTOM_EFFECT: &str = "// params(0).x = amplitude (points), params(0).y = frequency
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let p = params(0u);
    let amp = p.x * fl.pixels_per_point;
    let wave = sin(uv.y * p.y + fl.time * 3.0) * amp;
    let r = src_px(vec2<f32>(wave + 2.0, 0.0)).r;
    let g = src_px(vec2<f32>(wave, 0.0)).g;
    let b = src_px(vec2<f32>(wave - 2.0, 0.0)).b;
    let a = src_px(vec2<f32>(wave, 0.0)).a;
    return vec4<f32>(r, g, b, a);
}
";

/// A preset the side panel can add to the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Blur,
    BrightnessContrast,
    Saturation,
    Grayscale,
    Invert,
    Tint,
    Sharpen,
    Pixelate,
    Ripple,
    Lens,
    Vignette,
    KawaseBlur,
    Bloom,
    Custom,
}

impl Kind {
    /// Every kind, in menu order.
    const ALL: [Kind; 14] = [
        Kind::Blur,
        Kind::BrightnessContrast,
        Kind::Saturation,
        Kind::Grayscale,
        Kind::Invert,
        Kind::Tint,
        Kind::Sharpen,
        Kind::Pixelate,
        Kind::Ripple,
        Kind::Lens,
        Kind::Vignette,
        Kind::KawaseBlur,
        Kind::Bloom,
        Kind::Custom,
    ];

    /// Command-line name (`--stack`).
    fn key(self) -> &'static str {
        match self {
            Kind::Blur => "blur",
            Kind::BrightnessContrast => "brightness_contrast",
            Kind::Saturation => "saturation",
            Kind::Grayscale => "grayscale",
            Kind::Invert => "invert",
            Kind::Tint => "tint",
            Kind::Sharpen => "sharpen",
            Kind::Pixelate => "pixelate",
            Kind::Ripple => "ripple",
            Kind::Lens => "lens",
            Kind::Vignette => "vignette",
            Kind::KawaseBlur => "kawase_blur",
            Kind::Bloom => "bloom",
            Kind::Custom => "custom",
        }
    }

    /// Menu / list label.
    fn name(self) -> &'static str {
        match self {
            Kind::Blur => "Gaussian blur",
            Kind::BrightnessContrast => "Brightness / contrast",
            Kind::Saturation => "Saturation",
            Kind::Grayscale => "Grayscale",
            Kind::Invert => "Invert",
            Kind::Tint => "Tint",
            Kind::Sharpen => "Sharpen",
            Kind::Pixelate => "Pixelate",
            Kind::Ripple => "Ripple (animated)",
            Kind::Lens => "Lens",
            Kind::Vignette => "Vignette",
            Kind::KawaseBlur => "Dual-Kawase blur (named params)",
            Kind::Bloom => "Bloom (named params)",
            Kind::Custom => "Custom WGSL (animated)",
        }
    }

    /// `(label, min, max, default)` of each slider. For named-parameter kinds these are the
    /// preset's constructor arguments.
    fn sliders(self) -> &'static [(&'static str, f32, f32, f32)] {
        match self {
            Kind::Blur => &[("sigma (pt)", 0.0, 30.0, 6.0)],
            Kind::BrightnessContrast => &[("brightness", -1.0, 1.0, 0.1), ("contrast", 0.0, 3.0, 1.3)],
            Kind::Saturation => &[("factor", 0.0, 3.0, 1.8)],
            Kind::Grayscale | Kind::Invert => &[("amount", 0.0, 1.0, 1.0)],
            Kind::Tint => &[("amount", 0.0, 1.0, 0.3)],
            Kind::Sharpen => &[("amount", 0.0, 3.0, 1.0)],
            Kind::Pixelate => &[("cell (pt)", 1.0, 40.0, 10.0)],
            Kind::Ripple => &[("amplitude (pt)", 0.0, 20.0, 4.0), ("wavelength (pt)", 5.0, 100.0, 30.0), ("speed", 0.0, 3.0, 0.8)],
            Kind::Lens => &[("magnification", 1.0, 4.0, 2.0)],
            Kind::Vignette => &[("strength", 0.0, 1.0, 0.8), ("inner", 0.0, 1.0, 0.3), ("softness", 0.0, 1.0, 0.6)],
            Kind::KawaseBlur => &[("radius (pt)", 0.0, 60.0, 12.0)],
            Kind::Bloom => &[("threshold", 0.0, 1.0, 0.6), ("intensity", 0.0, 4.0, 1.5), ("radius (pt)", 0.0, 60.0, 16.0)],
            Kind::Custom => &[("amplitude (pt)", 0.0, 20.0, 5.0), ("frequency", 1.0, 80.0, 30.0)],
        }
    }

    /// Whether the entry keeps built passes whose named parameters the side panel edits.
    fn keeps_passes(self) -> bool {
        match self {
            Kind::KawaseBlur | Kind::Bloom | Kind::Custom => true,
            Kind::Blur
            | Kind::BrightnessContrast
            | Kind::Saturation
            | Kind::Grayscale
            | Kind::Invert
            | Kind::Tint
            | Kind::Sharpen
            | Kind::Pixelate
            | Kind::Ripple
            | Kind::Lens
            | Kind::Vignette => false,
        }
    }
}

/// The passes a named-parameter entry was built into, and what they were built from.
#[derive(Clone, Debug)]
struct Built {
    /// The passes; their named parameters are edited in place.
    chain: FilterChain,
    /// Slider values the chain was built from (constructor arguments; unused by `Custom`).
    values: Vec<f32>,
    /// `StackEditor::custom_generation` the chain was built for (`Custom` only).
    generation: u64,
}

/// One entry of the filter stack.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Which preset.
    kind: Kind,
    /// Whether it participates in the chain.
    enabled: bool,
    /// Slider values, parallel to `kind.sliders()`.
    values: Vec<f32>,
    /// Colour for `Kind::Tint`.
    color: Color32,
    /// Built passes of a named-parameter kind (`Kind::keeps_passes`); `None` until the first
    /// sync, or while the custom effect fails to compile.
    built: Option<Built>,
}

impl Entry {
    /// An entry with default values.
    fn new(kind: Kind) -> Self {
        Self { kind, enabled: true, values: kind.sliders().iter().map(|s| s.3).collect(), color: Color32::from_rgb(120, 170, 255), built: None }
    }

    /// Slider value `i` (0 if out of range, which cannot happen for entries built by `new`).
    fn value(&self, i: usize) -> f32 {
        self.values.get(i).copied().unwrap_or(0.0)
    }

    /// Brings `built` up to date: rebuilds a named preset whose constructor sliders changed, and
    /// re-creates the custom pass after a recompile (`generation` changed), carrying over the
    /// named values whose name and type still match.
    fn sync(&mut self, custom: Option<&Effect>, generation: u64) {
        let v = |i| self.value(i);
        match self.kind {
            Kind::KawaseBlur | Kind::Bloom => {
                if self.built.as_ref().is_some_and(|built| built.values == self.values) {
                    return;
                }
                let chain = if self.kind == Kind::KawaseBlur { presets::kawase_blur(v(0)) } else { presets::bloom(v(0), v(1), v(2)) };
                self.built = Some(Built { chain, values: self.values.clone(), generation });
            }
            Kind::Custom => {
                if self.built.as_ref().is_some_and(|built| built.generation == generation) {
                    return;
                }
                let old = self.built.take();
                self.built = custom.map(|effect| {
                    let mut pass = Pass::new(effect);
                    if let Some(old_pass) = old.as_ref().and_then(|built| built.chain.passes().first()) {
                        carry_named(old_pass, &mut pass);
                    }
                    Built { chain: pass.into(), values: Vec::new(), generation }
                });
            }
            Kind::Blur
            | Kind::BrightnessContrast
            | Kind::Saturation
            | Kind::Grayscale
            | Kind::Invert
            | Kind::Tint
            | Kind::Sharpen
            | Kind::Pixelate
            | Kind::Ripple
            | Kind::Lens
            | Kind::Vignette => {}
        }
    }

    /// The passes of this entry, or `None` when it cannot run (custom effect failed to compile,
    /// or a named entry not yet synced).
    fn chain(&self) -> Option<FilterChain> {
        let v = |i| self.value(i);
        Some(match self.kind {
            Kind::Blur => presets::gaussian_blur(v(0)),
            Kind::BrightnessContrast => presets::brightness_contrast(v(0), v(1)).into(),
            Kind::Saturation => presets::saturation(v(0)).into(),
            Kind::Grayscale => presets::grayscale(v(0)).into(),
            Kind::Invert => presets::invert(v(0)).into(),
            Kind::Tint => presets::tint(self.color, v(0)).into(),
            Kind::Sharpen => presets::sharpen(v(0)).into(),
            Kind::Pixelate => presets::pixelate(v(0)).into(),
            Kind::Ripple => presets::ripple(v(0), v(1), v(2)).into(),
            Kind::Lens => presets::lens(v(0)).into(),
            Kind::Vignette => presets::vignette(v(0), v(1), v(2)).into(),
            Kind::KawaseBlur | Kind::Bloom => self.built.as_ref()?.chain.clone(),
            Kind::Custom => {
                let pass = self.built.as_ref()?.chain.passes().first()?.clone();
                let pass = pass.with_param(0, [v(0), v(1), 0.0, 0.0]).ok()?;
                // The wave shifts samples by up to `amplitude` points plus a 2 px channel split.
                pass.with_sample_radius(v(0) + 2.0).into()
            }
        })
    }

    /// The side-panel controls of this entry. `action` receives `(index, -1 | 0 | 1)` for
    /// up / remove / down.
    fn ui(&mut self, ui: &mut egui::Ui, index: usize, action: &mut Option<(usize, i8)>) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.enabled, self.kind.name());
                if ui.small_button("^").clicked() {
                    *action = Some((index, -1));
                }
                if ui.small_button("v").clicked() {
                    *action = Some((index, 1));
                }
                if ui.small_button("x").clicked() {
                    *action = Some((index, 0));
                }
            });
            for (value, (label, min, max, _)) in self.values.iter_mut().zip(self.kind.sliders()) {
                ui.add(egui::Slider::new(value, *min..=*max).text(*label));
            }
            if self.kind == Kind::Tint {
                ui.horizontal(|ui| {
                    ui.label("colour");
                    ui.color_edit_button_srgba(&mut self.color);
                });
            }
            let Some(built) = &mut self.built else {
                return;
            };
            if built.chain.passes().iter().all(|pass| pass.effect().params().is_empty()) {
                return;
            }
            egui::CollapsingHeader::new("named parameters").id_salt("named").default_open(false).show(ui, |ui| {
                for (pass_index, pass) in built.chain.passes_mut().iter_mut().enumerate() {
                    ui.push_id(pass_index, |ui| {
                        ui.label(format!("pass {pass_index}: {}", pass.effect().label()));
                        pass_params_ui(ui, pass);
                    });
                }
            });
        });
    }
}

/// Copies every named parameter of `from` into `to` whose name, type and length match `to`'s
/// effect (after a recompile); the others keep their zero default.
fn carry_named(from: &Pass, to: &mut Pass) {
    let effect = to.effect().clone();
    for info in effect.params() {
        let Some(values) = from.get_slice(&info.name) else {
            continue;
        };
        let fits = values.len() == usize::try_from(info.count).unwrap_or(usize::MAX) && values.iter().all(|value| value.ty() == info.ty);
        if !fits {
            continue;
        }
        if let Err(error) = to.set_slice(&info.name, &values) {
            // Cannot happen after the check above; logged rather than silently dropped.
            log::warn!("demo: could not carry named parameter `{}` over: {error}", info.name);
        }
    }
}

/// Parses one `--stack` item: `name` or `name:v1:v2...` (values override the sliders in order).
fn parse_stack_entry(item: &str) -> Result<Entry, String> {
    let mut parts = item.trim().split(':');
    let name = parts.next().unwrap_or_default();
    let kind = Kind::ALL.into_iter().find(|kind| kind.key() == name).ok_or_else(|| format!("unknown filter {name:?}; known: {}", Kind::ALL.map(Kind::key).join(",")))?;
    let mut entry = Entry::new(kind);
    for (index, value) in parts.enumerate() {
        let slot = entry.values.get_mut(index).ok_or_else(|| format!("{name:?} takes at most {} values", kind.sliders().len()))?;
        *slot = value.parse().map_err(|error| format!("{name:?} value {value:?}: {error}"))?;
    }
    Ok(entry)
}

/// Parses a `--stack` list: comma-separated `name[:v1:v2...]` items.
///
/// # Errors
/// The message to print for an unknown name, too many values or an unparsable number.
pub fn parse_stack(list: &str) -> Result<Vec<Entry>, String> {
    list.split(',').map(parse_stack_entry).collect()
}

/// The preset stack and the custom effect it may use.
pub struct StackEditor {
    /// The preset stack, applied top to bottom.
    stack: Vec<Entry>,
    /// Editable source of the custom effect.
    custom_source: String,
    /// Last compile result of `custom_source`.
    custom_effect: Result<Effect, EffectError>,
    /// Bumped on every compile, so custom entries re-create their pass.
    custom_generation: u64,
}

impl StackEditor {
    /// An editor over `stack`, or the default blur + saturation when `None`.
    pub fn new(stack: Option<Vec<Entry>>) -> Self {
        Self {
            stack: stack.unwrap_or_else(|| vec![Entry::new(Kind::Blur), Entry::new(Kind::Saturation)]),
            custom_source: CUSTOM_EFFECT.to_owned(),
            custom_effect: compile_custom(CUSTOM_EFFECT),
            custom_generation: 0,
        }
    }

    /// Brings every entry's built passes up to date (see `Entry::sync`).
    fn sync(&mut self) {
        let custom = self.custom_effect.as_ref().ok();
        for entry in self.stack.iter_mut().filter(|entry| entry.kind.keeps_passes()) {
            entry.sync(custom, self.custom_generation);
        }
    }

    /// The chain of every enabled entry, top to bottom (entries that cannot run are skipped).
    pub fn chain(&mut self) -> FilterChain {
        self.sync();
        self.stack.iter().filter(|entry| entry.enabled).filter_map(Entry::chain).fold(FilterChain::new(), FilterChain::then)
    }

    /// The side-panel editor: add menu, entries, custom effect source and compile status.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Add filter", |ui| {
            for kind in Kind::ALL {
                if ui.button(kind.name()).clicked() {
                    self.stack.push(Entry::new(kind));
                    ui.close();
                }
            }
        });
        ui.label("Applied top to bottom. Drag the rect in the scene; resize from its corner.");

        // Named entries show their built passes' parameters, so build them before drawing.
        self.sync();
        let mut action: Option<(usize, i8)> = None;
        for (index, entry) in self.stack.iter_mut().enumerate() {
            ui.push_id(index, |ui| entry.ui(ui, index, &mut action));
        }
        match action {
            Some((index, 0)) => {
                self.stack.remove(index);
            }
            Some((index, -1)) if index > 0 => self.stack.swap(index, index - 1),
            Some((index, 1)) if index + 1 < self.stack.len() => self.stack.swap(index, index + 1),
            _ => {}
        }

        ui.separator();
        ui.label("Custom effect (WGSL). Add \"Custom WGSL\" to the stack to use it. A `struct Params` in the source gets editable named parameters.");
        egui::ScrollArea::vertical().id_salt("custom-source").max_height(260.0).show(ui, |ui| {
            ui.add(egui::TextEdit::multiline(&mut self.custom_source).code_editor().desired_width(f32::INFINITY));
        });
        if ui.button("Compile").clicked() {
            self.custom_effect = compile_custom(&self.custom_source);
            self.custom_generation += 1;
        }
        match &self.custom_effect {
            Ok(_) => {
                ui.label("compiled OK");
            }
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, error.to_string());
            }
        }
    }
}

/// Compiles the custom effect (animated automatically when it reads `fl.time`).
fn compile_custom(source: &str) -> Result<Effect, EffectError> {
    Effect::new("custom", source)
}
