/*
File: examples/demo/params_ui.rs

Purpose:
A generic editor for the NAMED parameters of a pass (`struct Params` members, reflected by
`Effect::params()`), built from the effect alone: one control per `ParamInfo`.

Key functions:
- pass_params_ui(): edits every named parameter of one `Pass` in place.

Notes:
Controls per `ParamType`: `DragValue` for scalars and vector components; a colour picker for a
`vec4<f32>` whose name contains `color` / `colour` (premultiplied gamma-space RGBA, the space
`ParamValue::from(Color32)` uses); a 4x4 grid (one row per column) for `mat4x4<f32>`; arrays sit
in a collapsed header with one row per element. Values are read with `Pass::get` / `get_slice`
and written back only when a control changed, so an untouched colour is never re-quantized.
*/

use eframe::egui::{self, Color32};
use egui_shader_layers::{ParamInfo, ParamValue, Pass};

/// Edits every named parameter of `pass` in place; returns whether any value changed.
/// A pass whose effect declares no `struct Params` shows a short note instead.
pub fn pass_params_ui(ui: &mut egui::Ui, pass: &mut Pass) -> bool {
    // A cheap clone (the compiled module is shared): the parameter list must outlive `&mut pass`.
    let effect = pass.effect().clone();
    if effect.params().is_empty() {
        ui.weak(format!("`{}`: no named parameters", effect.label()));
        return false;
    }
    let mut changed = false;
    for info in effect.params() {
        ui.push_id(&info.name, |ui| {
            changed |= if info.is_array { array_ui(ui, pass, info) } else { member_ui(ui, pass, info) };
        });
    }
    changed
}

/// One plain member: read, edit, write back when changed.
fn member_ui(ui: &mut egui::Ui, pass: &mut Pass, info: &ParamInfo) -> bool {
    let Some(mut value) = pass.get(&info.name) else {
        ui.colored_label(ui.visuals().error_fg_color, format!("{}: unreadable", info.name));
        return false;
    };
    if !ui.horizontal(|ui| value_ui(ui, &info.name, is_color_name(&info.name), &mut value)).inner {
        return false;
    }
    match pass.set(&info.name, value) {
        Ok(()) => true,
        Err(error) => {
            // Cannot happen: the value's type came from the member itself.
            log::error!("demo: could not set named parameter `{}`: {error}", info.name);
            false
        }
    }
}

/// One array member, collapsed by default: one row per element.
fn array_ui(ui: &mut egui::Ui, pass: &mut Pass, info: &ParamInfo) -> bool {
    let Some(mut values) = pass.get_slice(&info.name) else {
        ui.colored_label(ui.visuals().error_fg_color, format!("{}: unreadable", info.name));
        return false;
    };
    let color = is_color_name(&info.name);
    let header = format!("{}: array<{}, {}>", info.name, info.ty, info.count);
    let changed = egui::CollapsingHeader::new(header)
        .id_salt("array")
        .default_open(false)
        .show(ui, |ui| {
            let mut changed = false;
            for (index, value) in values.iter_mut().enumerate() {
                ui.push_id(index, |ui| {
                    changed |= ui.horizontal(|ui| value_ui(ui, &format!("[{index}]"), color, value)).inner;
                });
            }
            changed
        })
        .body_returned
        .unwrap_or(false);
    if !changed {
        return false;
    }
    match pass.set_slice(&info.name, &values) {
        Ok(()) => true,
        Err(error) => {
            // Cannot happen: the element types and count came from the member itself.
            log::error!("demo: could not set named parameter `{}`: {error}", info.name);
            false
        }
    }
}

/// The control(s) for one value, labelled `label`; `color` turns a `Vec4` into a colour picker.
/// Returns whether the value changed.
fn value_ui(ui: &mut egui::Ui, label: &str, color: bool, value: &mut ParamValue) -> bool {
    ui.label(label);
    match value {
        ParamValue::F32(v) => drag_f32(ui, v),
        ParamValue::I32(v) => ui.add(egui::DragValue::new(v).speed(0.1)).changed(),
        ParamValue::U32(v) => ui.add(egui::DragValue::new(v).speed(0.1)).changed(),
        ParamValue::Vec2(v) => drag_all(ui, v),
        ParamValue::Vec3(v) => drag_all(ui, v),
        ParamValue::Vec4(v) if color => color_ui(ui, v),
        ParamValue::Vec4(v) => drag_all(ui, v),
        ParamValue::Mat4(columns) => {
            let mut changed = false;
            ui.vertical(|ui| {
                for column in columns.iter_mut() {
                    ui.horizontal(|ui| changed |= drag_all(ui, column));
                }
            });
            changed
        }
        // `ParamValue` is `#[non_exhaustive]`: a type added by a later crate version has no
        // control here yet; show it instead of silently skipping it.
        other => {
            ui.weak(format!("{:?} (no editor)", other.ty()));
            false
        }
    }
}

/// A drag control whose speed follows the value's magnitude (1 % of it, at least 0.001).
fn drag_f32(ui: &mut egui::Ui, value: &mut f32) -> bool {
    let speed = (value.abs() * 0.01).max(0.001);
    ui.add(egui::DragValue::new(value).speed(speed)).changed()
}

/// One drag control per component.
fn drag_all(ui: &mut egui::Ui, values: &mut [f32]) -> bool {
    values.iter_mut().fold(false, |changed, value| drag_f32(ui, value) | changed)
}

/// Whether a member name asks for a colour picker (`*color*` / `*colour*`, any case).
fn is_color_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("color") || name.contains("colour")
}

/// A colour picker over premultiplied gamma-space RGBA floats in `0..=1`.
fn color_ui(ui: &mut egui::Ui, rgba: &mut [f32; 4]) -> bool {
    // Round-trips through 8-bit `Color32`; written back only on an edit, so an untouched value
    // keeps its exact floats.
    let [r, g, b, a] = rgba.map(unit_to_byte);
    let mut color = Color32::from_rgba_premultiplied(r, g, b, a);
    if ui.color_edit_button_srgba(&mut color).changed() {
        *rgba = color.to_normalized_gamma_f32();
        return true;
    }
    false
}

/// `0..=1` -> `0..=255`, clamped (non-finite -> 0).
fn unit_to_byte(value: f32) -> u8 {
    let scaled = if value.is_finite() { (value.clamp(0.0, 1.0) * 255.0).round() } else { 0.0 };
    // `scaled` is an integer in 0..=255 after the clamp and round, so the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = scaled as u8;
    byte
}
