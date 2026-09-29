/*
File: wgpu_backend/resources.rs

Purpose:
The UI callback's wgpu objects: the chain executor's device objects (`ChainResources`), the
blended composite pipelines built for egui's render pass, the mesh re-render pipeline (gamma or,
on `*Srgb` targets with view-format support, linear through an sRGB view of the capture), and the
pool of per-layer targets of the frame being prepared.

Key structures:
- `WgpuResources`: stored in egui-wgpu's `CallbackResources`; one per install generation.
- `LayerTargets`: capture, chain targets and mesh buffers of ONE layer of the current frame.
- `StateTargets`: the two state textures of one stateful layer (F7), kept across frames in
  `WgpuResources::states` (NOT in the per-frame pool), so they die with the install's resources.

Key functions:
- `WgpuResources::create()`, `WgpuResources::next_targets()`.

Notes:
All layers' `prepare` calls are recorded before egui's single render pass, and every layer's
result must survive until its `paint`, so each layer of a frame gets its own `LayerTargets`
(pool index = order of `prepare`, reset in `finish_prepare`). Reusing an entry in the next
submission is safe: queue writes and passes of a later submission execute after the earlier one.
Object creation runs inside wgpu validation error scopes: on native wgpu-core a scope resolves
immediately, so a rejected shader becomes a latched failure instead of wgpu's default panic.
*/

use super::WgpuOptions;
use super::chain::{ChainResources, ChainTargets, CompositeOver};
use super::coverage::{EGUI_BLEND, MESH_WGSL, mesh_locals_layout, mesh_pipeline};
use super::gpu::{GrowBuffer, Target, egui_texture_entries, shader, with_error_scope};
use crate::state::StateRegistry;

/// The two state textures of one stateful layer (`Slot::StateA` / `StateB`), created and grown by
/// the chain executor.
pub(super) type StateTargets = [Option<Target>; 2];

/// Format of the re-rendered backdrop: 8-bit gamma values, exactly what egui writes to a
/// non-sRGB surface (and the bytes a `*Srgb` surface stores). Effects always sample this format.
pub(super) const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The view format the meshes are rendered through when [`WgpuResources::srgb_capture`] is set:
/// the hardware then blends in linear and stores the sRGB encoding, as on a `*Srgb` screen.
pub(super) const CAPTURE_SRGB_VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;


/// Everything ONE filter layer of the current frame renders into and draws from.
#[derive(Debug)]
pub(super) struct LayerTargets {
    /// The re-rendered backdrop (8-bit gamma); the chain's source.
    pub capture: Option<Target>,
    /// The chain's ping-pong intermediates and uniforms.
    pub chain: ChainTargets,
    /// The mesh shader's `MeshLocals` block.
    pub mesh_locals: GrowBuffer,
    /// Vertices of the beneath meshes.
    pub vertices: GrowBuffer,
    /// Indices of the beneath meshes.
    pub indices: GrowBuffer,
    /// The composite pass's bind group, built by `prepare` for `paint`.
    pub composite: Option<wgpu::BindGroup>,
}

impl LayerTargets {
    /// Empty targets; textures and buffers are allocated on first use.
    fn new() -> Self {
        Self {
            capture: None,
            chain: ChainTargets::new(),
            mesh_locals: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_filter_layers mesh locals"),
            vertices: GrowBuffer::new(wgpu::BufferUsages::VERTEX, "egui_filter_layers vertices"),
            indices: GrowBuffer::new(wgpu::BufferUsages::INDEX, "egui_filter_layers indices"),
            composite: None,
        }
    }
}

/// All GPU objects of one install, kept in egui-wgpu's `CallbackResources`.
#[derive(Debug)]
pub(super) struct WgpuResources {
    /// `Shared::generation` these were built for.
    pub generation: u64,
    /// The chain executor's device objects (sampler, layouts, effect pipeline cache).
    pub chain: ChainResources,
    /// The blended composite pipelines (egui's target format, sample count and depth format).
    pub composite: CompositeOver,
    /// Layout of the mesh pipeline's group 0 (`MeshLocals`).
    pub mesh_locals_layout: wgpu::BindGroupLayout,
    /// The mesh re-render pipeline (into `CAPTURE_FORMAT`, or its `CAPTURE_SRGB_VIEW_FORMAT` view
    /// when `srgb_capture`).
    pub mesh_pipeline: wgpu::RenderPipeline,
    /// Whether the capture is rendered through its sRGB view (`*Srgb` target and the device
    /// supports `DownlevelFlags::VIEW_FORMATS`): semi-transparent meshes then blend in linear like
    /// egui's own draws. `false` on a `*Srgb` target is the documented GL / WebGL2 gap.
    pub srgb_capture: bool,
    /// Per-layer targets of the frame being prepared.
    pub pool: Vec<LayerTargets>,
    /// Next free `pool` entry in the frame being prepared.
    pub cursor: usize,
    /// Persistent layer states (F7), by viewport and layer id; untouched by the per-frame pool
    /// reset, dropped with these resources (a re-install builds new ones).
    pub states: StateRegistry<StateTargets>,
}

impl WgpuResources {
    /// Builds the chain objects, the blended composite pipelines for egui's pass and the mesh
    /// pipeline for one install.
    /// Warns (once per install, since this runs once per install) when a `*Srgb` target has to fall
    /// back to gamma-space re-rendering for lack of `DownlevelFlags::VIEW_FORMATS`.
    ///
    /// # Errors
    /// A message when the crate's composite module is broken or the device rejects any object.
    pub(super) fn create(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter, target_format: wgpu::TextureFormat, options: &WgpuOptions, generation: u64) -> Result<Self, String> {
        let chain = ChainResources::create(device, queue, adapter)?;
        // Mirrors egui-wgpu's depth state so the pipeline is compatible with its pass.
        let depth_stencil = options.depth_stencil_format().map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let multisample = wgpu::MultisampleState { count: options.msaa_samples(), mask: !0, alpha_to_coverage_enabled: false };
        let composite = chain.composite_over_pipelines(device, target_format, multisample, depth_stencil)?;

        // On a `*Srgb` screen egui blends in linear; the capture reproduces that by rendering
        // through an sRGB view of its unorm texture, which needs view-format reinterpretation.
        let srgb_capture = target_format.is_srgb() && adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::VIEW_FORMATS);
        // Logged here, i.e. once per install: `create` runs once per install generation (a
        // failure latches the backend instead of retrying), so no latch is needed.
        if target_format.is_srgb() && !srgb_capture {
            log::warn!(
                "egui_filter_layers (wgpu): the {target_format:?} target is sRGB but this device cannot reinterpret texture formats \
                 (no DownlevelFlags::VIEW_FORMATS; GL / WebGL2). Effect: the backdrop under filters is re-rendered with gamma-space \
                 blending, so semi-transparent edges under a filter visibly differ from the unfiltered screen (about 25 8-bit \
                 steps measured for 50% white over dark grey). Fix: prefer an Rgba8Unorm / Bgra8Unorm surface (egui-wgpu's own \
                 preference). Logged once per install."
            );
        }
        let (mesh_format, mesh_entry) = if srgb_capture { (CAPTURE_SRGB_VIEW_FORMAT, "fs_main_linear") } else { (CAPTURE_FORMAT, "fs_main") };

        let (mesh_locals_layout, mesh_pipeline) = with_error_scope(device, || {
            let mesh_locals_layout = mesh_locals_layout(device);
            // Identical descriptor to egui-wgpu's `egui_texture_bind_group_layout`: wgpu-core
            // deduplicates equal layouts and WebGPU treats them as group-equivalent, so egui's
            // own per-texture bind groups can be bound to this pipeline unchanged.
            let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_filter_layers egui texture"), entries: &egui_texture_entries() });
            let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("egui_filter_layers mesh"),
                bind_group_layouts: &[Some(&mesh_locals_layout), Some(&texture_layout)],
                immediate_size: 0,
            });
            let mesh_shader = shader(device, "egui_filter_layers mesh", MESH_WGSL);
            let mesh_pipeline = mesh_pipeline(device, &mesh_pipeline_layout, &mesh_shader, mesh_format, mesh_entry, Some(EGUI_BLEND), "egui_filter_layers mesh");
            (mesh_locals_layout, mesh_pipeline)
        })?;

        Ok(Self { generation, chain, composite, mesh_locals_layout, mesh_pipeline, srgb_capture, pool: Vec::new(), cursor: 0, states: StateRegistry::default() })
    }

    /// The next free per-layer targets of this frame.
    pub(super) fn next_targets(&mut self) -> usize {
        let index = self.cursor;
        if self.pool.len() <= index {
            self.pool.push(LayerTargets::new());
        }
        self.cursor += 1;
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mesh_shader_validates_and_matches_the_locals_size() {
        let source = MESH_WGSL;
        let module = naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source));
        let Ok(module) = module else { panic!("{module:?}") };
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module);
        assert!(info.is_ok(), "{info:?}");
        let (_, ty) = module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some("MeshLocals")).unwrap_or_else(|| panic!("MeshLocals missing"));
        let naga::TypeInner::Struct { span, .. } = &ty.inner else { panic!("MeshLocals is not a struct") };
        assert_eq!(usize::try_from(*span).unwrap_or(0), crate::plan::MESH_LOCALS_SIZE);
        for entry in ["vs_main", "fs_main", "fs_main_linear"] {
            assert!(module.entry_points.iter().any(|point| point.name == entry), "entry point `{entry}` missing");
        }
    }
}
