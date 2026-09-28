/*
File: wgpu_backend/gpu.rs

Purpose:
Plain wgpu helpers with no egui state: grow-only textures and buffers, bind group layout
entries (and the ONE descriptor of egui-wgpu's texture bind group layout, `egui_texture_entries`),
the egui-wgpu sampler mapping of `egui::TextureOptions` (plain data), shader / full-viewport
pipeline creation, single-attachment render passes, format capability checks, and error scopes.
Shared by the chain executor (`chain.rs`), the UI callback's own objects (`resources.rs`,
`textures.rs`) and the offline renderer.

Key structures:
- `Target`: grow-only 2D render target that is sampled afterwards (optionally with a second,
  sRGB-reinterpreting view to render through).
- `GrowBuffer`: grow-only buffer written with `Queue::write_buffer`.

Key functions:
- `ErrorScopes` (out-of-memory + validation + internal), `with_error_scope()`, `pass_pipeline()`, `begin_pass()`, `renderable_and_filterable()`.

Notes:
Everything here takes `&wgpu::Device` / `&wgpu::Queue` explicitly, so it works on any device.
*/

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use crate::effect::{CompiledModule, FRAGMENT_ENTRY, VERTEX_ENTRY};

/// Error scopes for every error kind (out-of-memory, validation, internal) around a block of GPU
/// work: [`ErrorScopes::push`] before it, [`ErrorScopes::pop`] right after it.
///
/// wgpu routes an error only to the innermost scope with its exact filter and panics in the
/// default uncaptured handler otherwise (`wgpu` 30, `backend/wgpu_core.rs`), so a validation
/// scope alone lets an out-of-memory error abort the process. Scopes are thread-local and must be
/// popped in reverse order on the thread that pushed them.
#[derive(Debug)]
pub(crate) struct ErrorScopes {
    // Declared innermost first: if the work panics, the guards drop (and pop) in field order,
    // which is then the reverse push order wgpu requires.
    /// Innermost scope.
    internal: wgpu::ErrorScopeGuard,
    /// Middle scope.
    validation: wgpu::ErrorScopeGuard,
    /// Outermost scope.
    out_of_memory: wgpu::ErrorScopeGuard,
}

impl ErrorScopes {
    /// Pushes the three scopes on the current thread.
    pub(crate) fn push(device: &wgpu::Device) -> Self {
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        Self { internal, validation, out_of_memory }
    }

    /// Pops the scopes (immediately, in reverse order) and returns a future of the first error
    /// they caught. The future holds no scope, so it may be awaited later, across `.await`
    /// points and after other GPU work; it resolves at once on native (wgpu-core) and
    /// asynchronously on the web. An out-of-memory error is preferred over the validation errors
    /// it usually causes downstream (use of an invalid resource).
    pub(crate) fn pop(self) -> impl Future<Output = Option<wgpu::Error>> {
        let Self { internal, validation, out_of_memory } = self;
        let internal = internal.pop();
        let validation = validation.pop();
        let out_of_memory = out_of_memory.pop();
        async move {
            let (out_of_memory, validation, internal) = (out_of_memory.await, validation.await, internal.await);
            out_of_memory.or(validation).or(internal)
        }
    }
}

/// Runs `create` inside [`ErrorScopes`] and reports what the device rejected, synchronously.
///
/// On native (wgpu-core) the popped scopes resolve immediately. On the web they resolve
/// asynchronously; there the error cannot be observed here and `Ok` is returned (the browser logs
/// it and the draw is dropped by the device, it does not abort). Callers that can await pop
/// [`ErrorScopes`] themselves instead.
pub(crate) fn with_error_scope<T>(device: &wgpu::Device, create: impl FnOnce() -> T) -> Result<T, String> {
    let scopes = ErrorScopes::push(device);
    let value = create();
    let mut future = pin!(scopes.pop());
    match future.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(Some(error)) => Err(error.to_string()),
        Poll::Ready(None) | Poll::Pending => Ok(value),
    }
}

/// A grow-only 2D texture used as a render target and sampled afterwards.
#[derive(Debug)]
pub(crate) struct Target {
    /// The texture (kept for its lifetime; views hold it too).
    _texture: wgpu::Texture,
    /// Default view of the whole texture (the texture's own format); what passes sample.
    pub view: wgpu::TextureView,
    /// A view in another format of the same texture (e.g. `Rgba8UnormSrgb` of an `Rgba8Unorm`
    /// texture) to render through, when one was requested; see [`Target::render_view`].
    render_view: Option<wgpu::TextureView>,
    /// Format of `render_view` (`None` when there is none).
    render_view_format: Option<wgpu::TextureFormat>,
    /// Allocated size in pixels.
    pub size: [u32; 2],
}

impl Target {
    /// Makes `slot` hold a target of `format` at least `needed` pixels large (grow-only, so a
    /// layer that changes size every frame does not reallocate every frame).
    ///
    /// `render_view_format`, when set, adds a second view in that format (listed in the texture's
    /// `view_formats`; it must differ from `format` only by the sRGB suffix and the device needs
    /// `DownlevelFlags::VIEW_FORMATS` — the caller checks). A slot holding a target with another
    /// render view format is recreated.
    pub(crate) fn ensure<'a>(slot: &'a mut Option<Target>, device: &wgpu::Device, format: wgpu::TextureFormat, render_view_format: Option<wgpu::TextureFormat>, label: &str, needed: [u32; 2]) -> &'a Target {
        let fits = slot.as_ref().is_some_and(|target| target.size[0] >= needed[0] && target.size[1] >= needed[1] && target.render_view_format == render_view_format);
        let previous = slot.as_ref().map_or([0, 0], |target| target.size);
        if !fits {
            *slot = None;
        }
        slot.get_or_insert_with(|| {
            let size = [needed[0].max(previous[0]).max(1), needed[1].max(previous[1]).max(1)];
            let view_formats = render_view_format.as_slice();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats,
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let render_view = render_view_format.map(|view_format| texture.create_view(&wgpu::TextureViewDescriptor { format: Some(view_format), ..wgpu::TextureViewDescriptor::default() }));
            Target { _texture: texture, view, render_view, render_view_format, size }
        })
    }

    /// The view to render into: the `render_view_format` view when one was requested, else the
    /// default view.
    #[must_use]
    pub(crate) fn render_view(&self) -> &wgpu::TextureView {
        self.render_view.as_ref().unwrap_or(&self.view)
    }

    /// Format of the second render view, if one was requested.
    #[must_use]
    pub(crate) fn render_view_format(&self) -> Option<wgpu::TextureFormat> {
        self.render_view_format
    }

}

/// A grow-only GPU buffer written with `Queue::write_buffer`.
#[derive(Debug)]
pub(crate) struct GrowBuffer {
    /// The buffer, once something was written.
    buffer: Option<wgpu::Buffer>,
    /// Allocated bytes.
    capacity: u64,
    /// Usage (always includes `COPY_DST`).
    usage: wgpu::BufferUsages,
    /// Debug label.
    label: &'static str,
}

impl GrowBuffer {
    /// An empty buffer slot.
    pub(crate) fn new(usage: wgpu::BufferUsages, label: &'static str) -> Self {
        Self { buffer: None, capacity: 0, usage: usage | wgpu::BufferUsages::COPY_DST, label }
    }

    /// Uploads `bytes` (length a multiple of 4) at offset 0, growing the buffer first if needed.
    pub(crate) fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> &wgpu::Buffer {
        let needed = u64::try_from(bytes.len()).unwrap_or(u64::MAX).max(4);
        if self.capacity < needed {
            self.buffer = None;
        }
        let buffer = self.buffer.get_or_insert_with(|| {
            let capacity = needed.next_power_of_two().max(1024);
            self.capacity = capacity;
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(self.label), size: capacity, usage: self.usage, mapped_at_creation: false })
        });
        if !bytes.is_empty() {
            queue.write_buffer(buffer, 0, bytes);
        }
        buffer
    }
}

/// A uniform-buffer entry; `min_size` is the block's size in bytes when fixed (validated at
/// pipeline creation), `None` when modules may declare blocks of any size there.
pub(crate) fn uniform_entry(binding: u32, min_size: Option<usize>) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: min_size.and_then(|size| NonZeroU64::new(u64::try_from(size).unwrap_or(u64::MAX))) },
        count: None,
    }
}

/// A fragment-stage filterable 2D float texture entry.
pub(crate) fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { multisampled: false, sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2 },
        count: None,
    }
}

/// A fragment-stage filtering sampler entry.
pub(crate) fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// The entries of egui-wgpu 0.36's texture bind-group layout (`renderer.rs`,
/// `egui_texture_bind_group_layout`): a filterable 2D float texture at 0 and a filtering sampler
/// at 1, fragment stage. Every layout that binds egui's per-texture bind groups (the mesh
/// pipeline's group 1, the user texture groups 1..=3 of effects) is created from these, so
/// egui's bind groups bind unchanged (wgpu-core deduplicates equal layouts; WebGPU
/// group-equivalence).
pub(crate) fn egui_texture_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [texture_entry(0), sampler_entry(1)]
}

/// Maps egui texture options to a sampler descriptor exactly as egui-wgpu 0.36 does
/// (`renderer.rs`, `create_sampler`: mipmap mode ignored).
pub(crate) fn egui_sampler_descriptor(options: egui::TextureOptions) -> wgpu::SamplerDescriptor<'static> {
    let filter = |filter: egui::TextureFilter| match filter {
        egui::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        egui::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode = match options.wrap_mode {
        egui::TextureWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        egui::TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        egui::TextureWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };
    wgpu::SamplerDescriptor {
        label: Some("egui_filter_layers egui texture sampler"),
        mag_filter: filter(options.magnification),
        min_filter: filter(options.minification),
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..Default::default()
    }
}

/// Creates a WGSL shader module.
pub(crate) fn shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)) })
}

/// A pipeline running an assembled effect / composite module (attribute-less full-viewport
/// triangle) into `format`. `blend: None` replaces the target's pixels; `Some` blends with them.
pub(crate) fn pass_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &CompiledModule,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    multisample: wgpu::MultisampleState,
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    let shader_module = shader(device, &module.label, &module.source);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&module.label),
        layout: Some(layout),
        vertex: wgpu::VertexState { module: &shader_module, entry_point: Some(VERTEX_ENTRY), compilation_options: wgpu::PipelineCompilationOptions::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample,
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some(FRAGMENT_ENTRY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Whether `format` can be rendered to and bilinearly sampled on this adapter.
pub(crate) fn renderable_and_filterable(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> bool {
    let features = adapter.get_texture_format_features(format);
    features.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING) && features.flags.contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

/// Begins a single-attachment render pass on `view` that clears it to `clear`.
pub(crate) fn begin_pass<'e>(encoder: &'e mut wgpu::CommandEncoder, view: &wgpu::TextureView, label: &str, clear: wgpu::Color) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store } })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samplers_mirror_egui_wgpu() {
        let descriptor = egui_sampler_descriptor(egui::TextureOptions::NEAREST_REPEAT);
        assert_eq!(descriptor.mag_filter, wgpu::FilterMode::Nearest);
        assert_eq!(descriptor.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(egui_sampler_descriptor(egui::TextureOptions::LINEAR).min_filter, wgpu::FilterMode::Linear);
        assert_eq!(egui_sampler_descriptor(egui::TextureOptions::LINEAR_MIRRORED_REPEAT).address_mode_v, wgpu::AddressMode::MirrorRepeat);
    }

    /// A refused allocation is reported by `ErrorScopes` instead of reaching wgpu's default
    /// uncaptured-error handler (which panics), and the scopes leave the stack balanced.
    #[test]
    #[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
    fn error_scopes_capture_a_refused_allocation() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap_or_else(|error| panic!("no wgpu adapter: {error}"));
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap_or_else(|error| panic!("no wgpu device: {error}"));
        let too_large = device.limits().max_buffer_size.saturating_add(4);
        let scopes = ErrorScopes::push(&device);
        drop(device.create_buffer(&wgpu::BufferDescriptor { label: None, size: too_large, usage: wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        let error = pollster::block_on(scopes.pop());
        assert!(error.is_some(), "a buffer above max_buffer_size must be refused");
        assert!(with_error_scope(&device, || ()).is_ok(), "the scopes were popped");
    }
}
