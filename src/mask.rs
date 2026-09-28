/*
File: mask.rs

Purpose:
The layer mask (F9): which part of a layer's rect the filter replaces. The public `Mask` is
either the analytic rounded rect (the default, drawn by the composite's SDF) or an arbitrary
egui `Shape` in layer points, which is tessellated here on the calling thread (the GUI thread in
the UI, the caller's thread offline) into `CoverageMeshes`; a backend rasterizes those into a
coverage texture at capture resolution (the coverage step of `plan.rs`) and the composite reads
its alpha as the mask.

Key structures:
- `Mask` (public): `RoundedRect(CornerRadius)` / `Shape(egui::Shape)`.
- `MaskError` (public): a shape the mask cannot use (paint callbacks anywhere; offline also text
  and textured meshes).
- `JobMask` (crate): the mask of one `LayerJob` / offline run, resolved (rounded rect radii, or
  tessellated meshes).
- `CoverageMeshes` (crate): the tessellated shape, in layer points, one texture per mesh.
- `PackedCoverage` (crate): the meshes as one vertex / index upload (egui's 20-byte vertex layout,
  absolute `u32` indices) plus one draw per texture, shared by every backend.

Key functions:
- `Mask::check()`: the UI's validation (no callbacks); `Mask::check_offline()`: also no text and
  no textured meshes (no font atlas offline).
- `tessellate()`: the one tessellation of a mask shape (both the UI and offline).
- `CoverageMeshes::pack()`.

Notes:
Coverage is the ALPHA of the tessellated shape (vertex colour alpha x texture alpha): paint the
mask opaque for a full-strength filter; a translucent fill filters partially. The feathered edge
of egui's tessellator is the anti-aliasing. The tessellation uses the given pixels per point
(the context's in the UI) WITHOUT egui's pre-rasterized discs, so solid shapes never sample the
font atlas and the UI and offline runs tessellate identical meshes; text samples the atlas (UI
only). Under a layer transform the mesh is mapped by that transform at paint time, so the
feathered edge scales with the zoom (tessellation happens before the transform is known).
*/

use std::sync::Arc;

use egui::epaint::{ClippedShape, Mesh, Primitive, TessellationOptions, Tessellator};
use egui::{CornerRadius, Rect, Shape, TextureId};

/// Which part of the layer rect the filter replaces; outside it the backdrop stays untouched.
/// The mask is always intersected with the layer rect.
///
/// ```no_run
/// # fn demo(ui: &mut egui::Ui, rect: egui::Rect) {
/// use egui_filter_layers::{FilterLayer, Mask, presets};
/// let circle = egui::Shape::circle_filled(rect.center(), rect.height() * 0.5, egui::Color32::WHITE);
/// FilterLayer::new(rect, presets::gaussian_blur(6.0)).mask(Mask::Shape(circle)).paint(ui.painter());
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Mask {
    /// The layer rect with rounded corners (layer points), anti-aliased over one pixel. The
    /// default ([`CornerRadius::ZERO`]: the plain rect); what `FilterLayer::corner_radius` sets.
    RoundedRect(CornerRadius),
    /// Any egui shape in LAYER points (the space the layer's rect is in): paths, circles, rects,
    /// meshes, text galleys (UI only), or a `Shape::Vec` of them. Its alpha is the coverage:
    /// paint it opaque (e.g. `Color32::WHITE`) for a full-strength filter; egui's feathered edge
    /// anti-aliases it. Paint callbacks are rejected ([`MaskError::Callback`]). egui fills convex
    /// paths only: build a concave outline from OVERLAPPING convex pieces (pieces meeting edge to
    /// edge leave a seam where both feathered edges fade) or from a mesh.
    Shape(Shape),
}

impl Default for Mask {
    fn default() -> Self {
        Self::RoundedRect(CornerRadius::ZERO)
    }
}

/// Why a [`Mask`] cannot be used.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MaskError {
    /// The shape contains a paint callback, which has no tessellated geometry.
    #[error("a mask shape cannot contain a paint callback (it has no geometry to rasterize)")]
    Callback,
    /// A `Shape::Mesh` is malformed: an index past its vertices (`Mesh::is_valid` is false) or an
    /// index count that is not a multiple of 3. Drawing it would address vertices it does not own.
    #[error("a mask mesh is invalid: {indices} indices (must be a multiple of 3, each below the vertex count) over {vertices} vertices")]
    InvalidMesh {
        /// The mesh's vertex count.
        vertices: usize,
        /// The mesh's index count.
        indices: usize,
    },
    /// Offline: the shape contains text. Text needs egui's font atlas, which an offline run does
    /// not have (a documented 0.2 limitation; the UI supports text masks).
    #[error("text masks are not supported offline (they need egui's font atlas); convert the text to a path or mesh first")]
    Text,
    /// Offline: a `Shape::Mesh` samples a texture (another texture id than the default, or uvs
    /// other than `epaint::WHITE_UV`); offline masks are untextured.
    #[error("offline mask meshes must be untextured (texture {texture:?} with uvs other than epaint::WHITE_UV)")]
    TexturedMesh {
        /// The mesh's texture.
        texture: TextureId,
    },
}

impl Mask {
    /// Whether the UI can rasterize this mask: a shape must contain no paint callback and no
    /// malformed mesh.
    ///
    /// # Errors
    /// [`MaskError::Callback`] or [`MaskError::InvalidMesh`].
    pub fn check(&self) -> Result<(), MaskError> {
        match self {
            Self::RoundedRect(_) => Ok(()),
            Self::Shape(shape) => visit(shape, &mut |shape| match shape {
                Shape::Callback(_) => Err(MaskError::Callback),
                Shape::Mesh(mesh) if !mesh_is_drawable(mesh) => Err(MaskError::InvalidMesh { vertices: mesh.vertices.len(), indices: mesh.indices.len() }),
                _ => Ok(()),
            }),
        }
    }

    /// Whether an offline run can rasterize this mask: [`Mask::check`], plus no text and only
    /// untextured meshes (offline runs have no font atlas).
    ///
    /// # Errors
    /// [`MaskError::Callback`], [`MaskError::InvalidMesh`], [`MaskError::Text`] or
    /// [`MaskError::TexturedMesh`].
    pub fn check_offline(&self) -> Result<(), MaskError> {
        self.check()?;
        match self {
            Self::RoundedRect(_) => Ok(()),
            Self::Shape(shape) => visit(shape, &mut |shape| match shape {
                Shape::Text(_) => Err(MaskError::Text),
                Shape::Mesh(mesh) if mesh.texture_id != TextureId::default() || mesh.vertices.iter().any(|vertex| vertex.uv != egui::epaint::WHITE_UV) => {
                    Err(MaskError::TexturedMesh { texture: mesh.texture_id })
                }
                _ => Ok(()),
            }),
        }
    }
}

/// Whether every index of `mesh` addresses one of its own vertices and the indices form whole
/// triangles (a draw of it, or of meshes merged after it, then never reads a foreign vertex).
fn mesh_is_drawable(mesh: &Mesh) -> bool {
    mesh.indices.len().is_multiple_of(3) && mesh.is_valid()
}

/// Calls `check` on `shape` and, for a `Shape::Vec`, on every nested shape (depth first).
fn visit(shape: &Shape, check: &mut impl FnMut(&Shape) -> Result<(), MaskError>) -> Result<(), MaskError> {
    check(shape)?;
    if let Shape::Vec(shapes) = shape {
        for shape in shapes {
            visit(shape, check)?;
        }
    }
    Ok(())
}

/// The tessellated mask shape: meshes in LAYER points (the space of the layer's rect), each with
/// one texture (the font atlas for text, egui's white pixel of it, or a mesh's own texture).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CoverageMeshes {
    /// Non-empty meshes, in paint order.
    pub meshes: Vec<Mesh>,
}

/// The mask of one layer run, resolved for the backend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JobMask {
    /// The analytic rounded rect (the composite's SDF; no coverage texture).
    RoundedRect(CornerRadius),
    /// A shape, tessellated: a coverage step rasterizes it (`plan::ChainPlan::with_coverage`).
    Coverage(Arc<CoverageMeshes>),
}

impl JobMask {
    /// The rounded-rect radii of the analytic mask (`ZERO` for a shape mask, whose SDF inputs the
    /// composite ignores).
    #[must_use]
    pub(crate) fn corner_radius(&self) -> CornerRadius {
        match self {
            Self::RoundedRect(radius) => *radius,
            Self::Coverage(_) => CornerRadius::ZERO,
        }
    }

    /// The meshes of a shape mask.
    #[must_use]
    pub(crate) fn coverage(&self) -> Option<&CoverageMeshes> {
        match self {
            Self::RoundedRect(_) => None,
            Self::Coverage(meshes) => Some(meshes),
        }
    }
}

/// Tessellates `shape` (layer points) for a layer whose rect is `rect`, at `pixels_per_point`
/// (feathering: one physical pixel at this scale) with `options`, WITHOUT pre-rasterized discs
/// (see the file notes). `font_tex_size` normalizes text uvs (the font atlas size; any value for a
/// shape without text). Shapes wholly outside `rect` (expanded by two points) are culled; callback
/// primitives are dropped (callers reject them first with [`Mask::check`]).
#[must_use]
pub(crate) fn tessellate(shape: Shape, rect: Rect, pixels_per_point: f32, options: TessellationOptions, font_tex_size: [usize; 2]) -> CoverageMeshes {
    let mut tessellator = Tessellator::new(pixels_per_point, options, font_tex_size, Vec::new());
    // The clip rect only culls: coverage outside the layer rect is never composited.
    let primitives = tessellator.tessellate_shapes(vec![ClippedShape { clip_rect: rect.expand(2.0), shape }]);
    let meshes = primitives
        .into_iter()
        .filter_map(|primitive| match primitive.primitive {
            Primitive::Mesh(mesh) if !mesh.indices.is_empty() => Some(mesh),
            Primitive::Mesh(_) | Primitive::Callback(_) => None,
        })
        .collect();
    CoverageMeshes { meshes }
}

/// `shaders/mesh.wgsl`: draws egui meshes; the coverage rasterizer on every backend (glow
/// translates it with naga) and the wgpu backdrop re-render.
#[cfg(any(feature = "glow", feature = "wgpu"))]
pub(crate) const MESH_WGSL: &str = include_str!("shaders/mesh.wgsl");

/// Size in bytes of one packed vertex: egui's `epaint::Vertex` (pos `f32x2`, uv `f32x2`, colour
/// as four bytes `r, g, b, a` read as one `u32`, i.e. `mesh.wgsl`'s `unpack_color`).
pub(crate) const VERTEX_BYTES: usize = 20;

/// One draw of a packed coverage upload: the indices of the meshes sharing one texture run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CoverageDraw {
    /// The texture every vertex of the draw samples.
    pub texture: TextureId,
    /// First index (into the index upload, in indices).
    pub first_index: u32,
    /// Number of indices.
    pub index_count: u32,
}

/// Every mesh of a mask as one vertex upload and one index upload; indices are absolute (into
/// the whole vertex upload), so a draw needs no base vertex (GLES 3.0 has none).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PackedCoverage {
    /// `VERTEX_BYTES` per vertex, native endianness (egui's own vertex bytes).
    pub vertices: Vec<u8>,
    /// `u32` indices, native endianness.
    pub indices: Vec<u8>,
    /// The draws, in paint order (consecutive meshes with the same texture merged).
    pub draws: Vec<CoverageDraw>,
}

impl CoverageMeshes {
    /// Every texture the meshes sample.
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // glow resolves the packed draws' textures instead.
    pub(crate) fn textures(&self) -> impl Iterator<Item = TextureId> + '_ {
        self.meshes.iter().map(|mesh| mesh.texture_id)
    }

    /// Packs the meshes for upload. A mesh whose vertices or indices would overflow `u32`
    /// addressing is skipped (never reached by real masks: 4 G vertices), and so is a malformed
    /// or empty one (an index past its vertices, a partial triangle, no triangles), so no index
    /// outside the upload ever reaches a GPU draw ([`Mask::check`] rejects malformed user meshes
    /// before this; the skip is the defensive half).
    #[must_use]
    pub(crate) fn pack(&self) -> PackedCoverage {
        let mut packed = PackedCoverage::default();
        let (mut vertex_count, mut index_count) = (0_u32, 0_u32);
        for mesh in &self.meshes {
            if mesh.indices.is_empty() || !mesh_is_drawable(mesh) {
                continue;
            }
            let (Ok(mesh_vertices), Ok(mesh_indices)) = (u32::try_from(mesh.vertices.len()), u32::try_from(mesh.indices.len())) else {
                continue;
            };
            let (Some(next_vertices), Some(next_indices)) = (vertex_count.checked_add(mesh_vertices), index_count.checked_add(mesh_indices)) else {
                continue;
            };
            for vertex in &mesh.vertices {
                for value in [vertex.pos.x, vertex.pos.y, vertex.uv.x, vertex.uv.y] {
                    packed.vertices.extend_from_slice(&value.to_ne_bytes());
                }
                packed.vertices.extend_from_slice(&vertex.color.to_array());
            }
            // `mesh_is_drawable`: every index is below `mesh_vertices`, so the sum stays below
            // `next_vertices` (checked above) and inside this mesh's own vertices.
            for &index in &mesh.indices {
                packed.indices.extend_from_slice(&(vertex_count + index).to_ne_bytes());
            }
            match packed.draws.last_mut() {
                Some(draw) if draw.texture == mesh.texture_id => draw.index_count += mesh_indices,
                Some(_) | None => packed.draws.push(CoverageDraw { texture: mesh.texture_id, first_index: index_count, index_count: mesh_indices }),
            }
            (vertex_count, index_count) = (next_vertices, next_indices);
        }
        packed
    }
}

#[cfg(test)]
mod tests {
    use egui::epaint::{PathShape, Vertex, WHITE_UV};
    use egui::{Color32, Pos2, Stroke, pos2, vec2};

    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(pos2(10.0, 10.0), vec2(100.0, 100.0))
    }

    #[test]
    fn callbacks_are_rejected_everywhere_text_and_textures_offline_only() {
        let circle = Shape::circle_filled(pos2(50.0, 50.0), 20.0, Color32::WHITE);
        assert_eq!(Mask::default(), Mask::RoundedRect(CornerRadius::ZERO));
        assert_eq!(Mask::Shape(circle.clone()).check(), Ok(()));
        assert_eq!(Mask::Shape(circle.clone()).check_offline(), Ok(()));
        let callback = Shape::Callback(egui::PaintCallback { rect: rect(), callback: Arc::new(()) });
        let nested = Shape::Vec(vec![circle.clone(), Shape::Vec(vec![callback])]);
        assert_eq!(Mask::Shape(nested.clone()).check(), Err(MaskError::Callback));
        assert_eq!(Mask::Shape(nested).check_offline(), Err(MaskError::Callback));

        let mut textured = Mesh::with_texture(TextureId::User(3));
        textured.add_rect_with_uv(rect(), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        assert_eq!(Mask::Shape(Shape::mesh(textured)).check(), Ok(()));
        let mut atlas_uv = Mesh::default();
        atlas_uv.vertices.push(Vertex { pos: Pos2::ZERO, uv: pos2(0.5, 0.5), color: Color32::WHITE });
        assert_eq!(Mask::Shape(Shape::mesh(atlas_uv)).check_offline(), Err(MaskError::TexturedMesh { texture: TextureId::default() }));
        let mut plain = Mesh::default();
        plain.add_colored_rect(rect(), Color32::WHITE);
        assert_eq!(Mask::Shape(Shape::Vec(vec![circle, Shape::mesh(plain)])).check_offline(), Ok(()));
    }

    /// Solid shapes tessellate to meshes on the default texture at `WHITE_UV` only (no
    /// pre-rasterized discs), with a feathered (partially transparent) edge.
    #[test]
    fn solid_shapes_tessellate_untextured_with_a_feathered_edge() {
        let shapes = Shape::Vec(vec![
            Shape::circle_filled(pos2(40.0, 40.0), 3.0, Color32::WHITE),
            Shape::Path(PathShape::convex_polygon(vec![pos2(20.0, 20.0), pos2(80.0, 20.0), pos2(50.0, 90.0)], Color32::WHITE, Stroke::NONE)),
        ]);
        let meshes = tessellate(shapes, rect(), 2.0, TessellationOptions::default(), [1, 1]);
        assert!(!meshes.meshes.is_empty());
        assert!(meshes.meshes.iter().all(|mesh| mesh.texture_id == TextureId::default() && mesh.vertices.iter().all(|vertex| vertex.uv == WHITE_UV)));
        let alphas: Vec<u8> = meshes.meshes.iter().flat_map(|mesh| mesh.vertices.iter().map(|vertex| vertex.color.a())).collect();
        assert!(alphas.contains(&255) && alphas.contains(&0), "opaque interior and a transparent feather: {alphas:?}");
        let outside = tessellate(Shape::circle_filled(pos2(500.0, 500.0), 5.0, Color32::WHITE), rect(), 1.0, TessellationOptions::default(), [1, 1]);
        assert!(outside.meshes.is_empty(), "culled outside the layer rect");
    }

    /// Malformed user meshes are rejected by both checks and skipped by the packer, so no index
    /// past the vertex upload reaches a draw.
    #[test]
    fn malformed_meshes_are_rejected_and_never_packed() {
        let no_vertices = Mesh { indices: vec![0, 1, 2], ..Mesh::default() };
        let out_of_range = {
            let mut mesh = Mesh::default();
            mesh.add_colored_rect(rect(), Color32::WHITE);
            mesh.indices.push(7);
            mesh.indices.extend([0, 1]);
            mesh
        };
        let partial = {
            let mut mesh = Mesh::default();
            mesh.add_colored_rect(rect(), Color32::WHITE);
            mesh.indices.push(0);
            mesh
        };
        // Built as `Shape::Mesh` directly: `Shape::mesh` debug-asserts validity itself.
        for mesh in [&no_vertices, &out_of_range, &partial] {
            let error = MaskError::InvalidMesh { vertices: mesh.vertices.len(), indices: mesh.indices.len() };
            let shape = Shape::Vec(vec![Shape::circle_filled(pos2(50.0, 50.0), 5.0, Color32::WHITE), Shape::Mesh(Arc::new(mesh.clone()))]);
            assert_eq!(Mask::Shape(shape.clone()).check(), Err(error.clone()));
            assert_eq!(Mask::Shape(shape).check_offline(), Err(error));
        }
        let mut valid = Mesh::default();
        valid.add_colored_rect(rect(), Color32::WHITE);
        let packed = CoverageMeshes { meshes: vec![no_vertices, valid, out_of_range, partial, Mesh::default()] }.pack();
        assert_eq!(packed.vertices.len(), 4 * VERTEX_BYTES, "only the valid mesh is uploaded");
        assert_eq!(packed.draws, [CoverageDraw { texture: TextureId::default(), first_index: 0, index_count: 6 }]);
        assert!(packed.indices.chunks_exact(4).all(|bytes| u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) < 4));
    }

    #[test]
    fn packing_makes_indices_absolute_and_merges_draws_per_texture() {
        let mut first = Mesh::default();
        first.add_colored_rect(rect(), Color32::from_rgba_premultiplied(1, 2, 3, 4));
        let mut second = Mesh::default();
        second.add_colored_rect(rect(), Color32::WHITE);
        let mut third = Mesh::with_texture(TextureId::User(1));
        third.add_rect_with_uv(rect(), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        let packed = CoverageMeshes { meshes: vec![first, second, third] }.pack();
        assert_eq!(packed.vertices.len(), 12 * VERTEX_BYTES);
        assert_eq!(packed.draws, [CoverageDraw { texture: TextureId::default(), first_index: 0, index_count: 12 }, CoverageDraw { texture: TextureId::User(1), first_index: 12, index_count: 6 }]);
        let index = |at: usize| u32::from_ne_bytes([packed.indices[at * 4], packed.indices[at * 4 + 1], packed.indices[at * 4 + 2], packed.indices[at * 4 + 3]]);
        assert!((0..6).all(|at| index(at) < 4) && (6..12).all(|at| (4..8).contains(&index(at))) && (12..18).all(|at| (8..12).contains(&index(at))));
        // Vertex 0: pos (10, 10), uv (0, 0), colour bytes r, g, b, a.
        let f32_at = |at: usize| f32::from_ne_bytes([packed.vertices[at], packed.vertices[at + 1], packed.vertices[at + 2], packed.vertices[at + 3]]);
        assert!((f32_at(0) - 10.0).abs() < f32::EPSILON && (f32_at(4) - 10.0).abs() < f32::EPSILON && f32_at(8).abs() < f32::EPSILON);
        assert_eq!(&packed.vertices[16..20], &[1, 2, 3, 4]);
    }
}
