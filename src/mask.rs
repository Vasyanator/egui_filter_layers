/*
File: mask.rs

Purpose:
The layer mask (F9): which part of a layer's rect the filter replaces. The public `Mask` is
either the analytic rounded rect (the default, drawn by the composite's SDF), an arbitrary
egui `Shape` in layer points, or a list of REGIONS (R1): shapes whose union is the mask and whose
list index is an id effects read with the prelude's `region()`. Shapes are tessellated here on the
calling thread (the GUI thread in the UI, the caller's thread offline) into `CoverageMeshes`; a
backend rasterizes those into a coverage texture at capture resolution (the coverage step of
`plan.rs`) and the composite reads its alpha as the mask; for regions the same meshes are drawn
twice more into a region-id texture (the plan's id passes).

Key structures:
- `Mask` (public): `RoundedRect(CornerRadius)` / `Shape(egui::Shape)` / `Regions(Vec<Region>)`.
- `Region` (public): one region of a `Mask::Regions` mask (an egui shape in layer points).
- `MaskError` (public): a shape the mask cannot use (paint callbacks anywhere; offline also text
  and textured meshes; more than `MAX_REGIONS` regions).
- `JobMask` (crate): the mask of one `LayerJob` / offline run, resolved (rounded rect radii, or
  tessellated meshes, or tessellated meshes per region).
- `CoverageMeshes` (crate): the tessellated shape, in layer points, one texture per mesh;
  `RegionMeshes`: one `CoverageMeshes` per region, in id order.
- `PackedCoverage` (crate): the meshes as one vertex / index upload (egui's 20-byte vertex layout,
  absolute `u32` indices) plus one draw per texture, shared by every backend; `PackedRegions`
  adds the draw range of every region (empty for a plain shape mask).

Key functions:
- `Mask::check()`: the UI's validation (no callbacks); `Mask::check_offline()`: also no text and
  no textured meshes (no font atlas offline).
- `tessellate()`: the one tessellation of a mask shape (both the UI and offline).
- `CoverageMeshes::pack()`, `RegionMeshes::pack()`, `JobMask::pack()` (what backends upload).

Notes:
Coverage is the ALPHA of the tessellated shape (vertex colour alpha x texture alpha), summed over
its meshes and capped at 1 (the backends' additive union): paint the mask opaque for a
full-strength filter; a translucent fill filters partially; abutting pieces leave no seam.
Regions: the union of all regions is the coverage; the region-id texture gives each pixel the id
of the LAST region (list order) covering it by more than half, else of the last one touching it
at all (so no pixel of the mask's antialiased edge is left without an id). The feathered edge
of egui's tessellator is the anti-aliasing. The tessellation uses the given pixels per point
(the context's in the UI) WITHOUT egui's pre-rasterized discs, so solid shapes never sample the
font atlas and the UI and offline runs tessellate identical meshes; text samples the atlas (UI
only). Under a layer transform the mesh is mapped by that transform at paint time, so the
feathered edge scales with the zoom (tessellation happens before the transform is known).
*/

use std::sync::Arc;

use egui::epaint::{ClippedShape, Mesh, Primitive, TessellationOptions, Tessellator};
use std::ops::Range;

use egui::{Color32, CornerRadius, Rect, Shape, TextureId};

/// The largest number of regions a [`Mask::Regions`] mask may have: ids are `0..MAX_REGIONS`,
/// stored in one byte of the region-id texture, whose value 255 is [`REGION_NONE`].
pub const MAX_REGIONS: usize = 255;

/// The region id the prelude's `region()` / `region_at(uv)` return outside every region and on
/// layers without regions (the WGSL prelude's `REGION_NONE`).
pub const REGION_NONE: u32 = 255;

/// One region of a [`Mask::Regions`] mask: an egui shape in LAYER points whose alpha is its
/// coverage (as [`Mask::Shape`]). Its index in the region list is its id (the prelude's
/// `region()`); where regions overlap, the later one in the list owns the pixels.
///
/// ```no_run
/// # fn demo(ui: &mut egui::Ui, rect: egui::Rect) {
/// use egui_filter_layers::{FilterLayer, Region, presets};
/// let (left, right) = rect.split_left_right_at_fraction(0.5);
/// FilterLayer::new(rect, presets::gaussian_blur(6.0)).regions([Region::rounded_rect(left, 8), Region::rounded_rect(right, 8)]).paint(ui.painter());
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Region {
    /// The region's shape, in layer points.
    shape: Shape,
}

impl Region {
    /// A rounded rect region (`rect` in layer points), opaque: full coverage inside, egui's
    /// one-pixel antialiased edge outside.
    #[must_use]
    pub fn rounded_rect(rect: Rect, corner_radius: impl Into<CornerRadius>) -> Self {
        Self { shape: Shape::rect_filled(rect, corner_radius, Color32::WHITE) }
    }

    /// Any egui shape in layer points; its alpha is its coverage (as [`Mask::Shape`]: paint it
    /// opaque for a full-strength filter; paint callbacks are refused by [`Mask::check`]).
    #[must_use]
    pub fn shape(shape: Shape) -> Self {
        Self { shape }
    }

    /// The region's shape.
    pub fn shape_ref(&self) -> &Shape {
        &self.shape
    }
}

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
    /// anti-aliases it. Paint callbacks are rejected ([`MaskError::Callback`]). The coverage of
    /// several pieces is the SUM of their alphas, capped at 1: pieces meeting edge to edge leave no
    /// seam (their feathered edges are complementary and add up to full coverage), overlapping
    /// translucent pieces add up. egui fills convex paths only: build a concave outline from convex
    /// pieces (abutting or overlapping) or from a mesh.
    Shape(Shape),
    /// Several REGIONS (at most [`MAX_REGIONS`]): their union is the mask (coverage as for
    /// [`Mask::Shape`], so abutting regions leave no seam), and an effect reads which region a
    /// pixel belongs to with the prelude's `region()` / `region_at(uv)` (the region's index in
    /// this list; `REGION_NONE` outside every region), e.g. to index per-region parameters
    /// (`p.tint[region()]`). Where regions overlap, the LATER one owns the pixels; the boundary
    /// between two regions is pixel-hard at the later region's 50 % coverage contour. An empty
    /// list is a mask covering nothing. `FilterLayer::regions` is the shorthand.
    Regions(Vec<Region>),
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
    /// A [`Mask::Regions`] mask has more regions than ids exist (ids are one byte, 255 = none).
    #[error("a region mask has {count} regions; at most {max} are supported")]
    TooManyRegions {
        /// How many regions the mask has.
        count: usize,
        /// [`MAX_REGIONS`].
        max: usize,
    },
}

impl Mask {
    /// Whether the UI can rasterize this mask: every shape (of every region) must contain no
    /// paint callback and no malformed mesh, and a region mask has at most [`MAX_REGIONS`]
    /// regions.
    ///
    /// # Errors
    /// [`MaskError::Callback`], [`MaskError::InvalidMesh`] or [`MaskError::TooManyRegions`].
    pub fn check(&self) -> Result<(), MaskError> {
        if let Self::Regions(regions) = self
            && regions.len() > MAX_REGIONS
        {
            return Err(MaskError::TooManyRegions { count: regions.len(), max: MAX_REGIONS });
        }
        self.visit_shapes(&mut |shape| match shape {
            Shape::Callback(_) => Err(MaskError::Callback),
            Shape::Mesh(mesh) if !mesh_is_drawable(mesh) => Err(MaskError::InvalidMesh { vertices: mesh.vertices.len(), indices: mesh.indices.len() }),
            _ => Ok(()),
        })
    }

    /// Whether an offline run can rasterize this mask: [`Mask::check`], plus no text and only
    /// untextured meshes (offline runs have no font atlas), in every region too.
    ///
    /// # Errors
    /// [`MaskError::Callback`], [`MaskError::InvalidMesh`], [`MaskError::TooManyRegions`],
    /// [`MaskError::Text`] or [`MaskError::TexturedMesh`].
    pub fn check_offline(&self) -> Result<(), MaskError> {
        self.check()?;
        self.visit_shapes(&mut |shape| match shape {
            Shape::Text(_) => Err(MaskError::Text),
            Shape::Mesh(mesh) if mesh.texture_id != TextureId::default() || mesh.vertices.iter().any(|vertex| vertex.uv != egui::epaint::WHITE_UV) => Err(MaskError::TexturedMesh { texture: mesh.texture_id }),
            _ => Ok(()),
        })
    }

    /// Calls `check` on every shape of the mask (each region's, nested `Shape::Vec`s depth
    /// first), stopping at the first error. A rounded rect has none.
    fn visit_shapes(&self, check: &mut impl FnMut(&Shape) -> Result<(), MaskError>) -> Result<(), MaskError> {
        match self {
            Self::RoundedRect(_) => Ok(()),
            Self::Shape(shape) => visit(shape, check),
            Self::Regions(regions) => regions.iter().try_for_each(|region| visit(&region.shape, check)),
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

/// The tessellated regions of a [`Mask::Regions`] mask, in id order (index = region id).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RegionMeshes {
    /// One entry per region (possibly without meshes: culled or empty).
    pub regions: Vec<CoverageMeshes>,
}

/// The mask of one layer run, resolved for the backend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JobMask {
    /// The analytic rounded rect (the composite's SDF; no coverage texture).
    RoundedRect(CornerRadius),
    /// A shape, tessellated: a coverage step rasterizes it (`plan::ChainPlan::with_coverage`).
    Coverage(Arc<CoverageMeshes>),
    /// Regions, tessellated: the coverage step rasterizes their union, the id passes their ids
    /// (`plan::ChainPlan::with_regions`).
    Regions(Arc<RegionMeshes>),
}

impl JobMask {
    /// The rounded-rect radii of the analytic mask (`ZERO` for a shape or region mask, whose SDF
    /// inputs the composite ignores).
    #[must_use]
    pub(crate) fn corner_radius(&self) -> CornerRadius {
        match self {
            Self::RoundedRect(radius) => *radius,
            Self::Coverage(_) | Self::Regions(_) => CornerRadius::ZERO,
        }
    }

    /// The meshes of a shape mask (`None` for a rounded rect or regions).
    #[cfg_attr(not(test), allow(dead_code))] // Backends use `pack` / `textures`; tests inspect the meshes.
    #[must_use]
    pub(crate) fn coverage(&self) -> Option<&CoverageMeshes> {
        match self {
            Self::RoundedRect(_) | Self::Regions(_) => None,
            Self::Coverage(meshes) => Some(meshes),
        }
    }

    /// The regions of a region mask (`None` otherwise).
    #[cfg_attr(not(feature = "glow"), allow(dead_code))] // glow refuses region masks by this (until it runs them).
    #[must_use]
    pub(crate) fn regions(&self) -> Option<&RegionMeshes> {
        match self {
            Self::RoundedRect(_) | Self::Coverage(_) => None,
            Self::Regions(regions) => Some(regions),
        }
    }

    /// What a backend uploads for this mask: `None` for the rounded rect (no coverage step), the
    /// packed shape (no region ranges) or the packed regions.
    #[cfg_attr(not(any(feature = "glow", feature = "wgpu")), allow(dead_code))] // Only backends upload masks.
    #[must_use]
    pub(crate) fn pack(&self) -> Option<PackedRegions> {
        match self {
            Self::RoundedRect(_) => None,
            Self::Coverage(meshes) => Some(PackedRegions { packed: meshes.pack(), regions: Vec::new() }),
            Self::Regions(regions) => Some(regions.pack()),
        }
    }

    /// Every texture the mask's meshes sample (every region's), in mesh order, with repeats.
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // glow resolves the packed draws' textures instead.
    pub(crate) fn textures(&self) -> Vec<TextureId> {
        match self {
            Self::RoundedRect(_) => Vec::new(),
            Self::Coverage(meshes) => meshes.textures().collect(),
            Self::Regions(regions) => regions.regions.iter().flat_map(CoverageMeshes::textures).collect(),
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

/// Tessellates every region of a [`Mask::Regions`] mask with [`tessellate`] (same arguments), in
/// id order: entry `i` holds region `i`'s meshes (empty when it lies outside `rect`).
#[must_use]
pub(crate) fn tessellate_regions(regions: Vec<Region>, rect: Rect, pixels_per_point: f32, options: TessellationOptions, font_tex_size: [usize; 2]) -> RegionMeshes {
    RegionMeshes { regions: regions.into_iter().map(|region| tessellate(region.shape, rect, pixels_per_point, options, font_tex_size)).collect() }
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
    /// The draws, in paint order (consecutive meshes with the same texture merged, never across
    /// two regions).
    pub draws: Vec<CoverageDraw>,
}

/// A packed mask plus, for a region mask, which draws belong to which region: `regions[id]` is a
/// range of `packed.draws` (the ranges partition the draws in id order; a region without meshes
/// has an empty range). `regions` is empty for a plain shape mask.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PackedRegions {
    /// Every mesh of every region, in id order (the coverage step draws all of them).
    pub packed: PackedCoverage,
    /// The draw range of each region (the id passes draw `packed.draws[range]` per region).
    pub regions: Vec<Range<usize>>,
}

impl RegionMeshes {
    /// Packs every region's meshes into one upload, in id order, and records each region's draw
    /// range (draws never merge across two regions).
    #[must_use]
    pub(crate) fn pack(&self) -> PackedRegions {
        let mut out = PackedRegions::default();
        for region in &self.regions {
            let start = out.packed.draws.len();
            region.append_to(&mut out.packed);
            out.regions.push(start..out.packed.draws.len());
        }
        out
    }
}

impl CoverageMeshes {
    /// Every texture the meshes sample.
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // glow resolves the packed draws' textures instead.
    pub(crate) fn textures(&self) -> impl Iterator<Item = TextureId> + '_ {
        self.meshes.iter().map(|mesh| mesh.texture_id)
    }

    /// Packs the meshes for upload (see [`CoverageMeshes::append_to`]).
    #[must_use]
    pub(crate) fn pack(&self) -> PackedCoverage {
        let mut packed = PackedCoverage::default();
        self.append_to(&mut packed);
        packed
    }

    /// Appends the meshes to `packed` (after what it holds); consecutive meshes of THIS call with
    /// the same texture share a draw, a draw already in `packed` is never extended. A mesh whose
    /// vertices or indices would overflow `u32` addressing is skipped (never reached by real
    /// masks: 4 G vertices), and so is a malformed or empty one (an index past its vertices, a
    /// partial triangle, no triangles), so no index outside the upload ever reaches a GPU draw
    /// ([`Mask::check`] rejects malformed user meshes before this; the skip is the defensive half).
    pub(crate) fn append_to(&self, packed: &mut PackedCoverage) {
        let first_draw = packed.draws.len();
        // `packed` only ever grows through this function, which keeps both counts within `u32`.
        let (Ok(mut vertex_count), Ok(mut index_count)) = (u32::try_from(packed.vertices.len() / VERTEX_BYTES), u32::try_from(packed.indices.len() / 4)) else {
            return;
        };
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
            let own_last = packed.draws.len().checked_sub(1).filter(|&last| last >= first_draw);
            match own_last.and_then(|last| packed.draws.get_mut(last)) {
                Some(draw) if draw.texture == mesh.texture_id => draw.index_count += mesh_indices,
                Some(_) | None => packed.draws.push(CoverageDraw { texture: mesh.texture_id, first_index: index_count, index_count: mesh_indices }),
            }
            (vertex_count, index_count) = (next_vertices, next_indices);
        }
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

    /// R1: regions tessellate in id order (a region outside the rect keeps its slot, empty),
    /// `pack` ranges partition the draws in order and never merge two regions' draws, and the
    /// union keeps every region's meshes.
    #[test]
    fn regions_tessellate_in_order_and_pack_into_partitioning_ranges() {
        let regions = vec![
            Region::rounded_rect(Rect::from_min_max(pos2(10.0, 10.0), pos2(60.0, 60.0)), 0),
            Region::shape(Shape::circle_filled(pos2(500.0, 500.0), 5.0, Color32::WHITE)),
            Region::rounded_rect(Rect::from_min_max(pos2(60.0, 10.0), pos2(110.0, 60.0)), 4),
            Region::shape(Shape::Vec(vec![Shape::circle_filled(pos2(30.0, 90.0), 8.0, Color32::WHITE), Shape::circle_filled(pos2(80.0, 90.0), 8.0, Color32::WHITE)])),
        ];
        let meshes = tessellate_regions(regions, rect(), 1.0, TessellationOptions::default(), [1, 1]);
        assert_eq!(meshes.regions.len(), 4);
        assert!(meshes.regions[1].meshes.is_empty(), "outside the rect: culled, the id stays");
        assert!([0, 2, 3].iter().all(|&id| !meshes.regions[id].meshes.is_empty()));
        let packed = meshes.pack();
        assert_eq!(packed.regions.len(), 4);
        assert_eq!(packed.regions[0].start, 0);
        assert!(packed.regions.windows(2).all(|pair| pair[0].end == pair[1].start), "contiguous: {:?}", packed.regions);
        assert_eq!(packed.regions[3].end, packed.packed.draws.len());
        assert!(packed.regions[1].is_empty());
        // Every region's meshes share the default texture, yet each region has its own draw.
        assert!([0, 2, 3].iter().all(|&id| packed.regions[id].len() == 1), "{:?}", packed.regions);
        let whole: usize = meshes.regions.iter().map(|region| region.pack().indices.len()).sum();
        assert_eq!(packed.packed.indices.len(), whole, "the union holds every region's indices");
        let first = &packed.packed.draws[packed.regions[2].start];
        assert_eq!(first.first_index * 4, u32::try_from(meshes.regions[0].pack().indices.len()).unwrap_or(0), "region 2's draw starts after region 0's indices");
        assert_eq!(JobMask::Regions(Arc::new(meshes)).pack().map(|packed| packed.regions.len()), Some(4));
    }

    /// R1: more than `MAX_REGIONS` regions are refused (UI and offline); an empty list is valid;
    /// every region's shape is checked.
    #[test]
    fn region_masks_are_checked_per_region() {
        let square = Region::rounded_rect(rect(), 0);
        assert_eq!(Mask::Regions(Vec::new()).check(), Ok(()));
        assert_eq!(Mask::Regions(vec![square.clone(); MAX_REGIONS]).check(), Ok(()));
        let too_many = Mask::Regions(vec![square.clone(); MAX_REGIONS + 1]);
        assert_eq!(too_many.check(), Err(MaskError::TooManyRegions { count: 256, max: 255 }));
        assert_eq!(too_many.check_offline(), Err(MaskError::TooManyRegions { count: 256, max: 255 }));
        let callback = Shape::Callback(egui::PaintCallback { rect: rect(), callback: Arc::new(()) });
        assert_eq!(Mask::Regions(vec![square.clone(), Region::shape(callback)]).check(), Err(MaskError::Callback));
        let ctx = egui::Context::default();
        let mut text = Shape::Noop;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| text = Shape::galley(Pos2::ZERO, ui.painter().layout_no_wrap("R".to_owned(), egui::FontId::default(), Color32::WHITE), Color32::WHITE));
        output.textures_delta.clear();
        let with_text = Mask::Regions(vec![square, Region::shape(text)]);
        assert_eq!(with_text.check(), Ok(()));
        assert_eq!(with_text.check_offline(), Err(MaskError::Text));
        assert_eq!(REGION_NONE, 255);
        assert_eq!(JobMask::Regions(Arc::new(RegionMeshes::default())).corner_radius(), CornerRadius::ZERO);
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
