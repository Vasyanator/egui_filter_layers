//! Shared scene builder, timing loop and report of the `bench_many_elements` benchmarks in
//! `wgpu_headless.rs` and `glow_headless.rs` (roadmap §19.5, R2): N small frosted glass tiles,
//! each with its own tint, over a busy backdrop, painted in one of the [`Mode`]s. It only paints
//! into an egui `Painter`; each test file renders and reads back frames with its own helper, so
//! a timed frame includes egui's CPU work, tessellation, the GPU work and its completion.
//!
//! The UNIFORM modes run the SAME chain on every tile ([`glass_chain`]): a Gaussian blur, then a
//! tint read from `p.tint[region()]`, or `p.tint[p.index]` on a layer without regions. Only the
//! parameters, the number of layers and the number of entries differ, so the modes are
//! comparable. The MIXED modes give every odd tile a different chain (pixelate, [`PIXELATE_CELL`])
//! — the case entries exist for (R3); their Regions equivalent is one "uber" chain that blurs
//! every tile and branches on the region id ([`MIXED_TINT_WGSL`]).
//!
//! Included as `mod common;` by both files; each uses every item (a test crate warns on dead code).

use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Stroke, pos2, vec2};
use egui_shader_layers::{Effect, Entry, FilterChain, Overlap, Pass, Region, ShaderLayer, presets};

/// The element counts every benchmark runs.
pub const ELEMENT_COUNTS: [usize; 3] = [4, 16, 64];

/// Frames rendered before timing starts (pipelines, targets and caches warm).
pub const WARM_UP_FRAMES: u32 = 10;

/// Frames timed per row.
pub const TIMED_FRAMES: u32 = 100;

/// Tint slots of [`GLASS_TINT_WGSL`]: the largest element count benchmarked.
pub const TINT_SLOTS: usize = 64;

/// Tint strength of every tile (`0..1`).
const TINT_AMOUNT: f32 = 0.35;

/// Tile corner radius, points.
const TILE_CORNER: u8 = 3;

/// The tint pass: mixes the (blurred) backdrop towards `p.tint[i].rgb` by `p.tint[i].a`, where
/// `i` is the pixel's region id, or `p.index` on a layer without regions.
pub const GLASS_TINT_WGSL: &str = "struct Params { tint: array<vec4<f32>, 64>, index: u32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let r = region();
    let i = min(select(r, p.index, r == REGION_NONE), 63u);
    let t = p.tint[i];
    let c = unpremultiply(src(uv));
    return premultiply(vec4<f32>(mix(c.rgb, t.rgb, t.a), c.a));
}";

/// Cell size of the pixelate chain of the odd tiles in the mixed modes, points.
pub const PIXELATE_CELL: f32 = 4.0;

/// The mixed-modes Regions equivalent, run after the blur: even region ids get
/// [`GLASS_TINT_WGSL`]'s tint, odd ids pixelate the ORIGINAL (unblurred) backdrop with cells of
/// `p.cell` points aligned to the layer rect. The blur still runs under the odd tiles: that waste
/// is what one chain for two looks costs.
pub const MIXED_TINT_WGSL: &str = "struct Params { tint: array<vec4<f32>, 64>, index: u32, cell: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let r = region();
    let i = min(select(r, p.index, r == REGION_NONE), 63u);
    if (i % 2u) == 1u {
        let cell = max(p.cell * fl.pixels_per_point, 1.0) * fl.texel;
        let origin = fl.rect_uv.xy;
        return original((floor((uv - origin) / cell) + vec2<f32>(0.5)) * cell + origin);
    }
    let t = p.tint[i];
    let c = unpremultiply(src(uv));
    return premultiply(vec4<f32>(mix(c.rgb, t.rgb, t.a), c.a));
}";

/// How the N elements are painted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// N `ShaderLayer`s, one per tile (`p.index` selects its tint).
    Layers,
    /// ONE `ShaderLayer` over the tiles' union with a `Mask::Regions` of N regions (`region()`
    /// selects the tint).
    Regions,
    /// ONE `ShaderLayer` with N entries (`Overlap::Independent`), one per tile, each its own
    /// capture of its tile and the glass chain with its own `p.index`.
    Entries,
    /// As [`Mode::Entries`] with `Overlap::Stacked`. The tiles do not overlap, but every tile's
    /// padded capture (the blur radius) reaches its neighbours, so on wgpu each earlier neighbour
    /// is handed off into it: the cost of the stacked contract, not of overlap.
    EntriesStacked,
    /// N layers, odd tiles pixelate instead of the glass.
    LayersMixed,
    /// ONE layer, N regions, one chain: blur + [`MIXED_TINT_WGSL`].
    RegionsMixed,
    /// ONE layer, N entries (independent) alternating the glass chain and pixelate.
    EntriesMixed,
}

impl Mode {
    /// Every benchmarked mode, in table order.
    pub const ALL: [Self; 7] = [Self::Layers, Self::Regions, Self::Entries, Self::EntriesStacked, Self::LayersMixed, Self::RegionsMixed, Self::EntriesMixed];

    /// The table label.
    pub fn name(self) -> &'static str {
        match self {
            Self::Layers => "layers",
            Self::Regions => "regions",
            Self::Entries => "entries",
            Self::EntriesStacked => "entries-stacked",
            Self::LayersMixed => "layers-mixed",
            Self::RegionsMixed => "regions-mixed",
            Self::EntriesMixed => "entries-mixed",
        }
    }
}

/// `value` as `f32`, exactly: the scene's counts and grid indices are at most [`TINT_SLOTS`], far
/// below `u16::MAX` (saturates there, never reached).
fn small_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

/// The tint of tile `index` of `count`: hues spread around the colour wheel, alpha = strength.
pub fn tile_tint(index: usize, count: usize) -> [f32; 4] {
    let hue = small_f32(index) / small_f32(count.max(1));
    let channel = |offset: f32| {
        let phase = (hue + offset).rem_euclid(1.0);
        (1.0 - (phase * 6.0 - 3.0).abs().min(3.0) / 1.5).clamp(0.0, 1.0) * 0.8 + 0.2
    };
    [channel(0.0), channel(2.0 / 3.0), channel(1.0 / 3.0), TINT_AMOUNT]
}

/// The glass chain: a Gaussian blur (sigma 3 points) then [`GLASS_TINT_WGSL`] with the tints of
/// `count` tiles and `index` as the tint of a layer without regions.
///
/// # Panics
/// On a WGSL or parameter error: the snippet and its parameters are fixed test data.
pub fn glass_chain(count: usize, index: u32) -> FilterChain {
    let effect = match Effect::new("bench glass tint", GLASS_TINT_WGSL) {
        Ok(effect) => effect,
        Err(error) => panic!("{error}"),
    };
    let tints: Vec<[f32; 4]> = (0..count.min(TINT_SLOTS)).map(|tile| tile_tint(tile, count)).collect();
    let mut tint = Pass::new(&effect);
    if let Err(error) = tint.set_slice("tint", &tints).and_then(|()| tint.set("index", index)) {
        panic!("{error}");
    }
    presets::gaussian_blur(3.0).then(tint)
}

/// The mixed-modes Regions chain: a Gaussian blur (sigma 3 points) then [`MIXED_TINT_WGSL`] with
/// the tints of `count` tiles and [`PIXELATE_CELL`].
///
/// # Panics
/// On a WGSL or parameter error: the snippet and its parameters are fixed test data.
pub fn mixed_region_chain(count: usize) -> FilterChain {
    let effect = match Effect::new("bench mixed tint", MIXED_TINT_WGSL) {
        Ok(effect) => effect,
        Err(error) => panic!("{error}"),
    };
    let tints: Vec<[f32; 4]> = (0..count.min(TINT_SLOTS)).map(|tile| tile_tint(tile, count)).collect();
    let mut pass = Pass::new(&effect).with_sample_radius(PIXELATE_CELL);
    if let Err(error) = pass.set_slice("tint", &tints).and_then(|()| pass.set("cell", PIXELATE_CELL)) {
        panic!("{error}");
    }
    presets::gaussian_blur(3.0).then(pass)
}

/// N glass tiles laid out in a grid over an area, ready to paint in either [`Mode`].
#[derive(Debug)]
pub struct GridScene {
    /// The area the backdrop fills and the tiles are laid out in, points.
    area: Rect,
    /// The tiles, in id order.
    tiles: Vec<Rect>,
    /// [`Mode::Layers`] / [`Mode::Entries`]: one chain per tile (its own `p.index`).
    layer_chains: Vec<FilterChain>,
    /// The mixed modes: the glass chain on even tiles, pixelate on odd ones.
    mixed_chains: Vec<FilterChain>,
    /// [`Mode::RegionsMixed`]: the one chain.
    mixed_region_chain: FilterChain,
    /// [`Mode::Regions`]: the one chain.
    region_chain: FilterChain,
    /// [`Mode::Regions`]: one region per tile, in id order.
    regions: Vec<Region>,
    /// [`Mode::Regions`]: the layer rect (the tiles' union).
    union: Rect,
}

impl GridScene {
    /// `count` tiles (at most [`TINT_SLOTS`]) in a grid of about twice as many columns as rows
    /// filling `area` with a 2-point inset per cell.
    pub fn new(count: usize, area: Rect) -> Self {
        let count = count.clamp(1, TINT_SLOTS);
        // Rows = round(sqrt(N / 2)) for a 2:1 area, in integers: the largest r >= 1 with
        // r - 1/2 <= sqrt(N / 2), i.e. (2r - 1)^2 <= 2N.
        let rows = (1..=count).take_while(|rows| (2 * rows - 1) * (2 * rows - 1) <= 2 * count).last().unwrap_or(1);
        let cols = count.div_ceil(rows);
        let cell = vec2(area.width() / small_f32(cols), area.height() / small_f32(rows));
        let tiles: Vec<Rect> = (0..count)
            .map(|tile| {
                let min = area.min + vec2(small_f32(tile % cols) * cell.x, small_f32(tile / cols) * cell.y);
                Rect::from_min_size(min, cell).shrink(2.0)
            })
            .collect();
        let union = tiles.iter().fold(Rect::NOTHING, |union, tile| union.union(*tile));
        let layer_chains: Vec<FilterChain> = (0..count).map(|tile| glass_chain(count, u32::try_from(tile).unwrap_or(0))).collect();
        let mixed_chains = layer_chains.iter().enumerate().map(|(tile, chain)| if tile % 2 == 1 { FilterChain::from(presets::pixelate(PIXELATE_CELL)) } else { chain.clone() }).collect();
        let regions = tiles.iter().map(|tile| Region::rounded_rect(*tile, TILE_CORNER)).collect();
        Self { area, tiles, layer_chains, mixed_chains, mixed_region_chain: mixed_region_chain(count), region_chain: glass_chain(count, 0), regions, union }
    }

    /// The centre of tile 0 (a probe pixel for "was the glass applied").
    pub fn first_tile_centre(&self) -> Pos2 {
        self.tiles.first().map_or(self.area.center(), Rect::center)
    }

    /// The busy backdrop: a diagonal gradient, thin stripes and a lattice of circles.
    pub fn paint_backdrop(&self, painter: &egui::Painter) {
        let area = self.area;
        let mut mesh = egui::Mesh::default();
        for (pos, color) in [(area.left_top(), Color32::from_rgb(28, 30, 80)), (area.right_top(), Color32::from_rgb(120, 30, 90)), (area.right_bottom(), Color32::from_rgb(230, 140, 40)), (area.left_bottom(), Color32::from_rgb(20, 120, 110))] {
            mesh.colored_vertex(pos, color);
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);
        let mut x = area.left();
        while x < area.right() {
            painter.vline(x, area.y_range(), Stroke::new(1.0, Color32::from_white_alpha(70)));
            x += 6.0;
        }
        let mut y = area.top() + 5.0;
        let mut row = 0_u8;
        while y < area.bottom() {
            let mut x = area.left() + if row.is_multiple_of(2) { 5.0 } else { 11.0 };
            while x < area.right() {
                painter.circle_filled(pos2(x, y), 3.0, if row.is_multiple_of(3) { Color32::YELLOW } else { Color32::from_rgb(40, 200, 255) });
                x += 12.0;
            }
            y += 10.0;
            row = row.wrapping_add(1);
        }
    }

    /// Paints the tiles' glass in `mode` (the backdrop must already be painted).
    pub fn paint(&self, painter: &egui::Painter, mode: Mode) {
        match mode {
            Mode::Layers => self.paint_layers(painter, &self.layer_chains),
            Mode::LayersMixed => self.paint_layers(painter, &self.mixed_chains),
            Mode::Regions => ShaderLayer::new(self.union, self.region_chain.clone()).regions(self.regions.clone()).paint(painter),
            Mode::RegionsMixed => ShaderLayer::new(self.union, self.mixed_region_chain.clone()).regions(self.regions.clone()).paint(painter),
            Mode::Entries => self.entries_layer(&self.layer_chains).paint(painter),
            Mode::EntriesStacked => self.entries_layer(&self.layer_chains).overlap(Overlap::Stacked).paint(painter),
            Mode::EntriesMixed => self.entries_layer(&self.mixed_chains).paint(painter),
        }
    }

    /// One layer per tile, tile `i` running `chains[i]`.
    fn paint_layers(&self, painter: &egui::Painter, chains: &[FilterChain]) {
        for (tile, chain) in self.tiles.iter().zip(chains) {
            ShaderLayer::new(*tile, chain.clone()).corner_radius(TILE_CORNER).paint(painter);
        }
    }

    /// One layer with one entry per tile, tile `i` running `chains[i]` (overlap: independent).
    fn entries_layer(&self, chains: &[FilterChain]) -> ShaderLayer {
        ShaderLayer::from_entries(self.tiles.iter().zip(chains).map(|(tile, chain)| Entry::new(*tile, chain.clone()).corner_radius(TILE_CORNER)))
    }
}

/// Whether pixel `probe` of the `width`-pixel-wide RGBA8 frame `filtered` differs from the same
/// pixel of `backdrop` by more than 3 steps in any channel (the glass was applied there).
pub fn glass_applied(backdrop: &[u8], filtered: &[u8], width: u32, probe: Pos2) -> bool {
    // The probe is a tile centre inside the frame; anything else is "not applied". Checked to be
    // in 0..65536 first, so the float -> integer truncation (the intended floor) cannot saturate.
    let pixel = |coordinate: f32| (0.0..65536.0).contains(&coordinate).then(|| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // In range, checked above.
        let floored = coordinate as usize;
        floored
    });
    let (Some(x), Some(y)) = (pixel(probe.x), pixel(probe.y)) else {
        return false;
    };
    let index = (y * usize::try_from(width).unwrap_or(0) + x) * 4;
    match (backdrop.get(index..index + 3), filtered.get(index..index + 3)) {
        (Some(a), Some(b)) => a.iter().zip(b).any(|(a, b)| a.abs_diff(*b) > 3),
        _ => false,
    }
}

/// Renders [`WARM_UP_FRAMES`] untimed frames, then returns the mean wall time of
/// [`TIMED_FRAMES`] frames. `frame` must render one frame and wait for the GPU to finish it.
pub fn time_per_frame(mut frame: impl FnMut()) -> Duration {
    for _ in 0..WARM_UP_FRAMES {
        frame();
    }
    let start = Instant::now();
    for _ in 0..TIMED_FRAMES {
        frame();
    }
    start.elapsed() / TIMED_FRAMES
}

/// One row of the report.
#[derive(Debug)]
pub struct BenchRow {
    /// The mode, or `None` for the backdrop-only baseline.
    pub mode: Option<Mode>,
    /// Element count (0 for the baseline).
    pub count: usize,
    /// Mean wall time per frame.
    pub per_frame: Duration,
}

/// Prints the rows as a table under a `backend` heading (visible with `--nocapture`).
pub fn print_table(backend: &str, frame_px: [u32; 2], rows: &[BenchRow]) {
    println!("\nbench_many_elements [{backend}], {}x{} px, {WARM_UP_FRAMES} warm-up + {TIMED_FRAMES} timed frames, wall time incl. GPU completion + readback", frame_px[0], frame_px[1]);
    println!("{:<16} {:>4} {:>12}", "mode", "N", "ms/frame");
    for row in rows {
        let mode = row.mode.map_or("backdrop", Mode::name);
        println!("{mode:<16} {:>4} {:>12.3}", row.count, row.per_frame.as_secs_f64() * 1000.0);
    }
}
