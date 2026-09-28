# egui_filter_layers 0.2 — design and roadmap

Baseline: 0.1.0 (commit 1e1908b). Versions verified in the registry: egui / egui-wgpu / egui_glow
0.36.2, wgpu 30.0.1, naga 30.0.1. Breaking changes are allowed (0.x) and are listed in §10.

## 0. Architectural stop condition (§2.5 trigger 2) — refactor FIRST

"Which texture is bound to which binding for pass `i`, with which storage-uv mapping, and what
goes into its uniform block" is decided in TWO places today:

- wgpu: `src/wgpu_backend/chain.rs:206-277` (`record`: bind group per pass, ping-pong choice,
  viewport) and `chain.rs:383-403` (`chain_uniforms`: per-slot `src_*`/`aux_*`/frag offsets);
- glow: `src/glow_backend/render.rs:460-490` (`run_passes`: same choice, `frame.src_scale =
  input.scale(size)` at :474) and `render.rs:493-530` (`composite`: units 0/1, aux scale).

F1 (original in every pass), F6 (per-pass viewport size), F7 (state ping-pong, a third
input, a second output chain), F8 (extra inputs), F9 (coverage input) each add a row to that
wiring. Implemented as today, every one of them lands twice and the copies drift (this is the
exact failure documented in AGENTS.md §2.5). Therefore **Phase 1 is a behaviour-preserving
refactor that creates one owner**, verified on its own (bit-identical GPU tests), before any
feature phase.

The owner: a backend-neutral `src/plan.rs`:

```rust
/// One texture the executor knows. Backends map every variant to their own object.
pub(crate) enum Slot { Source, Ping(u8) /*0|1*/, StatePrev, StateNext, Coverage, User(u8) }
pub(crate) enum StepKind { Display, State }
pub(crate) struct PassStep {
    pub pass: usize,               // index into job.passes
    pub kind: StepKind,
    pub input: Slot,               // src()
    pub output: Slot,              // render target
    pub scale: f32,                // F6; 1.0 today
    pub used_px: [i32; 2],         // output viewport = ceil(window * scale)
}
pub(crate) struct ChainPlan { pub steps: Vec<PassStep>, pub composite_input: Slot, pub state_final: Option<Slot>, pub has_state: bool, pub max_scale_targets: ... }
pub(crate) fn plan(passes: &[Pass], capture: [i32;2], window: PxRect, state: Option<StatePlanInput>) -> Result<ChainPlan, PlanError>;
/// The complete uniform block of one step: FrameUniforms::for_layer + per-slot mappings.
pub(crate) fn step_uniforms(frame: &FrameUniforms, step: &PassStep, maps: &SlotMappings, pass: &Pass) -> [u8; UNIFORM_BLOCK_SIZE];
pub(crate) fn composite_uniforms(frame: &FrameUniforms, plan: &ChainPlan, maps: &SlotMappings, geometry: &LayerGeometry) -> [u8; UNIFORM_BLOCK_SIZE];
/// `TextureMapping::of(used, allocated, window_origin)` — the ONE formula for storage uv.
```

`SlotMappings` is a small struct `{ source, ping: [;2], state: [;2], coverage, user: [;3] }` of
`TextureMapping { scale, offset }` (today's `chain.rs:359-364` type, moved to `plan.rs`).
Backends keep: object creation, binding the slot's texture, drawing. They lose: every decision
about *what* to bind and *what* to upload. `chain_uniforms` becomes a loop over
`plan.steps` calling `step_uniforms`; glow `run_passes` becomes the same loop. The existing
`uniform_slots_follow_the_ping_pong_order` test moves to `plan.rs` and is joined by
`glow_and_wgpu_upload_identical_blocks` (call both packers on the same plan; assert equal
bytes except `flip_y`/`frag_mode`).

## 1. Shared foundations (designed once)

### 1.1 Binding layout (every effect pass AND the composite; one bind group layout on wgpu)

| binding | name (WGSL prelude) | type | filled by | notes |
|---|---|---|---|---|
| 0 | `fl` | uniform `FlUniforms` | both | grown block, §1.2 |
| 1 | `fl_src_texture` | texture_2d | both | pass input (unchanged) |
| 2 | `fl_sampler` | sampler linear/clamp | both | unchanged |
| 3 | `fl_original_texture` | texture_2d | both | **F1**: now bound for effects too (today composite-only, `effect.rs:472-475`) |
| 4 | `fl_params` | uniform, user `struct Params` | both | **F4**; a 16-byte zero buffer when the effect declares none |
| 5 | `fl_state_texture` | texture_2d | both | **F7**: previous state (1x1 transparent dummy when the chain has no state) |
| 6 | `fl_coverage_texture` | texture_2d | both | **F9**: mask coverage (1x1 white dummy for a rounded-rect / no mask) |
| 7 | `fl_nearest_sampler` | sampler nearest/clamp | both | state and pixel-exact reads |
| group 1..3, binding 0/1 | `fl_user0..2`, `fl_user0_sampler..` | texture_2d + sampler | wgpu: egui's own bind group; glow: units 4..6 + own samplers | **F8**, §6 |

Rationale for one layout: wgpu allows a bind-group layout to contain entries a shader does not
use, so ONE `effect_layout` serves every effect (dummies fill unused slots); pipelines never
depend on which features an effect uses. WebGL2 budget (wgpu `downlevel_defaults`, inherited by
`downlevel_webgl2_defaults`, `wgpu-types-30.0.1/src/limits.rs:584-597`): 4 bind groups, 16
sampled textures / stage, 16 samplers, 16 KiB per uniform binding. We use 4 groups, 7 textures,
5 samplers. GLES 3.0 guarantees 16 fragment texture units: we use 7. `glsl.rs:124-138` already
assigns units from naga's reflection, so glow follows automatically.

Interface check (`effect.rs:455-489`) changes from "binding <= max" to "every bound global is a
prelude-declared name" (the user may declare no binding at all; `struct Params` is the only
user-declared thing the wrapper binds, §5). `ModuleKind::Composite` disappears: the composite is
just an effect with the full layout.

### 1.2 `FlUniforms` growth (byte offsets, pinned by `offsets_match_the_wgsl_struct_layout`)

Today: 352 bytes, `params` at 96 (`pass.rs:211,240`). New (naga std140-like uniform layout):

| offset | member | type | feature |
|---|---|---|---|
| 0 | `rect_uv` | vec4 | — |
| 16 / 24 | `texel`, `size_px` | vec2, vec2 | — (still the CAPTURE's) |
| 32 / 36 | `time`, `pixels_per_point` | f32 | — |
| 40 / 44 | `internal_flip_y`, `internal_frag_mode` | f32 | — |
| 48 / 56 | `internal_src_scale`, `internal_aux_scale` | vec2 | — |
| 64 / 72 / 80 | `internal_frag_offset`, `internal_src_offset`, `internal_aux_offset` | vec2 | — |
| 88 | `delta_time` | f32 | F2 |
| 92 | `frame` | u32 | F2 |
| 96 | `pointer` | vec4 (`xy` capture uv, `z` pressed 0/1, `w` present 0/1) | F2 |
| 112 / 120 | `internal_state_scale`, `internal_state_offset` | vec2 | F7 |
| 128 / 136 | `internal_cover_scale`, `internal_cover_offset` | vec2 | F9 |
| 144 | `pass_scale` | f32 | F6 (1.0 for full-res passes) |
| 148 | `pass_index` | u32 | F7 substeps (index of this step within the frame's steps of its kind) |
| 152 | `internal_frag_texel` | vec2 (one OUTPUT pixel of the draw; `texel` at full res) | F6: mode-1 uv derivation, `out_texel()` |
| 160 | `internal_src_texel` | vec2 (one pixel of the draw's `src` input) | F6 (phase 3b): `src_px`, `src_texel()`, `src()` edge clamp |
| 176 | `params` | array<vec4, 16> | — (kept, §5) |
| 432 | (size) | | |

User-texture mappings (F8) are not needed: user textures are sampled in THEIR OWN uv (0..1 of
the texture), never in capture uv. State (F7) and coverage (F9) use capture uv like `src`.
Public names added to the prelude table: `fl.delta_time`, `fl.frame`, `fl.pointer`,
`fl.pass_scale`. `fl.frame`/`delta_time` are NOT animation triggers (`reads_time` stays the only
detector; F3's helper requests repaints explicitly).

### 1.3 Per-layer persistent identity (F7 only)

`FilterLayer::id(egui::Id)` (public). Only chains containing a state pass need it: without an
id such a layer is skipped and an error is logged once per context (`layer.rs` latch pattern,
`layer.rs:22-42`), and `status()` is unaffected (it is a per-layer misuse, not a backend fault).
F2 needs no identity: `delta_time` = `input.stable_dt` clamped to `0..=0.1` s, `frame` =
`ctx.cumulative_frame_nr()` (u64 → u32 wrapping, documented) — both read in
`FilterLayer::paint` next to `time` (`layer.rs:126`) and carried in `LayerJob`.

### 1.4 New/changed core types

- `backend::LayerJob` gains: `id: Option<Id>`, `mask: Mask`, `frame_input: FrameInput
  { delta_time, frame, pointer: Option<(Pos2 /*screen points*/, bool)> }`, `textures:
  [Option<TextureId>; 3]`, `state: Option<StateOptions>`.
- `pass::Pass` gains: `scale: f32` (F6), `named: NamedParams` (F4).
- `effect::Effect` gains: `kind: EffectKind { Display, State }`, `params: Arc<ParamLayout>`
  (F4), `textures: u8` (how many user slots it reads; from naga globals used).

## 2. F1 — `original(uv)` in every pass (S)

- Prelude: `fn original(uv) -> vec4` = sample binding 3 through `internal_aux_*` (today's
  composite code, `composite.wgsl:56`); `composite.wgsl` drops its own binding-3 declaration.
- `plan.rs`: every step's `aux` mapping = source mapping (today only the composite's:
  `chain.rs:397-398`, `render.rs:501`).
- wgpu: `chain.rs` bind group entry 3 for every step. glow: `render.rs:460-490` binds the capture
  on unit 1 for every pass (today only in `composite`, `render.rs:508`). Offline: nothing (goes
  through `chain.rs`).
- Tests: unit — interface check accepts `original`; GPU (`tests/wgpu_headless.rs`) — an effect
  returning `original(uv)` after an invert pass equals the unfiltered backdrop;
  `offline_equivalence.rs` case `original_passthrough`.
- Docs: README/lib.rs prelude table, `shaders/MODULE_README.md` (bindings), `src/MODULE_README.md`
  contract line "Bindings in @group(0)".

## 3. F2 — `delta_time`, `frame`, pointer (S)

- Values: §1.3. Pointer: `input.pointer.latest_pos()` (screen points) and `primary_down()`
  (`egui-0.36.2/src/input_state/mod.rs:1310,1539`) read in `FilterLayer::paint`; the backend
  converts screen points → capture uv (`(p * ppp - capture.xy) / size_px`; no layer transform
  needed since the capture is in screen pixels). `w = 0` when there is no pointer.
- Offline: `OfflineOptions::with_delta_time(f32)` (default 0), `with_frame(u32)` (default 0),
  `with_pointer(Option<PointerState { uv: [f32;2], pressed: bool }>)` (default absent, w = 0).
  `VideoProcessor` sets `delta_time = time - previous_time` (0 for the first frame; clamped to
  `>= 0`) and `frame = index` automatically in `run_frame` (`offline/video/mod.rs:181`), unless
  the user overrides through options — document that the override wins.
- Tests: `pack_uniforms` field test; `layer.rs` test that a headless `Context` run yields
  `delta_time <= 0.1`; offline test for the video counters.

## 4. F3 — `Impulses` helper + click-ripple preset (S)

`src/impulses.rs` (public):

```rust
pub struct Impulses { life: f32, max: usize, items: Vec<Impulse> }
pub struct Impulse { pub uv: [f32; 2], pub started: f64, pub strength: f32 }
impl Impulses {
    pub fn new(life_seconds: f32) -> Self;             // max defaults to 16
    pub fn push(&mut self, uv: [f32; 2], strength: f32, now: f64);
    pub fn push_point(&mut self, pos: Pos2, layer_rect: Rect, strength: f32, now: f64); // rect-relative uv
    pub fn retire(&mut self, now: f64);                // drop expired
    pub fn is_alive(&self) -> bool;
    pub fn pack(&self, now: f64) -> ImpulsePack;       // [[uv.x, uv.y, age_seconds, strength]; 16] + count
    pub fn apply(&self, pass: &mut Pass, now: f64) -> Result<(), ParamError>; // sets `impulses` + `impulse_count`
}
```

`pack` writes the named params `impulses: array<vec4<f32>, 16>` and `impulse_count: u32` (F4
arrays). uv is rect-relative (`fl.rect_uv` converts inside the shader). The app calls
`if impulses.is_alive() { ctx.request_repaint() }` — documented; the helper never touches the
`Context` (keeps it usable offline). Preset `presets::click_ripples(&impulses, now, amplitude,
wavelength, speed) -> Pass`: displacement rings with decay; named params. Tests: pack layout,
retire, count clamp to 16.

## 5. F4 — named parameters (M)

**Decision: add named params, keep `params(i)`.** Presets and existing user code keep working;
the anonymous block costs 256 bytes per pass and one branch nowhere. Migrating presets buys
nothing for users and would churn every preset test. `params(i)` is documented as "the untyped
block; prefer named".

- WGSL contract: the snippet MAY declare `struct Params { ... }`. `effect.rs::compile` detects
  it after the parse (naga `module.types` has a struct named `Params`), then re-assembles with a
  tail `@group(0) @binding(4) var<uniform> p: Params;` and re-parses (two parses; a few ms, at
  construction only). The effect reads `p.sigma`. Layout = naga's computed member offsets /
  struct span; naga validation already rejects layouts illegal for `var<uniform>`; the error is
  surfaced as `EffectError::Validation` located in the snippet.
- Public API:
  ```rust
  pub struct ParamInfo { pub name: String, pub ty: ParamType, pub offset: u32, pub count: u32 /*1 or array len*/ }
  pub enum ParamType { F32, I32, U32, Vec2, Vec3, Vec4, Mat4 }
  impl Effect { pub fn params(&self) -> &[ParamInfo]; pub fn param(&self, name) -> Option<&ParamInfo>; }
  pub enum ParamValue { F32(f32), I32(i32), U32(u32), Vec2([f32;2]), Vec3([f32;3]), Vec4([f32;4]), Mat4([[f32;4];4]) }
  impl Pass {
      pub fn set(&mut self, name: &str, value: impl Into<ParamValue>) -> Result<(), ParamError>;
      pub fn with(self, name, value) -> Result<Self, ParamError>;
      pub fn set_slice(&mut self, name, values: &[impl Into<ParamValue> + Copy]) -> Result<(), ParamError>; // arrays
      pub fn get(&self, name) -> Option<ParamValue>;
  }
  pub enum ParamError { Unknown { name }, Type { name, expected: ParamType, given: ParamType }, Length { name, expected, given } }
  ```
  Storage: `Pass.named: Vec<u8>` sized to the struct span (0 when none). Unsupported member
  types (bool, f16, nested structs, arrays of arrays) → `EffectError::Interface` naming the
  member. Size cap: span ≤ 16 KiB (`EffectError::Interface`, WebGL2 limit).
- Backends: wgpu — binding 4 in the same uniform `GrowBuffer` at its own stride slot (`chain.rs`
  `uniform_stride` logic reused: slot layout `[fl_0, p_0, fl_1, p_1, …]`); glow — a second UBO
  bound at `UNIFORM_BINDING_POINT + 1`, `glsl.rs` maps the second reflected block. Dummy: 16 zero
  bytes when the effect has no `Params`.
- Tests: layout table vs naga for a struct with every supported type; array member; type
  mismatch; GPU test reading `p.color` on wgpu; glow translation test in `glsl.rs` for a
  two-block program.
- Docs: README "Custom effects" gains a named-params example; prelude table row `p.<name>`.

## 6. F8 — user textures (M)

- API: `FilterLayer::texture(slot: usize /*0..3*/, id: egui::TextureId)`; prelude
  `fl_user0/1/2` + their samplers, helpers `tex(i, uv)` (own sampler = the texture's egui
  `TextureOptions`), `tex_linear(i, uv)` / `tex_nearest(i, uv)` (crate samplers, group 0). `uv`
  is the texture's own 0..1. An effect that reads slot `i` with no texture bound → the layer is
  skipped and logged once (`failed_effects`-style entry `EffectFailure` with a clear reason).
- wgpu (verified `egui-wgpu-0.36.2/src/renderer.rs:161-170`): `Renderer::texture(&id)` gives a
  `Texture { texture: Option<wgpu::Texture>, bind_group, options }`; native textures have
  `texture = None`, so the crate binds egui's **bind group** as group `1 + slot` (its layout is the
  one already replicated for the mesh pipeline, `resources.rs:622`). Staleness: the crate's
  `textures.rs::snapshot_textures` (`textures.rs:221`) already yields the right bind group for a
  texture uploaded in a not-yet-applied pass; the plugin (`plugin.rs:119-142`) resolves the
  layer's user ids the same way and stores them in `BeneathPlan.user_textures`. Lifetime: the
  bind group `Arc` is held in the plan for the frame; a freed id → skipped + logged once.
- glow (verified `egui_glow-0.36.2/src/painter.rs:645`): `Painter::texture(id) ->
  Option<glow::Texture>`; textures of the frame are applied BEFORE `paint_primitives`
  (`painter.rs:356-372`), so no staleness handling. Bind on units 4..6; the "own" sampler is
  the texture's own GL parameters (egui_glow sets them per `TextureOptions`), the crate samplers
  are two GL sampler objects (GL 3.3 / ES 3.0 have `glGenSamplers`). The `!Send` uniform-location
  rule (`glow_backend/MODULE_README.md`) is respected: units are set at link time as today.
- Offline: `OfflineOptions::with_texture(slot, FrameRef, TextureOptions)`; uploaded once per
  `process` call into renderer-owned `Rgba8Unorm` textures (re-uploaded only when the bytes'
  pointer/len/size change is not detectable → document "uploaded every call"; `VideoProcessor`
  offers `set_texture` that uploads once and keeps it).
- Tests: interface check counts slots; GPU test sampling a 2x2 texture (nearest) both UI and
  offline (`offline_equivalence.rs` gains a textured case).

## 7. F6 — per-pass scale, dual-Kawase blur, bloom (M)

- `Pass::with_scale(s)`: `s` clamped to `(0, 1]` (non-finite → 1). The step renders
  `ceil(window * s)` pixels into its ping-pong slot; `src()` is unchanged (capture uv; the
  smaller texture is bilinear-sampled). Prelude adds `fl.pass_scale` and `fn out_texel()` (=
  `fl.texel / fl.pass_scale`), which kernels use to step in THEIR pixels. Frag-mode 1 uv =
  `(position + frag_offset * s) * internal_frag_texel` (§1.2). glow (mode 0) interpolates: nothing
  to do beyond the viewport size.
- `plan.rs` owns `used_px` per step; both backends size the viewport from it.
- Offline tiling: a chain with any `scale < 1` **refuses tiling** (`OfflineError::NotTileable
  { reason }`) — window edges do not round consistently across tiles at fractional scales; the
  user sets `with_max_tile_size(None)` (frame must fit the device). Explicit, no approximation.
- Presets: `presets::kawase_blur(radius_points) -> FilterChain` (N down passes at 1/2, 1/4, … then
  up passes; N from radius; declares its reach on the first pass like `gaussian_blur`,
  `presets.rs:243-257`), `presets::bloom(threshold, intensity, radius) -> FilterChain`
  (bright-pass at 1/2 → kawase → final pass `original(uv) + glow` using F1). Both use named
  params (F4).
- Tests: plan test for step sizes; GPU: kawase at large radius vs gaussian within a tolerance
  band on a smooth gradient; offline: `NotTileable` error test.

## 8. F7 — persistent state (L)

Contract:

- `Effect::new_state(label, wgsl)` builds a **state pass**: `fn effect(uv) -> vec4` whose output
  is written to the layer's state (RGBA16F), not to the display chain. It reads `state(uv)`
  (previous state, nearest-by-default through `fl_nearest_sampler`; `state_linear(uv)` also
  provided), `src(uv)` (= the display chain's input at that point), `original(uv)`, params,
  `fl.delta_time`, `fl.pass_index`.
- Display passes read `state(uv)` = the state as written by the LAST state step preceding them
  in the chain (or last frame's if none precedes). A chain may hold any number of state steps in
  any positions; N consecutive copies of one state pass = N substeps (`fl.pass_index` = 0..N-1;
  the effect divides `delta_time` by the substep count it is told via a named param).
- `FilterLayer::id(id)` required (§1.3). `FilterLayer::state(StateOptions { scale: f32 /*0..1,
  default 1*/, format: StateFormat::Rgba16Float | Rgba8Unorm /*explicit opt-in for packed 8-bit
  effects*/ })`. State size = `ceil(capture * scale)`.
- Format policy: `Rgba16Float` must be renderable + filterable on the device (wgpu:
  `renderable_and_filterable`, `gpu.rs:1026`; glow: the `probe` at `resources.rs:580`, which
  on WebGL2 requires `EXT_color_buffer_float`, `wgpu-hal-30.0.1/src/gles/adapter.rs:668-676`).
  If it is not, stateful layers are **refused**: `BackendStatus` stays usable, the layer draws
  nothing, and a `StateFailure`-kind entry appears in `failed_effects()` ("state needs
  RGBA16F; this device has none — opt in to `StateFormat::Rgba8Unorm` if the effect packs its
  state in 8 bits"). No silent RGBA8 downgrade (AGENTS.md §14).
- Resize / move: state textures are grow-only per id; the READ mapping uses the previous
  frame's used size, the WRITE uses the new one — so a resized layer reads a stretched previous
  state for free (no copy) and never resets on the ±1 px jitter of a dragged rect. A moved rect
  keeps its uv-relative state (documented: state is in layer uv, not screen space).
- Reset / eviction: `egui_filter_layers::reset_state(ctx, id)` sets a flag consumed at the next
  paint (clears both state textures to 0). A state not painted for 120 frames or 2 s (whichever
  first, checked per paint) is freed; `destroy_glow` frees all. `status()` gains nothing;
  `state_count(ctx)` is a debug accessor.
- Where it lives: wgpu — `WgpuResources.states: HashMap<Id, LayerState { ping_pong: [Option<Target>;2], used: [u32;2], last_seen: u64, reset: bool }>` (in `CallbackResources`, so it dies with
  the renderer like everything else); `LayerTargets` (per-frame pool) is untouched. glow —
  `GlowState.states` with two `RenderTarget`s each. Offline — `OfflineRenderer.state:
  Option<OfflineState>` carried across `process` calls of the same renderer; `OfflineOptions::
  with_state(StateOptions)` enables it; `OfflineRenderer::reset_state()`; `VideoProcessor` carries
  it frame to frame automatically. **Tiling is refused for stateful chains** (`OfflineError::
  NotTileable`), as for F6.
- Executor: `plan.rs` emits `StepKind::State` steps with `input: StatePrev`, `output:
  StateNext` and swaps the state ping-pong after each; the composite ignores state. glow and wgpu
  each add ONE new slot binding (5) and one target pair — no new decisions.
- Uniforms: `internal_state_*` mappings (§1.2) from the state textures' used/allocated sizes.
- Tests: plan test (substeps, read-after-write ordering); GPU: a state pass `state(uv) +
  0.25` over 4 frames reads 1.0 on frame 4 (UI headless with a fixed id; offline via 4
  `process` calls); reset test; eviction test (unit, on the map); refusal test with
  `StateFormat` on a device lacking RGBA16F is documented as not runnable on CI (see Not
  verified).
- Demo scene "water around the menu": state pass = height-field wave equation (two channels:
  height, velocity; 2 substeps), obstacles = up to 8 rounded rects as `array<vec4,8>` +
  `radii` named params (the central menu's rect), `Impulses` (F3) inject clicks and the moving
  pointer (`fl.pointer` with `z` pressed injects a continuous wake); display pass refracts
  `original(uv)` by the height gradient and adds a specular term ("reflections" = the refracted
  original plus a highlight; true environment reflection is out of scope). Trails scene: a state
  pass `max(state * 0.96, luma(src))`.

## 9. F9 — arbitrary-shape mask (M)

- API: `FilterLayer::mask(Mask)`, `enum Mask { RoundedRect(CornerRadius) /*default*/,
  Shape(egui::Shape) /*in layer points; Path, Circle, Text galleys, Mesh…*/ }`; `corner_radius`
  stays as the shorthand.
- Coverage rasterization (backend-neutral input): `FilterLayer::paint` tessellates the shape
  with `ctx.tessellate(vec![ClippedShape{clip: rect, shape}], ppp)` (`egui-0.36.2/src/
  context.rs:2859`) into `Vec<ClippedPrimitive>` stored in `LayerJob.mask` — meshes whose vertex
  colour alpha × texture alpha IS the coverage (egui's feathering gives the AA edge). Text
  shapes reference the font atlas, so the rasterizer samples a texture; solid shapes reference
  the white pixel of the same atlas. Under a layer transform the feather width scales with the
  zoom (documented approximation; tessellation happens before the transform is known).
- Rasterizer = a mesh draw into an RGBA8 `Coverage` target (capture-sized, grow-only; coverage
  read from `.a`): wgpu reuses `mesh.wgsl` + the mesh pipeline (`resources.rs:617-655`) with
  the font atlas bind group from `snapshot_textures`; glow gets `mesh.wgsl` translated by
  `glsl.rs` (it unpacks the colour by hand, `mesh.wgsl:35-40`, so naga glsl-out handles it) plus
  a VBO/EBO pair in `resources.rs` and the atlas from `Painter::texture(TextureId::default())`.
  `mesh.wgsl` moves to `src/shaders/mesh.wgsl` (shared). Risk: naga glsl-out attribute
  translation must be verified in the first glow test; the fallback is a hand-written 30-line
  GLSL pair in `glsl.rs`.
- Composite: `fl_coverage(uv)` (`composite.wgsl:33-37`) becomes `select(sdf_coverage,
  textureSampleLevel(fl_coverage_texture, …).a, fl.params[3].x > 0.5)`; both paths keep the
  1/1024 snap. The blended wgpu composite (erase/add) is unchanged. Effects may also read
  `mask(uv)` (glass edge lighting).
- Offline: `OfflineOptions::with_mask(Mask)` tessellated by `epaint::Tessellator::new(ppp,
  TessellationOptions::default(), font_tex_size, vec![])` (`epaint-0.36.2/src/tessellator.rs:
  1329`); text masks offline require the caller to pass a `Fonts`-rendered galley — supported
  for `Shape::Mesh`/`Path`/`Circle`/`Rect` only in 0.2 (`OfflineError::InvalidOption` for
  `Shape::Text`; documented).
- Tests: unit — coverage packing, mask kind flag; GPU headless: a circle mask leaves the
  corners untouched (reuse `a_rounded_corner_keeps_what_is_not_re_rendered` structure); offline:
  star mask vs UI within the documented 1-px edge band.

## 10. F5 — hot reload (S, native only, no new dependency)

- `src/hot.rs` behind feature `hot-reload` (native only; no crate dependency: a `std::thread`
  polls `metadata().modified()` every 250 ms — `notify` would add a dependency tree for a dev
  tool, so it is not used). API:
  ```rust
  pub struct HotEffect { .. }
  impl HotEffect {
      pub fn watch(label: &str, path: impl Into<PathBuf>) -> Result<Self, HotReloadError>; // first compile is synchronous, errors returned
      pub fn watch_state(..)                                   // F7 state effects
      pub fn poll(&mut self) -> HotReloadEvent;                // GUI thread, per frame: Unchanged | Reloaded | Failed(EffectError) | IoError(..)
      pub fn effect(&self) -> &Effect;                         // last GOOD compile
      pub fn error(&self) -> Option<&HotReloadError>;           // last failure, if the file is currently broken
      pub fn status_text(&self) -> String;                     // for a label / overlay
  }
  ```
  Compilation (`Effect::new`, naga) happens on the watcher thread; the GUI thread only swaps an
  `Arc`. The layer keeps rendering the last good effect while the file is broken; the app shows
  `error()` (the demo draws it as an overlay on the layer). Named-param values survive a reload
  when the names/types still match (`Pass::rebind(&new_effect)` returns which names were dropped).
- Tests: a temp file edited by the test; poll observes `Reloaded`, then `Failed` with the naga
  location, then `Reloaded`.

## 11. Phase order (each phase: worker → `cargo check-all`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, GPU tests `--ignored` where a GPU exists, commit)

| # | phase | size | files owned (disjoint) | depends on | parallel with |
|---|---|---|---|---|---|
| 1 | **Refactor: `plan.rs` single owner** (§0); binding layout + uniform block grown to §1.1/§1.2 with dummies; interface check by prelude names; `ModuleKind::Composite` removed. Behaviour-preserving: GPU tests bit-identical. | M | `src/plan.rs` (new), `pass.rs`, `effect.rs`, `shaders/*.wgsl`, `wgpu_backend/chain.rs`, `glow_backend/render.rs`, `glow_backend/resources.rs`, readmes | — | — (serial; everything builds on it) |
| 2a | F1 `original()` | S | prelude, `plan.rs` aux mapping, tests | 1 | 2b, 2c |
| 2b | F2 uniforms + `LayerJob.frame_input` + offline/video options | S | `layer.rs`, `backend.rs`, `offline/mod.rs`, `offline/video/mod.rs`, both backends' pointer conversion (one fn in `geometry.rs`) | 1 | 2a, 2c |
| 2c | F4 named params | M | `effect.rs` (Params detection/layout), `pass.rs` (`ParamValue`, storage), `chain.rs` binding 4 slot, `glow_backend/{glsl.rs,render.rs,resources.rs}` second UBO | 1 | 2a, 2b (touches `pass.rs`/`effect.rs`: sequence AFTER 2b if the manager wants zero merge risk) |
| 3a | F3 `Impulses` + `click_ripples` preset | S | `src/impulses.rs`, `presets.rs` | 2b, 2c | 3b, 3c |
| 3b | F6 scale + kawase + bloom presets + offline `NotTileable` | M | `plan.rs` sizes, `chain.rs`/`render.rs` viewports, `presets.rs` (kawase/bloom section), `offline/mod.rs` refusal | 2a, 2c | 3a, 3c |
| 3c | F8 user textures | M | `layer.rs` API, `wgpu_backend/{plugin.rs,textures.rs,chain.rs}` groups 1..3, `glow_backend/{render.rs,resources.rs}` units 4..6 + samplers, `offline/mod.rs` texture inputs | 1 | 3a, 3b (shares `chain.rs`/`render.rs` with 3b — give 3b the viewport code, 3c the bind code, or sequence) |
| 4 | F7 state | L | `effect.rs` (`new_state`), `plan.rs` state steps, `layer.rs` id/state API, `backend.rs`, `wgpu_backend/{resources.rs,render.rs,chain.rs}` state map, `glow_backend/{mod.rs,render.rs,resources.rs}`, `offline/{mod.rs,video/mod.rs}` | 2b, 2c, 3b | — (serial: touches every backend) |
| 5 | F9 shape mask | M | `layer.rs` (`Mask`), `shaders/mesh.wgsl` move, `composite.wgsl`, `wgpu_backend/{resources.rs,render.rs}` coverage draw, `glow_backend/{glsl.rs,resources.rs,render.rs}` mesh program, `offline/mod.rs` tessellation | 1 | 6 |
| 6 | F5 hot reload (`src/hot.rs`, feature `hot-reload`) | S | `hot.rs`, `Cargo.toml`, `lib.rs` | 2c (rebind) | 5 |
| 7 | Demo split + scenes (§12) | L (split into 7a scaffold, 7b scenes ripples/water/trails, 7c bloom/textures/mask, 7d hot-reload/params UI/record) | `examples/demo/**` | all | 7b/7c/7d parallel after 7a |
| 8 | Docs + release: README, CHANGELOG, `lib.rs` prelude table, every `MODULE_README.md`, version 0.2.0, cross-pool review of the whole diff (Sol on chain/state/tiling contracts) | M | docs only | 7 | — |

Verification per phase, beyond the standard commands: phases 1, 2a, 3b, 4, 5 must run
`cargo test --all-features -- --ignored` on a GPU (the bit-identity suites in
`tests/offline_equivalence.rs`), and the manager re-runs them before accepting (AGENTS.md §2.3
rule 2). Every phase adds tests to `tests/wgpu_headless.rs` / `tests/offline_equivalence.rs`
for the new contract; glow has no headless test — the demo `--screenshot` on
`--renderer glow` is the manual check, stated in the phase report.

## 12. Demo plan (phase 7)

`examples/demo.rs` (788 lines) becomes a directory example `examples/demo/`: `main.rs` (cli,
logger, eframe boot), `record.rs` (the `--record`/`--screenshot` machinery), `backdrop.rs`,
`stack.rs` (the preset stack editor, `Kind`/`Entry`), `params_ui.rs` (generic editor over
`Effect::params()`: DragValue per `ParamInfo`, colour picker for `Vec4` named `*color*`),
`scenes/{mod.rs, ripples.rs, water.rs, trails.rs, bloom.rs, textures.rs, mask.rs, hot.rs}`, a
`scenes/*.wgsl` file per custom effect (also the hot-reload targets). Scene selector in the side
panel; `--scene name` on the CLI; `--record` works per scene (each scene exposes a periodic
`phase` like today's `record_rect`).

Scenes: (1) click ripples — F3/F4/F2; (2) water around a central menu — F7 wave sim with the
menu's rect as an obstacle, reflections, pointer wake, substep slider; (3) trails — F7 decay;
(4) bloom / fast blur — F1/F6, gaussian vs kawase toggle with a frame-time readout; (5)
texture-driven — F8: noise texture displacement, a normal-map refraction and a 3D LUT colour
grade (a 16x16x16 LUT flattened to 256x16); (6) shape-mask glass — F9: star path, circle and a
text galley ("GLASS"), frosted preset underneath; (7) hot-reload editor — F5: watches
`examples/demo/scenes/hot.wgsl`, in-app text editor writes the file (`std::fs::write` on a
worker thread), error overlay on the layer; (8) the existing preset stack with the named-params
UI. The demo requires both backend features as today.

## 13. Infeasible / refused cases (explicit errors, no fakes)

- **RGBA16F not renderable** (WebGL2 without `EXT_color_buffer_float`, some GLES): F7 state is
  refused per layer with a reason in `failed_effects()`; display chains still run on RGBA8 as in
  0.1 (that fallback is pre-existing and documented).
- **WebGL2 bind groups (4)**: 3 user texture slots, everywhere (same API on all backends).
- **Uniform block 16 KiB**: `struct Params` span capped at 16 KiB at `Effect::new`.
- **Tiling** with state (F7) or scaled passes (F6): `OfflineError::NotTileable`.
- **Text masks offline** (F9): rejected in 0.2 (needs a font pipeline; UI supports them).
- **Hot reload on wasm32**: feature is native-only (`compile_error!` if enabled on wasm? No —
  the module is `cfg(not(wasm32))`; on wasm the type is absent and documented).
- **wgpu re-render of paint callbacks** stays a documented gap (unchanged from 0.1).

## 14. File-size and module notes

- `src/effect.rs` (752) gains Params detection + kind: expect ~950 — split `effect/params.rs`
  (layout table, `ParamValue`) in phase 2c: `src/effect/` directory with `mod.rs`, `params.rs`,
  `interface.rs` (the check), `diagnostics.rs` (render_diagnostic), each with headers; `MODULE_README.md` added.
- `src/wgpu_backend/chain.rs` (474) shrinks in phase 1 (decisions move to `plan.rs`) then grows
  by bindings; expect ≤ 600. `glow_backend/render.rs` (547) likewise ≤ 650; `resources.rs`
  (741) gains samplers, mesh program, state targets: expect ~1000 → split `glow_backend/
  targets.rs` (RenderTarget/ResolveTarget/grown_size) out in phase 5.
- `src/offline/mod.rs` (751) gains options and refusals: keep under 900 by moving
  `OfflineOptions` to `offline/options.rs` in phase 2b.
- `examples/demo.rs` → directory (§12); `examples/MODULE_README.md` updated.
- `README_AGENT.md`: this crate has none (standalone crate with `src/MODULE_README.md` as root
  doc); the roadmap's contracts land in `src/MODULE_README.md` and the nested readmes.

## 15. Risks (ranked)

1. **Phase 1 refactor regressions** — mitigated by bit-identity GPU tests before/after and the
   `glow_and_wgpu_upload_identical_blocks` unit test; run on both wgpu (headless) and glow (demo
   screenshot diff of a fixed `--stack`).
2. **F7 state on wgpu UI**: the per-layer state must survive `finish_prepare` pool reset and a
   re-install (`generation`), and a layer painted twice per frame with the same id (two
   viewports) must not double-step — plan: the second paint of an id in one
   `cumulative_frame_nr` reuses the first's result (no state step), logged once.
3. **naga glsl-out of `mesh.wgsl`** for the glow coverage rasterizer (vertex attributes, `u32`
   colour) — verify in the first glow build of phase 5; fallback is hand-written GLSL.
4. **F4 double parse** (detect `struct Params`, re-assemble) — if naga's error spans shift,
   `snippet_location` must subtract the tail; the tail is appended AFTER the snippet so line
   numbers are unaffected.
5. **F8 on wgpu with `Managed` textures uploaded this pass** — covered by `textures.rs`, but the
   font atlas special case (size mismatch → skipped) also applies to user slots; documented.
6. **Frame time**: dummies for unused bindings add bind-group entries, not draws; the state
   pair and coverage target are per layer — memory grows with the number of stateful/masked
   layers (documented; eviction bounds it).

## 16. Owner decisions (fixed for 0.2)
- (a) Anonymous `params(i)` stays beside named params; presets are not migrated.
- (b) `FilterLayer::id` is mandatory for stateful chains; a stateful chain without an id is refused with a reason.
- (c) 3 user-texture slots on every backend (WebGL2 `max_bind_groups = 4`).
- (d) Hot reload polls file mtimes on a std thread; no new dependency.
- (e) One directory demo (`examples/demo/`) with a scene selector.

## 17. Progress
Each phase is marked done here in the same commit that lands it.
- Phase 1 (plan.rs single owner, full `@group(0)` layout with dummies, 416-byte `FlUniforms`, prelude-name interface check, `ModuleKind` removed): done.
- Phase 2a (F1 `original(uv)` in every pass: prelude helper, the plan binds the capture as every draw's original with its own `aux` mapping; GPU tests UI + offline incl. tiles): done.
- Phase 2b (F2: `LayerJob::frame_input` read in `FilterLayer::paint`; `geometry::pointer_uniform` the one pointer -> capture-uv conversion, resolved per paint callback on both backends; `pass::InputUniforms` + `FrameUniforms::with_input`; `OfflineOptions::with_delta_time` / `with_frame` / `with_pointer` + `PointerState` in `offline/options.rs`; `VideoProcessor` frame clock, explicit options win; not animation triggers): done.
- Phase 2c (F4 named parameters: `struct Params` detected by a pre-parse token scan, crate tail `@group(0) @binding(4) var<uniform> p: Params;`, `ParamLayout` = naga offsets (= std140), unsupported members / arrays with a non-16 stride / > 16 KiB rejected as located `EffectError::Interface`; `Effect::params()` / `param()`, `Pass::set` / `with` / `set_slice` / `get` / `get_slice`, `ParamError`; `plan::params_block` owns binding 4's bytes; wgpu `[fl_i, p_i]` at offset alignment, glow second UBO at binding point 1; `effect.rs` split into `src/effect/`): done.
- Phase 3b (F6: `Pass::with_scale` clamped to `[MIN_PASS_SCALE = 1/1024, 1]`, non-finite = 1; `plan.rs` owns per-step `Content` (resolution + window), viewports `scaled_px = ceil(capture * s)` (f32 rounding error removed), the pool size `target_px`, per-step `src_texel` / `frag_texel` / `pass_scale`; the composite upsamples a scaled last output; `internal_src_texel` added to `FlUniforms` (432 bytes); prelude `src_texel()` / `out_texel()`, `src_px` in input pixels; both backends size targets and viewports from the plan; offline `OfflineError::NotTileable` when a scaled chain would need tiles; presets `kawase_blur(radius)` (dual-Kawase pyramid, up to 8 levels) and `bloom(threshold, intensity, radius)`; GPU tests wgpu + glow + offline, UI == offline bit for bit for scaled chains): done.
- Phase 3a (F3 `Impulses` in `src/impulses.rs` + `presets::click_ripples`: layer-rect uv, oldest-first drop at 16 / `with_capacity`, `pack` / `apply` into named params `impulses` / `impulse_count`, `request_repaint_if_alive` the only `Context` use; compact-support wave packet clamped to `|amplitude|` = sample radius; GPU tests wgpu + glow): done.
- Phase 6 (F5 hot reload, feature `hot-reload`, `src/hot.rs`: `HotEffect::watch` / `watch_with` + `HotOptions` (interval, repaint `Context`, set-up closure), `poll` -> `HotReloadEvent`, `current` / `error` / `status_text`, `HotReloadError` (missing / io / compile with diagnostic / spawn / stopped), free fn `rebind(&Pass, &Effect) -> Rebound`; not compiled on wasm32): done.
- Phase 3c (F8 user textures: `FilterLayer::texture(slot, TextureId)` / `texture_at` / `check_textures`, `TEXTURE_SLOTS` = 3; prelude `tex(i, uv)` (the texture's own sampler = its egui `TextureOptions`), `tex_size(i)`, `tex_load(i, px)` (`textureLoad`); `fl_user0..2` + samplers at `@group(1..=3)` = egui-wgpu's texture layout; binding 7 (`fl_nearest_sampler`) removed; `Effect::reads_texture` from an IR scan (constant slot = that slot, computed = all); `plan::check_textures` refuses a layer / offline run reading an unattached slot (`TextureError`, logged once per context in the UI); `plan::Bindings::user`; wgpu binds egui's own bind group per slot (kept copies for full uploads not yet applied), dummy groups elsewhere, composites group 0 only; glow units 4..6 from `Painter::texture`; unresolvable ids skip the layer and are reported by `failed_textures()` (`TextureFailure`, latched per id); offline `OfflineTexture` (`new` / `from_color_image`) + `OfflineOptions::with_texture`, uploaded once per texture identity, `OfflineError::Texture` / `TextureTooLarge`; glow params UBO grow-only + `glBufferSubData`; GPU tests wgpu + glow + offline (UI == offline bit for bit, tiled == whole), glow scenes for `original()` in later passes and `fl.pointer` / `delta_time` / `frame`): done.

## 18. Findings during execution
- Phase 1: naga's GLSL writer rejects one texture read through two samplers in one module
  ("A image was used with multiple samplers"; pinned by a test in `glow_backend/glsl.rs`). So
  `state(uv)` + `state_linear(uv)` on the same texture, and `tex_linear` + `tex_nearest` on the
  same user texture, cannot both be used in one effect on glow. Resolve in phases 3c / 4 before
  writing the API: prefer exact reads via `textureLoad` (texelFetch, no sampler) for
  pixel-exact / state access, so each texture keeps exactly ONE sampler; drop binding 7
  (`fl_nearest_sampler`) if nothing needs it any more. Verify the translation with a glsl.rs test.
- glow has no headless runtime test; a harness is phase 1b.
- Phase 2c (F4): the "detect `struct Params` in `module.types` after the parse, re-assemble,
  re-parse" plan (§5, risk 4 in §15) is INFEASIBLE: a snippet that reads `p.x` does not parse
  without the appended `var<uniform> p: Params;` ("no definition in scope for identifier: `p`",
  prototyped). Detection is therefore a comment-aware token scan of the snippet for
  `struct Params` BEFORE the single parse, confirmed by naga's IR afterwards; there is no second
  parse. Consequences: `struct /* c */ Params` counts, a commented-out one does not, an `alias
  Params = ...` does not (the snippet then fails on `p` with a located parse error). naga
  validation errors of the tail (layout illegal in a uniform block, a snippet-level `p`) are
  located at the `struct Params` declaration with a hint; the crate pre-checks member types and
  array strides itself so most of them are reported by member name. `EffectError::Interface`
  gained `location`. API additions beyond §5: `ParamError::Array` (single value on an array
  member), `Pass::get_slice`, `set_slice` zeroes the unset tail, `ParamInfo::{is_array,
  stride}`, `From<egui::Vec2 | egui::Color32> for ParamValue` (gamma, premultiplied) plus `ParamValue::linear_rgba`; no default values (WGSL has none).
- Phase 2c: `tests/wgpu_headless.rs` with 14 parallel tests crashed (SIGSEGV inside `libvulkan`
  / `libVkLayer_khronos_validation`, concurrent `vkCreateInstance`, Mesa radv, debug build with
  wgpu's validation layer). The test helper `gpu()` now serializes instance / device creation
  with a test-only mutex; rendering still runs in parallel.
- Phase 3a (F3): API deviations from §4, all additive or safer. `is_alive(now)` takes the clock
  (true iff an impulse's age < life at `now`), so a caller that forgets `retire` cannot keep
  repainting; `pack` also skips expired impulses. `request_repaint_if_alive(ctx, now)` is the one
  method touching a `Context` (per the phase brief; `pack` / `apply` stay `Context`-free, so the
  §4 offline guarantee holds). `push` / `push_point` return `bool` (false: non-finite input or a
  rect without area, nothing stored). `with_capacity(1..=16)`, `IMPULSE_SLOTS`,
  `IMPULSES_PARAM` / `IMPULSE_COUNT_PARAM` constants. No existing geometry helper fits the
  point -> layer-rect-uv conversion (`pointer_uniform` targets capture uv, which depends on the
  chain's padding), so it is a private helper in `impulses.rs`. `click_ripples` takes `life` from
  the `Impulses` as a named param; the ring fades by `(1 - age/life)^2`, so expiry never pops.
- Phase 3b (F6): §7's "`fl.texel` stays the capture's, kernels step with `out_texel()`" left
  `src_px` stepping CAPTURE pixels, which on a scaled INPUT is a fraction of an input pixel and
  whose `src()` edge clamp (half a capture texel) read past the used region of a smaller image
  (garbage of the grow-only target). Resolution: `fl.texel` / `fl.size_px` /
  `fl.pixels_per_point` stay the capture's (effect distance maths is scale-invariant; the
  `original()` clamp stays right), and a new member `internal_src_texel` (one pixel of the
  draw's input) drives `src_px`, `src_texel()` and the `src()` clamp. At full resolution it is
  `fl.texel` bit for bit, so unscaled chains are unchanged (bit-identity suites green).
  `gaussian_blur` / `ripple` now offset by `d * fl.texel` instead of `src_px(d)` (same float
  operations, capture-pixel distances under any scale).
- Phase 3b: the kawase level count is fixed when the chain is built, but the pixels per point
  is known only at paint time. The tap offset adapts at runtime (`o = radius_px / span - 1`,
  clamped to `0..4`), so the reach is `radius` points from about 0.75 to 3 pixels per point;
  outside that range the blur is slightly wider (the pyramid's own reach) or narrower (taps
  capped). Declared sample radius = `max(radius, span)` (pixels = points at >= 1 px/pt).
- Phase 3b: `tests/offline_equivalence.rs` crashed (SIGSEGV inside
  `libVkLayer_khronos_validation`, `vkSetDebugUtilsObjectNameEXT` during a buffer creation)
  in about half the `--ignored` runs, ALSO without the new tests (reproduced 5/6 with them
  skipped). Locking only instance creation did not help; its `gpu()` now returns a lock held
  for the whole test (the file runs serially, ~6 s).
- Phase 3b: the ping-pong pool of a plan (`ChainPlan::target_px`) is the WINDOW even for a chain
  of scaled passes only: sizing it to the largest scaled viewport would let the UI allocate
  smaller targets than the offline run (which pins them to the frame), changing the bilinear
  weights of the upsample; with the window both match bit for bit (tested).
- Phase 6 (F5): deviations from §10. The FIRST compile is asynchronous too (`current()` is
  `Option<&Effect>`, `None` until it succeeds; `watch` fails only when the thread cannot spawn):
  a synchronous first read + naga compile would be file I/O and heavy work on the GUI thread
  (AGENTS.md §5). The re-bind helper is the free function `hot::rebind` (re-exported), not
  `Pass::rebind`: `pass.rs` belonged to the parallel phase 3c, and the public `Pass` API suffices,
  except that a sample-radius override is inferred (effective radius != old effect's radius).
  Moving it to `Pass::rebind` later is a pure relocation. `watch_state` is left to F7 (phase 4).
  `HotReloadEvent` carries no payload (the error is read through `error()`); `Reloaded` wins when
  one poll drains both a reload and a later failure, so a caller never misses a rebind.
  On wasm32 the feature is accepted and compiles to nothing (a `compile_error!` would break
  `--all-features` wasm builds).
- Phase 3c (F8): §6's `tex_linear(i, uv)` / `tex_nearest(i, uv)` (crate samplers) are NOT
  provided: with the texture's own sampler they would read one texture through two samplers,
  which naga's GLSL writer rejects (Phase 1 finding). Each texture has ONE sampler (its egui
  `TextureOptions`, which the app chooses at `load_texture`) and exact reads go through
  `tex_load(i, px)` = `textureLoad`, which naga's writer combines with that same sampler (one
  GLSL `sampler2D`, `textureLod` + `texelFetch`; pinned by
  `user_texture_helpers_translate_one_unit_per_slot`). Nothing else used `fl_nearest_sampler`, so
  binding 7 is gone (layout: 7 entries in group 0; glow units 0..=6, no sampler objects); F7's
  `state(uv)` should read with `textureLoad` too. The glow failure-scene test now provokes the
  "multiple samplers" error with `fl_user0_sampler` on `fl_src_texture` (the layers attach slot 0);
  `textureNumLevels` was no alternative (naga translates it for GLSL 330).
- Phase 3c: missing vs. unresolvable textures are two different contracts. A slot an effect
  reads with nothing attached is known on the GUI thread from the IR, so it is a typed
  `TextureError` from `FilterLayer::check_textures` (what `paint` runs: the layer draws nothing,
  logged once per context and message; offline `OfflineError::Texture`) — never a dummy. Whether an
  attached id is alive is only known where the renderer is (wgpu: the end-of-pass plugin; glow:
  the paint callback), so an unknown / freed id skips the layer there, is logged once per id and
  listed by the new `failed_textures(ctx)` (`TextureFailure { id, error }`), permanent per id since
  egui never reuses `TextureId`s. Slot detection is conservative: a computed slot index (or a
  helper call in an unused function) demands every slot; the helpers take the slot as their first
  argument by contract.
- Phase 3c: API deviations from §6. Offline textures are an owned `OfflineTexture` (not a
  borrowed `FrameRef` in the options, which would give `OfflineOptions` a lifetime and break its
  `Clone` / video use): `OfflineOptions::with_texture(slot, OfflineTexture)`. The renderer keeps
  each slot's upload while the same `OfflineTexture` (identity, `PartialEq`) stays attached, so
  the roadmap's "uploaded every call" and a separate `VideoProcessor::set_texture` are unneeded: a
  video passes the same texture with every frame's options. `OfflineTexture::from_color_image`
  holds egui's upload bytes, the equivalence path. `FilterLayer::texture_at` (accessor) added.
- Phase 3c: the wgpu effect pipelines need groups 1..=3 set for EVERY draw (wgpu validates every
  group of the pipeline layout), so empty slots get a dummy bind group (1x1 transparent + the
  linear sampler); composite pipelines keep a group-0-only layout and bind nothing more. egui's
  per-texture bind groups (and `TextureCopies`' kept copies, and native textures) bind to the
  crate's layout because every such layout is created from one descriptor
  (`gpu::egui_texture_entries`); egui-wgpu textures are `Rgba8Unorm`, so `tex()` returns the same
  gamma bytes as `src()` on every backend (egui_glow uploads RGBA8 too, `srgb_textures = false`).
