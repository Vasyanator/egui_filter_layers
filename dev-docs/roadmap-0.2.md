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
- Phase 4 (F7 persistent state: `Effect::new_state` / `is_state`; `FilterLayer::id` (required, `StateError::MissingId` via `check_state`) / `state(StateOptions)`; `StateFormat` RGBA16F (renderable + filterable or refused: `failed_states` / `StateFailure`, offline `OfflineError::State`) / opt-in RGBA8; `reset_state(ctx, id)`, `state_count(ctx)`; `src/state.rs` (`StateRegistry`: per `(ViewportId, Id)`, one step per frame, eviction after > 120 frames AND > 2 s); `plan.rs` state steps (`StepKind`, `Slot::StateA/B`, `StatePlanInput` / `StatePlan`: clear, sizes, final slot); prelude `state` / `state_load` / `state_px` / `state_size` / `state_texel`, `fl.pass_index` documented; `FlUniforms` 448 B; wgpu `WgpuResources::states` + `Target::ensure_kept` + pipelines keyed by format; glow `GlowState::states` + `RenderTarget::ensure_kept`; offline state carried across runs, `reset_state`, `with_state`, `NotTileable`; `VideoProcessor::reset_state`, per-stream reset; `HotOptions::with_state`; presets `trails`, `wave_sim` + `wave_display` + `WaveObstacle`; GPU tests wgpu (Vulkan + GL) / glow / offline / video): done.
- Phase 5 (F9 shape masks: public `Mask` (`RoundedRect` default / `Shape`), `MaskError`, `FilterLayer::mask` / `layer_mask` / `check_mask` (`corner_radius` = shorthand), `OfflineOptions::with_mask`, `OfflineError::Mask`; `src/mask.rs` (`tessellate`, `JobMask`, `CoverageMeshes::pack` -> `PackedCoverage`); `LayerJob::mask`; `geometry::mask_transform` (layer points -> capture pixels through the layer transform); `plan.rs` `Slot::Coverage`, `CoverageStep`, `ChainPlan::with_coverage`, `COVERAGE_MASK_FLAG` (`params[3].x`), `mesh_locals` / `coverage_locals`, `Allocations::coverage`; `composite.wgsl` `fl_coverage` selects SDF / coverage alpha (snap kept); `mesh.wgsl` moved to `src/shaders/`; wgpu `coverage.rs` (shared mesh pipeline, rasterizer used by `chain::record`, UI and offline); glow `coverage.rs` + `glsl::translate_mesh` (naga translation, no hand-written GLSL); GPU tests wgpu (Vulkan + GL) / glow (incl. 4x MSAA) / offline (UI equivalence, tiled == whole, refusals)): done.
- Phase 7a (demo scaffold: `examples/demo/` directory example — `main.rs` shell + `--scene`, `record.rs`, `backdrop.rs`, `stack.rs` (+ kawase / bloom entries and custom `struct Params` via `params_ui.rs`), `scenes/mod.rs` `Scene` trait + `SCENES` registry, scene "stack"; no new scenes yet): done.
- Phase 8 (docs + release prep: version 0.2.0, CHANGELOG `[0.2.0]` with every breaking change, README feature overview / prelude table / compatibility / demo scenes / Wayland capture note, `WaveObstacle` re-exported at the root, crate docs overview, direct-minimal-versions re-checked (dev-dep `glutin` floor 0.32.3 for eframe 0.36), package list + publish dry run): done.
- Final review fixes (per-viewport eviction clocks in `StateRegistry`; `reset_state` as a `ResetTicket` claimed where the registry's `begin` runs, so an unrun paint keeps it; malformed user mask meshes refused (`MaskError::InvalidMesh`) and skipped by `CoverageMeshes::pack`; explicit `LayerOutcome` arms in glow; `#[non_exhaustive]` on `Mask`, `HotReloadEvent`, `WaveObstacle` (+ `new`), `PointerState` (+ `new`), `Impulse`, `ImpulsePack`, `Rebound`; demo captures written on a `FrameWriter` thread; hot-reload mtime granularity documented): done.
- Phase R0 (§19 D1: additive coverage union — wgpu `COVERAGE_BLEND` (`One, One`), `mesh_pipeline` takes the blend (backdrop keeps `EGUI_BLEND`), glow `blend_func(ONE, ONE)`; mask / lib.rs / README / CHANGELOG text: abutting pieces leave no seam; GPU tests `abutting_mask_pieces_leave_no_seam` wgpu + glow (0 / 4x MSAA); no F9 expectation changed): done.
- Phase R1a (§19.3 regions core + wgpu: `Region`, `Mask::Regions`, `MaskError::TooManyRegions`, `MAX_REGIONS`, `REGION_NONE`, `FilterLayer::regions`, `Effect::reads_regions`; `mask.rs` `RegionMeshes` / `PackedRegions` / `tessellate_regions` / `JobMask::{pack, textures, regions}`; binding 7 `fl_region_texture` + prelude `region()` / `region_at(uv)` / `REGION_NONE`; `FlUniforms` 464 B (`internal_region_scale` 184, `_offset` 192, `params` 208); `mesh.wgsl` `fs_region`, `MeshLocals.region_id` / `alpha_threshold`; `plan.rs` `Slot::RegionId`, `Bindings::region`, `CoverageStep::regions`, `ChainPlan::with_regions`, `region_locals`, `REGION_PASS_THRESHOLDS`, `Allocations::region`; wgpu `coverage.rs` region pipeline + target + dynamic-offset locals, `chain.rs` 8-entry layout; glow units shifted (group 0 = 0..=4, user = 5..=7) and region masks refused (`Unsupported::RegionMasks`, logged once) until R1b; offline refuses region masks (`OfflineError::InvalidOption`, option `mask`) until R1c; GPU tests wgpu (Vulkan + GL) grid / NONE / overlap, glow refusal): done.

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
- Phase 4 (F7): §8's `state(uv)` "nearest by default" + `state_linear(uv)` is infeasible (binding 7
  is gone and naga's GLSL writer rejects one texture through two samplers). `state(uv)` samples
  through the one linear `fl_sampler` (smooth display reads; exact at texel centres), and the
  simulation reads exact texels with `state_load(px)` / `state_px(offset)` (`textureLoad`, clamped
  = reflecting border, row flipped on GL). State + src + a user texture in one module translates
  for every GLSL dialect (`state_helpers_translate_with_src_and_a_user_texture`). `FlUniforms` grew
  by `internal_state_texel` (168) and `internal_state_size` (176): `params` moved 176 -> 192, the
  block is 448 bytes. `fl.pass_index` is now set (index among the chain's passes of its kind) and
  documented. A display pass reading the state in a chain with no state pass sees the 1x1 zero
  dummy — identical to a state never written, so no id is required there.
- Phase 4: eviction is "not painted for > 120 frames AND > 2 s", not "whichever first": time alone
  would evict every state of an app that idled > 2 s on its first frame back, frames alone every
  state of a fast app within 0.5 s at 240 fps. Checked at every paint (both UI backends). Frames
  are counted on the ENTRY'S OWN viewport (the registry keeps each viewport's latest frame): the
  first version treated every foreign-viewport entry as frame-stale, so an idle child viewport
  lost its state after 2 s of root animation (final review).
- Phase 4: states are keyed by `(ViewportId, Id)`: egui 0.36 counts `cumulative_frame_nr` per
  viewport, and eframe renders every viewport through one renderer (shared `CallbackResources` on
  wgpu), so "same frame" is only meaningful per viewport. One id in two viewports = two states
  (documented). The same id painted twice in one frame of a viewport: the first paint steps it,
  later ones skip the state steps and show the state as it is (debug-logged once per key) — §15
  risk 2's plan.
- Phase 4: grow-only state textures must KEEP the latest content when they grow (the read mapping
  uses the previous size). Growing the write target lazily per step would change allocations
  after the uniform buffer is built, so both backends grow the two state textures up front with a
  copy (wgpu `Target::ensure_kept`: new texture, `copy_texture_to_texture` of the old one, state
  textures carry `COPY_SRC | COPY_DST`; glow `RenderTarget::ensure_kept`: new texture + `NEAREST`
  blit) — tested by layers growing far past their allocation at scale 1 and 0.5 on wgpu (Vulkan,
  GL) and glow.
- Phase 4: reset is backend-neutral: `reset_state` stores the id in the context's temp data and
  the next stateful paint of that id consumes it into `LayerJob::reset_state`; a new or reset
  state is cleared by the plan (`StatePlan::clear`), so no draw ever reads a dummy as "zero
  state" with the wrong size. wgpu effect pipelines are cached per `(module, output format)`:
  an RGBA8 state pass on a float-intermediate device renders into another format.
- Phase 4: the RGBA16F policy is the float-intermediates check on both backends (wgpu
  `renderable_and_filterable`; glow: the renderability probe — half-float linear filtering is core
  in GL 3.3 / GLES 3.0). Refusals are latched per format in `failed_states()` (`StateFailure`), a
  new accessor beside `failed_effects()` (a format is not an effect). Not runnable here: no device
  at hand lacks RGBA16F (see Not verified in the phase report).
- Phase 4 (offline): `OfflineOptions::with_state` only CONFIGURES the state (scale, format); a
  chain with a state pass always carries it (the renderer is the identity). `process_video`
  resets the state with the frame clock at the start of every stream; a failed run drops it.
- Phase 4 (`wave_sim`): the velocity-driven impulses inject momentum and the discrete wave
  equation conserves `sum(v)` (reflecting borders), so the whole surface drifted without bound
  (measured: displayed energy grew linearly for 120 frames; f16 would overflow under a held
  pointer). A restoring term `restore` (default 0.002 per step, stability `speed + restore / 8 <=
  0.5`) pulls the mean back to 0. Obstacles are rounded rects in layer-rect uv (`WaveObstacle`, at
  most 8); their radii are packed into `array<vec4<f32>, 2>` (F4 rejects `array<f32, N>`). Steps are
  per paint; `trails(decay)` takes the fraction kept per SECOND (`pow(decay, fl.delta_time)`), so
  it is frame-rate independent. Neither preset is animated by itself (documented: repaint while
  moving). `HotEffect::watch_state` became `HotOptions::with_state(true)`.
- Phase 5 (F9): roadmap risk 3 did not materialise: naga's GLSL writer translates `mesh.wgsl`
  (`vs_main` + `fs_main`) for GLSL 330, ES 300 and WebGL2 with explicit attribute locations and the
  `u32` colour as an integer attribute (`glVertexAttribIPointer`); pinned by
  `the_mesh_shader_translates_for_every_target`, no hand-written fallback. naga names the
  `MeshLocals` block PER STAGE, so both names are bound to the binding point.
- Phase 5: deviations from §9. Tessellation is `mask::tessellate` (an `epaint::Tessellator` with the
  context's pixels per point, tessellation options and font atlas size) instead of `ctx.tessellate`:
  the latter uses egui's pre-rasterized discs, so small circles would sample the font atlas and the
  UI would differ from the offline tessellation; without them only text samples the atlas and UI and
  offline tessellate identical meshes. The coverage texture holds the run's WINDOW at its origin
  (like the source) instead of the whole capture, so offline tiles rasterize their own window
  (tiled == whole, tested) and a huge frame never needs a whole-frame coverage texture. Only the
  composite binds the coverage; effects get no `mask(uv)` helper in 0.2 (for a rounded rect it would
  read the white dummy = 1, a silent wrong answer; left out rather than faked). The coverage is the
  shape's alpha, so a translucent fill filters partially (documented feature).
- Phase 5: egui fills CONVEX paths only; a concave outline (the star of the tests) must be built
  from convex pieces or a mesh. Adjacent pieces' feathered edges overlap into faint interior seams
  (union of two ~0.5 coverages = 0.75); documented in the README, not worked around. Superseded
  by phase R0 (§19 D1): the union is additive now, abutting pieces leave no seam.
- Phase 5: mask texture failures. glow: an id `Painter::texture` does not know skips the layer and is
  reported by `failed_textures`. wgpu: the same for user texture ids; a font atlas the plugin cannot
  resolve yet (size mismatch with the CPU atlas) only skips that frame (it cannot be permanent).
  Paint callbacks in a mask are refused on every path (`MaskError::Callback`, logged once per
  context and message; offline `OfflineError::Mask`), offline also text and textured meshes.
- Phase 5: equivalence measured: a star mask over invert + blur, UI vs offline: exact where the
  coverage is 0 or 1, max 1 step on the anti-aliased outline (~450 values at 1 and 2 ppp); tiled vs
  whole offline: bit-identical.

- Phase R0 (D1): no F9 test expectation changed — the existing star / circle / text / transformed
  checks sample interiors and exteriors only, and the offline F9 cases compare UI vs offline, which
  share the rasterizer. The new seam test was confirmed to detect the old rule: with egui's blend
  restored the shared column read `[64, 191, 0]` (0.75 coverage), with the additive union
  `[0, 255, 0]`.
- Phase R1a: the D1 over-estimate at OUTER junctions is larger than §19.1 says (≤ 0.25): where two
  abutting pieces' outer corners meet, egui's feathered corner is 0.5 at its geometric corner for
  each piece, so the sum is 1.0 where the true coverage is about 0.5 — one pixel per junction
  (pinned in `regions_tint_each_cell_and_own_their_edges`, row 60 of the left edge).
- Phase R1a: a mid-pixel shared edge stores two ~0.5 feathers as 8-bit alphas that sum to ~254/255,
  not 1 (the composite snaps only within 1/1024). On gamma targets that is ≤ 1 step; on `*Srgb`
  UI targets the linear blend magnifies it to ~14 gamma steps of backdrop in that one column
  (measured `[14, 254, 254]` over red). Not worked around; the sRGB variant of the grid test
  compares only fully covered pixels.
- Phase R1a deviations from §19.3: `JobMask::coverage()` stays the plain-shape accessor (tests);
  backends call `JobMask::pack()` (-> `PackedRegions`, empty ranges for a shape mask) and
  `JobMask::textures()`, so there is no separate "packed union" accessor. `FlUniforms` has 28
  members (26 + 2; §19.3.8's "27" miscounted), 464 bytes as designed. `Region::shape_ref` is not
  `#[must_use]` (clippy: `&Shape` already is). Temporary refusals: glow `Unsupported::RegionMasks`
  in `glow_backend/mod.rs` (typed, latched, logged once; removed by R1b), offline
  `OfflineError::InvalidOption { option: "mask" }` from `OfflineOptions::validate` (no new public
  variant; removed by R1c). The prelude now reserves `region`, `region_at` and `REGION_NONE`: a
  snippet declaring one of them fails to parse (redefinition), like any prelude name.
- Phase R1a: the id passes are verified by mutation: with pass A dropped (thresholds `[0.5, 0.5]`)
  the grid test and `region_none_outside_regions` fail on `REGION_NONE` fringe pixels.

# 19. Multi-region layers (R1 regions, R2 benchmark, R3 entries) — design and phases

Draft section for `dev-docs/roadmap-0.2.md` (append as §19; §17 gets one progress line per phase
below). Baseline: commit 223daf1, version 0.2.0 prepared and unpublished, so changes against the
0.2 working tree are free; the only 0.1.0 surface touched is listed in §19.9.

Owner's request, restated: several effects / areas under ONE layer instead of many layers —
(R1) one chain whose effect behaves differently per REGION of the layer (non-overlapping regions,
priority when they overlap), (R3) several independent (mask, chain) ENTRIES in one layer, each
capturing only its own window, and (R2) a benchmark that tells users when either pays off.

## 19.0 Architectural check (AGENTS.md §2.5)

* Trigger 1 (no seam): none. R1 is a second rasterized texture next to the F9 coverage step,
  read by the prelude; R3 is a loop over entries around the existing single-entry pipeline.
* Trigger 2 (one decision, many owners): R3 WOULD fire it if written naively — "in which order
  are entries captured, run and composited, and which earlier entry is copied into a later one's
  capture" would be decided in `wgpu_backend/render.rs`, `glow_backend/render.rs` and
  `offline/mod.rs`, the exact failure `plan.rs` was created against (§0). The design therefore
  puts that schedule in `plan.rs` (`plan::schedule`, §19.4.3) BEFORE any backend loops over
  entries; no refactor of existing code is required first, so no §2.5 stop is raised.
* File size: `plan.rs` 956 → ~1250, `layer.rs` 692 (new `Entry` goes to a new `src/entry.rs`),
  `glow_backend/render.rs` 845 → ~1100, `offline/mod.rs` 853 → ~1050. All under the 3000 line
  awareness mark; report sizes in each phase.
* §14 temporary: phase R3a leaves glow refusing multi-entry layers (logged once, typed
  `LayerOutcome`), removed by R3b in the same release. Stated here, bounded by the phase table.

## 19.1 Decisions the manager must confirm (D1–D9)

| # | decision (recommended first) | alternative |
|---|---|---|
| D1 | Mask coverage union becomes ADDITIVE (`One, One`, saturating) instead of egui "over": abutting pieces / regions get coverage 1 at the shared edge (feathers are complementary ramps), so adjacent regions show no seam. Changes F9 output only at junction pixels where two pieces' OUTER feathers cross (over-estimate ≤ 0.25, one pixel) | keep "over"; document "regions must overlap by ≥ 1 px" and accept the 0.75 seam of §18 phase 5 |
| D2 | Regions are a mask variant `Mask::Regions(Vec<Region>)` (+ `FilterLayer::regions(..)` shorthand); the union of the regions IS the mask | a separate `FilterLayer::regions` field beside the mask (two sources of coverage) |
| D3 | R3 API: `Entry` per (rect, chain, mask, textures, state); `FilterLayer::new(rect, chain)` = one entry; layer-level builders (`mask`, `corner_radius`, `texture`, `state`) apply to EVERY entry; accessors `chain()` / `layer_mask()` / `texture_at()` return the FIRST entry's (documented) | make the layer-level builders errors on multi-entry layers |
| D4 | glow composites every entry of a multi-entry layer with the BLENDED erase+add composite (the wgpu UI's two modules, GL blend funcs); single-entry layers keep the replace composite unchanged | replace in `Stacked`, blended only in `Independent` (two glow paths) |
| D5 | State per entry, key `layer_id.with(entry_index)`; `reset_state(ctx, layer_id)` resets all entries; offline keeps one state per entry index | refuse stateful chains in multi-entry layers in 0.2 (`StateError::MultiEntry`, later removed = churn) |
| D6 | Offline multi-entry layers refuse tiling (`OfflineError::NotTileable`): the frame must fit `max_tile_size` (default 4096; raise it for larger frames). R1 (regions) tiles like F9 | per-entry tiling (+M, a second tiler over sub-rects) |
| D7 | R2 measures wall time per frame including GPU completion (the harnesses already wait / read back), K frames after warm-up; no timestamp queries | `TIMESTAMP_QUERY` in every pass (intrusive in `chain.rs`, native only) |
| D8 | `Stacked` on wgpu (UI and offline) = a HAND-OFF: the replace composite of an earlier entry drawn into a later entry's capture texture (one implementation in `chain.rs`); glow = natural blit order. Both meet one contract (§19.4.2); glow differs ≤ 1 step at AA edges | offline copies from its output texture (natural, but UI ≠ offline by ≤ 1 step at overlaps) |
| D9 | Phases R3a (core + schedule + wgpu UI + offline) then R3b (glow) sequential, glow refusing multi-entry in between | one L worker for both backends |

## 19.2 Shared foundations touched

### 19.2.1 Binding layout: binding 7 returns as the region-id texture
Binding 7 (`fl_nearest_sampler`) was removed in phase 3c (§18), so `@group(0)` has a free slot.
New row of the §1.1 table:

| binding | name | type | filled by | notes |
|---|---|---|---|---|
| 7 | `fl_region_texture` | texture_2d | both | R1: region ids, RGBA8, `.r * 255` = id, 255 = none; 1x1 WHITE dummy (id 255) without regions |

`effect/mod.rs`: `BINDING_REGION_TEXTURE = 7`; `effect/interface.rs:25` `PRELUDE_RESOURCES` gains
`("fl_region_texture", 0, 7)` (13 entries). wgpu `chain.rs:162-173` layout gets an 8th
`texture_entry`; every draw's bind group (`chain.rs:404`) binds the region target or the white
dummy. glow `glsl.rs:42` `GROUP0_TEXTURE_BINDINGS` becomes `[src, original, state, coverage,
region]` (units 0..=4), `glsl.rs:46` `USER_UNIT_BASE = 5` (user units 5..=7; 8 < 16 GLES 3.0
units). `glow_backend/render.rs` `bind_textures` / `cleanup` follow the table (no other change:
the unit order IS the table). Budget check: WebGL2 4 bind groups unchanged, 8 sampled textures,
4 samplers.

### 19.2.2 `FlUniforms` growth (pinned by `offsets_match_the_wgsl_struct_layout`)
Two `vec2<f32>` after `internal_state_size` (176): `internal_region_scale` at 184,
`internal_region_offset` at 192; `params` moves 192 → 208; `UNIFORM_BLOCK_SIZE` 448 → 464
(`pass.rs:348`, `offsets::PARAMS`, `FrameUniforms.region: [[f32; 2]; 2]` neutral identity).
`plan::draw_uniforms` sets it exactly like `cover` (`plan.rs:513-516`) for a draw whose
`Bindings.region` is `Some`.

### 19.2.3 `mesh.wgsl` and `MeshLocals`
`MeshLocals.reserved: vec2<f32>` (`mesh.wgsl:30`) becomes `region_id: f32, alpha_threshold: f32`
— same 32 bytes, `MESH_LOCALS_SIZE` unchanged; `plan::mesh_locals` (`plan.rs:558`) takes the two
values (the backdrop re-render and the coverage draw pass `0.0, 0.0`). New fragment entry:

```wgsl
// R1: writes the region id where the mesh covers the pixel more than `alpha_threshold`.
@fragment
fn fs_region(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = in.color.a * textureSample(tex_color, tex_sampler, in.tex_coord).a;
    if a <= locals.alpha_threshold { discard; }
    return vec4<f32>(locals.region_id / 255.0, 0.0, 0.0, 1.0);
}
```
No blend (replace), so the LAST region drawn owns the pixel. glow's `glsl::translate_mesh`
takes the fragment entry name (`fs_main` | `fs_region`); the existing
`the_mesh_shader_translates_for_every_target` test covers both.

## 19.3 R1 — regions inside one chain

### 19.3.1 Public API (`mask.rs`, `layer.rs`, `lib.rs`)
```rust
/// One region of a `Mask::Regions` mask, in LAYER points. Its index in the list is its id.
#[derive(Clone, Debug, PartialEq)] #[non_exhaustive]
pub struct Region { shape: egui::Shape }
impl Region {
    pub fn rounded_rect(rect: Rect, corner_radius: impl Into<CornerRadius>) -> Self; // Shape::rect_filled(rect, r, WHITE)
    pub fn shape(shape: egui::Shape) -> Self;   // any shape, alpha = its coverage (as Mask::Shape)
    pub fn shape_ref(&self) -> &egui::Shape;
}
pub enum Mask { RoundedRect(CornerRadius), Shape(Shape), /* new */ Regions(Vec<Region>) }
pub enum MaskError { .., /* new */ TooManyRegions { count: usize, max: usize } /* max = MAX_REGIONS */ }
/// Ids are `0..MAX_REGIONS`; `REGION_NONE` (255) is "no region here" (the prelude's `region()`).
pub const MAX_REGIONS: usize = 255; pub const REGION_NONE: u32 = 255;
impl FilterLayer { pub fn regions(self, regions: impl IntoIterator<Item = Region>) -> Self; } // = mask(Mask::Regions(..))
impl Effect { pub fn reads_regions(&self) -> bool; }  // IR scan for `fl_region_texture` (like reads_texture); informational
```
Semantics: the union of the regions is the layer's mask (coverage texture, F9 path); the region
texture holds, per capture pixel, the id of the region that owns it. `Mask::Regions(vec![])` is
a mask covering nothing (valid, draws nothing). `check` / `check_offline` visit every region's
shape (callbacks, malformed meshes; offline: text, textured meshes) and reject > 255 regions.
Per-region parameters need NO new mechanism: F4 arrays (`struct Params { tint: array<vec4<f32>,
8> }`, `Pass::set_slice`) indexed by `region()`; scalars are packed four per `vec4`
(`p.gain[r / 4u][r % 4u]`; `array<f32, N>` is rejected by F4, `effect/params.rs:403-407`).
Documented in the README with this idiom.

### 19.3.2 Prelude additions (`shaders/prelude.wgsl`)
```wgsl
@group(0) @binding(7) var fl_region_texture: texture_2d<f32>;
const REGION_NONE: u32 = 255u;
// Id of the region owning capture uv `uv` (exact texel: ids must never be interpolated), or
// REGION_NONE outside every region and on layers without regions.
fn region_at(uv: vec2<f32>) -> u32 {
    let storage = fl_storage_uv(uv, fl.texel, fl.internal_region_scale, fl.internal_region_offset);
    let px = vec2<i32>(floor(storage * vec2<f32>(textureDimensions(fl_region_texture))));
    return u32(round(textureLoad(fl_region_texture, px, 0).r * 255.0));
}
fn region() -> u32 { return region_at(fl_frag_uv); }
```
One read path (`textureLoad`) → one GLSL combined sampler, no "two samplers" conflict (§18
phase 1). `u8 → unorm → f32 * 255 → round` is exact. `region_at` lets a kernel stay inside its
own region (`if region_at(uv + d) == region()`), the tool for "blur that does not bleed across
regions". Public names added to the crate-docs table: `region()`, `region_at(uv)`, `REGION_NONE`.

### 19.3.3 Ownership rule at edges (the seam / halo design)
Two facts drive it: (a) effects run on every capture pixel, and the composite mixes the filtered
value by coverage, so a pixel with coverage 0 < c < 1 whose id is NONE would be filtered with the
wrong (sentinel) parameters and show as a fringe; (b) an id must switch HARD between two
regions (an interpolated id is garbage). Rule, implemented as two id passes over all regions in
list order into the region texture cleared to id 255:

1. pass A, `alpha_threshold = 0.0`: every pixel a region touches at all gets that region's id
   (later regions overwrite) — no touched pixel is ever NONE, so no sentinel fringe;
2. pass B, `alpha_threshold = 0.5`: every pixel a region covers by more than half gets that
   region's id — interior boundaries sit at the 50 % contour of the LATER region ("later in the
   list wins"), not shifted by feathers.

Result: within the mask the boundary between two regions is pixel-hard at the geometric edge;
the silhouette of the union is antialiased by the coverage texture as in F9; with D1 an
abutting edge has coverage 1 (no backdrop seam). A soft transition BETWEEN two regions with
different parameters is not a region feature (it needs the effect evaluated twice and blended):
that is R3 (two entries), documented as such. Cost: 2 × (mesh draws of all regions) per frame,
meshes are tiny.

### 19.3.4 Plan changes (`plan.rs`, single owner)
```rust
pub(crate) enum Slot { Source, Ping, Pong, StateA, StateB, Coverage, /* new */ RegionId }
pub(crate) struct Bindings { .., /* new */ pub region: Option<Slot> }   // Some(RegionId) for every PASS step of a region run; None for the composite
pub(crate) struct CoverageStep { pub used_px, pub window_origin, /* new */ pub regions: u32 } // 0 = plain F9 mask; else id passes A and B run after the coverage draw
impl ChainPlan { pub(crate) fn with_regions(self, count: u32) -> Self }  // with_coverage() + binds RegionId on every step; composite unchanged (COVERAGE_MASK_FLAG)
pub(crate) struct Allocations { .., pub region: [u32; 2] }
pub(crate) fn region_locals(transform: &MaskTransform, step: &CoverageStep, region: u32, threshold: f32) -> [u8; MESH_LOCALS_SIZE];
```
Both textures hold the run's WINDOW at their origin like the coverage (offline tiles rasterize
their own window ⇒ R1 tiles exactly like F9). `draw_uniforms` maps `region` with the source's
`Content` in the region allocation. Unit tests: `with_regions` binds `RegionId` on every pass
step and not on the composite; the region mapping equals the coverage mapping when both
allocations are equal and differs otherwise; `region_locals` packs id and threshold at bytes
24 / 28.

### 19.3.5 Mask plumbing (`mask.rs`, `layer.rs`, `backend.rs`)
* `JobMask::Regions(Arc<RegionMeshes>)`, `RegionMeshes { regions: Vec<CoverageMeshes> }` —
  `resolve_mask` tessellates each region with the one `mask::tessellate` (same ppp / options /
  atlas size; a rounded-rect region is `Shape::rect_filled`, tessellated as a path since the
  tessellator has no pre-rasterized discs).
* `RegionMeshes::pack() -> PackedRegions { packed: PackedCoverage /* all meshes, list order */,
  regions: Vec<Range<usize>> /* draw-index range per region */ }` — the coverage draw runs
  `packed.draws` as today; id passes run `packed.draws[range]` per region with its locals.
* `JobMask::corner_radius()` → `ZERO`, `coverage()` → the packed union (unchanged callers);
  new `regions()` accessor. `CoverageJob` (wgpu) / glow `LayerTextures.coverage` gain
  `regions: &[Range<usize>]` (empty for F9).

### 19.3.6 Backends
* wgpu `coverage.rs`: `CoverageRasterizer` gains `region_pipeline` (`mesh_pipeline` with
  `fs_region`, `blend: None`) — `mesh_pipeline` takes `blend: Option<BlendState>`; the coverage
  pipeline uses `COVERAGE_BLEND` (D1, additive), the backdrop keeps `EGUI_BLEND`.
  `CoverageTargets` gains `region: Option<Target>` and a region-locals `GrowBuffer` whose bind
  group has `has_dynamic_offset: true` (one 32-byte block per region per pass at the device's
  uniform offset alignment, `set_bind_group(0, &group, &[offset])`). `record` clears the region
  target to `(1, 0, 0, 0)` and runs passes A and B after the coverage draw; returns both targets.
  `chain.rs` maps `Slot::RegionId` (view_of) and reports `Allocations.region`.
* glow `coverage.rs`: `region_program` (translate_mesh `fs_region`), `region_target`
  (`RenderTarget`, RGBA8, cleared to `(1, 0, 0, 0)`), blend disabled for id passes, locals
  re-uploaded per draw (`buffer_sub_data`, as `upload_uniforms` does); coverage draw uses
  `blend_func(ONE, ONE)` (D1). `render.rs`: `target_of(Slot::RegionId)`, `allocations().region`.
* offline: nothing beyond `Mask::check_offline` visiting regions; the rasterizer is `chain.rs`'s.

### 19.3.7 Refusals
`MaskError::TooManyRegions` (UI: logged once, layer skipped; offline: `OfflineError::Mask`).
`region()` on a layer without regions is DEFINED (NONE), not refused: a layer without regions
is a valid input; `Effect::reads_regions` exists for UIs that want to warn.

### 19.3.8 Tests
* unit (`mask.rs`): regions tessellate in order, `pack` ranges partition the draws, > 255 refused,
  `Mask::Regions(vec![])` valid; (`plan.rs`) §19.3.4; (`pass.rs`) offsets 27 members / 464;
  (`effect/`) `reads_regions`, `region()` compiles, interface accepts binding 7 only from the
  prelude; (`glsl.rs`) `region()` + `src()` + `tex()` translate with units `[0, 4, 5..]`;
  `glow_and_wgpu_upload_identical_blocks` extended with a region plan.
* `tests/wgpu_headless.rs`: `regions_tint_each_cell_and_own_their_edges` — a 2×2 grid of
  abutting rounded-rect regions, effect `mix(src, p.tint[region()], 0.5)`; interior of each
  cell has its tint, the shared edge shows NO backdrop (D1: coverage 1) and switches tint within
  1 px of the geometric edge; `region_none_outside_regions` — an effect returning red where
  `region() == REGION_NONE` paints nothing (masked out) and a plain rounded-rect layer with the
  same effect is fully red (dummy = NONE).
* `tests/glow_headless.rs`: the same two scenes (+ 4x MSAA).
* `tests/offline_equivalence.rs`: `region_masks_offline_equal_the_ui` (exact at coverage 0 / 1,
  ≤ 1 step on the outline, as F9) and `tiled_region_masks_equal_whole` (bit-identical).

## 19.4 R3 — several (mask, chain) entries in one layer

### 19.4.1 Public API (`src/entry.rs` new, `layer.rs`, `lib.rs`)
```rust
/// One filtered area of a `FilterLayer`: its own rect, mask, chain, textures and state options.
#[derive(Clone, Debug)]
pub struct Entry { rect: Rect, chain: FilterChain, mask: Mask, textures: [Option<TextureId>; 3], invalid_slot: Option<usize>, state: StateOptions }
impl Entry {
    pub fn new(rect: Rect, chain: impl Into<FilterChain>) -> Self;
    pub fn mask(self, Mask) -> Self; pub fn corner_radius(self, impl Into<CornerRadius>) -> Self;
    pub fn texture(self, slot: usize, id: TextureId) -> Self; pub fn state(self, StateOptions) -> Self;
    pub fn rect(&self) -> Rect; pub fn chain(&self) -> &FilterChain; pub fn layer_mask(&self) -> &Mask; pub fn texture_at(&self, usize) -> Option<TextureId>; pub fn state_options(&self) -> StateOptions;
}
/// How entries that overlap see each other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)] #[non_exhaustive]
pub enum Overlap { #[default] Independent, Stacked }
impl FilterLayer {
    pub fn new(rect, chain) -> Self;                         // one entry (unchanged call)
    pub fn from_entries(entries: impl IntoIterator<Item = Entry>) -> Self; // rect() = union of entry rects
    pub fn with_entry(self, entry: Entry) -> Self;           // appends; rect() grows
    pub fn overlap(self, Overlap) -> Self; pub fn overlap_mode(&self) -> Overlap;
    pub fn entries(&self) -> &[Entry];
    // existing: mask / corner_radius / texture / state apply to EVERY entry (D3); chain() / layer_mask() / texture_at() / state_options() = the first entry's; rect() = union.
    // check_textures / check_state / check_mask check every entry (error names the entry index: TextureError::NotAttached gains `entry: usize`, same for StateError::MissingId and a MaskError wrapper in the log message).
}
```
`paint` skips the layer when ANY entry fails a check (one layer, one decision; logged once).
An empty entry list or all entries with empty chains paints nothing. Contract: entries are
painted in list order; a later entry is composited over an earlier one where they overlap,
blended by its own coverage — identical on every backend.

### 19.4.2 Overlap modes — exact contract
Let `backdrop` be the screen content beneath the layer, `f_k` entry k's filtered image, `c_k`
its coverage. Composite in list order: `screen ← mix(screen, f_k, c_k)`.
* `Independent`: entry k's capture is `backdrop` (no entry's result). Where entries overlap,
  the later entry's filtered backdrop wins, blended by `c_k` at its AA edge.
* `Stacked`: entry k's capture is `backdrop` with every earlier entry j < k composited:
  `mix(capture_k, f_j, c_j)` over `region_j ∩ capture_k`, in gamma space with the replace
  composite's exact-at-0/1 mix. Like separate layers on glow today; UNLIKE separate layers on
  wgpu today (§13 gap: a lower layer is a skipped callback) — within one layer both backends
  give this contract.
How each backend meets it (D4, D8):
* glow: `Independent` = blit every entry's capture FIRST (a pool of capture targets), then per
  entry run + BLENDED composite (erase+add programs, `glBlendFuncSeparate(ZERO, ONE_MINUS_SRC_
  ALPHA, ZERO, ONE_MINUS_SRC_ALPHA)` then `(ONE, ONE, ONE, ONE)`, scissor on, into the caller's
  framebuffer). `Stacked` = per entry: blit (sees the framebuffer with earlier composites), run,
  blended composite. Single-entry layers: unchanged replace path (bit-identity of the suite).
* wgpu UI: captures re-rendered per entry from ONE beneath selection / tessellation (the union of
  the entries' padded capture rects clips the selection; per-entry scissors); `Independent`:
  run every entry, composites (erase+add, existing) drawn in list order in `paint`. `Stacked`:
  after entry j's chain, for every k > j whose capture intersects `region_j`: a HAND-OFF draw =
  the replace composite of j (`composite_module`, gamma, `Rgba8Unorm`) into `capture_k` with
  `LoadOp::Load`, viewport `region_j ∩ capture_k` in k's capture pixels, uniform block from
  `plan::composite_for_origin(&plan_j, origin = capture_j.xy - capture_k.xy)`. Outside j's mask
  the composite writes `original_j` = j's re-render, byte-identical to k's re-render there (same
  meshes, same clear, dithering keyed on screen position), so the hand-off is exactly the
  contract's `mix`.
* offline: the same hand-off code (`chain.rs`), see §19.4.6 ⇒ UI == offline bit for bit for
  both modes on a non-sRGB target; glow ≤ 1 step at AA pixels of overlaps (hardware blend
  rounding), no cross-backend bit-identity is claimed today either.

### 19.4.3 Schedule — the single owner (`plan.rs`)
```rust
/// Whether a backend's capture sees the results of earlier composites (a framebuffer blit) or not (a re-render / a copy of the pristine frame).
pub(crate) enum CaptureSource { Framebuffer, Pristine }
pub(crate) enum Op { Capture(usize), Run(usize), HandOff { from: usize, into: usize, viewport: PxRect /* in `into`'s capture pixels */, origin: [i32; 2] }, Composite(usize) }
/// The ops of one multi-entry paint, in order. `geometry[k]` = entry k's LayerGeometry (screen pixels).
pub(crate) fn schedule(geometry: &[LayerGeometry], overlap: Overlap, source: CaptureSource) -> Vec<Op>;
pub(crate) fn composite_for_origin(plan: &ChainPlan, composite_origin: [i32; 2]) -> CompositeStep;
```
Emitted orders: `Framebuffer`+`Stacked` → `[Cap0 Run0 Comp0 Cap1 Run1 Comp1 …]`;
`Framebuffer`+`Independent` → `[Cap0..CapN, Run0 Comp0, Run1 Comp1, …]`; `Pristine` (either
mode) → `[Cap0..CapN, Run0, HandOff(0→k)*, Comp0, Run1, HandOff(1→k)*, Comp1, …]` with hand-offs
only for `Stacked` and only where `region_j ∩ capture_k ≠ ∅`. Backends execute ops in order
(wgpu UI defers every `Composite` to `paint`, keeping list order). Unit tests pin all three
orders and the overlap test (touching rects, disjoint rects, a region fully inside a later
capture). Also `entry_state_key(layer: StateKey, index) -> StateKey` (`id.with(index)`, D5).

### 19.4.4 Core plumbing (`backend.rs`, `layer.rs`, `geometry.rs`, `state.rs`)
* `LayerJob { rect: Rect /* union */, entries: Vec<EntryJob>, overlap, time, frame_input, id, viewport, reset_state }`,
  `EntryJob { rect, mask: JobMask, passes, padding_points, textures, state: StateOptions }`;
  `LayerJob::is_stateful()` any entry; `state_key(index)`.
* `geometry::entry_rect(painted_union: Rect, rendered_union: Rect, entry: Rect) -> Rect` — the
  entry's rect after the layer transform, derived like `mask_transform` (`geometry.rs:178`);
  per entry then `layer_geometry(entry_rect, clip, ppp, screen, entry.padding, radius, scale)`.
* Reset: `job.claim_reset()` once per paint (hoisted out of the entry loop), the bool passed to
  every stateful entry's `StateRegistry::begin` (`state.rs:288`).

### 19.4.5 wgpu UI (`wgpu_backend/{mod,plugin,beneath,render,resources,chain}.rs`)
* `Pending`: per-entry `padding_points`, `textures`, plus the mask textures union;
  `plan_layer` (`plugin.rs:128`): capture rect = union over entries of `capture_points(entry_rect,
  clip, padding_k, scale, ppp)`; `user_textures: Vec<[Option<BindGroup>; 3]>`.
* `LayerTargets` (`resources.rs:49`) becomes per ENTRY (one pool slot per entry; `next_targets`
  per entry); `Drawn { generation, entries: Vec<(pool_index, region)> }`; `paint` draws every
  entry's `draw_composite_over` in order after checking every region as today.
* `render.rs::record`: pack the beneath meshes once (vertex / index buffers in the FIRST entry's
  targets), render the backdrop per entry with per-entry scissors (`scissor_in_capture` with
  `capture_k`); then execute `plan::schedule(.., CaptureSource::Pristine)`: `Run(k)` =
  `chain.record` into `pool[k]`; `HandOff` = `chain.record_handoff(rec, &pool[from], &pool[into].capture, op)`;
  states per `entry_state_key`.
* `chain.rs`: `record` returns a `RecordedRun { composite: BindGroup, state: Option<StatePlan>, plan: ChainPlan, allocations }` (the plan is needed by hand-offs); new
  `record_handoff` (replace composite pipeline into `CAPTURE_FORMAT`, cached in
  `ChainResources`, `gpu::begin_pass_load`); `chain_uniforms` gains the hand-off composite blocks
  (`extra_origins: &[[i32; 2]]`).

### 19.4.6 offline (`offline/{mod,options}.rs`)
```rust
pub struct OfflineEntry { rect: Rect /* frame points */, chain: FilterChain, mask: Mask, textures: [Option<OfflineTexture>; 3], state: StateOptions } // builders as Entry
pub struct OfflineLayer { entries: Vec<OfflineEntry>, overlap: Overlap }
impl OfflineRenderer { pub fn process_layer(&mut self, frame, layer: &OfflineLayer, options) -> Result<RgbaFrame>; process_layer_into; process_layer_async; }
```
`process(frame, chain, options)` stays the single-entry-over-the-frame path, untouched
(bit-identity suites). `process_layer`: refuse tiling when the frame exceeds the tile limit
(D6, `NotTileable { reason: "multi-entry layers need the whole frame" }`); `output` texture =
frame (`copy_texture_to_texture` from the uploaded source first); per entry a capture texture
(pool) filled by `copy_texture_to_texture` of the entry's capture rect from the pristine source
(`Pristine` schedule ⇒ hand-offs for `Stacked`); composites = `composite_over_pipelines(device,
FRAME_FORMAT, default, None)` via `draw_composite_over` over `region_k`; readback of the whole
output. `states: Vec<Option<OfflineState>>` per entry index (`reset_state` clears all).
Video: `VideoProcessor` gets `process_layer` too only if trivial; otherwise documented as
"0.2: images only" (manager call; it is a thin wrapper over `run`).

### 19.4.7 glow (`glow_backend/{render,resources,coverage,mod}.rs`)
* `Resources.capture` → `captures: Vec<RenderTarget>` (grown per entry index); lazily built
  `composite_erase` / `composite_add` programs (`composite_erase_module()` /
  `composite_add_module()` un-gated from `cfg(feature = "wgpu")`).
* `paint_saved`: programs for every entry's passes first; per-entry `LayerTextures`; geometries
  per entry (`entry_rect`); `plan::schedule(.., CaptureSource::Framebuffer)`; ops: `Capture(k)`
  = today's `capture` into `captures[k]`; `Run(k)` = `run_passes` with `Slot::Source →
  captures[k]`; `Composite(k)` = blended (multi-entry) or replace (single entry).
* `LayerOutcome::PassFailed { index }` gains the entry index; suspects per module unchanged.

### 19.4.8 Tests
* unit: `plan::schedule` orders (§19.4.3); `entry_rect` under a TSTransform; `LayerJob` checks
  name the entry; `FilterLayer::rect()` = union; layer-level builders reach every entry.
* `tests/wgpu_headless.rs`: `two_entries_far_apart_capture_only_their_windows` (an inverting
  entry left, a red-tint entry right; the middle stays untouched; `Independent`);
  `overlapping_entries_independent_later_wins` (two overlapping rounded rects, invert + tint:
  overlap shows tint(backdrop) not tint(invert(backdrop)); AA edge blends over the earlier
  entry's result); `overlapping_entries_stacked_compose` (overlap shows tint(invert(backdrop)));
  `stateful_entries_keep_separate_states` (two trails entries, one id, different decays);
  `entry_textures_are_per_entry`.
* `tests/glow_headless.rs`: the same four scenes (+ MSAA), single-entry scenes unchanged.
* `tests/offline_equivalence.rs`: `entries_offline_equal_the_ui_in_both_modes` (bit for bit,
  non-sRGB, no dithering), `multi_entry_layers_refuse_tiling`.

## 19.5 R2 — benchmark: N layers vs 1 layer / N regions vs 1 layer / N entries

* Demo scene "grid" (`examples/demo/scenes/grid.rs`, registry `SCENES`): an `N = cols × rows`
  grid of small frosted glass tiles (`presets::frosted` + a tint effect `mix(src, p.tint[…], k)`)
  over the busy backdrop; mode selector: **Layers** (N `FilterLayer`s), **Regions** (one layer,
  `Mask::Regions`, tint by `region()`), **Entries / Independent**, **Entries / Stacked** (after
  R3); N slider (4..=256); frame-time readout = a `FrameTimer` helper moved into
  `scenes/mod.rs` from bloom's inline code (`bloom.rs:141-148`), plus the count of egui
  callbacks. `--record` path periodic (tiles pulse).
* Headless: `#[ignore]` tests `bench_many_elements` in `tests/wgpu_headless.rs` (using
  `render_frame`, which waits for the GPU) and `tests/glow_headless.rs` (`Harness::frame`, which
  reads pixels back): for N in [16, 64] and each mode, 10 warm-up frames then K = 100 timed
  frames, printing `mode, N, ms/frame` with `--nocapture`. Same scene builder shared by both
  files' own code (tests cannot share modules without a `tests/common/`; add
  `tests/common/mod.rs` holding the scene builder used by both). The numbers land in the README
  ("Many elements" section) and §18 with the machine named; the docs state the expected shape
  (Layers: N beneath selections + tessellations + re-renders on wgpu, N callbacks on glow;
  Regions: one of everything but a union-bbox capture; Entries: one selection, N small captures).

## 19.6 Documentation
`src/MODULE_README.md` (pipeline: entries, schedule owner, binding 7), `shaders/MODULE_README.md`
(binding 7, `fs_region`, locals), `wgpu_backend/MODULE_README.md` (per-entry pool, hand-off),
`glow_backend/MODULE_README.md` (capture pool, blended composite, units 0..=4 / 5..=7),
`offline/MODULE_README.md` (`process_layer`, tiling refusal), `effect/MODULE_README.md`
(`reads_regions`), `examples/MODULE_README.md` (grid scene, bench), `lib.rs` prelude table and a
"Regions and entries" section, README (feature list, coverage rule change D1, bench numbers),
CHANGELOG `[0.2.0]` (Added + the D1 behaviour note), roadmap §17 progress lines and §18 findings.

## 19.7 Refusals (explicit, no fakes)
* > 255 regions: `MaskError::TooManyRegions`. Regions with callbacks / malformed meshes: as F9.
* Offline multi-entry layers that would need tiles: `OfflineError::NotTileable` (D6). Offline
  text / textured-mesh regions: as F9.
* A multi-entry layer whose ANY entry fails `check_textures` / `check_state` / `check_mask`:
  the whole layer is skipped (error names the entry).
* Stateful entries without a layer id: `StateError::MissingId { entry, .. }`.
* glow between R3a and R3b: multi-entry layers skipped with a once-logged reason (temporary,
  removed in R3b).

## 19.8 Phases (each: worker → `cargo check-all`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, GPU suites `--ignored` where a GPU exists; manager re-runs the GPU suites)

| # | phase | size | files owned (disjoint within a parallel group) | depends on | parallel with |
|---|---|---|---|---|---|
| R0 | Coverage union additive (D1): `COVERAGE_BLEND`, glow blend func, doc text (`mask.rs` docs, lib.rs / README "overlap" sentences), re-check F9 tests | S | `wgpu_backend/coverage.rs` (blend const + `mesh_pipeline` blend param), `glow_backend/coverage.rs`, docs | — | — (first; tiny, verified alone) |
| R1a | Regions core + wgpu: `mask.rs` (`Region`, `Mask::Regions`, `RegionMeshes`, `PackedRegions`), `layer.rs::regions`, `entry`-free `backend.rs` untouched, `plan.rs` (§19.3.4), `pass.rs` offsets, `shaders/prelude.wgsl` + `mesh.wgsl`, `effect/{mod,interface}.rs`, wgpu `chain.rs` layout/dummy/slot, wgpu `coverage.rs` region pipeline + targets, `render.rs` CoverageJob ranges, unit + wgpu headless tests | L | all of the above | R0 | — |
| R1b | Regions on glow: `glsl.rs` units + `translate_mesh(entry)`, `coverage.rs` region program/target/id passes, `render.rs` slot/allocations/bind, `resources.rs` if needed; glow headless tests | M | `glow_backend/**` | R1a | R1c, R2a |
| R1c | Regions offline + equivalence: `Mask::check_offline` over regions, `offline/options.rs` docs, `tests/offline_equivalence.rs` region cases (UI vs offline, tiled vs whole) | S | `offline/**`, `tests/offline_equivalence.rs` | R1a | R1b, R2a |
| R2a | Bench, part 1: `tests/common/mod.rs` scene builder, `bench_many_elements` in both headless files (modes Layers / Regions), demo scene `grid.rs` + `FrameTimer` in `scenes/mod.rs`, `examples/MODULE_README.md` | M | `tests/common/`, bench fns appended to both headless files, `examples/demo/**` | R1a (+R1b for the glow bench) | R1b, R1c |
| R3a | Entries core + schedule + wgpu UI + offline: `src/entry.rs` (new), `layer.rs`, `backend.rs`, `geometry.rs::entry_rect`, `plan.rs` (`schedule`, `composite_for_origin`, `entry_state_key`), `effect/mod.rs` (un-gate erase/add modules), wgpu `{mod,plugin,beneath,render,resources,chain}.rs`, `gpu.rs::begin_pass_load`, `offline/{mod,options}.rs` (`OfflineEntry`, `OfflineLayer`, `process_layer*`, per-entry states); glow adapted to compile with the temporary multi-entry refusal; unit + wgpu headless + offline equivalence tests | L | all listed | R1a–R1c | — |
| R3b | Entries on glow: capture pool, blended composite programs, schedule execution, per-entry textures / states, remove the R3a refusal; glow headless tests | M | `glow_backend/**` | R3a | R2b |
| R2b | Bench, part 2: Entries modes in the bench tests and the grid scene; run both benches on the dev machine; record numbers in README + §18 | S | `tests/common/`, bench fns, `grid.rs`, README perf section | R3a (+R3b for glow) | R3b (files disjoint) |
| R4 | Docs + review: every `MODULE_README.md`, `lib.rs` docs, README, CHANGELOG, roadmap §17/§18; cross-pool review of `plan::schedule` + hand-off + glow blended composite (Sol) | M | docs only (+ review fixes) | all | — |

Ownership rule for parallel groups: R1b / R1c / R2a touch disjoint directories; R2a appends
bench functions at the END of the two headless test files while R1b/R1c add scene tests — the
manager sequences the glow headless file (R1b before R2a's glow bench) or gives R2a the wgpu
bench only and R2b both.

## 19.9 API surface changes vs 0.1.0 (for the CHANGELOG)
Additive: `Region`, `Mask::Regions`, `MaskError::TooManyRegions`, `MAX_REGIONS`, `REGION_NONE`,
`FilterLayer::regions` / `from_entries` / `with_entry` / `overlap` / `overlap_mode` /
`entries`, `Entry`, `Overlap`, `Effect::reads_regions`, `OfflineEntry`, `OfflineLayer`,
`OfflineRenderer::process_layer{,_into,_async}`, prelude `region()` / `region_at()` /
`REGION_NONE`; `TextureError::NotAttached` and `StateError::MissingId` gain `entry: usize`
(both `#[non_exhaustive]`: additive). Behaviour: `FilterLayer::rect()` = union of entries (equal
to the rect for one entry); `chain()` = the first entry's; `OfflineRenderer::reset_state` resets
every entry's state. Breaking vs 0.1.0: none beyond what 0.2 already lists (§10); vs the 0.2
working tree: the coverage union rule (D1, F9 junction pixels).

## 19.10 Risks (ranked)
1. **Hand-off correctness** (wgpu `Stacked`): the viewport must be clipped to `capture_k` and the
   uniform origin derived by `plan.rs`, not the backend; `LoadOp::Load` on a texture written by
   the backdrop pass in the same encoder is ordinary wgpu. Test: three overlapping entries where
   entry 0 overlaps 2 but not 1 (the case analysed in §19.4.2).
2. **D1 changes F9 pixels**: only junction pixels; the F9 suites assert inside / outside and UI
   vs offline (both change) — re-run before accepting R0; README text about "overlapping pieces"
   must be rewritten in R0, not left stale.
3. **Two id passes vs text regions**: thin glyph strokes with atlas alpha < 0.5 keep their pass-A
   id unless a later region covers them by ≥ 0.5 — intended; document "text regions own their
   antialiased pixels".
4. **wgpu pool growth**: N entries per frame × layers; pool slots are per entry now; a layer with
   many entries and large rects allocates N captures + N ping-pong pairs — inherent (the owner
   refused union captures); documented, bounded by entry sizes.
5. **glow blended composite on odd framebuffers**: MSAA + scissor + blend into the caller's
   framebuffer is what egui_glow itself does; the RGB565 / RGB8 read formats already handled by
   `ReadFormat` are unaffected (blend happens on the draw side).
6. **Offline `process_layer` memory**: whole-frame output + per-entry captures; bounded by the
   tile cap refusal (D6) — an 8K export needs `with_max_tile_size(None)` and enough VRAM.
7. **Bench interpretation**: wall time in a 256×128 harness understates fill cost; the demo
   scene at window size is the second data point; both are reported together.
8. **`mesh_locals` reuse of `reserved`**: byte layout pinned by `mesh_locals_follow_the_wgsl_layout`
   and `the_mesh_shader_validates_and_matches_the_locals_size`.

Manager decision: D1–D9 accepted as recommended (additive coverage union; `Mask::Regions`; layer-level builders apply to every entry; glow blended composite for multi-entry layers; state per entry keyed `id.with(index)`; offline multi-entry refuses tiling; wall-time bench; wgpu Stacked via hand-off; R3a then R3b with a temporary once-logged glow refusal in between).
