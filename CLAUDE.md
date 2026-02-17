# Rktri

Voxel-based 3D game engine using SVO (Sparse Voxel Octree) raytracing.

## Development Principles

- **Never decide something is "too hard"** - when stuck, ask questions instead of taking shortcuts
- **Don't simplify by making things worse** - fix properly or ask for guidance
- **When stuck on a bug, investigate the code** - read and understand rather than guessing
- **No reverting to known-broken workarounds** - if a fix breaks something, fix it properly
- **When unsure, ask the user** - don't make unilateral decisions about scope

## Build & Test

```bash
cargo build --release                              # game binary
cargo build --release --bin generate_world          # world generator
cargo build --release --features dlss               # with DLSS (requires NVIDIA RTX + SDK)
cargo test -p rktri -p rktri-debug -p rktri-mcp    # run all tests
```

**Known test issue**: `test_biome_classifier_evaluate` is a pre-existing failure (material 0 vs expected 4).

## Run

```bash
# Generate a pre-built world (required for large worlds)
cargo run --release --bin generate_world -- --size 1000 --name my_world

# Run the game
cargo run --release --bin rktri -- --world my_world   # load pre-generated world
cargo run --release --bin rktri -- --size 100          # runtime generation (small worlds only)
cargo run --release --bin rktri                        # default 32m radius runtime generation
```

## Architecture

### Workspace

- `src/` — Main engine (`rktri` crate)
- `crates/rktri-debug/` — TCP debug server library (`DebugHandler` trait, JSON protocol, port 9742)
- `crates/rktri-mcp/` — MCP server (stdio JSON-RPC 2.0, forwards to TCP debug server)
- `shaders/` — WGSL compute/render shaders

### Module Map

| Module | Purpose |
|--------|---------|
| `voxel/` | Core data structures: `Voxel`, `VoxelBrick` (8x8x8), `Chunk`, `Octree`, `World` |
| `voxel/svo/` | Sparse Voxel Octree: `Octree`, `OctreeNode`, `AdaptiveOctreeBuilder`, `SvdagBuilder`, `RegionClassifier` |
| `mask/` | Generic mask octrees: `MaskOctree<T>`, `MaskValue` trait, `BiomeId`, `MaskBuilder`, `MaskGenerator<T>` |
| `generation/` | World generation pipeline: `GenerationPipeline`, `BiomeNoiseGenerator`, `MaskDrivenTerrainClassifier` |
| `terrain/` | Noise-based terrain: `TerrainGenerator`, `TerrainParams`, `BiomeMap`, `BiomeTerrainClassifier` |
| `scene/` | Scene graph: `SceneManager`, `SceneGraph`, `SceneConfig`, `LayerId`, flatten for GPU upload |
| `render/` | wgpu rendering: pipelines, buffers, textures, culling, profiling |
| `render/pipeline/` | Per-pass pipelines: `SvoTrace`, `Shadow`, `GodRays`, `Clouds`, `Lighting`, `Tonemap`, `Display` |
| `atmosphere/` | Day/night cycle, sun/moon, fog, weather, wind, clouds, `ColorRamp<T>` interpolation |
| `grass/` | Procedural grass: `GrassSystem`, `GrassConfig`, `GrassProfileTable` (volumetric Beer's law in shader) |
| `streaming/` | Chunk streaming, disk I/O (rkyv + LZ4), brick pool, LOD |
| `animation/` | Skeletal animation: clips, skeleton, keyframes, skinning |
| `entity/` | Entity system |
| `math/` | `Aabb`, `Camera`, `FpsCameraController`, `FrameTimer` |

### Render Pipeline (pass order)

1. **SVO Trace** — Raycast through chunk grid (DDA), traverse octrees, output depth + material
2. **Shadow** — Shadow ray from hit point toward sun, half-res
3. **God Rays** — Screen-space volumetric scattering, half-res, depth-based occlusion
4. **Clouds** — GPU compute raymarching through cloud slab, half-res
5. **Lighting** — PBR shading, fog, aerial perspective, moon disc
6. **(DLSS)** — Optional upscaling (after lighting, before tonemap)
7. **Tonemap** — HDR to LDR (ACES)
8. **Display** — Final blit to swapchain

### Generation Pipeline

The world generation system uses a two-stage approach:

1. **Biome mask** — `BiomeNoiseGenerator` implements `MaskGenerator<BiomeId>`, builds a `MaskOctree<BiomeId>` per chunk (depth 3 = 8 cells/side)
2. **Terrain octree** — `MaskDrivenTerrainClassifier` implements `RegionClassifier`, reads biome from mask, builds voxel octree via `AdaptiveOctreeBuilder` (128 voxels/side)

`GenerationPipeline` orchestrates both stages. `SceneManager` delegates to it.

Height-guided Y-level filtering: only generates 2-3 Y levels per XZ column near the terrain surface.

### World Format (V2)

Pre-generated worlds are stored under `assets/worlds/<name>/`:

```
manifest.json           # World metadata, per-layer chunk lists
terrain/                # Terrain layer (independently loadable)
  chunk_0_10_0.rkc      # rkyv-serialized + LZ4-compressed octree
  chunk_1_10_0.rkc
  ...
vegetation/             # (future)
structures/             # (future)
```

Each layer directory is self-contained for independent streaming. The manifest includes per-layer metadata (chunk count, byte size, Y-level distribution).

### Key Technical Details

- **Chunk size**: 4m (CHUNK_SIZE = 4)
- **Voxel resolution**: 128 voxels/side in terrain chunks (3.125cm)
- **Depth buffer**: Stores linear distance (`hit.t`), NOT normalized. Sky = 1000.0 (camera far)
- **Shadow buffer**: Lit/shadow state (1.0=lit, 0.0=shadow)
- **Render scale**: 0.5x internal resolution; shadow & godrays at half of that
- **Chunk grid**: 3D DDA acceleration buffer for raymarching through chunk space
- **SVDAG**: Shared brick deduplication (6-20x compression typical)
- **Grass**: Volumetric Beer's law in `svo_trace.wgsl`, 80 march steps, 80m range, triggered on terrain materials 9/10/11
- **Multi-layer rendering**: See "Multi-Layer Octree Rendering" section below

### Multi-Layer Octree Rendering

Layers remain as separate octrees in the GPU buffer. The raycaster checks all layers at each grid cell and returns the highest-priority hit.

**Layer Priority:**
| Layer | ID | Priority |
|-------|-----|----------|
| Terrain | 0 | Lowest |
| Rocks | 2 | Medium |
| Vegetation | 3 | Highest |

**Chunk Grid Structure:**
```
chunk_grid[idx] -> LayerIndices { terrain: u32, rocks: u32, vegetation: u32 }
// 0 = not present, non-zero = index into octree buffer
```

**Raycasting:** For each grid cell, check all non-zero layer indices, track closest hit by layer priority. Higher layer ID wins for overlapping voxels.

**See `docs/multi-layer-architecture.md` for full details.

### Debug System

- TCP server on port 9742 (`rktri-debug` crate)
- MCP server (`rktri-mcp` crate) wraps TCP as JSON-RPC 2.0 over stdio
- `SharedDebugState` with `Arc<StdMutex<>>` bridges render thread and async handlers
- Commands: camera control, god rays, time of day, weather, fog, wind, grass, DLSS, screenshots

### Conventions

- Rust 2024 edition — `gen` is a reserved keyword, don't use it as a variable name
- wgpu v28 API: `immediate_size: 0` (not `push_constant_ranges`), `multiview_mask: None`
- WGSL uniform structs must match Rust layout exactly (byte-for-byte)
- Brick index 0 is a sentinel — `is_empty()` checks `brick_offset == 0`
- `AdaptiveOctreeBuilder` adds a padding brick at index 0
- Feature-gated DLSS: `--features dlss` (requires `~/.config/nvidia-ngx-conf.json`)
