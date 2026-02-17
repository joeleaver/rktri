# Layer Masking Implementation Plan

## Overview

Implement layer masking system to enable multi-pass SVO rendering with z-fighting prevention for overlapping layers.

## Goals

1. Enable selective layer rendering (e.g., opaque-only, water-only passes)
2. Add layer priority to resolve z-fighting when layers overlap
3. Support shadow optimization with coarse LOD bias
4. Maintain backward compatibility (no changes when not used)
5. Keep performance impact minimal (3-7% for layer-targeted passes)

## Architecture Decisions

### Layer Priority System
- Use integer priority array per layer (lower = wins ties)
- Default priorities match LayerCompositor: Terrain=0, Static=1, Dynamic=2, Water=10, Effects=20
- Applied in shader during chunk hit resolution

### Layer Masking
- Bitmask in `TraceParams` (32 bits = 32 layers max)
- Shader filters chunks by `layer_mask & (1 << layer_id)`
- Early exit for non-matching chunks in DDA loop

### Shadow Optimization
- Add `coarse_lod_bias` to skip N LOD levels
- Only target layers cast shadows (terrain + static objects by default)

---

## File Changes

### 1. src/render/pipeline/svo_trace.rs

**Changes to TraceParams struct**:
```rust
pub struct TraceParams {
    pub width: u32,
    pub height: u32,
    pub chunk_count: u32,
    pub _pad0: u32,
    pub lod_distances: [f32; 4],
    pub lod_distances_ext: [f32; 2],

    // NEW: Layer targeting
    pub layer_mask: u32,      // Bitmask of layers to trace (0xFFFFFFFF = all)
    pub layer_priority: [i32; 5], // Priority per layer (lower = wins ties)
    pub flags: u32,            // Flags for special modes

    // Existing grid fields
    pub grid_min_x: i32,
    pub grid_min_y: i32,
    pub grid_min_z: i32,
    pub chunk_size: f32,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub _pad2: u32,
}
```

**Changes to SvoTracePipeline struct**:
- Add `params: TraceParams` field to store current params
- Needed for modifying `layer_mask` and `layer_priority` before dispatch

**New methods**:
```rust
/// Update layer mask without changing other params
pub fn set_layer_mask(&self, queue: &wgpu::Queue, layer_mask: u32) {
    let mut params = self.params;
    params.layer_mask = layer_mask;
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    self.params = params;
}

/// Update layer priorities
pub fn set_layer_priorities(&self, queue: &wgpu::Queue, priorities: &[i32; 5]) {
    let mut params = self.params;
    params.layer_priority.copy_from_slice(priorities);
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    self.params = params;
}

/// Dispatch with layer targeting
pub fn dispatch_with_layers(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    octree_buffer: &OctreeBuffer,
    output_bind_group: &wgpu::BindGroup,
    layer_mask: u32,
    width: u32,
    height: u32,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    self.set_layer_mask(queue, layer_mask);
    self.dispatch(encoder, octree_buffer, output_bind_group, width, height, timestamp_writes);
}

/// Dispatch single layer
pub fn dispatch_layer(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    octree_buffer: &OctreeBuffer,
    output_bind_group: &wgpu::BindGroup,
    layer_id: u32,
    width: u32,
    height: u32,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    self.dispatch_with_layers(encoder, octree_buffer, output_bind_group, 1u << layer_id, width, height, timestamp_writes);
}
```

**Update new() to initialize params**:
```rust
pub fn new(
    device: &wgpu::Device,
    camera_buffer: &CameraBuffer,
    octree_buffer: &OctreeBuffer,
) -> Self {
    // ... existing buffer creation ...

    let params = TraceParams {
        width: 0,
        height: 0,
        chunk_count: 0,
        _pad0: 0,
        lod_distances: [64.0, 128.0, 256.0, 512.0],
        lod_distances_ext: [1024.0, f32::MAX],
        layer_mask: 0xFFFFFFFFu, // NEW: default = all layers
        layer_priority: [0, 1, 2, 10, 20], // NEW: default priorities
        flags: 0u, // NEW
        grid_min_x: 0,
        grid_min_y: 0,
        grid_min_z: 0,
        chunk_size: 0.0,
        grid_size_x: 0,
        grid_size_y: 0,
        grid_size_z: 0,
        _pad2: 0,
    };

    Self {
        pipeline,
        params_buffer,
        grass_buffer,
        profile_table_buffer,
        params_bind_group_layout,
        params_bind_group,
        output_bind_group_layout,
        params, // NEW: store params
    }
}
```

**Update update_params() to sync stored params**:
```rust
pub fn update_params(&self, queue: &wgpu::Queue, params: &TraceParams) {
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    self.params = *params; // NEW: sync stored copy
}
```

---

### 2. src/render/pipeline/shadow.rs

**Changes to ShadowParams struct**:
```rust
pub struct ShadowParams {
    pub light_dir: [f32; 3],
    pub _pad1: f32,
    pub shadow_bias: f32,
    pub soft_shadow_samples: u32,
    pub soft_shadow_angle: f32,
    pub chunk_count: u32,
    pub width: u32,
    pub height: u32,
    pub leaf_opacity: f32,

    // NEW: Layer targeting for shadows
    pub layer_mask: u32,         // Bitmask of shadow-casting layers
    pub coarse_lod_bias: i32,   // Skip N LOD levels for faster shadows

    pub _pad2: f32,

    // Existing grid fields
    pub grid_min_x: i32,
    pub grid_min_y: i32,
    pub grid_min_z: i32,
    pub chunk_size: f32,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub _pad3: u32,
}
```

**New methods**:
```rust
/// Update shadow parameters
pub fn update_params(&self, queue: &wgpu::Queue, params: &ShadowParams) {
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
}

/// Dispatch shadow pass with layer targeting and LOD bias
pub fn dispatch_with_lod_bias(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    camera_buffer: &CameraBuffer,
    octree_buffer: &OctreeBuffer,
    gbuffer_bind_group: &wgpu::BindGroup,
    output_bind_group: &wgpu::BindGroup,
    layer_mask: u32,
    coarse_lod_bias: i32,
    width: u32,
    height: u32,
) {
    let mut params = self.params;
    params.layer_mask = layer_mask;
    params.coarse_lod_bias = coarse_lod_bias;
    self.update_params(queue, &params);

    // ... existing dispatch logic ...
}
```

---

### 3. shaders/svo_trace.wgsl

**Update TraceParams struct**:
```wgsl
struct TraceParams {
    width: u32,
    height: u32,
    chunk_count: u32,
    pad0: u32,
    lod_distances: array<f32, 4>,
    lod_distances_ext: array<f32, 2>,

    // NEW: Layer targeting
    layer_mask: u32,      // Bitmask of layers to trace
    layer_priority: array<i32, 5>, // Priority per layer (lower = wins ties)
    flags: u32,

    // Chunk grid acceleration
    grid_min_x: i32,
    grid_min_y: i32,
    grid_min_z: i32,
    chunk_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    pad2: u32,
}
```

**Update ChunkInfo struct**:
```wgsl
struct ChunkInfo {
    world_min: vec3<f32>,
    root_size: f32,
    root_node: u32,
    max_depth: u32,
    layer_id: u32,  // Already exists, ensure it's here
    flags: u32,
}
```

**Update grid_lookup() with layer filtering**:
```wgsl
// Look up chunk index from 3D grid with layer filtering.
fn grid_lookup(cx: i32, cy: i32, cz: i32) -> u32 {
    let gx = cx - params.grid_min_x;
    let gy = cy - params.grid_min_y;
    let gz = cz - params.grid_min_z;

    if (gx < 0 || gy < 0 || gz < 0 ||
        u32(gx) >= params.grid_size_x ||
        u32(gy) >= params.grid_size_y ||
        u32(gz) >= params.grid_size_z) {
        return 0xFFFFFFFFu;
    }

    let idx = u32(gx) + u32(gy) * params.grid_size_x + u32(gz) * params.grid_size_x * params.grid_size_y;
    let chunk_idx = chunk_grid[idx];

    // NEW: Layer filtering
    if (chunk_idx != 0xFFFFFFFFu && params.layer_mask != 0xFFFFFFFFu) {
        let chunk = chunk_infos[chunk_idx];
        let layer_bit = 1u << chunk.layer_id;
        if ((params.layer_mask & layer_bit) == 0u) {
            return 0xFFFFFFFFu; // Skip this chunk
        }
    }

    return chunk_idx;
}
```

**Update trace_all_chunks() for layer priority**:
```wgsl
// In the for loop where chunk_result is checked (around line 1754):

if (chunk_result.hit) {
    let chunk = chunk_infos[chunk_idx];
    let chunk_priority = params.layer_priority[chunk.layer_id];

    // NEW: Layer priority for tie-breaking
    if (result.hit) {
        let result_chunk = chunk_infos[result.chunk_idx];
        let result_priority = params.layer_priority[result_chunk.layer_id];

        let is_same_layer = (chunk.layer_id == result_chunk.layer_id);

        if (!is_same_layer) {
            // Different layer: use priority, not just distance
            if (chunk_priority < result_priority) {
                result = chunk_result;
                // Continue - may find even higher priority chunk
            } else if (chunk_result.t < result.t) {
                result = chunk_result;
            }
        } else {
            // Same layer: use distance (original behavior)
            if (chunk_result.t < result.t) {
                result = chunk_result;
            }
        }
    } else {
        // First hit
        result = chunk_result;
    }

    // DDA marches front-to-back, so first hit is usually closest
    // But priority can override for overlapping layers
    if (result.hit && chunk_result.t < result.t) {
        // For same-priority layers, closest distance still wins
        return result;
    }
}
```

---

### 4. shaders/shadow.wgsl

**Update ShadowParams struct**:
```wgsl
struct ShadowParams {
    light_dir: vec3<f32>,
    pad1: f32,
    shadow_bias: f32,
    soft_shadow_samples: u32,
    soft_shadow_angle: f32,
    chunk_count: u32,
    width: u32,
    height: u32,
    leaf_opacity: f32,

    // NEW: Layer targeting for shadows
    layer_mask: u32,       // Bitmask of shadow-casting layers
    coarse_lod_bias: i32,   // Skip N LOD levels

    pad2: f32,

    // Chunk grid acceleration
    grid_min_x: i32,
    grid_min_y: i32,
    grid_min_z: i32,
    chunk_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    pad3: u32,
}
```

**Add LOD bias function**:
```wgsl
// NEW: Get max depth with LOD bias for shadow optimization
fn get_max_depth_for_shadow(lod_level: u32, chunk_max_depth: u32, bias: i32) -> u32 {
    if (bias < 0) { return chunk_max_depth; }
    let biased_lod = lod_level + u32(bias);
    if (biased_lod >= chunk_max_depth) { return 0u; }
    return chunk_max_depth - biased_lod;
}
```

**Add layer filtering to shadow's grid_lookup** (or reuse from svo_trace.wgsl):
```wgsl
// In trace_chunk_octree_shadow(), use biased max_depth:
let max_depth = get_max_depth_for_shadow(lod, chunk.max_depth, params.coarse_lod_bias);
```

---

### 5. src/main.rs

**Update render loop to use layer masking**:

Find the render loop (around lines 2240-2430) and add:

```rust
// After SVO trace pass setup, add layer masking options

// Example: Shadow optimization (terrain + static objects cast shadows)
let shadow_casters = (1 << LayerId::TERRAIN.0) | (1 << LayerId::STATIC_OBJECTS.0);
shadow_pipeline.dispatch_with_lod_bias(
    encoder,
    &resources.camera_buffer,
    &resources.octree_buffer,
    &gbuffer_bind_group,
    &shadow_output_bind_group,
    shadow_casters,
    2, // Skip 2 LOD levels (coarser shadows)
    width / 2, height / 2, // Half-res shadows
);

// Example: Water separate pass
// Pass 1: Opaque layers (terrain, static, dynamic)
let opaque_mask = 0xFFFFFFFFu ^ (1 << LayerId::WATER.0);
svo_trace_pipeline.dispatch_with_layers(
    encoder,
    &resources.octree_buffer,
    &opaque_output_bind_group,
    opaque_mask,
    width,
    height,
    None,
);

// Pass 2: Water layer
svo_trace_pipeline.dispatch_layer(
    encoder,
    &resources.octree_buffer,
    &water_output_bind_group,
    LayerId::WATER.0,
    width,
    height,
    None,
);

// Then compose results in lighting pass
```

---

## Testing Strategy

### 1. Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_mask_bitmask() {
        // Single layer
        assert_eq!(1u << 0, 1u);      // Terrain
        assert_eq!(1u << 3, 8u);      // Water
        // Multiple layers
        assert_eq!((1u << 0) | (1u << 1), 3u);
        // All layers
        assert_eq!(0xFFFFFFFFu, 4294967295u);
    }

    #[test]
    fn test_default_params() {
        let params = TraceParams::default();
        assert_eq!(params.layer_mask, 0xFFFFFFFFu); // All layers
        assert_eq!(params.layer_priority, [0, 1, 2, 10, 20]); // Default priorities
    }
}
```

### 2. Integration Tests

1. **Backward compatibility test**:
   - Run with `layer_mask = 0xFFFFFFFF`
   - Verify output matches current rendering exactly
   - No visual regression

2. **Single layer test**:
   - Set `layer_mask = 1 << LayerId::TERRAIN.0`
   - Verify only terrain renders
   - Water, objects should be invisible

3. **Layer priority test**:
   - Create overlapping rock in terrain
   - Verify rock (priority 1) wins over terrain (priority 0)

4. **Shadow optimization test**:
   - Run shadow pass with `coarse_lod_bias = 2`
   - Verify 2-3× speedup
   - Check shadow quality is acceptable

### 3. Performance Profiling

1. Baseline profile (all layers):
   ```
   cargo run --release -- --world test_trees
   ```
   Measure: FPS, frame time, GPU time

2. Layer-targeted profile (single layer):
   ```
   // Set layer_mask = terrain only
   ```
   Measure: FPS, frame time, GPU time
   Verify: < 10% overhead vs baseline

3. Shadow optimization profile:
   ```
   // Set coarse_lod_bias = 2
   ```
   Measure: FPS for shadow pass
   Verify: 2-3× speedup

---

## Implementation Order

1. **Phase 1: Rust API changes** (1 day)
   - Update `TraceParams` in `svo_trace.rs`
   - Update `ShadowParams` in `shadow.rs`
   - Add new methods to `SvoTracePipeline`
   - Add new methods to `ShadowPipeline`

2. **Phase 2: Shader changes** (1 day)
   - Update `TraceParams` in `svo_trace.wgsl`
   - Update `ShadowParams` in `shadow.wgsl`
   - Add layer filtering to `grid_lookup()`
   - Add layer priority to hit resolution
   - Add LOD bias function to shadow shader

3. **Phase 3: Integration** (1 day)
   - Update `main.rs` render loop
   - Add shadow optimization call
   - Add example multi-pass usage

4. **Phase 4: Testing** (1 day)
   - Run unit tests
   - Run integration tests
   - Profile performance
   - Fix any issues found

**Total estimated time**: 4 days

---

## Rollback Plan

If any phase causes issues:
1. Revert shader changes to original
2. Remove new methods from Rust structs
3. Keep `layer_mask = 0xFFFFFFFF` as default (backward compatible)
4. All changes are additive; can be disabled by using default values

---

## Success Criteria

1. ✅ Unit tests pass
2. ✅ Backward compatibility verified (no regression with default values)
3. ✅ Single layer filtering works
4. ✅ Layer priority resolves z-fighting
5. ✅ Shadow optimization provides 2-3× speedup
6. ✅ Performance overhead < 10% for layer-targeted passes
7. ✅ Code compiles and runs without errors
