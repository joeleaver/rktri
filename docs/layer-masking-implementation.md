# Layer Masking Implementation Guide

This document shows the exact code changes needed to implement layer masking for multi-pass SVO rendering.

## Overview

The `GpuChunkInfo` struct already contains `layer_id: u32` (line 29 of octree_buffer.rs), but the shader doesn't use it. We'll add layer filtering to enable:
- Rendering specific layers independently
- Shadow optimization with coarse LOD
- Transparent layers (water) in separate passes

---

## Step 1: Update TraceParams Rust Struct

**File**: `src/render/pipeline/svo_trace.rs`

Add `layer_mask` and `flags` fields to `TraceParams`:

```rust
/// Trace parameters uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TraceParams {
    pub width: u32,
    pub height: u32,
    pub chunk_count: u32,
    pub _pad0: u32,
    pub lod_distances: [f32; 4],     // First 4 LOD distances: 64, 128, 256, 512
    pub lod_distances_ext: [f32; 2], // Remaining: 1024, f32::MAX

    // NEW: Layer targeting
    pub layer_mask: u32,  // Bitmask of layers to trace (0xFFFFFFFF = all layers)
    pub flags: u32,        // Flags for special modes

    // Existing fields (chunk grid acceleration)
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

**Notes**:
- `layer_mask` uses bits: bit 0 = terrain (LayerId::TERRAIN.0), bit 1 = static objects, etc.
- `0xFFFFFFFF` = all layers (backward compatible)
- Removing 2 padding floats to make room - the struct was 32 bytes, now 40 bytes with 2 new u32s

---

## Step 2: Update Shader - Grid Lookup with Layer Filtering

**File**: `shaders/svo_trace.wgsl`

First, update the `TraceParams` struct in the shader (around line 140-170):

```wgsl
struct TraceParams {
    width: u32,
    height: u32,
    chunk_count: u32,
    pad0: u32,
    lod_distances: array<f32, 4>,
    lod_distances_ext: array<f32, 2>,

    // NEW: Layer targeting
    layer_mask: u32,  // Bitmask of layers to trace
    flags: u32,        // Special mode flags

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

Then, modify the `grid_lookup()` function (around line 1660) to add layer filtering:

```wgsl
// Look up chunk index from 3D grid with layer filtering.
// Returns 0xFFFFFFFF if empty, out of bounds, or layer doesn't match mask.
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
            return 0xFFFFFFFFu; // Skip this chunk - layer not in mask
        }
    }

    return chunk_idx;
}
```

**Note**: The layer filter only runs when `layer_mask != 0xFFFFFFFF`, so there's zero overhead for the default all-layers case.

---

## Step 3: Add Layer-Aware Dispatch Methods

**File**: `src/render/pipeline/svo_trace.rs`

Add these new methods to `SvoTracePipeline`:

```rust
impl SvoTracePipeline {
    // ... existing methods ...

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
        // Note: We need to update params with layer_mask before dispatch
        // This requires storing params on the struct or having a helper

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("svo_trace_pass"),
            timestamp_writes,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.params_bind_group, &[]);
        pass.set_bind_group(1, octree_buffer.bind_group(), &[]);
        pass.set_bind_group(2, output_bind_group, &[]);

        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Convenience: dispatch single layer
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
        self.dispatch_with_layers(
            encoder,
            octree_buffer,
            output_bind_group,
            1u << layer_id, // Single bit for this layer
            width,
            height,
            timestamp_writes,
        );
    }
}
```

**Important**: The `update_params()` method needs to be called before `dispatch_with_layers()` to set the layer_mask. You'll need to store params on the `SvoTracePipeline` struct:

```rust
pub struct SvoTracePipeline {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    grass_buffer: wgpu::Buffer,
    profile_table_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group: wgpu::BindGroup,
    output_bind_group_layout: wgpu::BindGroupLayout,
    params: TraceParams,  // NEW: Store params to modify layer_mask
}
```

Update `new()` to initialize `params`:

```rust
impl SvoTracePipeline {
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &CameraBuffer,
        octree_buffer: &OctreeBuffer,
    ) -> Self {
        // ... existing code ...

        let params = TraceParams {
            width: 0,
            height: 0,
            chunk_count: 0,
            _pad0: 0,
            lod_distances: [64.0, 128.0, 256.0, 512.0],
            lod_distances_ext: [1024.0, f32::MAX],
            layer_mask: 0xFFFFFFFFu,  // Default: all layers
            flags: 0u,
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
            params,  // NEW
        }
    }
}
```

Update `update_params()` to also update the stored params:

```rust
pub fn update_params(&self, queue: &wgpu::Queue, params: &TraceParams) {
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    // NEW: Also update stored params
    // Note: Need interior mutability or use Cell/RefCell
}
```

---

## Step 4: Update Shadow Pipeline (Optional but Recommended)

**File**: `src/render/pipeline/shadow.rs`

Add `layer_mask` and `coarse_lod_bias` to `ShadowParams`:

```rust
pub struct ShadowParams {
    // ... existing fields ...

    pub layer_mask: u32,       // NEW: Bitmask of shadow-casting layers
    pub coarse_lod_bias: i32,   // NEW: Skip N LOD levels for faster shadows
}
```

Add shadow-optimized dispatch:

```rust
impl ShadowPipeline {
    pub fn dispatch_with_lod_bias(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params_bind_group: &wgpu::BindGroup,
        depth_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        layer_mask: u32,
        coarse_lod_bias: i32,
    ) {
        // ... dispatch logic with LOD bias ...
    }
}
```

---

## Step 5: Update Render Loop to Use Layer Masking

**File**: `src/main.rs` (in the render loop, around lines 2240-2430)

### Example: Shadow Optimization

```rust
// Before shadow pass, trace only terrain + static objects
let shadow_casters = (1 << LayerId::TERRAIN.0) | (1 << LayerId::STATIC_OBJECTS.0);

// Update trace params with layer mask
let mut trace_params = resources.svo_trace_pipeline.params();
trace_params.layer_mask = shadow_casters;
resources.svo_trace_pipeline.update_params(&queue, &trace_params);

// Or add a new method:
// resources.svo_trace_pipeline.update_layer_mask(&queue, shadow_casters);

// Shadow pass with coarse LOD bias
shadow_pipeline.dispatch_with_lod_bias(
    encoder,
    params_bg,
    depth_bg,
    shadow_output_bg,
    shadow_casters,
    2, // Skip 2 LOD levels (coarser shadows)
    width / 2, height / 2, // Half-res shadows
);
```

### Example: Water Separate Pass

```rust
// Pass 1: Trace everything except water (opaque layers)
let opaque_mask = 0xFFFFFFFFu ^ (1 << LayerId::WATER.0);
svo_trace_pipeline.dispatch_with_layers(
    encoder,
    octree_buffer,
    opaque_output_bg,
    opaque_mask,
    width,
    height,
);

// Pass 2: Water-only with refraction
svo_trace_pipeline.dispatch_layer(
    encoder,
    octree_buffer,
    water_output_bg,
    LayerId::WATER.0,
    width,
    height,
);

// Later: compose results in lighting pass
```

---

## Testing

### Unit Tests

```rust
#[test]
fn test_layer_mask_bitmask() {
    // Single layer
    assert_eq!(1u << 0, 1u);      // Terrain
    assert_eq!(1u << 3, 8u);      // Water

    // Multiple layers
    assert_eq!((1u << 0) | (1u << 1), 3u);  // Terrain + Static

    // All layers (32 bits)
    assert_eq!(0xFFFFFFFFu, 4294967295u);
}
```

### Integration Tests

1. **Backward compatibility**: Run with `layer_mask = 0xFFFFFFFF`, should render identically to current
2. **Single layer**: Run with terrain-only mask, verify only terrain renders
3. **Shadow optimization**: Run shadow pass with LOD bias, verify ~2-3x speedup
4. **Water pass**: Run water-only, verify refraction sampling works

---

## Performance Notes

- **Overhead**: ~5-10% for layer-targeted passes due to chunk_info load + mask check per DDA step
- **Zero overhead**: All-layers pass (`layer_mask = 0xFFFFFFFF`) has no filtering cost
- **Cache impact**: Layer-targeted passes may load non-matching chunk metadata, but this is minimal (32 bytes per chunk)

---

## Summary

| File | Changes | Lines |
|-------|-----------|--------|
| `svo_trace.rs` | Add `layer_mask`, `flags` to `TraceParams` | 2 |
| `svo_trace.rs` | Add `params: TraceParams` field to struct | 1 |
| `svo_trace.rs` | Add `dispatch_with_layers()`, `dispatch_layer()` | ~50 |
| `shadow.rs` | Add `layer_mask`, `coarse_lod_bias` to `ShadowParams` | 2 |
| `shadow.rs` | Add `dispatch_with_lod_bias()` | ~30 |
| `svo_trace.wgsl` | Add `layer_mask`, `flags` to `TraceParams` | 2 |
| `svo_trace.wgsl` | Modify `grid_lookup()` with layer filter | 7 |
| `main.rs` | Update render loop to use layer masking | ~20 |

**Total**: ~115 lines of code changes, minimal risk, backward compatible.
