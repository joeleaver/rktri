# Multi-Layer Octree Rendering Plan (v2)

## Decision: Keep Layers Separate with Layer Priority

Layers remain as separate octrees in the GPU buffer. The raycaster checks all layers at each grid cell and returns the highest-priority hit.

## Layer Priority

| Layer | ID | Priority | Notes |
|-------|-----|----------|-------|
| Terrain | 0 | Lowest | Base ground |
| Grass | 1 | - | Not voxel geometry |
| Rocks | 2 | Medium | Sits on terrain |
| Vegetation | 3 | Highest | Trees on top |

## Data Structure Changes

### GPU Buffer (chunk_grid + layer_data)

**chunk_grid:** Stores descriptor pointing to layer data
```wgsl
struct LayerDescriptor {
    base_index: u32,    // first index in layer_data for this cell
    layer_count: u32,   // how many layers at this cell (0 = empty)
}
chunk_grid[idx] -> LayerDescriptor
```

**layer_data:** Flat array of octree indices
```wgsl
// Flat array indexed by descriptor.base_index + layer_offset
layer_data: array<u32>,  // octree indices into octree buffer
```

### Rust Structures

```rust
struct LayerDescriptor {
    base_index: u32,    // first index in layer_data for this cell
    layer_count: u32,   // how many layers at this cell (0 = empty)
}

// Grid is vec<LayerDescriptor>
// layer_data is Vec<u32> of octree indices
```

## Raycasting Algorithm

```
For each grid cell along ray:
    1. Read LayerIndices from chunk_grid
    2. For each non-zero layer_index:
        a. Traverse octree at that index
        b. If hit found, record hit with layer_priority
    3. After checking all layers in cell:
        a. Return hit with highest layer_priority
    4. Continue to next grid cell
```

## Overlapping Voxel Resolution

When multiple layers have voxels at the same position:
- Layer with higher ID wins (rendered on top)
- Example: Rock (layer 2) at same position as Stone (layer 0) → Rock wins
- This is correct: rocks sit ON TOP of terrain

## Disk I/O

### Loading
- Load terrain chunks → upload to buffer, store index in grid
- Load rock chunks → upload to buffer, store index in grid
- Load vegetation chunks → upload to buffer, store index in grid

### File Structure (already correct)
```
terrain/chunk_x_y_z.rkc
rocks/chunk_x_y_z.rkc
vegetation/chunk_x_y_z.rkc
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/render/buffer/octree_buffer.rs` | Change `GpuChunkInfo` to include layer indices, change grid upload |
| `shaders/svo_trace.wgsl` | Change `chunk_grid` to store `LayerIndices`, update raycaster |
| `shaders/shadow.wgsl` | Same changes for shadows |
| `src/main.rs` | Load layers separately, populate grid with LayerIndices |
| `docs/multi-layer-architecture.md` | Update to reflect this decision |

## Implementation Steps

1. **Update GPU buffer structures**
   - Change `GpuChunkInfo` or create new `GpuLayerIndices`
   - Change chunk_grid storage format

2. **Update shader chunk grid access**
   - Modify `chunk_grid` binding to return LayerIndices
   - Update all places that read chunk_grid

3. **Update raycaster**
   - Check all 3 layer indices per grid cell
   - Track best hit by layer priority

4. **Update main.rs loading**
   - Don't merge octrees
   - Upload each layer to GPU buffer separately
   - Populate grid with layer indices

## Verification

- Rocks render ON TOP of terrain at same position
- No "filled chunk" artifacts
- Multiple layers at same coord work correctly
