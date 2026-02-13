// Common types for Rktri shaders
// These must match the Rust struct layouts exactly

struct Camera {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    position: vec3<f32>,
    near: f32,
    far: f32,
    world_offset: vec3<f32>,
}

struct OctreeNode {
    // flags: bits 0-7 child_valid, 8-15 child_leaf, 16-31 reserved
    flags: u32,
    // Offset to first child or brick index
    child_offset: u32,
    // Packed bounds and LOD data
    bounds_min_x: u32,  // u16 in lower bits
    bounds_min_y: u32,
    bounds_min_z: u32,
    lod_color: u32,     // RGB565 in lower 16 bits
    bounds_max_x: u32,
    bounds_max_y: u32,
    bounds_max_z: u32,
    lod_material: u32,
    _padding: vec2<u32>,
}

struct Voxel {
    // Packed: color (u16 RGB565), material_id (u8), flags (u8)
    data: u32,
}

struct VoxelBrick {
    voxels: array<Voxel, 8>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
}

// Helper functions

fn unpack_rgb565(color: u32) -> vec3<f32> {
    let r = f32((color >> 11u) & 0x1Fu) / 31.0;
    let g = f32((color >> 5u) & 0x3Fu) / 63.0;
    let b = f32(color & 0x1Fu) / 31.0;
    return vec3<f32>(r, g, b);
}

fn unpack_voxel_color(voxel: Voxel) -> vec3<f32> {
    return unpack_rgb565(voxel.data & 0xFFFFu);
}

fn unpack_voxel_material(voxel: Voxel) -> u32 {
    return (voxel.data >> 16u) & 0xFFu;
}

fn is_voxel_empty(voxel: Voxel) -> bool {
    return voxel.data == 0u;
}

fn get_child_valid_mask(node: OctreeNode) -> u32 {
    return node.flags & 0xFFu;
}

fn get_child_leaf_mask(node: OctreeNode) -> u32 {
    return (node.flags >> 8u) & 0xFFu;
}

fn is_child_valid(node: OctreeNode, index: u32) -> bool {
    return (get_child_valid_mask(node) & (1u << index)) != 0u;
}

fn is_child_leaf(node: OctreeNode, index: u32) -> bool {
    return (get_child_leaf_mask(node) & (1u << index)) != 0u;
}
