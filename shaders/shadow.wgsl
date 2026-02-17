// Shadow Ray Tracing Compute Shader
// Traces rays toward the light to determine shadow visibility
// Output: shadow mask (1.0 = fully lit, 0.0 = fully shadowed)

struct Camera {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    position: vec3<f32>,
    _pos_pad: f32,
    near: f32,
    far: f32,
    _pad2: vec2<f32>,
    world_offset: vec3<f32>,
}

struct ShadowParams {
    light_dir: vec3<f32>,      // Normalized direction TO the light
    _pad1: f32,
    shadow_bias: f32,          // Small offset to prevent self-shadowing
    soft_shadow_samples: u32,  // 1 for hard shadows, 4-16 for soft
    soft_shadow_angle: f32,    // Cone angle for soft shadows (radians)
    chunk_count: u32,          // Number of chunks (legacy, unused with grid)
    width: u32,
    height: u32,
    leaf_opacity: f32,         // How opaque leaf voxels are for shadows (0.0-1.0)
    _pad2: f32,
    // Chunk grid acceleration structure
    grid_min_x: i32,
    grid_min_y: i32,
    grid_min_z: i32,
    chunk_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    _pad3: u32,
}

struct ChunkInfo {
    world_min: vec3<f32>,
    root_size: f32,
    root_node: u32,
    max_depth: u32,
    layer_id: u32,
    flags: u32,
}

struct OctreeNode {
    flags: u32,
    child_offset: u32,
    brick_offset: u32,
    bounds_min_xy: u32,  // packed u16 x, y
    bounds_min_z_lod_color: u32,  // packed u16 z, lod_color
    bounds_max_xy: u32,  // packed u16 x, y
    bounds_max_z_lod_mat: u32,  // packed u16 z, lod_material
    _padding: u32,
}

struct Voxel {
    data: u32,
}

struct VoxelBrick {
    voxels: array<Voxel, 8>,
}

// Group 0: Camera + Shadow params
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> shadow_params: ShadowParams;

// Group 1: Octree data (same as svo_trace)
@group(1) @binding(0) var<storage, read> nodes: array<OctreeNode>;
@group(1) @binding(1) var<storage, read> bricks: array<VoxelBrick>;
@group(1) @binding(4) var<storage, read> chunk_infos: array<ChunkInfo>;
// Layer descriptor: first index in layer_data + count of layers at this cell
struct LayerDescriptor {
    base_index: u32,
    layer_count: u32,
}
@group(1) @binding(5) var<storage, read> chunk_grid: array<LayerDescriptor>;
@group(1) @binding(9) var<storage, read> layer_data: array<u32>;

// Group 2: G-buffer inputs (for world position reconstruction)
@group(2) @binding(0) var t_depth: texture_2d<f32>;
@group(2) @binding(1) var t_normal: texture_2d<f32>;
@group(2) @binding(2) var t_material: texture_2d<f32>;

// Group 3: Output shadow mask
@group(3) @binding(0) var output_shadow: texture_storage_2d<r32float, write>;

// Get child valid mask from node flags
fn get_child_valid_mask(node: OctreeNode) -> u32 {
    return node.flags & 0xFFu;
}

// Check if child at index is valid
fn is_child_valid(node: OctreeNode, index: u32) -> bool {
    return (get_child_valid_mask(node) & (1u << index)) != 0u;
}

// Check if child at index is a leaf
fn is_child_leaf(node: OctreeNode, index: u32) -> bool {
    return ((node.flags >> 8u) & (1u << index)) != 0u;
}

// Check if voxel is empty
fn is_voxel_empty(voxel: Voxel) -> bool {
    let color = voxel.data & 0xFFFFu;
    let material_id = (voxel.data >> 16u) & 0xFFu;
    return color == 0u && material_id == 0u;
}

// Get voxel material ID
fn get_voxel_material(voxel: Voxel) -> u32 {
    return (voxel.data >> 16u) & 0xFFu;
}

// Get brick voxel using Morton order index
fn get_brick_voxel(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> Voxel {
    let morton_idx = (lz << 2u) | (ly << 1u) | lx;
    return bricks[brick_idx].voxels[morton_idx];
}

// Ray-AABB intersection (slab method)
fn ray_aabb_intersect(
    ray_origin: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>
) -> vec2<f32> {
    let t1 = (box_min - ray_origin) * ray_inv_dir;
    let t2 = (box_max - ray_origin) * ray_inv_dir;

    let t_min = min(t1, t2);
    let t_max = max(t1, t2);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_near, t_far);
}

// Look up layer descriptor from the 3D grid.
fn grid_lookup(cx: i32, cy: i32, cz: i32) -> LayerDescriptor {
    let gx = cx - shadow_params.grid_min_x;
    let gy = cy - shadow_params.grid_min_y;
    let gz = cz - shadow_params.grid_min_z;
    if (gx < 0 || gy < 0 || gz < 0 ||
        u32(gx) >= shadow_params.grid_size_x ||
        u32(gy) >= shadow_params.grid_size_y ||
        u32(gz) >= shadow_params.grid_size_z) {
        var desc: LayerDescriptor;
        desc.base_index = 0u;
        desc.layer_count = 0u;
        return desc;
    }
    let idx = u32(gx) + u32(gy) * shadow_params.grid_size_x + u32(gz) * shadow_params.grid_size_x * shadow_params.grid_size_y;
    return chunk_grid[idx];
}

// Get chunk index at a specific layer offset within a grid cell
fn get_layer_chunk_index(desc: LayerDescriptor, layer_offset: u32) -> u32 {
    if (layer_offset >= desc.layer_count) {
        return 0xFFFFFFFFu;
    }
    return layer_data[desc.base_index + layer_offset];
}

// Trace a single chunk for shadow opacity. Returns opacity contribution.
fn trace_shadow_chunk(ray_origin: vec3<f32>, ray_inv_dir: vec3<f32>, chunk_idx: u32, opacity_in: f32, leaf_hits_in: u32) -> vec3<f32> {
    // Returns: (new_opacity, new_leaf_hits, did_hit) packed as vec3
    var shadow_opacity = opacity_in;
    var leaf_hits = leaf_hits_in;
    let max_leaf_hits = 8u;

    let chunk = chunk_infos[chunk_idx];
    let root_min = chunk.world_min;
    let root_max = chunk.world_min + vec3<f32>(chunk.root_size);

    // Check if ray hits this chunk's AABB
    let root_hit = ray_aabb_intersect(ray_origin, ray_inv_dir, root_min, root_max);
    if (root_hit.x > root_hit.y || root_hit.y < 0.0) {
        return vec3<f32>(shadow_opacity, f32(leaf_hits), 0.0);
    }

    // Stack for traversal (max 16 levels)
    var stack_node: array<u32, 16>;
    var stack_min: array<vec3<f32>, 16>;
    var stack_size: array<f32, 16>;
    var stack_ptr = 0;

    stack_node[0] = chunk.root_node;
    stack_min[0] = root_min;
    stack_size[0] = chunk.root_size;
    stack_ptr = 1;

    while (stack_ptr > 0) {
        if (shadow_opacity >= 0.95) {
            return vec3<f32>(shadow_opacity, f32(leaf_hits), 1.0);
        }

        stack_ptr -= 1;
        let node_idx = stack_node[stack_ptr];
        let node_min = stack_min[stack_ptr];
        let node_size = stack_size[stack_ptr];

        let node = nodes[node_idx];
        let child_valid = get_child_valid_mask(node);

        if (child_valid == 0u) {
            continue;
        }

        let child_size = node_size * 0.5;

        for (var i = 0u; i < 8u; i++) {
            if (!is_child_valid(node, i)) {
                continue;
            }

            let child_offset_v = vec3<f32>(
                f32(i & 1u),
                f32((i >> 1u) & 1u),
                f32((i >> 2u) & 1u)
            ) * child_size;
            let child_min = node_min + child_offset_v;
            let child_max = child_min + vec3<f32>(child_size);

            let child_hit = ray_aabb_intersect(ray_origin, ray_inv_dir, child_min, child_max);
            if (child_hit.x > child_hit.y || child_hit.y < 0.0) {
                continue;
            }

            if (is_child_leaf(node, i)) {
                var leaf_offset = 0u;
                for (var j = 0u; j < i; j++) {
                    if (is_child_valid(node, j) && is_child_leaf(node, j)) {
                        leaf_offset++;
                    }
                }
                let brick_idx = node.brick_offset + leaf_offset;

                var solid_count = 0u;
                var leaf_count = 0u;

                for (var vz = 0u; vz < 2u; vz++) {
                    for (var vy = 0u; vy < 2u; vy++) {
                        for (var vx = 0u; vx < 2u; vx++) {
                            let voxel = get_brick_voxel(brick_idx, vx, vy, vz);
                            if (!is_voxel_empty(voxel)) {
                                let mat_id = get_voxel_material(voxel);
                                if (mat_id == 3u) {
                                    leaf_count++;
                                } else {
                                    solid_count++;
                                }
                            }
                        }
                    }
                }

                if (solid_count > 0u) {
                    let fill_ratio = f32(solid_count) / 8.0;
                    shadow_opacity += fill_ratio * (1.0 - shadow_opacity);
                    if (shadow_opacity > 0.95) {
                        return vec3<f32>(1.0, f32(leaf_hits), 1.0);
                    }
                }

                if (leaf_count > 0u) {
                    leaf_hits++;
                    let fill_ratio = f32(leaf_count) / 8.0;
                    let brick_opacity = shadow_params.leaf_opacity * fill_ratio;
                    shadow_opacity += brick_opacity * (1.0 - shadow_opacity);
                    if (leaf_hits >= max_leaf_hits) {
                        return vec3<f32>(shadow_opacity, f32(leaf_hits), 1.0);
                    }
                }
            } else {
                if (stack_ptr < 15) {
                    var child_count = 0u;
                    for (var j = 0u; j < i; j++) {
                        if (is_child_valid(node, j) && !is_child_leaf(node, j)) {
                            child_count++;
                        }
                    }
                    stack_node[stack_ptr] = node.child_offset + child_count;
                    stack_min[stack_ptr] = child_min;
                    stack_size[stack_ptr] = child_size;
                    stack_ptr++;
                }
            }
        }
    }

    return vec3<f32>(shadow_opacity, f32(leaf_hits), 0.0);
}

// Shadow ray tracing using DDA grid traversal
// Returns shadow opacity (0.0 = fully lit, 1.0 = fully shadowed)
fn trace_shadow_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    let ray_inv_dir = 1.0 / ray_dir;
    let cs = shadow_params.chunk_size;

    var shadow_opacity = 0.0;
    var leaf_hits = 0u;

    // Grid world-space bounds
    let grid_world_min = vec3<f32>(
        f32(shadow_params.grid_min_x) * cs,
        f32(shadow_params.grid_min_y) * cs,
        f32(shadow_params.grid_min_z) * cs
    );
    let grid_world_max = vec3<f32>(
        f32(i32(shadow_params.grid_size_x) + shadow_params.grid_min_x) * cs,
        f32(i32(shadow_params.grid_size_y) + shadow_params.grid_min_y) * cs,
        f32(i32(shadow_params.grid_size_z) + shadow_params.grid_min_z) * cs
    );

    let t_bounds = ray_aabb_intersect(ray_origin, ray_inv_dir, grid_world_min, grid_world_max);
    let t_entry = max(t_bounds.x, 0.0);
    let t_exit = t_bounds.y;

    if (t_entry >= t_exit) {
        return 0.0;
    }

    let entry_pos = ray_origin + ray_dir * (t_entry + 0.001);
    var cell = vec3<i32>(floor(entry_pos / cs));
    cell = clamp(cell,
        vec3<i32>(shadow_params.grid_min_x, shadow_params.grid_min_y, shadow_params.grid_min_z),
        vec3<i32>(shadow_params.grid_min_x + i32(shadow_params.grid_size_x) - 1,
                  shadow_params.grid_min_y + i32(shadow_params.grid_size_y) - 1,
                  shadow_params.grid_min_z + i32(shadow_params.grid_size_z) - 1));

    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0)
    );
    let delta_t = abs(vec3<f32>(cs) / ray_dir);

    var next_t: vec3<f32>;
    if (ray_dir.x >= 0.0) { next_t.x = (f32(cell.x + 1) * cs - ray_origin.x) / ray_dir.x; }
    else { next_t.x = (f32(cell.x) * cs - ray_origin.x) / ray_dir.x; }
    if (ray_dir.y >= 0.0) { next_t.y = (f32(cell.y + 1) * cs - ray_origin.y) / ray_dir.y; }
    else { next_t.y = (f32(cell.y) * cs - ray_origin.y) / ray_dir.y; }
    if (ray_dir.z >= 0.0) { next_t.z = (f32(cell.z + 1) * cs - ray_origin.z) / ray_dir.z; }
    else { next_t.z = (f32(cell.z) * cs - ray_origin.z) / ray_dir.z; }

    let max_steps = shadow_params.grid_size_x + shadow_params.grid_size_y + shadow_params.grid_size_z;
    for (var iter = 0u; iter < max_steps; iter++) {
        let layer_desc = grid_lookup(cell.x, cell.y, cell.z);

        // Check all layers at this cell
        if (layer_desc.layer_count > 0u) {
            for (var layer_idx = 0u; layer_idx < layer_desc.layer_count; layer_idx++) {
                let chunk_idx = get_layer_chunk_index(layer_desc, layer_idx);
                if (chunk_idx != 0xFFFFFFFFu) {
                    let r = trace_shadow_chunk(ray_origin, ray_inv_dir, chunk_idx, shadow_opacity, leaf_hits);
                    shadow_opacity = r.x;
                    leaf_hits = u32(r.y);
                    if (shadow_opacity >= 0.95) {
                        return shadow_opacity;
                    }
                }
            }
        }

        if (next_t.x < next_t.y && next_t.x < next_t.z) {
            cell.x += step.x;
            next_t.x += delta_t.x;
        } else if (next_t.y < next_t.z) {
            cell.y += step.y;
            next_t.y += delta_t.y;
        } else {
            cell.z += step.z;
            next_t.z += delta_t.z;
        }

        if (cell.x < shadow_params.grid_min_x || cell.x >= shadow_params.grid_min_x + i32(shadow_params.grid_size_x) ||
            cell.y < shadow_params.grid_min_y || cell.y >= shadow_params.grid_min_y + i32(shadow_params.grid_size_y) ||
            cell.z < shadow_params.grid_min_z || cell.z >= shadow_params.grid_min_z + i32(shadow_params.grid_size_z)) {
            break;
        }
    }

    return shadow_opacity;
}

// Generate a random-looking 2D vector from an index (Hammersley sequence approximation)
fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    let fi = f32(i);
    let f_n = f32(n);
    return vec2<f32>(fi / f_n, radical_inverse_vdc(i));
}

// Radical inverse for quasi-random sampling
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Construct tangent space basis from a normal
fn make_tangent_frame(normal: vec3<f32>) -> mat3x3<f32> {
    // Find arbitrary perpendicular vector
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// Sample a direction within a cone around the light direction
fn sample_cone_direction(light_dir: vec3<f32>, cone_angle: f32, sample_2d: vec2<f32>) -> vec3<f32> {
    // Convert uniform sample to cone
    let cos_theta = mix(cos(cone_angle), 1.0, sample_2d.x);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let phi = sample_2d.y * 2.0 * 3.14159265359;

    // Local direction in cone space
    let local_dir = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    // Transform to world space (light_dir is the cone axis)
    let basis = make_tangent_frame(light_dir);
    return normalize(basis * local_dir);
}

// Reconstruct world position from depth and screen UV
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC coordinates
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,  // Flip Y for texture coordinates
        depth,
        1.0
    );

    // Transform to world space
    let world_pos = camera.view_proj_inv * ndc;
    return world_pos.xyz / world_pos.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);

    // Bounds check
    if (global_id.x >= shadow_params.width || global_id.y >= shadow_params.height) {
        return;
    }

    let dims = vec2<f32>(f32(shadow_params.width), f32(shadow_params.height));
    let uv = (vec2<f32>(global_id.xy) + 0.5) / dims;

    // Sample G-buffer
    let depth = textureLoad(t_depth, coords, 0).r;
    let normal_data = textureLoad(t_normal, coords, 0);
    let normal = normal_data.xyz;

    // Skip sky pixels (zero-length normal indicates sky)
    if (length(normal) < 0.001) {
        textureStore(output_shadow, coords, vec4<f32>(1.0, 0.0, 0.0, 0.0)); // Sky is always lit
        return;
    }

    // Reconstruct world position
    let world_pos = reconstruct_world_pos(uv, depth);

    // Transform to octree space and apply shadow bias
    let octree_pos = world_pos - camera.world_offset;
    let biased_pos = octree_pos + normal * shadow_params.shadow_bias;

    // Trace shadow rays
    var shadow_sum = 0.0;
    let num_samples = max(shadow_params.soft_shadow_samples, 1u);

    if (num_samples == 1u) {
        // Hard shadows - single ray
        let shadow_opacity = trace_shadow_ray(biased_pos, shadow_params.light_dir);
        shadow_sum = 1.0 - shadow_opacity;
    } else {
        // Soft shadows - multiple rays in a cone
        for (var i = 0u; i < num_samples; i++) {
            let sample_2d = hammersley_2d(i, num_samples);
            let sample_dir = sample_cone_direction(
                shadow_params.light_dir,
                shadow_params.soft_shadow_angle,
                sample_2d
            );
            let shadow_opacity = trace_shadow_ray(biased_pos, sample_dir);
            shadow_sum += 1.0 - shadow_opacity;
        }
        shadow_sum /= f32(num_samples);
    }

    // Write shadow factor (1.0 = fully lit, 0.0 = fully shadowed)
    textureStore(output_shadow, coords, vec4<f32>(shadow_sum, 0.0, 0.0, 0.0));
}
