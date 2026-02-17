// SVO Ray Tracing Compute Shader - Multi-Chunk

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

struct OctreeNode {
    flags: u32,
    child_offset: u32,   // offset to first internal child in node array (absolute)
    brick_offset: u32,   // offset to first leaf child in brick array (absolute)
    bounds_min_xy: u32,  // packed u16 x, y
    bounds_min_z_lod_color: u32,  // packed u16 z, lod_color
    bounds_max_xy: u32,  // packed u16 x, y
    bounds_max_z_lod_mat: u32,  // packed u16 z, lod_material
    _padding: u32,
}

// Voxel: packed as color(u16) | material_id(u8) | flags(u8)
struct Voxel {
    data: u32,
}

struct VoxelBrick {
    voxels: array<Voxel, 8>,
}

struct ChunkInfo {
    world_min: vec3<f32>,
    root_size: f32,
    root_node: u32,
    max_depth: u32,
    layer_id: u32,
    flags: u32,
}

// Sentinel value for brick not loaded (for streaming)
const BRICK_NOT_LOADED: u32 = 0xFFFFFFFFu;

// Unpack voxel color from packed data
fn unpack_voxel_color(voxel: Voxel) -> vec3<f32> {
    let color = voxel.data & 0xFFFFu;
    return unpack_rgb565(color);
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

struct TraceParams {
    width: u32,
    height: u32,
    chunk_count: u32,
    _pad0: u32,
    // LOD distances for 6 levels (64, 128, 256, 512, 1024, inf)
    lod_distances: vec4<f32>,
    lod_distances_ext: vec2<f32>,
    _pad: vec2<f32>,
    // Chunk grid acceleration structure
    grid_min_x: i32,
    grid_min_y: i32,
    grid_min_z: i32,
    chunk_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    _pad2: u32,
}

struct GrassParams {
    enabled: u32,
    max_distance: f32,
    fade_start: f32,
    time: f32,
    wind_direction: vec3<f32>,
    wind_speed: f32,
    profile_count: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct GrassProfileGpu {
    height_min: f32,
    height_max: f32,
    width: f32,
    density: f32,
    color_base: vec3<f32>,
    sway_amount: f32,
    color_variation: f32,
    sway_frequency: f32,
    blade_spacing: f32,
    slope_threshold: f32,
    coverage_scale: f32,
    coverage_amount: f32,
    _pad0: f32,
    _pad1: f32,
}

struct GrassMaskInfo {
    node_offset: u32,
    value_offset: u32,
    node_count: u32,
    max_depth: u32,
}

struct GrassMaskNode {
    masks: u32,
    child_offset: u32,
    value_offset: u32,
    lod_value_idx: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: TraceParams;
@group(0) @binding(2) var<uniform> grass: GrassParams;
@group(0) @binding(3) var<storage, read> grass_profiles: array<GrassProfileGpu>;

@group(1) @binding(0) var<storage, read> nodes: array<OctreeNode>;
@group(1) @binding(1) var<storage, read> bricks: array<VoxelBrick>;

struct FeedbackData {
    count: atomic<u32>,
    max_requests: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(2) var<storage, read_write> feedback_header: FeedbackData;
@group(1) @binding(3) var<storage, read_write> feedback_requests: array<u32>;
@group(1) @binding(4) var<storage, read> chunk_infos: array<ChunkInfo>;
// Layer descriptor: first index in layer_data + count of layers at this cell
struct LayerDescriptor {
    base_index: u32,
    layer_count: u32,
}
@group(1) @binding(5) var<storage, read> chunk_grid: array<LayerDescriptor>;
@group(1) @binding(9) var<storage, read> layer_data: array<u32>;
@group(1) @binding(6) var<storage, read> grass_mask_info: array<GrassMaskInfo>;
@group(1) @binding(7) var<storage, read> grass_mask_nodes: array<GrassMaskNode>;
@group(1) @binding(8) var<storage, read> grass_mask_values: array<u32>;

@group(2) @binding(0) var output_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var output_normal: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var output_depth: texture_storage_2d<r32float, write>;
@group(2) @binding(3) var output_material: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(4) var output_motion: texture_storage_2d<rgba16float, write>;

// Unpack RGB565 color
fn unpack_rgb565(color: u32) -> vec3<f32> {
    let r = f32((color >> 11u) & 0x1Fu) / 31.0;
    let g = f32((color >> 5u) & 0x3Fu) / 63.0;
    let b = f32(color & 0x1Fu) / 31.0;
    return vec3<f32>(r, g, b);
}

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

// Get LOD color from node
fn get_lod_color(node: OctreeNode) -> vec3<f32> {
    let color = (node.bounds_min_z_lod_color >> 16u) & 0xFFFFu;
    return unpack_rgb565(color);
}

// Get LOD material from node
fn get_lod_material(node: OctreeNode) -> u32 {
    return (node.bounds_max_z_lod_mat >> 16u) & 0xFFFFu;
}

// Hit result structure
struct HitResult {
    hit: bool,
    t: f32,
    color: vec3<f32>,
    normal: vec3<f32>,
    material_id: u32,
    roughness: f32,
    translucency: f32,
    chunk_idx: u32,
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

// Get child index for a point within a node
fn get_child_index(point: vec3<f32>, center: vec3<f32>) -> u32 {
    var index = 0u;
    if (point.x >= center.x) { index |= 1u; }
    if (point.y >= center.y) { index |= 2u; }
    if (point.z >= center.z) { index |= 4u; }
    return index;
}

// Compute smooth surface normal from an occupancy bitmask of 8 cells (2x2x2).
// The center of mass of filled cells indicates the solid direction;
// the normal points away from the filled region (toward the empty side).
// Works for both brick voxels (8 voxels) and node children (8 octants).
fn coverage_normal(filled_mask: u32) -> vec3<f32> {
    var center = vec3<f32>(0.0);
    var count = 0.0;
    for (var i = 0u; i < 8u; i++) {
        if ((filled_mask & (1u << i)) != 0u) {
            center += vec3<f32>(
                f32(i & 1u),
                f32((i >> 1u) & 1u),
                f32((i >> 2u) & 1u)
            );
            count += 1.0;
        }
    }
    if (count < 1.0 || count > 7.0) {
        return vec3<f32>(0.0, 1.0, 0.0); // fully empty or fully solid -> default up
    }
    let offset = center / count - vec3<f32>(0.5);
    let len = length(offset);
    if (len < 0.01) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(-offset);
}

// Material PBR properties lookup
// Returns vec3(roughness, metallic, translucency)
fn get_material_properties(material_id: u32) -> vec3<f32> {
    switch material_id {
        // Core material types (non-terrain objects like trees)
        case 1u: { return vec3<f32>(0.9, 0.0, 0.0); }   // Sand: very rough
        case 2u: { return vec3<f32>(0.85, 0.0, 0.0); }  // Bark: rough, matte
        case 3u: { return vec3<f32>(0.55, 0.0, 0.8); }  // Leaves: slight sheen, translucent
        case 4u: { return vec3<f32>(0.95, 0.0, 0.0); }  // Stone: very rough
        case 5u: { return vec3<f32>(0.8, 0.1, 0.0); }   // Snow: rough, slight metallic
        case 6u: { return vec3<f32>(0.9, 0.0, 0.0); }   // Dirt: very rough
        // Biome terrain surfaces (inherit PBR from base type)
        case 7u:  { return vec3<f32>(0.9, 0.0, 0.0); }  // Beach sand
        case 8u:  { return vec3<f32>(0.9, 0.0, 0.0); }  // Desert sand
        case 9u:  { return vec3<f32>(0.85, 0.0, 0.0); } // Grassland
        case 10u: { return vec3<f32>(0.85, 0.0, 0.0); } // Forest
        case 11u: { return vec3<f32>(0.85, 0.0, 0.0); } // Taiga
        case 12u: { return vec3<f32>(0.9, 0.0, 0.0); }  // Tundra
        case 13u: { return vec3<f32>(0.95, 0.0, 0.0); } // Mountains
        case 14u: { return vec3<f32>(0.8, 0.1, 0.0); }  // Snow terrain
        case 15u: { return vec3<f32>(0.95, 0.0, 0.0); } // Ocean floor
        case 16u: { return vec3<f32>(0.9, 0.0, 0.3); }  // Grass blade: rough, slight translucency
        default: { return vec3<f32>(0.5, 0.0, 0.0); }   // Default
    }
}

// Get voxel flags (sub-voxel height fraction for terrain)
fn get_voxel_flags(voxel: Voxel) -> u32 {
    return (voxel.data >> 24u) & 0xFFu;
}

// --- Procedural color variation for terrain ---

// Hash function for 3D → 1D
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth 3D value noise
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash31(i), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y),
        u.z
    );
}

// Multi-octave terrain color variation at a world position
fn terrain_color_variation(world_pos: vec3<f32>, material_id: u32) -> vec3<f32> {
    let base = terrain_base_color(material_id);

    // 3-octave FBM for natural variation (XZ only — terrain is horizontal)
    let p = vec3<f32>(world_pos.x, 0.0, world_pos.z);
    let n1 = value_noise(p * 0.5);    // ~2m patches
    let n2 = value_noise(p * 2.0);    // ~0.5m detail
    let n3 = value_noise(p * 8.0);    // ~0.125m fine grain
    let fbm = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;  // [0, 1]

    // Brightness variation: ±15%
    let brightness = 0.85 + fbm * 0.30;

    // Subtle hue shift toward warm/cool based on noise
    let warm = vec3<f32>(0.04, 0.02, -0.02) * (fbm - 0.5);

    return clamp(base * brightness + warm, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Base color for terrain biome surfaces (IDs 4, 7-15)
// Color field stores gradient, so visual color comes from this LUT
fn terrain_base_color(material_id: u32) -> vec3<f32> {
    switch material_id {
        case 4u:  { return vec3<f32>(0.35, 0.33, 0.32); } // Stone - dark grey-brown
        case 20u: { return vec3<f32>(0.35, 0.33, 0.32); } // Rock - same as stone for now
        case 7u:  { return vec3<f32>(0.93, 0.84, 0.69); } // Beach sand
        case 8u:  { return vec3<f32>(0.93, 0.79, 0.69); } // Desert sand
        case 9u:  { return vec3<f32>(0.39, 0.71, 0.31); } // Grassland
        case 10u: { return vec3<f32>(0.20, 0.47, 0.16); } // Forest
        case 11u: { return vec3<f32>(0.31, 0.39, 0.24); } // Taiga
        case 12u: { return vec3<f32>(0.63, 0.71, 0.67); } // Tundra
        case 13u: { return vec3<f32>(0.47, 0.47, 0.47); } // Mountains
        case 14u: { return vec3<f32>(0.94, 0.97, 1.00); } // Snow
        case 15u: { return vec3<f32>(0.12, 0.31, 0.59); } // Ocean floor
        default:  { return vec3<f32>(0.50, 0.50, 0.50); }
    }
}

// Procedural bark color variation — vertical streaks + moss patches
fn bark_color_variation(world_pos: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    // 3-octave FBM with stretched Y for vertical streaks
    let p = world_pos;
    let n1 = value_noise(vec3<f32>(p.x * 2.0, p.y * 0.5, p.z * 2.0));   // vertical streaks
    let n2 = value_noise(vec3<f32>(p.x * 6.0, p.y * 1.5, p.z * 6.0));   // medium detail
    let n3 = value_noise(vec3<f32>(p.x * 16.0, p.y * 4.0, p.z * 16.0)); // fine grain
    let fbm = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

    // Brightness variation: +-15%
    let brightness = 0.85 + fbm * 0.30;
    var color = base_color * brightness;

    // Subtle moss patches on north-facing or damp areas
    let moss_noise = value_noise(world_pos * 1.5 + vec3<f32>(100.0, 0.0, 100.0));
    if (moss_noise > 0.7) {
        let moss_strength = smoothstep(0.7, 0.85, moss_noise) * 0.3;
        let moss_color = vec3<f32>(0.15, 0.25, 0.08);
        color = mix(color, moss_color, moss_strength);
    }

    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Procedural leaf color variation — canopy height + per-voxel tint
fn leaf_color_variation(world_pos: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    // 2-octave noise for positional variation
    let n1 = value_noise(world_pos * 1.5);
    let n2 = value_noise(world_pos * 5.0 + vec3<f32>(50.0, 50.0, 50.0));
    let fbm = n1 * 0.6 + n2 * 0.4;

    // Height-based: lighter at canopy top (more sun)
    let height_factor = saturate(world_pos.y * 0.1);  // higher = lighter
    let sun_boost = mix(0.85, 1.1, height_factor);

    // Per-voxel hash for individual tint
    let voxel_hash = hash31(floor(world_pos * 8.0));
    let tint = vec3<f32>(
        (voxel_hash - 0.5) * 0.12,
        (fract(voxel_hash * 7.13) - 0.5) * 0.08,
        (fract(voxel_hash * 3.97) - 0.5) * 0.06
    );

    // Brightness variation from noise
    let brightness = 0.9 + fbm * 0.2;

    return clamp(base_color * brightness * sun_boost + tint, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Wind displacement for tree voxels — displaces AABB before ray intersection
fn tree_wind_displacement(world_pos: vec3<f32>, material_id: u32, chunk_min_y: f32, chunk_size: f32) -> vec3<f32> {
    let hfrac = saturate((world_pos.y - chunk_min_y) / chunk_size);
    let h2 = hfrac * hfrac;  // quadratic — base barely moves

    // Trunk sway (slow, all tree parts)
    let trunk_phase = grass.time * 0.7 + hash31(vec3<f32>(chunk_min_y, 0.0, 0.0));
    let trunk = vec2<f32>(sin(trunk_phase), cos(trunk_phase * 0.6)) * 0.03 * h2;

    // Branch oscillation (medium freq, bark only)
    var branch = vec2<f32>(0.0);
    if (material_id == 2u) {
        let bp = grass.time * 2.5 + hash31(world_pos * 0.5) * 6.28;
        branch = vec2<f32>(sin(bp), cos(bp * 0.8)) * 0.015 * hfrac;
    }

    // Leaf flutter (fast, leaves only)
    var flutter = vec2<f32>(0.0);
    if (material_id == 3u) {
        let fp = grass.time * 6.0 + hash31(world_pos * 2.0) * 6.28;
        flutter = vec2<f32>(sin(fp), cos(fp * 1.3)) * 0.025 * hfrac;
    }

    let wind_factor = grass.wind_speed * 0.5 + 0.15;
    let d = (trunk + branch + flutter) * wind_factor;
    return vec3<f32>(
        d.x * grass.wind_direction.x + d.y * grass.wind_direction.z,
        0.0,
        d.x * grass.wind_direction.z - d.y * grass.wind_direction.x
    );
}

// Decode terrain normal from gradient encoded in voxel color field
fn decode_terrain_normal(voxel: Voxel) -> vec3<f32> {
    let encoded = voxel.data & 0xFFFFu;
    let dx_enc = encoded & 0xFFu;
    let dz_enc = (encoded >> 8u) & 0xFFu;
    // Decode from [0,255] → [-4, 4]
    let dh_dx = f32(dx_enc) / 255.0 * 8.0 - 4.0;
    let dh_dz = f32(dz_enc) / 255.0 * 8.0 - 4.0;
    return normalize(vec3<f32>(-dh_dx, 1.0, -dh_dz));
}

// Decode bark surface normal from RGB565-encoded normal in voxel color field.
// Bark voxels store the radial direction (from branch axis to surface) in the color,
// just like terrain stores gradient normals. R(5)=nx, G(6)=ny, B(5)=nz.
fn decode_bark_normal(voxel: Voxel) -> vec3<f32> {
    let encoded = voxel.data & 0xFFFFu;
    let r5 = (encoded >> 11u) & 0x1Fu;
    let g6 = (encoded >> 5u) & 0x3Fu;
    let b5 = encoded & 0x1Fu;
    let nx = f32(r5) / 31.0 * 2.0 - 1.0;
    let ny = f32(g6) / 63.0 * 2.0 - 1.0;
    let nz = f32(b5) / 31.0 * 2.0 - 1.0;
    return normalize(vec3<f32>(nx, ny, nz));
}

// Compute box normal from hit point relative to voxel center
fn compute_box_normal(hit_point: vec3<f32>, center: vec3<f32>, half_size: f32) -> vec3<f32> {
    let rel = (hit_point - center) / half_size;
    let abs_rel = abs(rel);
    if (abs_rel.x > abs_rel.y && abs_rel.x > abs_rel.z) {
        return vec3<f32>(sign(rel.x), 0.0, 0.0);
    } else if (abs_rel.y > abs_rel.z) {
        return vec3<f32>(0.0, sign(rel.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign(rel.z));
}

// Build a bitmask of which voxels in a 2x2x2 brick are non-empty.
// Bit i corresponds to voxel (i&1, (i>>1)&1, (i>>2)&1).
fn get_brick_filled_mask(brick_idx: u32) -> u32 {
    var mask = 0u;
    for (var i = 0u; i < 8u; i++) {
        let lx = i & 1u;
        let ly = (i >> 1u) & 1u;
        let lz = (i >> 2u) & 1u;
        if (!is_voxel_empty(get_brick_voxel(brick_idx, lx, ly, lz))) {
            mask |= (1u << i);
        }
    }
    return mask;
}

// Trace ray through a 2x2x2 brick
fn trace_brick(
    brick_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    brick_min: vec3<f32>,
    brick_size: f32,
    max_t: f32,
    chunk_world_min_y: f32,
    chunk_root_size: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = max_t;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = 0xFFFFFFFFu;

    // Pre-compute filled mask for coverage normals (tree smoothing)
    let filled_mask = get_brick_filled_mask(brick_idx);

    for (var vz = 0u; vz < 2u; vz++) {
        for (var vy = 0u; vy < 2u; vy++) {
            for (var vx = 0u; vx < 2u; vx++) {
                let voxel = get_brick_voxel(brick_idx, vx, vy, vz);

                if (is_voxel_empty(voxel)) {
                    continue;
                }

                let material_id = get_voxel_material(voxel);
                let flags = get_voxel_flags(voxel);
                let voxel_size = brick_size * 0.5;
                let voxel_offset = vec3<f32>(f32(vx), f32(vy), f32(vz)) * voxel_size;
                var voxel_min = brick_min + voxel_offset;
                var voxel_max = voxel_min + vec3<f32>(voxel_size);

                let is_tree = material_id == 2u || material_id == 3u;

                // Wind displacement for tree materials
                if (is_tree) {
                    let world_center = voxel_min + vec3<f32>(voxel_size * 0.5);
                    let disp = tree_wind_displacement(world_center, material_id, chunk_world_min_y, chunk_root_size);
                    voxel_min += disp;
                    voxel_max += disp;
                }

                let hit = ray_aabb_intersect(ray_origin, ray_inv_dir, voxel_min, voxel_max);

                if (hit.x <= hit.y && hit.y >= 0.0 && hit.x < result.t) {
                    let t_entry = max(hit.x, 0.0);

                    // Terrain: flags > 0 means sub-voxel surface (height fraction encoded)
                    // Tree bark/leaves with flags > 0: SDF distance encoded
                    // Anything with flags == 0: standard box intersection
                    let is_terrain = !is_tree && flags > 0u;

                    var final_t = t_entry;
                    var final_normal: vec3<f32>;
                    var valid_hit = true;

                    if (is_terrain) {
                        // Sub-voxel surface: flags encodes height fraction (1-255 -> 0.0-1.0)
                        let h_frac = f32(flags) / 255.0;
                        let surface_y = voxel_min.y + h_frac * voxel_size;
                        let terrain_n = decode_terrain_normal(voxel);

                        var plane_hit = false;
                        if (abs(ray_dir.y) > 0.0001) {
                            let t_plane = (surface_y - ray_origin.y) / ray_dir.y;
                            if (t_plane >= t_entry - 0.001 && t_plane <= hit.y + 0.001) {
                                let p = ray_origin + ray_dir * t_plane;
                                if (p.x >= voxel_min.x - 0.001 && p.x <= voxel_max.x + 0.001 &&
                                    p.z >= voxel_min.z - 0.001 && p.z <= voxel_max.z + 0.001) {
                                    final_t = t_plane;
                                    final_normal = terrain_n;
                                    plane_hit = true;
                                }
                            }
                        }

                        if (!plane_hit) {
                            let entry_y = ray_origin.y + ray_dir.y * t_entry;
                            if (entry_y <= surface_y) {
                                final_t = t_entry;
                                final_normal = terrain_n;
                            } else {
                                valid_hit = false;
                            }
                        }
                    } else if (is_tree) {
                        // Tree voxels: precise encoded normals (like terrain).
                        // Bark: radial normal (branch axis → surface)
                        // Leaves: spherical normal (cloud center → voxel)
                        // Both encoded in the color field by the generator.

                        if (material_id == 2u && flags > 0u) {
                            // Bark: precise radial normal from generator
                            final_normal = decode_bark_normal(voxel);
                        } else if (material_id == 3u) {
                            // Leaves: soft edge culling + precise spherical normal
                            if (flags > 0u) {
                                let sdf_frac = f32(flags) / 255.0;
                                if (sdf_frac < 0.05) {
                                    valid_hit = false;
                                }
                            }
                            if (valid_hit) {
                                // Decode cloud-center-to-point normal from color field
                                final_normal = decode_bark_normal(voxel);
                            }
                        } else {
                            // Other tree materials: box normal
                            let center = voxel_min + vec3<f32>(voxel_size * 0.5);
                            let entry_point = ray_origin + ray_dir * t_entry;
                            final_normal = compute_box_normal(entry_point, center, voxel_size * 0.5);
                        }

                        // Hemisphere check: ensure shading normal faces toward the ray.
                        // Prevents pure-black back-faces at branch junctions and edges.
                        if (valid_hit && dot(final_normal, ray_dir) > 0.0) {
                            final_normal = -final_normal;
                        }
                    } else {
                        // Standard box intersection
                        final_t = t_entry;
                        let center = voxel_min + vec3<f32>(voxel_size * 0.5);
                        let entry_point = ray_origin + ray_dir * t_entry;
                        final_normal = compute_box_normal(entry_point, center, voxel_size * 0.5);
                    }

                    if (valid_hit && final_t < result.t) {
                        result.hit = true;
                        result.t = final_t;
                        let world_hit = ray_origin + ray_dir * final_t;

                        // Color: terrain procedural, bark procedural, leaf procedural, else voxel data
                        if (is_terrain) {
                            result.color = terrain_color_variation(world_hit, material_id);
                        } else if (material_id == 2u) {
                            // Bark color field stores encoded normal, not color — use fixed base
                            let bark_base = vec3<f32>(0.35, 0.25, 0.15);
                            result.color = bark_color_variation(world_hit, bark_base);
                        } else if (material_id == 3u) {
                            // Leaf color field stores encoded normal, not color — use fixed base
                            let leaf_base = vec3<f32>(0.20, 0.35, 0.08);
                            result.color = leaf_color_variation(world_hit, leaf_base);
                        } else {
                            result.color = unpack_voxel_color(voxel);
                        }
                        result.material_id = material_id;
                        result.normal = final_normal;

                        let mat_props = get_material_properties(material_id);
                        result.roughness = mat_props.x;
                        result.translucency = mat_props.z;
                    }
                }
            }
        }
    }

    return result;
}

// Calculate LOD level based on distance from camera
fn calculate_lod(distance: f32) -> u32 {
    if (distance < params.lod_distances.x) { return 0u; }
    if (distance < params.lod_distances.y) { return 1u; }
    if (distance < params.lod_distances.z) { return 2u; }
    if (distance < params.lod_distances.w) { return 3u; }
    if (distance < params.lod_distances_ext.x) { return 4u; }
    return 5u;
}

// Calculate maximum traversal depth for a given LOD level
fn get_max_depth_for_lod(lod_level: u32, chunk_max_depth: u32) -> u32 {
    if (lod_level >= chunk_max_depth) {
        return 0u;
    }
    return chunk_max_depth - lod_level;
}

// LOD crossfade for smooth transitions
fn lod_crossfade_factor(distance: f32) -> f32 {
    var transition_start: f32;
    var transition_end: f32;

    if (distance < params.lod_distances.x) {
        transition_start = params.lod_distances.x * 0.85;
        transition_end = params.lod_distances.x;
    } else if (distance < params.lod_distances.y) {
        transition_start = params.lod_distances.y * 0.85;
        transition_end = params.lod_distances.y;
    } else if (distance < params.lod_distances.z) {
        transition_start = params.lod_distances.z * 0.85;
        transition_end = params.lod_distances.z;
    } else if (distance < params.lod_distances.w) {
        transition_start = params.lod_distances.w * 0.85;
        transition_end = params.lod_distances.w;
    } else {
        return 0.0;
    }

    if (distance < transition_start) {
        return 0.0;
    }
    let t = (distance - transition_start) / (transition_end - transition_start);
    return t * t * (3.0 - 2.0 * t);
}

// Trace a single chunk's octree starting from its root node
fn trace_chunk_octree(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    chunk_idx: u32,
    pixel_hash: f32,
    best_t: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = best_t;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = chunk_idx;

    let chunk = chunk_infos[chunk_idx];
    let root_min = chunk.world_min;
    let root_max = chunk.world_min + vec3<f32>(chunk.root_size);

    // Check if ray hits this chunk
    let root_hit = ray_aabb_intersect(ray_origin, ray_inv_dir, root_min, root_max);
    if (root_hit.x > root_hit.y || root_hit.y < 0.0) {
        return result;
    }
    // Skip if chunk is entirely behind current best hit
    let chunk_entry = max(root_hit.x, 0.0);
    if (chunk_entry >= result.t) {
        return result;
    }

    // Stack for traversal (max 16 levels)
    var stack_node: array<u32, 16>;
    var stack_min: array<vec3<f32>, 16>;
    var stack_size: array<f32, 16>;
    var stack_depth: array<u32, 16>;
    var stack_ptr = 0;

    // Push root
    stack_node[0] = chunk.root_node;
    stack_min[0] = root_min;
    stack_size[0] = chunk.root_size;
    stack_depth[0] = 0u;
    stack_ptr = 1;

    while (stack_ptr > 0) {
        stack_ptr -= 1;
        let node_idx = stack_node[stack_ptr];
        let node_min = stack_min[stack_ptr];
        let node_size = stack_size[stack_ptr];
        let current_depth = stack_depth[stack_ptr];
        let node_max = node_min + vec3<f32>(node_size);

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

            let child_offset = vec3<f32>(
                f32(i & 1u),
                f32((i >> 1u) & 1u),
                f32((i >> 2u) & 1u)
            ) * child_size;
            let child_min = node_min + child_offset;
            let child_max = child_min + vec3<f32>(child_size);

            let child_hit = ray_aabb_intersect(ray_origin, ray_inv_dir, child_min, child_max);
            if (child_hit.x > child_hit.y || child_hit.y < 0.0) {
                continue;
            }
            let entry_t = max(child_hit.x, 0.0);
            if (entry_t >= result.t) {
                continue;
            }

            // Surface-adaptive LOD: only reduce detail for fully solid interior nodes.
            // Surface nodes (some children empty) always get full-depth traversal
            // so sub-voxel height fractions and terrain normals are used.
            let raw_lod_depth = get_max_depth_for_lod(calculate_lod(entry_t), chunk.max_depth);
            let lod_depth = max(raw_lod_depth, chunk.max_depth - 1u);
            let reached_lod_limit = current_depth >= lod_depth && child_valid == 0xFFu;

            if (is_child_leaf(node, i) || reached_lod_limit) {
                let t = entry_t;
                var used_brick_hit = false;
                var traced_empty_brick = false;

                if (is_child_leaf(node, i)) {
                    // In crossfade zone, some pixels skip brick tracing for smooth LOD transition
                    let fade = lod_crossfade_factor(entry_t);
                    let skip_brick = (fade > 0.0 && pixel_hash < fade);

                    if (!skip_brick) {
                        var leaf_offset = 0u;
                        for (var j = 0u; j < i; j++) {
                            if (is_child_valid(node, j) && is_child_leaf(node, j)) {
                                leaf_offset++;
                            }
                        }
                        let brick_idx = node.brick_offset + leaf_offset;

                        if (brick_idx == BRICK_NOT_LOADED) {
                            let request_idx = atomicAdd(&feedback_header.count, 1u);
                            if (request_idx < feedback_header.max_requests) {
                                feedback_requests[request_idx] = node_idx | (i << 24u);
                            }
                        } else {
                            let brick_hit = trace_brick(
                                brick_idx,
                                ray_origin,
                                ray_dir,
                                ray_inv_dir,
                                child_min,
                                child_size,
                                result.t,
                                chunk.world_min.y,
                                chunk.root_size
                            );

                            if (brick_hit.hit && brick_hit.t < result.t) {
                                result = brick_hit;
                                result.chunk_idx = chunk_idx; // Preserve chunk_idx (trace_brick doesn't know it)
                                used_brick_hit = true;
                            } else {
                                traced_empty_brick = true;
                            }
                        }
                    }
                }

                if (!used_brick_hit && !traced_empty_brick) {
                    if (t < result.t) {
                        result.hit = true;
                        result.t = t;

                        let hit_point = ray_origin + ray_dir * t;
                        let rel = (hit_point - child_min) / child_size;

                        // Calculate box normal from hit face
                        var box_normal: vec3<f32>;
                        if (rel.x < 0.01) { box_normal = vec3<f32>(-1.0, 0.0, 0.0); }
                        else if (rel.x > 0.99) { box_normal = vec3<f32>(1.0, 0.0, 0.0); }
                        else if (rel.y < 0.01) { box_normal = vec3<f32>(0.0, -1.0, 0.0); }
                        else if (rel.y > 0.99) { box_normal = vec3<f32>(0.0, 1.0, 0.0); }
                        else if (rel.z < 0.01) { box_normal = vec3<f32>(0.0, 0.0, -1.0); }
                        else { box_normal = vec3<f32>(0.0, 0.0, 1.0); }

                        // Read child node for better color when available
                        var lod_mat: u32;
                        if (!is_child_leaf(node, i)) {
                            var internal_count = 0u;
                            for (var j = 0u; j < i; j++) {
                                if (is_child_valid(node, j) && !is_child_leaf(node, j)) {
                                    internal_count++;
                                }
                            }
                            let child_node = nodes[node.child_offset + internal_count];
                            result.color = get_lod_color(child_node);
                            lod_mat = get_lod_material(child_node) & 0xFFu;
                        } else {
                            result.color = get_lod_color(node);
                            lod_mat = get_lod_material(node) & 0xFFu;
                        }
                        result.material_id = lod_mat;

                        // Terrain biome surfaces (7-15): procedural color + up normal
                        let is_lod_terrain = lod_mat >= 7u && lod_mat <= 15u;
                        if (is_lod_terrain) {
                            let world_hit = ray_origin + ray_dir * t;
                            result.color = terrain_color_variation(world_hit, lod_mat);
                            box_normal = vec3<f32>(0.0, 1.0, 0.0);
                        }

                        let mat_props = get_material_properties(result.material_id);
                        result.roughness = mat_props.x;
                        result.translucency = mat_props.z;

                        result.normal = box_normal;
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
                    stack_depth[stack_ptr] = current_depth + 1u;
                    stack_ptr++;
                }
            }
        }
    }

    return result;
}

// --- Procedural grass ---

// Hash vec2 → f32 in [0, 1]
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth 2D value noise for grass coverage (bilinear interpolation of hash values)
fn value_noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep for interpolation
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Multi-octave coverage noise — returns 0..1 where higher = more grass
fn coverage_noise(world_xz: vec2<f32>, cov_scale: f32, cov_amount: f32) -> f32 {
    if (cov_scale < 0.01 || cov_amount < 0.01) {
        return 1.0;
    }
    let p = world_xz / cov_scale;
    // Two octaves: large patches + medium clumping
    var n = value_noise_2d(p) * 0.65;
    n += value_noise_2d(p * 2.73 + vec2<f32>(17.1, 31.7)) * 0.35;
    // Shape: push toward 0 or 1 for patchier look
    n = smoothstep(0.3 * cov_amount, 1.0 - 0.2 * cov_amount, n);
    return n;
}

// Volumetric grass: march ray through grass volume, accumulate density via Beer's law.
// Each blade contributes soft density (smoothstep falloff) instead of binary hit/miss.
// Front-to-back compositing blends grass with terrain for natural anti-aliasing at 1spp.
// Wind shifts cross-section sampling positions at each height → visible blade curvature.
fn trace_grass_volumetric(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    terrain_t: f32,
    surface_y: f32,
    terrain_normal: vec3<f32>,
    profile: GrassProfileGpu,
    combined_density: f32,
    terrain_color: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = terrain_t;
    result.color = terrain_color;
    result.normal = normalize(terrain_normal);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = 0xFFFFFFFFu;

    let spacing = profile.blade_spacing;
    let max_height = profile.height_max;
    let gn = normalize(terrain_normal);
    let terrain_hit = ray_origin + ray_dir * terrain_t;

    // Normal-based grass volume entry: works for ALL viewing angles.
    // Height above surface along normal: h(t) = (t - terrain_t) * dot(ray_dir, gn)
    // At terrain_t: h=0 (ground). We want entry where h = top_height.
    let ndotd = dot(ray_dir, gn);
    let top_height = max_height * 1.3;

    var t_enter: f32;
    if (abs(ndotd) > 0.001) {
        t_enter = terrain_t + top_height / ndotd;
    } else {
        // Ray nearly parallel to surface — generous range
        t_enter = terrain_t - top_height * 3.0 / max(length(ray_dir.xz), 0.01);
    }
    t_enter = max(t_enter, 0.001);
    let t_exit = terrain_t;
    if (t_enter >= t_exit) { return result; }

    // Wind parameters
    let wind_str = grass.wind_speed * 0.4 + 0.2;
    let sway_phase = grass.time * profile.sway_frequency;

    // March through grass volume with Beer's law accumulation
    let num_steps = 80u;
    let total_dist = t_exit - t_enter;
    let step_len = total_dist / f32(num_steps);

    var accum_color = vec3<f32>(0.0);
    var accum_opacity: f32 = 0.0;
    var first_grass_t = terrain_t;
    var first_grass_normal = gn;
    var found_first = false;

    for (var step = 0u; step < num_steps; step++) {
        let t = t_enter + (f32(step) + 0.5) * step_len;
        let pos = ray_origin + ray_dir * t;

        // Height above terrain surface along normal (works for all view angles)
        let height = (t - terrain_t) * ndotd;

        if (height < 0.0 || height > max_height) { continue; }

        let hfrac = height / max_height;

        // Wind/lean: gentle swoop — cubic curve for smooth arc, capped lean angle.
        // Starts bending at 30% height, cubic gives gradual onset with smooth curvature.
        let lean_t = saturate((hfrac - 0.3) / 0.7);
        let lean_amount = lean_t * lean_t * lean_t;  // cubic — gentler than quadratic
        // Cap max displacement to ~55% of blade height (~30° lean at tip)
        let lean_strength = min(wind_str, 0.8) * max_height;
        let lean_dx = grass.wind_direction.x * lean_strength * lean_amount;
        let lean_dz = grass.wind_direction.z * lean_strength * lean_amount;
        let sway_dx = sin(sway_phase + pos.x * 3.0) * lean_amount * profile.sway_amount * 0.06;
        let sway_dz = cos(sway_phase * 0.7 + pos.z * 3.0) * lean_amount * profile.sway_amount * 0.06;

        // Arc-length stretching: derivative of lean curve compensates density
        let dlean_dhfrac = 3.0 * lean_t * lean_t / 0.7;  // d(lean_amount)/d(hfrac)
        let curve_scale = min(wind_str, 0.8);
        let curve_dx = grass.wind_direction.x * curve_scale * dlean_dhfrac;
        let curve_dz = grass.wind_direction.z * curve_scale * dlean_dhfrac;
        let arc_stretch = sqrt(1.0 + curve_dx * curve_dx + curve_dz * curve_dz);

        // "Undo" displacement to find source cell
        let sample_xz = pos.xz - vec2<f32>(lean_dx + sway_dx, lean_dz + sway_dz);

        // Accumulate soft density from nearby blade cells (2x2 neighborhood)
        let cell_f = sample_xz / spacing - 0.5;
        let cell_base = vec2<i32>(floor(cell_f));

        var total_density: f32 = 0.0;
        var weighted_color = vec3<f32>(0.0);
        var weighted_normal = vec3<f32>(0.0);
        var total_weight: f32 = 0.0;

        for (var cdx = 0; cdx <= 1; cdx++) {
            for (var cdz = 0; cdz <= 1; cdz++) {
                let c = cell_base + vec2<i32>(cdx, cdz);
                let ch = hash21(vec2<f32>(f32(c.x) * 0.7131 + 0.1, f32(c.y) * 0.5813 + 0.3));
                if (ch > combined_density) { continue; }

                let h2 = hash21(vec2<f32>(f32(c.x) * 1.331, f32(c.y) * 2.717));
                let h3 = hash21(vec2<f32>(f32(c.x) * 3.141, f32(c.y) * 1.618));
                let blade_h = mix(profile.height_min, profile.height_max, h2);
                if (height > blade_h) { continue; }

                let local_hf = height / blade_h;

                // Blade center (jittered within cell)
                let hex_off = select(0.0, 0.5, (c.y & 1) != 0);
                let jx = (h2 - 0.5) * spacing * 0.8;
                let jz = (h3 - 0.5) * spacing * 0.8;
                let blade_center = vec2<f32>(
                    (f32(c.x) + 0.5 + hex_off) * spacing + jx,
                    (f32(c.y) + 0.5) * spacing + jz
                );

                let d = length(sample_xz - blade_center);

                // Blade radius tapers with height (15% at tip).
                // In the bend region, widen slightly to compensate for
                // sparser hits along the curved path.
                let base_radius = profile.width * 0.5;
                let blade_radius = base_radius * (1.0 - local_hf * 0.85) * arc_stretch;

                let contribution = 1.0 - smoothstep(blade_radius * 0.8, blade_radius, d);
                if (contribution < 0.01) { continue; }

                // Thin-tip opacity boost: thinner blade → denser per hit
                // so thin tips still accumulate visible opacity across steps
                let thin_boost = base_radius / max(blade_radius, 0.001);
                total_density += contribution * min(thin_boost, 6.0) * arc_stretch;

                // Per-blade color: wide hue variation
                let wr = hash21(vec2<f32>(f32(c.x) * 5.331, f32(c.y) * 7.717));
                let br = hash21(vec2<f32>(f32(c.x) * 3.917, f32(c.y) * 11.213));
                let yr = hash21(vec2<f32>(f32(c.x) * 7.113, f32(c.y) * 2.937));

                // Large-scale patches + per-blade variation
                let world_patch = value_noise_2d(sample_xz * 0.4);
                let world_hue = value_noise_2d(sample_xz * 0.12 + vec2<f32>(50.0, 50.0));
                let world_stress = value_noise_2d(sample_xz * 0.25 + vec2<f32>(100.0, 200.0));

                let base_col = profile.color_base;
                var regional_color: vec3<f32>;
                if (world_hue < 0.35) {
                    regional_color = vec3<f32>(base_col.r * 0.6, base_col.g * 1.15, base_col.b * 1.4);
                } else if (world_hue < 0.65) {
                    regional_color = base_col;
                } else {
                    regional_color = vec3<f32>(base_col.r + 0.18, base_col.g * 0.85, base_col.b * 0.3);
                }
                regional_color *= 0.6 + world_patch * 0.8;

                if (world_stress > 0.75) {
                    let stress_blend = smoothstep(0.75, 0.90, world_stress);
                    regional_color = mix(regional_color,
                        vec3<f32>(0.35, 0.38, 0.12), stress_blend * 0.5);
                }

                var blade_color = regional_color + vec3<f32>(
                    (wr - 0.5) * 0.10,
                    (br - 0.5) * 0.08,
                    (yr - 0.5) * 0.05
                );

                // Individual dried/yellow blades scattered through (15%)
                if (yr > 0.85) {
                    blade_color = mix(blade_color, vec3<f32>(0.42, 0.40, 0.10), 0.7);
                }

                // Self-shadowing AO: dark at base
                let ao = smoothstep(0.0, 0.7, local_hf);
                let understory_col = vec3<f32>(0.04, 0.06, 0.02);
                blade_color = mix(understory_col, blade_color, ao * 0.85 + 0.15);

                weighted_color += clamp(blade_color, vec3<f32>(0.0), vec3<f32>(1.0)) * contribution;
                total_weight += contribution;

                // Normal: lean direction for shading
                let lean_dir = normalize(vec3<f32>(
                    lean_dx + sway_dx,
                    1.0 / max(local_hf + 0.1, 0.2),
                    lean_dz + sway_dz
                ));
                weighted_normal += lean_dir * contribution;
            }
        }

        if (total_density < 0.01) { continue; }

        // Beer's law: opacity for this step
        let density_scale = 8.0;
        let step_opacity = 1.0 - exp(-total_density * step_len * density_scale);

        // Weighted average color and normal
        let blade_color = weighted_color / max(total_weight, 0.001);
        let lean_dir = normalize(weighted_normal / max(total_weight, 0.001));

        // Front-to-back compositing
        accum_color += blade_color * step_opacity * (1.0 - accum_opacity);
        accum_opacity += step_opacity * (1.0 - accum_opacity);

        if (!found_first && accum_opacity > 0.05) {
            first_grass_t = t;
            first_grass_normal = lean_dir;
            found_first = true;
        }

        if (accum_opacity > 0.95) { break; }
    }

    // Blend grass with terrain via front-to-back compositing.
    // Terrain beneath grass is in shadow (understory) — darken it to prevent
    // bright terrain bleeding through thin blade tips as a white fringe.
    if (accum_opacity > 0.02) {
        result.hit = true;
        result.t = mix(terrain_t, first_grass_t, min(accum_opacity * 2.0, 1.0));
        let understory = terrain_color * 0.15;  // dark shadow beneath grass canopy
        result.color = accum_color + understory * (1.0 - accum_opacity);
        result.normal = normalize(mix(gn, first_grass_normal, min(accum_opacity * 2.0, 1.0)));
        result.material_id = 16u;
        result.roughness = 0.85;
        result.translucency = 0.3 * accum_opacity;
    }

    return result;
}

// AMD GPUOpen procedural grass — Bezier ribbon with normal-aligned growth.
// Reference: gpuopen.com/learn/mesh_shaders/mesh_shaders-procedural_grass_rendering
// Uses plane intersection + smoothstep coverage for analytical anti-aliasing.
// Each blade segment defines a plane (segment_dir × width_dir). Ray-plane intersection
// gives exact position on blade; distance from centerline → soft coverage via smoothstep.
// Coverage blends grass color with terrain_color for smooth edges at any resolution.
fn test_grass_blade_curved(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    cell_x: i32,
    cell_z: i32,
    surface_y: f32,
    max_t: f32,
    profile: GrassProfileGpu,
    slope_density: f32,
    gn: vec3<f32>,
    surface_tangent: vec3<f32>,
    surface_bitangent: vec3<f32>,
    terrain_color: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = max_t;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = 0xFFFFFFFFu;

    let spacing = profile.blade_spacing;
    let cell_hash = hash21(vec2<f32>(f32(cell_x) * 0.7131 + 0.1, f32(cell_z) * 0.5813 + 0.3));

    // Distance-based density fade
    let cell_center_xz = vec2<f32>((f32(cell_x) + 0.5) * spacing, (f32(cell_z) + 0.5) * spacing);
    let dist = length(vec3<f32>(cell_center_xz.x, surface_y, cell_center_xz.y) - ray_origin);

    var density = profile.density * slope_density;
    if (dist > grass.fade_start) {
        density *= 1.0 - saturate((dist - grass.fade_start) / (grass.max_distance - grass.fade_start));
    }
    let cov_noise = coverage_noise(cell_center_xz, profile.coverage_scale, profile.coverage_amount);
    density *= mix(1.0, cov_noise, profile.coverage_amount);
    if (cell_hash > density) { return result; }

    var dist_fade = 1.0;
    if (dist > grass.fade_start) {
        dist_fade = 1.0 - saturate((dist - grass.fade_start) / (grass.max_distance - grass.fade_start));
    }

    // LOD: widen blades at distance
    let lod_scale = 1.0 + clamp(dist / 8.0, 0.0, 3.0);

    let wind_dir_xz = normalize(vec2<f32>(grass.wind_direction.x + 0.001, grass.wind_direction.z + 0.001));

    let hex_offset = select(0.0, 0.5, (cell_z & 1) != 0);
    let cx = f32(cell_x);
    let cz = f32(cell_z);

    // Track best coverage across all blades in this cell
    var best_coverage = 0.0;
    var best_color = vec3<f32>(0.0);
    var best_normal = gn;
    var best_t = max_t;

    for (var blade_idx = 0u; blade_idx < 3u; blade_idx++) {
        let bi = f32(blade_idx);

        let h2 = hash21(vec2<f32>(cx * 1.331 + bi * 13.37, cz * 2.717 + bi * 7.89));
        let h3 = hash21(vec2<f32>(cx * 3.141 + bi * 5.43, cz * 1.618 + bi * 11.17));

        let blade_exist = hash21(vec2<f32>(cx * 0.913 + bi * 3.71, cz * 0.337 + bi * 9.13));
        if (blade_exist > density) { continue; }

        let blade_height = mix(profile.height_min, profile.height_max, h2);
        let arc_length = blade_height * dist_fade;
        if (arc_length < 0.005) { continue; }

        // Scatter within cell
        let jitter_x = (h2 - 0.5) * spacing * 0.9;
        let jitter_z = (h3 - 0.5) * spacing * 0.9;
        let base_x = (cx + 0.5 + hex_offset) * spacing + jitter_x;
        let base_z = (cz + 0.5) * spacing + jitter_z;

        // Width direction: fixed random angle in tangent plane
        let dir_a = h3 * 2.0 - 1.0;
        let dir_b = fract(h3 * 7.919) * 2.0 - 1.0;
        let width_dir = normalize(dir_a * surface_tangent + dir_b * surface_bitangent);

        // Lean direction
        let lean_hash = hash21(vec2<f32>(cx * 7.913 + bi * 2.31, cz * 4.271 + bi * 6.47));
        let rand_angle = lean_hash * 6.283185;
        let rand_dir = vec2<f32>(cos(rand_angle), sin(rand_angle));
        let lean_dir_2d = normalize(mix(wind_dir_xz, rand_dir, 0.4));
        let lean_dir_3d = lean_dir_2d.x * surface_tangent + lean_dir_2d.y * surface_bitangent;

        // Wind sway
        let wind_phase = grass.time * profile.sway_frequency + cell_hash * 6.283 + bi * 1.57;
        let sway = sin(wind_phase) * profile.sway_amount;
        let wind_scale = arc_length * 0.15;
        let sway_3d = lean_dir_3d * sway * arc_length * 0.08
                    + vec3<f32>(grass.wind_direction.x, 0.0, grass.wind_direction.z) * grass.wind_speed * wind_scale;

        // Bezier control points
        let lean_amount = 0.5 + lean_hash * 0.35;
        let p0 = vec3<f32>(base_x, surface_y, base_z);
        let p1 = p0 + gn * arc_length;
        let p2 = p1 + lean_dir_3d * arc_length * lean_amount + sway_3d;

        // Per-blade color (reduced variation for cohesive look)
        let ch_r = hash21(vec2<f32>(cx * 5.331 + bi * 4.17, cz * 7.717 + bi * 8.31));
        let ch_g = hash21(vec2<f32>(cx * 3.917 + bi * 6.93, cz * 11.213 + bi * 2.57));
        let vr = profile.color_variation;
        let color_var = vec3<f32>(
            (ch_r - 0.5) * vr * 0.3,
            (ch_g - 0.5) * vr * 0.15,
            -(ch_r - 0.5) * vr * 0.15
        );

        // Evaluate 4 centerline points along Bezier
        var center_pts: array<vec3<f32>, 4>;
        for (var i = 0u; i < 4u; i++) {
            let t = f32(i) / 3.0;
            let u = 1.0 - t;
            center_pts[i] = p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t);
        }

        // AMD width taper: 1.0 at base → 0.1 at tip
        let widths = array<f32, 4>(1.0, 0.7, 0.3, 0.1);

        // --- Plane intersection with curved-surface normal ---
        // Flat blade geometry (correct shape) but normal curves across the width
        // to simulate a folded/curved blade cross-section. This makes lighting
        // vary across each blade (3D shading) instead of flat uniform color.
        for (var seg = 0u; seg < 3u; seg++) {
            let A = center_pts[seg];
            let B = center_pts[seg + 1u];
            let seg_dir = B - A;
            let seg_len = length(seg_dir);
            if (seg_len < 0.001) { continue; }
            let seg_unit = seg_dir / seg_len;

            // Blade plane normal
            let plane_n = cross(seg_unit, width_dir);
            let plane_n_len = length(plane_n);
            if (plane_n_len < 0.0001) { continue; }
            let plane_normal = plane_n / plane_n_len;

            let denom = dot(ray_dir, plane_normal);
            if (abs(denom) < 0.00001) { continue; }

            let t_hit = dot(A - ray_origin, plane_normal) / denom;
            if (t_hit < 0.001 || t_hit > max_t) { continue; }

            let hit_pos = ray_origin + ray_dir * t_hit;
            let local = hit_pos - A;

            // Along-blade parameter
            let along = dot(local, seg_unit) / seg_len;
            if (along < -0.05 || along > 1.05) { continue; }
            let along_c = clamp(along, 0.0, 1.0);

            // Across-blade: signed distance from centerline
            let across = dot(local, width_dir);

            // Width taper
            let curve_t = (f32(seg) + along_c) / 3.0;
            let w_base = widths[seg];
            let w_next = widths[seg + 1u];
            let taper = mix(w_base, w_next, along_c);
            let half_w = profile.width * 0.5 * taper * lod_scale;

            let abs_across = abs(across);
            if (abs_across > half_w * 1.5) { continue; }

            // Soft AA at edges
            let edge_coverage = 1.0 - smoothstep(half_w * 0.6, half_w, abs_across);

            if (edge_coverage > best_coverage) {
                best_coverage = edge_coverage;
                best_t = t_hit;

                // Across-blade shading: darken edges, lighten center
                // Simulates a slightly curved/folded blade cross-section
                let cross_shade = 0.75 + 0.25 * (1.0 - abs_across / max(half_w, 0.001));

                // Base-to-tip color gradient
                let self_shadow = mix(0.45, 1.0, curve_t);
                let tip_warmth = vec3<f32>(0.06, 0.03, -0.02) * curve_t;
                best_color = clamp((profile.color_base + color_var + tip_warmth) * self_shadow * cross_shade, vec3<f32>(0.0), vec3<f32>(1.0));

                // Curved-surface normal: plane_normal + tilt toward edges
                // At center: normal = plane_normal (faces camera)
                // At edges: normal tilts toward width_dir (rolls off)
                let cross_frac = across / max(half_w, 0.001); // -1 to +1
                var blade_n = normalize(plane_normal + width_dir * cross_frac * 0.4);

                // Flip to face ray
                let fs = select(1.0, -1.0, dot(blade_n, ray_dir) > 0.0);
                blade_n = blade_n * fs;
                // Blend with ground normal
                best_normal = normalize(mix(blade_n, gn, 0.2));
            }
        }
    } // end blade_idx loop

    // Blend grass with terrain based on coverage (analytical AA)
    if (best_coverage > 0.02) {
        result.hit = true;
        result.t = best_t;
        result.color = mix(terrain_color, best_color, best_coverage);
        result.normal = normalize(mix(gn, best_normal, best_coverage));
        result.material_id = 16u;
        result.roughness = 0.9;
        result.translucency = 0.3;
    }

    return result;
}

// OLD: Billboard-based blade test (kept for reference)
// Per-profile parameters come from `profile`, global params from `grass`.
fn test_grass_blade(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    cell_x: i32,
    cell_z: i32,
    surface_y: f32,
    max_t: f32,
    profile: GrassProfileGpu,
    slope_density: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = max_t;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = 0xFFFFFFFFu;

    let spacing = profile.blade_spacing;

    // Existence hash — decides if this cell has a blade
    let cell_hash = hash21(vec2<f32>(f32(cell_x) * 0.7131 + 0.1, f32(cell_z) * 0.5813 + 0.3));

    // Distance-based density fade
    let cell_center_xz = vec2<f32>((f32(cell_x) + 0.5) * spacing, (f32(cell_z) + 0.5) * spacing);
    let dist = length(vec3<f32>(cell_center_xz.x, surface_y, cell_center_xz.y) - ray_origin);

    var density = profile.density * slope_density;
    if (dist > grass.fade_start) {
        density *= 1.0 - saturate((dist - grass.fade_start) / (grass.max_distance - grass.fade_start));
    }

    // Coverage noise — spatially varying density for natural patches
    let coverage = coverage_noise(cell_center_xz, profile.coverage_scale, profile.coverage_amount);
    density *= mix(1.0, coverage, profile.coverage_amount);

    if (cell_hash > density) {
        return result;
    }

    // Per-blade property hashes
    let h2 = hash21(vec2<f32>(f32(cell_x) * 1.331, f32(cell_z) * 2.717));
    let h3 = hash21(vec2<f32>(f32(cell_x) * 3.141, f32(cell_z) * 1.618));

    // Blade height with distance fade and slope reduction
    let blade_height = mix(profile.height_min, profile.height_max, h2);
    var effective_height = blade_height;
    if (dist > grass.fade_start) {
        effective_height *= 1.0 - saturate((dist - grass.fade_start) / (grass.max_distance - grass.fade_start));
    }
    if (effective_height < 0.005) {
        return result;
    }

    // Position jitter within cell + hex offset (odd rows shifted by half-spacing)
    let hex_offset = select(0.0, 0.5, (cell_z & 1) != 0);
    let jitter_x = (h2 - 0.5) * spacing * 0.8;
    let jitter_z = (h3 - 0.5) * spacing * 0.8;

    let base_x = (f32(cell_x) + 0.5 + hex_offset) * spacing + jitter_x;
    let base_z = (f32(cell_z) + 0.5) * spacing + jitter_z;

    // Random blade orientation (rotation around Y, 0..PI)
    let angle = h3 * 3.14159265;

    // Natural lean — each blade droops in a random direction, scaled by height
    // Taller grass droops much more under its own weight (cubic relationship)
    let lean_hash = hash21(vec2<f32>(f32(cell_x) * 7.913, f32(cell_z) * 4.271));
    let lean_angle = lean_hash * 6.283185;
    let lean_strength = effective_height * effective_height * 0.55;
    let lean_dx = cos(lean_angle) * lean_strength;
    let lean_dz = sin(lean_angle) * lean_strength;

    // Wind displacement — scales with blade height for realistic motion
    let wind_phase = grass.time * profile.sway_frequency + cell_hash * 6.283;
    let sway = sin(wind_phase) * profile.sway_amount;
    let wind_scale = effective_height * 0.2;
    let wind_dx = grass.wind_direction.x * grass.wind_speed * wind_scale
                + sway * effective_height * 0.08;
    let wind_dz = grass.wind_direction.z * grass.wind_speed * wind_scale
                + sway * effective_height * 0.08;

    // Combined tip displacement
    let tip_dx = lean_dx + wind_dx;
    let tip_dz = lean_dz + wind_dz;

    // Droop: blade tip hangs lower when leaning heavily
    // Arc of length L leaning by D has vertical extent ≈ L - D²/(2L)
    let total_lean = length(vec2<f32>(tip_dx, tip_dz));
    let droop = min(total_lean * total_lean / (2.0 * effective_height), effective_height * 0.4);
    effective_height -= droop;

    // Per-blade color variation: independent hashes for warmth + brightness + saturation
    let ch_r = hash21(vec2<f32>(f32(cell_x) * 5.331, f32(cell_z) * 7.717));
    let ch_g = hash21(vec2<f32>(f32(cell_x) * 3.917, f32(cell_z) * 11.213));
    let ch_b = hash21(vec2<f32>(f32(cell_x) * 8.461, f32(cell_z) * 2.593));
    let vr = profile.color_variation;
    // Warmth shift: some blades more yellow-green, others more blue-green
    let warmth = (ch_r - 0.5) * vr;
    // Brightness: overall lighter or darker
    let brightness_var = (ch_g - 0.5) * vr * 0.4;
    let color_var = vec3<f32>(
        warmth * 0.5 + brightness_var,
        brightness_var * 0.3,
        -warmth * 0.3 + (ch_b - 0.5) * vr * 0.2
    );

    let to_base = vec3<f32>(base_x - ray_origin.x, 0.0, base_z - ray_origin.z);

    // Cross-billboard: test two perpendicular quads for consistent density
    for (var q = 0u; q < 2u; q++) {
        let a = angle + f32(q) * 1.5707963; // +90° for second quad
        let nx = cos(a);
        let nz = sin(a);
        let tx = -nz;
        let tz = nx;

        let plane_n = vec3<f32>(nx, 0.0, nz);
        let denom = dot(ray_dir, plane_n);

        if (abs(denom) < 0.0001) { continue; } // ray parallel to this quad

        let t = dot(to_base, plane_n) / denom;
        if (t < 0.0 || t >= result.t) { continue; }

        let hit_pos = ray_origin + ray_dir * t;

        // Check Y bounds
        let local_y = hit_pos.y - surface_y;
        if (local_y < 0.0 || local_y > effective_height) { continue; }

        let height_frac = saturate(local_y / effective_height);

        // Gravity droop: gradual curve starting from low on the blade
        let bend = pow(height_frac, 1.5);
        let bent_x = base_x + tip_dx * bend;
        let bent_z = base_z + tip_dz * bend;

        // Width check along quad tangent from bent centerline
        let offset_x = hit_pos.x - bent_x;
        let offset_z = hit_pos.z - bent_z;
        let local_x = offset_x * tx + offset_z * tz;

        // Triangular taper: full width at base, zero at tip
        let taper = (1.0 - height_frac) * (1.0 - height_frac * 0.3);
        let half_w = profile.width * 0.5 * taper;

        if (abs(local_x) > half_w) { continue; }

        // --- Hit confirmed ---
        result.hit = true;
        result.t = t;

        // Height gradient: darker green at base, lighter/warmer at tips (drier)
        let tip_warmth = vec3<f32>(0.08, 0.02, -0.04) * height_frac;
        let height_brightness = mix(0.75, 1.15, height_frac);
        result.color = clamp((profile.color_base + color_var + tip_warmth) * height_brightness, vec3<f32>(0.0), vec3<f32>(1.0));

        // Normal: face the ray, with lean at tip
        let face_sign = -sign(denom);
        let lean_up = mix(0.4, 0.8, 1.0 - height_frac);
        result.normal = normalize(vec3<f32>(
            nx * face_sign + tip_dx * height_frac * 0.3,
            lean_up,
            nz * face_sign + tip_dz * height_frac * 0.3
        ));

        result.material_id = 16u;
        result.roughness = 0.9;
        result.translucency = 0.3;
    }

    return result;
}

// Compute octant index for a point relative to a center (for mask octree traversal)
fn grass_octant_index(pos: vec3<f32>, center: vec3<f32>) -> u32 {
    var idx = 0u;
    if (pos.x >= center.x) { idx |= 1u; }
    if (pos.y >= center.y) { idx |= 2u; }
    if (pos.z >= center.z) { idx |= 4u; }
    return idx;
}

// Sample the grass mask octree for a chunk to get packed grass cell at a world position.
// Returns packed u32: low 8 bits = profile index, bits 8-15 = density (0-255).
// Returns 0 if no grass mask exists or the position maps to no grass.
fn sample_grass_mask(chunk_idx: u32, world_pos: vec3<f32>) -> u32 {
    let info = grass_mask_info[chunk_idx];
    if (info.node_count == 0u) { return 0u; }

    // Convert world pos to local [0, chunk_size] coordinates
    let chunk = chunk_infos[chunk_idx];
    let local_pos = world_pos - chunk.world_min;
    let chunk_size = chunk.root_size;

    // Traverse octree
    var node_idx = info.node_offset;  // root node
    var center = vec3<f32>(chunk_size * 0.5);
    var half_size = chunk_size * 0.25;

    for (var depth = 0u; depth < info.max_depth; depth++) {
        let node = grass_mask_nodes[node_idx];
        let child_mask = node.masks & 0xFFu;
        let leaf_mask = (node.masks >> 8u) & 0xFFu;
        let octant = grass_octant_index(local_pos, center);
        let bit = 1u << octant;

        if ((child_mask & bit) == 0u) {
            // Child doesn't exist — use LOD value from this node
            if (node.lod_value_idx != 0xFFFFFFFFu) {
                return grass_mask_values[info.value_offset + node.lod_value_idx];
            }
            return 0u;
        }

        if ((leaf_mask & bit) != 0u) {
            // Leaf child — read value directly
            let leaf_rank = countOneBits(leaf_mask & (bit - 1u));
            return grass_mask_values[info.value_offset + node.value_offset + leaf_rank];
        }

        // Internal child — descend
        let internal_mask = child_mask & ~leaf_mask;
        let internal_rank = countOneBits(internal_mask & (bit - 1u));
        node_idx = info.node_offset + node.child_offset + internal_rank;

        // Update center for next level
        if (local_pos.x >= center.x) { center.x += half_size; } else { center.x -= half_size; }
        if (local_pos.y >= center.y) { center.y += half_size; } else { center.y -= half_size; }
        if (local_pos.z >= center.z) { center.z += half_size; } else { center.z -= half_size; }
        half_size *= 0.5;
    }

    // Reached max depth — use LOD value of final node
    let final_node = grass_mask_nodes[node_idx];
    if (final_node.lod_value_idx != 0xFFFFFFFFu) {
        return grass_mask_values[info.value_offset + final_node.lod_value_idx];
    }
    return 0u;
}


// Look up chunk index from the 3D grid. Returns 0xFFFFFFFF if empty or out of bounds.
fn grid_lookup(cx: i32, cy: i32, cz: i32) -> LayerDescriptor {
    let gx = cx - params.grid_min_x;
    let gy = cy - params.grid_min_y;
    let gz = cz - params.grid_min_z;
    if (gx < 0 || gy < 0 || gz < 0 ||
        u32(gx) >= params.grid_size_x ||
        u32(gy) >= params.grid_size_y ||
        u32(gz) >= params.grid_size_z) {
        var desc: LayerDescriptor;
        desc.base_index = 0u;
        desc.layer_count = 0u;
        return desc;
    }
    let idx = u32(gx) + u32(gy) * params.grid_size_x + u32(gz) * params.grid_size_x * params.grid_size_y;
    return chunk_grid[idx];
}

// Get chunk index at a specific layer offset within a grid cell
fn get_layer_chunk_index(desc: LayerDescriptor, layer_offset: u32) -> u32 {
    if (layer_offset >= desc.layer_count) {
        return 0xFFFFFFFFu;
    }
    return layer_data[desc.base_index + layer_offset];
}

// Trace ray through chunk grid using DDA (Digital Differential Analyzer).
// Instead of testing all N chunks per pixel, marches through grid cells front-to-back.
fn trace_all_chunks(ray_origin: vec3<f32>, ray_dir: vec3<f32>, pixel_hash: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = camera.far;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.material_id = 0u;
    result.roughness = 0.5;
    result.translucency = 0.0;
    result.chunk_idx = 0xFFFFFFFFu;

    let ray_inv_dir = 1.0 / ray_dir;
    let cs = params.chunk_size;

    // Grid world-space bounds
    let grid_world_min = vec3<f32>(
        f32(params.grid_min_x) * cs,
        f32(params.grid_min_y) * cs,
        f32(params.grid_min_z) * cs
    );
    let grid_world_max = vec3<f32>(
        f32(i32(params.grid_size_x) + params.grid_min_x) * cs,
        f32(i32(params.grid_size_y) + params.grid_min_y) * cs,
        f32(i32(params.grid_size_z) + params.grid_min_z) * cs
    );

    // Clip ray to grid bounds
    let t_bounds = ray_aabb_intersect(ray_origin, ray_inv_dir, grid_world_min, grid_world_max);
    let t_entry = max(t_bounds.x, 0.0);
    let t_exit = t_bounds.y;

    if (t_entry >= t_exit) {
        return result; // Ray misses grid entirely
    }

    // Entry point into the grid
    let entry_pos = ray_origin + ray_dir * (t_entry + 0.001);
    var cell = vec3<i32>(floor(entry_pos / cs));

    // Clamp to grid bounds (handles floating point edge cases)
    cell = clamp(cell,
        vec3<i32>(params.grid_min_x, params.grid_min_y, params.grid_min_z),
        vec3<i32>(params.grid_min_x + i32(params.grid_size_x) - 1,
                  params.grid_min_y + i32(params.grid_size_y) - 1,
                  params.grid_min_z + i32(params.grid_size_z) - 1));

    // DDA setup
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0)
    );
    let delta_t = abs(vec3<f32>(cs) / ray_dir);

    // Distance to next cell boundary along each axis
    var next_t: vec3<f32>;
    if (ray_dir.x >= 0.0) {
        next_t.x = (f32(cell.x + 1) * cs - ray_origin.x) / ray_dir.x;
    } else {
        next_t.x = (f32(cell.x) * cs - ray_origin.x) / ray_dir.x;
    }
    if (ray_dir.y >= 0.0) {
        next_t.y = (f32(cell.y + 1) * cs - ray_origin.y) / ray_dir.y;
    } else {
        next_t.y = (f32(cell.y) * cs - ray_origin.y) / ray_dir.y;
    }
    if (ray_dir.z >= 0.0) {
        next_t.z = (f32(cell.z + 1) * cs - ray_origin.z) / ray_dir.z;
    } else {
        next_t.z = (f32(cell.z) * cs - ray_origin.z) / ray_dir.z;
    }

    // March through grid cells (max iterations = grid diagonal)
    let max_steps = params.grid_size_x + params.grid_size_y + params.grid_size_z;
    for (var iter = 0u; iter < max_steps; iter++) {
        // Look up layer descriptor at this cell
        let layer_desc = grid_lookup(cell.x, cell.y, cell.z);

        // Check all layers at this cell
        // If multiple layers have hits, prefer higher layer_id (rocks over terrain)
        if (layer_desc.layer_count > 0u) {
            var best_hit_in_cell: HitResult;
            best_hit_in_cell.hit = false;
            best_hit_in_cell.t = camera.far;
            var best_layer_id: u32 = 0u;

            for (var layer_idx = 0u; layer_idx < layer_desc.layer_count; layer_idx++) {
                let chunk_idx = get_layer_chunk_index(layer_desc, layer_idx);
                if (chunk_idx != 0xFFFFFFFFu) {
                    // Get layer_id from chunk info
                    let chunk_info = chunk_infos[chunk_idx];
                    let this_layer_id = chunk_info.layer_id;

                    let chunk_result = trace_chunk_octree(
                        ray_origin, ray_dir, ray_inv_dir, chunk_idx, pixel_hash, result.t
                    );
                    if (chunk_result.hit) {
                        // If this is closer than current best, or same distance but higher layer
                        let is_closer = chunk_result.t < best_hit_in_cell.t;
                        let is_same_dist = abs(chunk_result.t - best_hit_in_cell.t) < 0.001;
                        let is_higher_layer = this_layer_id > best_layer_id;

                        if (!best_hit_in_cell.hit || is_closer || (is_same_dist && is_higher_layer)) {
                            best_hit_in_cell = chunk_result;
                            best_layer_id = this_layer_id;
                        }
                    }
                }
            }

            // If we found a hit in this cell that's better than global best, update
            if (best_hit_in_cell.hit && best_hit_in_cell.t < result.t) {
                result = best_hit_in_cell;
            }
        }

        // Step to the next cell along the axis with smallest next_t
        if (next_t.x < next_t.y && next_t.x < next_t.z) {
            if (next_t.x > result.t) { break; }
            cell.x += step.x;
            next_t.x += delta_t.x;
        } else if (next_t.y < next_t.z) {
            if (next_t.y > result.t) { break; }
            cell.y += step.y;
            next_t.y += delta_t.y;
        } else {
            if (next_t.z > result.t) { break; }
            cell.z += step.z;
            next_t.z += delta_t.z;
        }

        // Check if we've left the grid
        if (cell.x < params.grid_min_x || cell.x >= params.grid_min_x + i32(params.grid_size_x) ||
            cell.y < params.grid_min_y || cell.y >= params.grid_min_y + i32(params.grid_size_y) ||
            cell.z < params.grid_min_z || cell.z >= params.grid_min_z + i32(params.grid_size_z)) {
            break;
        }
    }

    return result;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));

    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }

    // Convert pixel to NDC (-1 to 1)
    let uv = (vec2<f32>(global_id.xy) + 0.5) / dims;
    let ndc = uv * 2.0 - 1.0;
    let ndc_flipped = vec2<f32>(ndc.x, -ndc.y);

    // Unproject to get ray direction
    let near_point = camera.view_proj_inv * vec4<f32>(ndc_flipped, 0.0, 1.0);
    let far_point = camera.view_proj_inv * vec4<f32>(ndc_flipped, 1.0, 1.0);

    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;

    // Ray in world space (chunks have world-space positions, no offset needed)
    let ray_origin = camera.position - camera.world_offset;
    let ray_dir = normalize(far_world - near_world);

    // Pre-compute pixel hash for dither-based crossfade
    let pixel_hash = fract(sin(dot(vec2<f32>(f32(global_id.x), f32(global_id.y)), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    var hit = trace_all_chunks(ray_origin, ray_dir, pixel_hash);

    // Procedural grass overlay via mask octree lookup with soft edges
    // Extended range: volumetric grass up to max_distance, ground tint up to 2x for smooth horizon
    let grass_tint_range = grass.max_distance * 2.0;
    if (hit.hit && grass.enabled != 0u && hit.t <= grass_tint_range) {
        let world_hit = ray_origin + ray_dir * hit.t;
        if (hit.chunk_idx != 0xFFFFFFFFu) {
            let grass_cell = sample_grass_mask(hit.chunk_idx, world_hit);
            let profile_idx = grass_cell & 0xFFu;
            let mask_density = f32((grass_cell >> 8u) & 0xFFu) / 255.0;

            if (profile_idx > 0u && profile_idx < grass.profile_count && mask_density > 0.01) {
                let slope_factor = smoothstep(0.15, 0.5, abs(hit.normal.y));
                let profile = grass_profiles[profile_idx];

                // Distance fade: smooth falloff between fade_start and max_distance
                let dist_fade = 1.0 - smoothstep(grass.fade_start, grass.max_distance, hit.t);
                let combined_density = slope_factor * mask_density * dist_fade;

                // Volumetric grass rendering (within fade range)
                if (combined_density > 0.01) {
                    let terrain_color = hit.color;
                    let surface_y = world_hit.y;

                    let vol_result = trace_grass_volumetric(
                        ray_origin, ray_dir, hit.t,
                        surface_y, hit.normal,
                        profile, combined_density,
                        terrain_color);
                    if (vol_result.hit) {
                        hit = vol_result;
                    }
                }

                // Ground tint: only for terrain pixels (not volumetric grass).
                // Tints bare terrain in grassy areas for smooth distance transition.
                if (hit.material_id != 16u) {
                    let tint_fade = 1.0 - smoothstep(grass.fade_start, grass_tint_range, hit.t);
                    let tint_strength = slope_factor * mask_density * tint_fade * 0.4;
                    if (tint_strength > 0.01) {
                        hit.color = mix(hit.color, profile.color_base * 0.75, tint_strength);
                    }
                }
            }
        }
    }

    if (hit.hit) {
        textureStore(output_albedo, pixel, vec4<f32>(hit.color, 1.0));
        textureStore(output_normal, pixel, vec4<f32>(hit.normal, hit.roughness));
        textureStore(output_depth, pixel, vec4<f32>(hit.t, 0.0, 0.0, 0.0));

        // Get material properties to extract metallic
        let mat_props = get_material_properties(hit.material_id);
        let metallic = mat_props.y;

        // Store material properties: metallic, roughness, translucency, material_id
        textureStore(output_material, pixel, vec4<f32>(
            metallic,
            hit.roughness,
            hit.translucency,
            f32(hit.material_id) / 255.0
        ));

        textureStore(output_motion, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    } else {
        let t = ray_dir.y * 0.5 + 0.5;
        let sky = mix(vec3<f32>(0.3, 0.4, 0.5), vec3<f32>(0.5, 0.7, 1.0), t);

        textureStore(output_albedo, pixel, vec4<f32>(sky, 1.0));
        textureStore(output_normal, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(output_depth, pixel, vec4<f32>(camera.far, 0.0, 0.0, 0.0));
        textureStore(output_material, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(output_motion, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
