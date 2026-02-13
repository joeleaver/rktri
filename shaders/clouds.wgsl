// Dynamic Cloud Rendering - Compute Shader
// Raymarches through a cloud slab using 3D noise for density

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _pos_pad: f32,
    near: f32,
    far: f32,
    _pad2: vec2<f32>,
    world_offset: vec3<f32>,
    _pad3: f32,
}

struct CloudParams {
    cloud_altitude: f32,
    cloud_thickness: f32,
    cloud_coverage: f32,
    cloud_density: f32,

    wind_offset: vec3<f32>,
    noise_scale: f32,

    cloud_color: vec3<f32>,
    detail_scale: f32,
    shadow_color: vec3<f32>,
    edge_sharpness: f32,

    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    time_of_day: f32,

    width: u32,
    height: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> params: CloudParams;
@group(1) @binding(0) var t_depth: texture_2d<f32>;
@group(2) @binding(0) var t_output: texture_storage_2d<rgba16float, write>;

// 3D hash function for noise
fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123);
}

// 3D value noise (voxel-friendly)
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Hermite interpolation
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = dot(hash33(i + vec3<f32>(0.0, 0.0, 0.0)), vec3<f32>(1.0)) / 3.0;
    let n100 = dot(hash33(i + vec3<f32>(1.0, 0.0, 0.0)), vec3<f32>(1.0)) / 3.0;
    let n010 = dot(hash33(i + vec3<f32>(0.0, 1.0, 0.0)), vec3<f32>(1.0)) / 3.0;
    let n110 = dot(hash33(i + vec3<f32>(1.0, 1.0, 0.0)), vec3<f32>(1.0)) / 3.0;
    let n001 = dot(hash33(i + vec3<f32>(0.0, 0.0, 1.0)), vec3<f32>(1.0)) / 3.0;
    let n101 = dot(hash33(i + vec3<f32>(1.0, 0.0, 1.0)), vec3<f32>(1.0)) / 3.0;
    let n011 = dot(hash33(i + vec3<f32>(0.0, 1.0, 1.0)), vec3<f32>(1.0)) / 3.0;
    let n111 = dot(hash33(i + vec3<f32>(1.0, 1.0, 1.0)), vec3<f32>(1.0)) / 3.0;

    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}

// FBM (Fractal Brownian Motion) - multiple noise octaves
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Cloud density at a world position
fn cloud_density(world_pos: vec3<f32>) -> f32 {
    // Offset by wind for movement
    let p = (world_pos + params.wind_offset) * params.noise_scale;

    // Base shape: low-frequency FBM
    let base = fbm(p, 4);

    // Detail: higher-frequency noise for wispy edges
    let detail_p = (world_pos + params.wind_offset) * params.detail_scale;
    let detail = fbm(detail_p, 2);

    // Combine: base shape with detail erosion
    var density = base - (1.0 - params.cloud_coverage);
    density -= detail * 0.3 * (1.0 - params.cloud_coverage);

    // Height gradient within cloud layer (denser in middle, wispy at edges)
    let height_in_layer = (world_pos.y - params.cloud_altitude) / params.cloud_thickness;
    let height_gradient = smoothstep(0.0, 0.2, height_in_layer) * smoothstep(1.0, 0.7, height_in_layer);
    density *= height_gradient;

    // Edge sharpness: higher values = blockier (more voxel-like)
    density = clamp(density * params.edge_sharpness, 0.0, 1.0);

    return density;
}

// Simple cloud lighting (toward sun = brighter)
fn cloud_lighting(world_pos: vec3<f32>, density: f32) -> vec3<f32> {
    // Sample density toward sun for self-shadowing
    let light_step = params.sun_direction * params.cloud_thickness * 0.3;
    let shadow_density = cloud_density(world_pos + light_step);

    // Beer's law absorption
    let shadow = exp(-shadow_density * params.cloud_density * 2.0);

    // Mix between shadow color (bottom) and lit color (top/sun-facing)
    let lit_color = params.cloud_color * params.sun_color * params.sun_intensity;
    let dark_color = params.shadow_color;

    return mix(dark_color, lit_color, shadow * 0.7 + 0.3);
}

// Ray-slab intersection (horizontal slab between altitude and altitude+thickness)
fn ray_slab_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    // Avoid division by zero for horizontal rays
    if (abs(ray_dir.y) < 0.0001) {
        // Check if ray is inside the slab
        if (ray_origin.y >= params.cloud_altitude && ray_origin.y <= params.cloud_altitude + params.cloud_thickness) {
            return vec2<f32>(0.0, 1000.0); // Arbitrary large range
        }
        return vec2<f32>(1.0, 0.0); // No intersection (t_near > t_far)
    }

    let inv_dir_y = 1.0 / ray_dir.y;
    let t0 = (params.cloud_altitude - ray_origin.y) * inv_dir_y;
    let t1 = (params.cloud_altitude + params.cloud_thickness - ray_origin.y) * inv_dir_y;

    let t_near = min(t0, t1);
    let t_far = max(t0, t1);

    return vec2<f32>(max(t_near, 0.0), t_far);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }

    let coords = vec2<i32>(global_id.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let uv = (vec2<f32>(global_id.xy) + 0.5) / dims;

    // Reconstruct ray direction
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let near_point = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_point = camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray_dir = normalize(far_point.xyz / far_point.w - near_point.xyz / near_point.w);
    let ray_origin = camera.view_pos;

    // Check depth buffer - don't render clouds behind terrain
    // Depth buffer is at render resolution, clouds at half-res - need to scale
    let depth_dims = textureDimensions(t_depth);
    let depth_coords = vec2<i32>(vec2<f32>(coords) * vec2<f32>(depth_dims) / dims);
    let scene_depth = textureLoad(t_depth, depth_coords, 0).r;

    // Intersect ray with cloud slab
    let slab_hit = ray_slab_intersect(ray_origin, ray_dir);

    // No intersection or cloud layer behind camera
    if (slab_hit.x >= slab_hit.y || slab_hit.y <= 0.0) {
        textureStore(t_output, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Clamp march to not go past scene geometry
    let max_t = min(slab_hit.y, scene_depth);
    if (slab_hit.x >= max_t) {
        textureStore(t_output, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Raymarch through cloud slab
    let num_steps = 24u;
    let step_size = (max_t - slab_hit.x) / f32(num_steps);

    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var t = slab_hit.x;

    for (var i = 0u; i < num_steps; i++) {
        if (accumulated_alpha > 0.95) {
            break; // Early out when nearly opaque
        }

        let sample_pos = ray_origin + ray_dir * t;
        let density = cloud_density(sample_pos);

        if (density > 0.01) {
            // Beer's law absorption for this step
            let step_alpha = 1.0 - exp(-density * params.cloud_density * step_size);
            let step_color = cloud_lighting(sample_pos, density);

            // Front-to-back compositing
            accumulated_color += step_color * step_alpha * (1.0 - accumulated_alpha);
            accumulated_alpha += step_alpha * (1.0 - accumulated_alpha);
        }

        t += step_size;
    }

    textureStore(t_output, coords, vec4<f32>(accumulated_color, accumulated_alpha));
}
