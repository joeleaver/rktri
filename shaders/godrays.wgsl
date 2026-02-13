// God Rays - Screen-Space Volumetric Light Scattering
// Post-process that ray-marches through the depth buffer toward the sun's
// screen position. Where depth reads sky (no geometry), light passes through
// and accumulates — creating light shafts through tree canopies and clouds.

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

struct GodRaysParams {
    sun_screen_pos: vec2<f32>,
    num_samples: u32,
    density: f32,
    decay: f32,
    exposure: f32,
    weight: f32,
    width: u32,
    height: u32,
    _pad: u32,
}

// Group 0: Camera + God rays params
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> params: GodRaysParams;

// Group 1: Input textures
@group(1) @binding(0) var t_shadow: texture_2d<f32>;
@group(1) @binding(1) var t_depth: texture_2d<f32>;

// Group 2: Output god rays texture
@group(2) @binding(0) var t_output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }

    let coords = vec2<i32>(global_id.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let uv = (vec2<f32>(global_id.xy) + 0.5) / dims;

    // Radial distance from sun (in UV space, aspect-corrected)
    let aspect = dims.x / dims.y;
    let diff = uv - params.sun_screen_pos;
    let corrected_diff = vec2<f32>(diff.x * aspect, diff.y);
    let dist_from_sun = length(corrected_diff);

    // Radial falloff — rays are strongest near the sun, fading outward
    // Fade from full at sun to zero at ~0.8 screen-widths away
    let radial_falloff = 1.0 - smoothstep(0.15, 0.8, dist_from_sun);

    // Skip pixels far from the sun (optimization + visual correctness)
    if (radial_falloff <= 0.001) {
        textureStore(t_output, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Delta UV from pixel toward sun position
    let delta_uv = (uv - params.sun_screen_pos) / f32(params.num_samples);

    // Start marching from this pixel toward the sun
    var sample_uv = uv;
    var illumination = 0.0;
    var current_weight = params.weight;

    for (var i = 0u; i < params.num_samples; i++) {
        // Step toward the sun
        sample_uv -= delta_uv;

        // Clamp to screen bounds
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_coords = vec2<i32>(clamped_uv * dims);

        // Sample depth buffer for occlusion (sky = far plane = light passes through)
        let depth = textureLoad(t_depth, sample_coords, 0).r;
        // Soft occlusion - gradual transition near far plane reduces temporal flicker
        let threshold = camera.far * 0.95;
        let occlusion = smoothstep(threshold * 0.85, threshold, depth);

        // Accumulate light where sky is visible (light passes between occluders)
        illumination += occlusion * current_weight;

        // Decay the weight for each step (farther samples contribute less)
        current_weight *= params.decay;
    }

    // Apply density, exposure, and radial falloff
    illumination = illumination * params.density * params.exposure * radial_falloff;

    // Clamp to reasonable range
    illumination = clamp(illumination, 0.0, 1.0);

    textureStore(t_output, coords, vec4<f32>(illumination, 0.0, 0.0, 0.0));
}
