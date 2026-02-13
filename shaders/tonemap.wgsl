// Tone mapping and color grading post-process
// Reads HDR lit texture, outputs LDR with ACES filmic tone mapping

struct TonemapParams {
    width: u32,
    height: u32,
    exposure: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: TonemapParams;
@group(1) @binding(0) var t_input: texture_2d<f32>;
@group(2) @binding(0) var t_output: texture_storage_2d<rgba8unorm, write>;

// ACES filmic tone mapping approximation (Krzysztof Narkowicz)
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Linear to sRGB gamma correction (kept for potential future use)
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let higher = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    let lower = color * 12.92;
    return select(higher, lower, color <= cutoff);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }

    let coords = vec2<i32>(global_id.xy);

    // Sample HDR input
    var color = textureLoad(t_input, coords, 0).rgb;

    // Apply exposure
    color = color * params.exposure;

    // ACES filmic tone mapping
    color = aces_tonemap(color);

    // Do NOT apply sRGB gamma here — the sRGB surface format handles it.
    // The chain is:
    // 1. Tone map writes linear ACES output to post_texture (Rgba8Unorm)
    // 2. Display shader samples post_texture -> linear value
    // 3. Hardware writes to sRGB surface -> applies gamma curve

    textureStore(t_output, coords, vec4<f32>(color, 1.0));
}
