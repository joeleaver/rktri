/// Normalizes linear depth values to [0,1] range for DLSS input.
/// Our SVO trace outputs linear distance (hit.t) in [0, far_plane].
/// DLSS expects normalized depth in [0,1] where 0=near, 1=far.

@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_depth: texture_storage_2d<r32float, write>;

const FAR_PLANE: f32 = 1000.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_depth);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let depth = textureLoad(input_depth, vec2<i32>(id.xy), 0).r;
    let normalized = clamp(depth / FAR_PLANE, 0.0, 1.0);
    textureStore(output_depth, vec2<i32>(id.xy), vec4<f32>(normalized, 0.0, 0.0, 1.0));
}
