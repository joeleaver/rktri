// Fullscreen triangle display shader
// Renders a texture to the screen using a single fullscreen triangle

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle - no vertex buffer needed
// Vertices: (-1,-1), (3,-1), (-1,3) cover entire screen
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle vertices
    let x = f32((vertex_index & 1u) << 2u) - 1.0;  // -1, 3, -1
    let y = f32((vertex_index & 2u) << 1u) - 1.0;  // -1, -1, 3

    out.position = vec4<f32>(x, y, 0.0, 1.0);

    // UV coordinates (0,0) to (1,1) with Y flipped for texture sampling
    out.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );

    return out;
}

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_color, s_color, in.uv);
}
