// Lighting Pass - PBR Compute Shader
// Reads G-buffer and applies physically-based lighting
// Output: final lit color

// ============================================
// PBR (Physically Based Rendering) Utilities
// Implements Cook-Torrance BRDF with GGX distribution
// ============================================

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz Normal Distribution Function
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith's Geometry Function with GGX
fn G_SchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let ggx_v = G_SchlickGGX(NdotV, roughness);
    let ggx_l = G_SchlickGGX(NdotL, roughness);
    return ggx_v * ggx_l;
}

// Fresnel-Schlick Approximation
fn F_Schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    let power = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (vec3<f32>(1.0) - F0) * power;
}

// Cook-Torrance Specular BRDF
fn brdf_cook_torrance(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotV = max(dot(N, V), 0.0001);
    let NdotL = max(dot(N, L), 0.0001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);

    let specular = (D * G * F) / (4.0 * NdotV * NdotL);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * albedo / PI;

    return (diffuse + specular) * NdotL;
}

// Approximate subsurface scattering for thin translucent materials (leaves)
fn subsurface_scattering(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    albedo: vec3<f32>,
    translucency: f32
) -> vec3<f32> {
    // Wrap diffuse: light wraps slightly around the surface
    let wrap = 0.25;
    let NdotL_wrap = (dot(N, L) + wrap) / (1.0 + wrap);
    let wrap_diffuse = max(NdotL_wrap, 0.0) * albedo * 0.15;

    // Back-lighting translucency: light transmitted through thin material
    // Narrow power curve (exponent 6) so only direct backlit angles contribute
    let back_light = max(dot(V, -L), 0.0);
    let sss_power = pow(back_light, 6.0) * translucency;
    // Subtle tint toward albedo color (e.g. green glow for leaves)
    let sss_color = albedo * sss_power * 0.4;

    // View-independent transmitted light (very subtle ambient translucency)
    let thickness_factor = translucency * 0.05;
    let ambient_through = albedo * thickness_factor;

    return wrap_diffuse + sss_color + ambient_through;
}

// ============================================
// Main Lighting Shader
// ============================================

// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Light configuration
struct LightUniform {
    direction: vec3<f32>,  // Normalized direction towards light
    _padding1: f32,
    color: vec3<f32>,      // Light color (RGB)
    intensity: f32,        // Sun intensity multiplier (0 when below horizon)
}

@group(0) @binding(1) var<uniform> light: LightUniform;

// Sky parameters
struct SkyParams {
    sun_direction: vec3<f32>,
    _pad1: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,

    sky_zenith_color: vec3<f32>,
    sky_intensity: f32,
    sky_horizon_color: vec3<f32>,
    _pad2: f32,

    ground_color: vec3<f32>,
    _pad3: f32,

    moon_direction: vec3<f32>,
    moon_phase: f32,
    moon_color: vec3<f32>,
    moon_size: f32,
    moon_count: u32,
    _pad4a: f32,
    _pad4b: f32,
    _pad4c: f32,

    fog_color: vec3<f32>,
    fog_density: f32,
    fog_height_falloff: f32,
    fog_height_base: f32,
    fog_inscattering: f32,
    _pad5: f32,

    ambient_color: vec3<f32>,
    ambient_intensity: f32,
}

@group(0) @binding(2) var<uniform> sky_params: SkyParams;

// Debug mode uniform
struct DebugParams {
    mode: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(3) var<uniform> debug_params: DebugParams;

// G-buffer inputs (sampled textures from geometry pass)
@group(1) @binding(0) var t_albedo: texture_2d<f32>;
@group(1) @binding(1) var t_normal: texture_2d<f32>;
@group(1) @binding(2) var t_depth: texture_2d<f32>;
@group(1) @binding(3) var t_material: texture_2d<f32>;
@group(1) @binding(4) var t_shadow: texture_2d<f32>;
@group(1) @binding(5) var t_godrays: texture_2d<f32>;
@group(1) @binding(6) var t_clouds: texture_2d<f32>;

// Output texture
@group(2) @binding(0) var t_output: texture_storage_2d<rgba16float, write>;

// Reconstruct world position from depth and screen coordinates
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC coordinates
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,  // Flip Y for texture coordinates
        depth,
        1.0
    );

    // Transform to world space
    let world_pos = camera.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Compute procedural sky color with dynamic atmosphere colors, moon, and fog
fn compute_sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let dir = normalize(ray_dir);
    let sun_dir = sky_params.sun_direction;
    let cos_sun = dot(dir, sun_dir);
    let sun_height = sun_dir.y;

    // Day/night factor (used for both ground and sky)
    let day_factor = smoothstep(-0.2, 0.1, sun_height);
    let night_color = vec3<f32>(0.005, 0.008, 0.02); // Deep night blue

    // Below horizon - ground (darkens at night)
    if (dir.y < 0.0) {
        let horizon_blend = smoothstep(-0.1, 0.0, dir.y);
        let dark_ground = sky_params.ground_color * 0.3;
        let ground = mix(dark_ground, sky_params.ground_color, horizon_blend) * sky_params.sky_intensity;
        return mix(night_color * 0.5, ground, day_factor);
    }

    // Dynamic sky gradient from atmosphere color ramps
    let horizon = 1.0 - max(dir.y, 0.0);
    var sky = mix(sky_params.sky_zenith_color, sky_params.sky_horizon_color, horizon * horizon);

    // Sun disc
    let sun_size = 0.995;
    if (cos_sun > sun_size) {
        let sun_factor = (cos_sun - sun_size) / (1.0 - sun_size);
        sky = mix(sky, sky_params.sun_color * sky_params.sun_intensity, sun_factor);
    }

    // Sun glow (Mie-like)
    let glow = pow(max(cos_sun, 0.0), 64.0) * 2.0;
    sky += sky_params.sun_color * glow;

    // Day/night transition (before moon so moon renders on top of dark sky)
    sky = mix(night_color, sky, day_factor);

    // Moon rendering (after day/night mix so it's visible at night)
    if (sky_params.moon_count > 0u) {
        let cos_moon = dot(dir, sky_params.moon_direction);
        let moon_angular_size = sky_params.moon_size;

        if (cos_moon > moon_angular_size) {
            // Moon disc with phase
            let moon_factor = (cos_moon - moon_angular_size) / (1.0 - moon_angular_size);
            let phase = sky_params.moon_phase; // 0=new, 0.5=full, 1=new
            // Phase illumination: 0 at new moon, 1 at full moon
            let phase_illumination = 0.5 - 0.5 * cos(phase * 2.0 * 3.14159);
            let moon_brightness = phase_illumination * moon_factor;
            sky += sky_params.moon_color * moon_brightness * 2.0;
        }

        // Moon glow
        let moon_glow = pow(max(cos_moon, 0.0), 128.0) * 0.3;
        let phase_illum = 0.5 - 0.5 * cos(sky_params.moon_phase * 2.0 * 3.14159);
        sky += sky_params.moon_color * moon_glow * phase_illum;
    }

    return max(sky * sky_params.sky_intensity, vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(t_output);
    let coords = vec2<i32>(global_id.xy);

    // Bounds check
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    let uv = (vec2<f32>(coords) + vec2<f32>(0.5)) / vec2<f32>(dims);

    // Sample G-buffer
    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal_data = textureLoad(t_normal, coords, 0);
    let depth = textureLoad(t_depth, coords, 0).r;
    let material = textureLoad(t_material, coords, 0);
    // Shadow and godrays are at half resolution - scale coords
    let shadow_dims = textureDimensions(t_shadow);
    let shadow_coords = vec2<i32>(vec2<f32>(coords) * vec2<f32>(shadow_dims) / vec2<f32>(dims));
    let shadow = textureLoad(t_shadow, shadow_coords, 0).r;
    let godrays = textureLoad(t_godrays, shadow_coords, 0).r;

    let normal = normal_data.xyz;
    // Sky is indicated by zero-length normal (roughness check was broken due to missing roughness data)
    let is_sky = length(normal) < 0.001;

    // Debug mode handling (skip for sky)
    if (!is_sky && debug_params.mode > 0u) {
        var debug_color = vec3<f32>(0.0);

        switch debug_params.mode {
            case 1u: {
                // Albedo
                debug_color = albedo;
            }
            case 2u: {
                // Normals (remap -1..1 to 0..1)
                debug_color = normal * 0.5 + 0.5;
            }
            case 3u: {
                // Depth (linearized)
                let linear_depth = depth;
                debug_color = vec3<f32>(linear_depth);
            }
            case 4u: {
                // Material (metallic, roughness, ao)
                debug_color = vec3<f32>(material.r, material.g, material.b);
            }
            default: {
                debug_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta for invalid
            }
        }

        textureStore(t_output, coords, vec4<f32>(debug_color, 1.0));
        return;
    }

    // Sky pixels - compute procedural sky color
    if (is_sky) {
        // Reconstruct ray direction from screen position
        let ndc = uv * 2.0 - 1.0;
        let ndc_flipped = vec2<f32>(ndc.x, -ndc.y);
        let near_point = camera.inv_view_proj * vec4<f32>(ndc_flipped, 0.0, 1.0);
        let far_point = camera.inv_view_proj * vec4<f32>(ndc_flipped, 1.0, 1.0);
        let ray_dir = normalize(far_point.xyz / far_point.w - near_point.xyz / near_point.w);

        let sky_color = compute_sky_color(ray_dir);
        // Add god rays to sky (visible as light shafts against sky)
        let sky_godrays = godrays * light.color * light.intensity;

        // Apply fog to sky (distant objects fade to fog color)
        var final_sky = sky_color + sky_godrays;
        if (sky_params.fog_density > 0.0) {
            let fog_amount = clamp(sky_params.fog_density * 2.0, 0.0, 0.8);
            final_sky = mix(final_sky, sky_params.fog_color, fog_amount);
        }

        // Blend clouds over sky (clouds at half-res, same coords as shadow)
        let cloud_data_sky = textureLoad(t_clouds, shadow_coords, 0);
        let cloud_color_sky = cloud_data_sky.rgb;
        let cloud_alpha_sky = cloud_data_sky.a;
        final_sky = mix(final_sky, cloud_color_sky, cloud_alpha_sky);

        textureStore(t_output, coords, vec4<f32>(final_sky, 1.0));
        return;
    }

    // Extract material properties
    let metallic = material.r;
    let roughness = material.g;
    let translucency = material.b;  // 0.0 = opaque, 0.8 = very translucent (leaves)
    let material_id = u32(material.a * 255.0 + 0.5);

    // Reconstruct world position
    let world_pos = reconstruct_world_pos(uv, depth);

    // View direction (from surface to camera)
    let V = normalize(camera.view_pos - world_pos);

    var N = normal;

    // Ambient light (from atmosphere system, varies with time of day)
    let ambient = sky_params.ambient_color * sky_params.ambient_intensity * albedo;

    // Directional light contribution (scaled by sun intensity)
    let effective_light_color = light.color * light.intensity;
    let L = normalize(light.direction);
    let NdotL = max(dot(N, L), 0.0);

    var direct_light = vec3<f32>(0.0);
    if (NdotL > 0.0) {
        // Apply PBR BRDF
        let brdf = brdf_cook_torrance(N, V, L, albedo, metallic, roughness);
        direct_light = brdf * effective_light_color;
    }

    // Subsurface scattering for translucent materials (leaves)
    var sss = vec3<f32>(0.0);
    if (translucency > 0.01) {
        sss = subsurface_scattering(N, V, L, albedo, translucency) * effective_light_color * shadow;
    }

    // Boost ambient for foliage (canopy self-illumination, scales with ambient)
    var ambient_boost = vec3<f32>(0.0);
    if (translucency > 0.01) {
        ambient_boost = albedo * translucency * 0.08 * sky_params.ambient_intensity;
    }

    // God rays contribution (additive volumetric light)
    let godrays_color = godrays * effective_light_color;

    // Final color
    let final_color = ambient + ambient_boost + (direct_light * shadow) + sss + godrays_color;

    // Aerial perspective - always-on atmospheric distance fade
    // Always blend toward horizon haze color (not ground/sky split)
    // This prevents a visible shading discontinuity at the camera's eye level
    let view_dir = normalize(world_pos - camera.view_pos);
    let aerial_dir = normalize(vec3<f32>(view_dir.x, max(view_dir.y, 0.01), view_dir.z));
    let aerial_sky = compute_sky_color(aerial_dir);
    let aerial_factor = 1.0 - exp(-depth * 0.003);
    var fogged_color = mix(final_color, aerial_sky, aerial_factor);
    if (sky_params.fog_density > 0.0) {
        let view_dist = length(world_pos - camera.view_pos);

        // Distance fog: exponential
        let dist_fog = 1.0 - exp(-sky_params.fog_density * view_dist * 0.01);

        // Height fog: denser at low altitudes
        let height_factor = exp(-sky_params.fog_height_falloff * max(world_pos.y - sky_params.fog_height_base, 0.0));
        let height_fog = height_factor * sky_params.fog_density * view_dist * 0.005;

        // Combined fog factor
        let fog_amount = clamp(max(dist_fog, height_fog), 0.0, 1.0);

        // Inscattering: light scattered into the fog volume toward the viewer
        let sun_dot = max(dot(view_dir, sky_params.sun_direction), 0.0);
        let inscatter = pow(sun_dot, 8.0) * sky_params.fog_inscattering * fog_amount;

        let fog_with_inscatter = sky_params.fog_color + sky_params.sun_color * inscatter;
        fogged_color = mix(fogged_color, fog_with_inscatter, fog_amount);
    }

    // Blend clouds over scene (clouds at half-res, same coords as shadow)
    let cloud_data = textureLoad(t_clouds, shadow_coords, 0);
    let cloud_color_sample = cloud_data.rgb;
    let cloud_alpha = cloud_data.a;
    fogged_color = mix(fogged_color, cloud_color_sample, cloud_alpha);

    // Store result
    textureStore(t_output, coords, vec4<f32>(fogged_color, 1.0));
}
