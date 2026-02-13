// Water surface ray tracing utilities
// This file provides functions for ray-water intersection, refraction, and reflection

struct WaterParams {
    sea_level: f32,
    time: f32,
    ior: f32,          // Index of refraction (1.333)
    roughness: f32,
    color: vec3<f32>,
    _pad1: f32,
    absorption: vec3<f32>,  // Beer-Lambert coefficients
    _pad2: f32,
    foam_threshold: f32,
    caustics_intensity: f32,
    _pad3: vec2<f32>,
}

struct WaterHit {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
}

// Gerstner wave displacement using multiple frequencies
// Matches Rust WaterSystem::wave_height exactly
fn wave_height(x: f32, z: f32, time: f32) -> f32 {
    var height = 0.0;

    // Wave 1: Large slow waves
    let amp1 = 0.5;
    let freq1_x = 0.02;
    let freq1_z = 0.015;
    let speed1_x = 0.5;
    let speed1_z = 0.3;
    height += amp1 * sin(freq1_x * x + speed1_x * time + freq1_z * z + speed1_z * time);

    // Wave 2: Medium waves
    let amp2 = 0.2;
    let freq2_x = 0.05;
    let freq2_z = 0.04;
    let speed2_x = 1.2;
    let speed2_z = 0.8;
    height += amp2 * sin(freq2_x * x + speed2_x * time + freq2_z * z + speed2_z * time);

    // Wave 3: Small fast ripples
    let amp3 = 0.05;
    let freq3_x = 0.15;
    let freq3_z = 0.12;
    let speed3_x = 2.5;
    let speed3_z = 1.8;
    height += amp3 * sin(freq3_x * x + speed3_x * time + freq3_z * z + speed3_z * time);

    return height;
}

// Compute water surface normal via finite differences
fn wave_normal(x: f32, z: f32, time: f32) -> vec3<f32> {
    let eps = 0.1;

    let h_center = wave_height(x, z, time);
    let h_right = wave_height(x + eps, z, time);
    let h_forward = wave_height(x, z + eps, time);

    let dx = h_right - h_center;
    let dz = h_forward - h_center;

    // Normal from partial derivatives
    let tangent_x = vec3<f32>(eps, dx, 0.0);
    let tangent_z = vec3<f32>(0.0, dz, eps);

    return normalize(cross(tangent_z, tangent_x));
}

// Find intersection of ray with animated water surface
// Uses iterative refinement with binary search
fn ray_water_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sea_level: f32, time: f32) -> WaterHit {
    var hit: WaterHit;
    hit.hit = false;
    hit.t = 0.0;

    // Quick reject if ray doesn't cross sea level plane
    if ray_dir.y == 0.0 {
        return hit;
    }

    // Find intersection with flat sea level as starting point
    let t_plane = (sea_level - ray_origin.y) / ray_dir.y;
    if t_plane < 0.0 {
        return hit;
    }

    // Binary search for accurate surface intersection
    var t_min = max(0.0, t_plane - 2.0);
    var t_max = t_plane + 2.0;

    for (var i = 0; i < 8; i++) {
        let t_mid = (t_min + t_max) * 0.5;
        let pos = ray_origin + ray_dir * t_mid;
        let surface_y = sea_level + wave_height(pos.x, pos.z, time);

        if pos.y > surface_y {
            t_min = t_mid;
        } else {
            t_max = t_mid;
        }
    }

    let t_final = (t_min + t_max) * 0.5;
    let final_pos = ray_origin + ray_dir * t_final;
    let surface_y = sea_level + wave_height(final_pos.x, final_pos.z, time);

    // Check if we're close enough to surface
    if abs(final_pos.y - surface_y) < 0.5 {
        hit.hit = true;
        hit.t = t_final;
        hit.position = vec3<f32>(final_pos.x, surface_y, final_pos.z);
        hit.normal = wave_normal(final_pos.x, final_pos.z, time);
    }

    return hit;
}

// Snell's law refraction with total internal reflection handling
fn refract_ray(incident: vec3<f32>, normal: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_i = -dot(incident, normal);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

    // Total internal reflection
    if sin2_t > 1.0 {
        return reflect(incident, normal);
    }

    let cos_t = sqrt(1.0 - sin2_t);
    return eta * incident + (eta * cos_i - cos_t) * normal;
}

// Schlick's approximation for Fresnel reflectance
fn fresnel_schlick(cos_theta: f32, ior: f32) -> f32 {
    let r0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
    let one_minus_cos = 1.0 - cos_theta;
    let one_minus_cos2 = one_minus_cos * one_minus_cos;
    let one_minus_cos5 = one_minus_cos2 * one_minus_cos2 * one_minus_cos;
    return r0 + (1.0 - r0) * one_minus_cos5;
}
