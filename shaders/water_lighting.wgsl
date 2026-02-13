// Water lighting and underwater effects
// This file provides functions for absorption, fog, and caustics

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

// Beer-Lambert law: exponential absorption with depth
fn apply_water_absorption(color: vec3<f32>, depth: f32, absorption: vec3<f32>) -> vec3<f32> {
    return color * exp(-absorption * depth);
}

// Exponential fog blending toward water color
fn underwater_fog(scene_color: vec3<f32>, water_color: vec3<f32>, depth: f32, max_visibility: f32) -> vec3<f32> {
    let fog_factor = exp(-depth / max_visibility);
    return mix(water_color, scene_color, fog_factor);
}

// Simple procedural caustics using overlapping wave patterns
// Returns brightness multiplier in range 0.5-1.5
fn caustics(pos: vec3<f32>, time: f32, intensity: f32) -> f32 {
    // Multiple overlapping wave patterns for caustic effect
    let scale1 = 3.0;
    let scale2 = 5.0;
    let scale3 = 7.0;

    let wave1 = sin(pos.x * scale1 + time * 2.0) * cos(pos.z * scale1 - time * 1.5);
    let wave2 = sin(pos.x * scale2 - time * 1.8) * cos(pos.z * scale2 + time * 2.2);
    let wave3 = sin(pos.x * scale3 + time * 1.3) * cos(pos.z * scale3 - time * 1.7);

    // Combine waves and normalize to 0-1 range
    let combined = (wave1 + wave2 * 0.5 + wave3 * 0.3) / 1.8;
    let normalized = (combined + 1.0) * 0.5;

    // Map to 0.5-1.5 range and apply intensity
    let base_brightness = 0.5 + normalized;
    return mix(1.0, base_brightness, intensity);
}

// Combined water effects: absorption, fog, and caustics
// Determines whether camera and hit point are underwater and applies appropriate effects
fn apply_water_effects(
    scene_color: vec3<f32>,
    hit_depth: f32,
    camera_depth: f32,
    water_params: WaterParams,
    world_pos: vec3<f32>,
    time: f32
) -> vec3<f32> {
    var result = scene_color;

    let camera_underwater = camera_depth < water_params.sea_level;
    let hit_underwater = hit_depth < water_params.sea_level;

    // Case 1: Both camera and hit point are underwater
    if camera_underwater && hit_underwater {
        let water_depth = abs(hit_depth - camera_depth);

        // Apply absorption
        result = apply_water_absorption(result, water_depth, water_params.absorption);

        // Apply underwater fog
        let max_visibility = 50.0;
        result = underwater_fog(result, water_params.color, water_depth, max_visibility);

        // Apply caustics if close enough to surface
        let depth_from_surface = water_params.sea_level - hit_depth;
        if depth_from_surface < 20.0 {
            let caustic_brightness = caustics(world_pos, time, water_params.caustics_intensity);
            result = result * caustic_brightness;
        }
    }
    // Case 2: Camera above water, looking at underwater object
    else if !camera_underwater && hit_underwater {
        let water_depth = water_params.sea_level - hit_depth;

        // Apply absorption through water
        result = apply_water_absorption(result, water_depth, water_params.absorption);

        // Apply fog
        let max_visibility = 50.0;
        result = underwater_fog(result, water_params.color, water_depth, max_visibility);

        // Caustics visible from above
        if water_depth < 20.0 {
            let caustic_brightness = caustics(world_pos, time, water_params.caustics_intensity);
            result = result * caustic_brightness;
        }
    }
    // Case 3: Camera underwater, looking at above-water object
    else if camera_underwater && !hit_underwater {
        let water_depth = water_params.sea_level - camera_depth;

        // Light travels through water to camera
        result = apply_water_absorption(result, water_depth, water_params.absorption);

        // Slight fog effect
        let max_visibility = 50.0;
        result = underwater_fog(result, water_params.color, water_depth, max_visibility);
    }
    // Case 4: Both above water - no water effects
    // result unchanged

    return result;
}
