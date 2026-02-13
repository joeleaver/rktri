// Procedural Sky Shader
// Implements atmospheric scattering for realistic sky rendering

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

// Physical constants for atmospheric scattering
const RAYLEIGH_SCALE: f32 = 8000.0;  // Scale height for Rayleigh scattering
const MIE_SCALE: f32 = 1200.0;       // Scale height for Mie scattering

// Wavelength-dependent Rayleigh scattering coefficient
// Approximates wavelength^-4 dependency for RGB channels
fn rayleigh_coefficient(wavelength: f32) -> f32 {
    // Simplified power law: shorter wavelengths scatter more
    let wl_normalized = wavelength / 650.0;  // Normalize to red wavelength
    return pow(wl_normalized, -4.0);
}

// Get RGB scattering coefficients
fn rayleigh_coefficients() -> vec3<f32> {
    // Approximate wavelengths: R=650nm, G=510nm, B=475nm
    return vec3<f32>(
        rayleigh_coefficient(650.0),  // Red
        rayleigh_coefficient(510.0),  // Green
        rayleigh_coefficient(475.0)   // Blue
    ) * 0.0005;  // Scale factor for visual appearance
}

// Rayleigh phase function (scattering pattern)
fn rayleigh_phase(cos_theta: f32) -> f32 {
    // Phase function: (3/16π)(1 + cos²θ)
    return 0.05968310365946075 * (1.0 + cos_theta * cos_theta);
}

// Mie scattering phase function (for sun glow/halo)
// g = anisotropy parameter (-1 to 1, typically 0.76 for atmosphere)
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let cos2 = cos_theta * cos_theta;

    // Henyey-Greenstein phase function
    let num = 1.0 - g2;
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);

    return (1.0 / (4.0 * 3.14159265359)) * (num / denom);
}

// Compute atmospheric density falloff
fn atmosphere_density(height: f32, scale_height: f32) -> f32 {
    return exp(-height / scale_height);
}

// Main sky color computation
fn compute_sky_color(ray_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    // Normalize ray direction
    let dir = normalize(ray_dir);

    // Check if we're below horizon (ground plane)
    if (dir.y < 0.0) {
        // Blend ground color with dark atmosphere near horizon
        let horizon_blend = smoothstep(-0.1, 0.0, dir.y);
        let dark_ground = params.ground_color * 0.3;
        return mix(dark_ground, params.ground_color, horizon_blend);
    }

    // Sun angle (cosine of angle between ray and sun direction)
    let cos_theta = dot(dir, params.sun_direction);

    // --- Rayleigh Scattering (Blue Sky) ---

    // Simplified optical depth (path length through atmosphere)
    // More atmosphere to look through near horizon (dir.y → 0)
    let zenith_angle = acos(dir.y);
    let optical_depth = 1.0 / (cos(zenith_angle) + 0.15 * pow(93.885 - degrees(zenith_angle), -1.253));

    // Wavelength-dependent scattering
    let beta_r = rayleigh_coefficients();
    let rayleigh = beta_r * rayleigh_phase(cos_theta) * optical_depth;

    // Sky base color (blue gradient)
    let sky_color = rayleigh * params.sky_intensity;

    // Adjust for sun position (sunset/sunrise effects)
    let sun_height = params.sun_direction.y;
    let sunset_factor = smoothstep(-0.1, 0.3, sun_height);

    // Add warm tones during sunset/sunrise
    let sunset_color = vec3<f32>(1.0, 0.6, 0.3) * (1.0 - sunset_factor) * 0.5;
    let atmospheric_color = sky_color + sunset_color;

    // --- Mie Scattering (Sun Glow) ---

    // Strong forward scattering for sun disc and halo
    let mie = mie_phase(cos_theta, 0.76) * params.sun_intensity;

    // Sun disc - sharper falloff
    let sun_angle = acos(cos_theta);
    let sun_disc_size = 0.05;  // ~3 degrees
    let sun_disc = smoothstep(sun_disc_size, sun_disc_size * 0.5, sun_angle);

    // Sun halo - softer falloff
    let sun_halo_size = 0.3;
    let sun_halo = smoothstep(sun_halo_size, 0.0, sun_angle) * 0.3;

    // Combine sun contribution
    let sun_contribution = (sun_disc + sun_halo + mie * 0.1) * params.sun_color;

    // --- Day/Night Transition ---

    // Fade sky intensity based on sun position
    let day_factor = smoothstep(-0.2, 0.1, sun_height);

    // Night sky (dark blue)
    let night_color = vec3<f32>(0.01, 0.02, 0.05) * sky_color * 0.1;

    // Combine all components
    let final_color = mix(
        night_color,
        atmospheric_color + sun_contribution,
        day_factor
    );

    // Return HDR color (can exceed 1.0 for bright sun)
    return max(final_color, vec3<f32>(0.0));
}

// Convenience function for simple sky rendering
fn sample_sky(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    var params: SkyParams;
    params.sun_direction = normalize(sun_dir);
    params.sun_color = vec3<f32>(1.0, 0.95, 0.9);
    params.sun_intensity = 20.0;
    params.sky_zenith_color = vec3<f32>(0.15, 0.35, 0.65);
    params.sky_intensity = 1.0;
    params.sky_horizon_color = vec3<f32>(0.45, 0.55, 0.7);
    params.ground_color = vec3<f32>(0.3, 0.25, 0.2);
    params.moon_count = 0u;
    params.fog_density = 0.0;

    return compute_sky_color(ray_dir, params);
}
