//! Atmosphere runtime state and GPU uniform.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Wind state
// ---------------------------------------------------------------------------

/// Runtime wind state computed each frame.
#[derive(Clone, Debug, Default)]
pub struct WindState {
    /// Current wind direction (normalized, Y=0).
    pub direction: [f32; 3],
    /// Current speed including gusts.
    pub speed: f32,
    /// Current gust intensity (0-1).
    pub gust_factor: f32,
    /// Integral of wind*dt (for cloud movement).
    pub accumulated_offset: [f32; 3],
}

// ---------------------------------------------------------------------------
// CPU-side state
// ---------------------------------------------------------------------------

/// Full atmosphere state computed each frame by [`super::AtmosphereSystem`].
#[derive(Clone, Debug)]
pub struct AtmosphereState {
    // Time
    pub time_of_day: f32,
    pub day_count: u32,

    // Sun
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,

    // Ambient
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,

    // Sky
    pub sky_zenith_color: [f32; 3],
    pub sky_horizon_color: [f32; 3],
    pub sky_intensity: f32,

    // Fog
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_height_base: f32,
    pub fog_inscattering: f32,

    // Moons (up to 4)
    pub moon_directions: [[f32; 3]; 4],
    pub moon_colors: [[f32; 3]; 4],
    pub moon_phases: [f32; 4],
    pub moon_sizes: [f32; 4],
    pub moon_count: u32,

    // Primary directional light (blended sun/moon - whichever is brighter)
    pub primary_light_direction: [f32; 3],
    pub primary_light_color: [f32; 3],
    pub primary_light_intensity: f32,

    // Weather
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub precipitation_intensity: f32,
    pub weather_wetness: f32,

    // Wind
    pub wind: WindState,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self {
            time_of_day: 10.0,
            day_count: 0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [1.0, 0.98, 0.95],
            sun_intensity: 1.5,
            ambient_color: [0.03, 0.04, 0.05],
            ambient_intensity: 1.0,
            sky_zenith_color: [0.15, 0.35, 0.65],
            sky_horizon_color: [0.45, 0.55, 0.7],
            sky_intensity: 1.0,
            fog_color: [0.5, 0.55, 0.6],
            fog_density: 0.0,
            fog_height_falloff: 0.05,
            fog_height_base: 0.0,
            fog_inscattering: 0.0,
            primary_light_direction: [0.0, 1.0, 0.0],
            primary_light_color: [1.0, 0.98, 0.95],
            primary_light_intensity: 1.5,
            moon_directions: [[0.0; 3]; 4],
            moon_colors: [[0.0; 3]; 4],
            moon_phases: [0.0; 4],
            moon_sizes: [0.0; 4],
            moon_count: 0,
            cloud_coverage: 0.0,
            cloud_density: 0.0,
            precipitation_intensity: 0.0,
            weather_wetness: 0.0,
            wind: WindState::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform
// ---------------------------------------------------------------------------

/// GPU-ready atmosphere uniform buffer.
///
/// All `vec3` fields are padded to 16-byte alignment for WGSL compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AtmosphereUniform {
    // -- Sun (16 + 16 = 32 bytes) --
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub time_of_day: f32,

    // -- Ambient (16 + 16 = 32 bytes) --
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub sky_zenith_color: [f32; 3],
    pub _pad1: f32,

    // -- Sky + fog start (16 + 16 = 32 bytes) --
    pub sky_horizon_color: [f32; 3],
    pub sky_intensity: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,

    // -- Fog cont. (16 bytes) --
    pub fog_height_falloff: f32,
    pub fog_height_base: f32,
    pub fog_inscattering: f32,
    pub _pad2: f32,

    // -- Moon (16 + 16 = 32 bytes) --
    pub moon_direction: [f32; 3],
    pub moon_phase: f32,
    pub moon_color: [f32; 3],
    pub moon_size: f32,

    // -- Weather (16 bytes) --
    pub moon_count: u32,
    pub cloud_coverage: f32,
    pub weather_wetness: f32,
    pub _pad3: f32,
}

impl Default for AtmosphereUniform {
    fn default() -> Self {
        Self {
            sun_direction: [0.0, 1.0, 0.0],
            sun_intensity: 1.5,
            sun_color: [1.0, 0.98, 0.95],
            time_of_day: 10.0,
            ambient_color: [0.03, 0.04, 0.05],
            ambient_intensity: 1.0,
            sky_zenith_color: [0.15, 0.35, 0.65],
            _pad1: 0.0,
            sky_horizon_color: [0.45, 0.55, 0.7],
            sky_intensity: 1.0,
            fog_color: [0.5, 0.55, 0.6],
            fog_density: 0.0,
            fog_height_falloff: 0.05,
            fog_height_base: 0.0,
            fog_inscattering: 0.0,
            _pad2: 0.0,
            moon_direction: [0.0, 0.0, 0.0],
            moon_phase: 0.0,
            moon_color: [0.0, 0.0, 0.0],
            moon_size: 0.0,
            moon_count: 0,
            cloud_coverage: 0.0,
            weather_wetness: 0.0,
            _pad3: 0.0,
        }
    }
}

impl From<&AtmosphereState> for AtmosphereUniform {
    fn from(s: &AtmosphereState) -> Self {
        // Primary moon (index 0 if present)
        let (moon_dir, moon_phase, moon_color, moon_size) = if s.moon_count > 0 {
            (
                s.moon_directions[0],
                s.moon_phases[0],
                s.moon_colors[0],
                s.moon_sizes[0],
            )
        } else {
            ([0.0; 3], 0.0, [0.0; 3], 0.0)
        };

        Self {
            sun_direction: s.sun_direction,
            sun_intensity: s.sun_intensity,
            sun_color: s.sun_color,
            time_of_day: s.time_of_day,
            ambient_color: s.ambient_color,
            ambient_intensity: s.ambient_intensity,
            sky_zenith_color: s.sky_zenith_color,
            _pad1: 0.0,
            sky_horizon_color: s.sky_horizon_color,
            sky_intensity: s.sky_intensity,
            fog_color: s.fog_color,
            fog_density: s.fog_density,
            fog_height_falloff: s.fog_height_falloff,
            fog_height_base: s.fog_height_base,
            fog_inscattering: s.fog_inscattering,
            _pad2: 0.0,
            moon_direction: moon_dir,
            moon_phase,
            moon_color,
            moon_size,
            moon_count: s.moon_count,
            cloud_coverage: s.cloud_coverage,
            weather_wetness: s.weather_wetness,
            _pad3: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_size_alignment() {
        // Must be a multiple of 16 bytes for GPU buffer alignment
        let size = std::mem::size_of::<AtmosphereUniform>();
        assert_eq!(
            size % 16,
            0,
            "AtmosphereUniform size {size} is not 16-byte aligned"
        );
    }

    #[test]
    fn test_bytemuck_cast() {
        let u = AtmosphereUniform::default();
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), std::mem::size_of::<AtmosphereUniform>());
    }

    #[test]
    fn test_from_state() {
        let state = AtmosphereState::default();
        let uniform = AtmosphereUniform::from(&state);
        assert_eq!(uniform.sun_intensity, state.sun_intensity);
        assert_eq!(uniform.sun_color, state.sun_color);
        assert_eq!(uniform.time_of_day, state.time_of_day);
    }
}
