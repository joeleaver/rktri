//! Atmosphere configuration with realistic color ramps.

use serde::{Deserialize, Serialize};

use crate::atmosphere::color_ramp::ColorRamp;

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Full atmosphere configuration. All color ramps are keyed over a 24-hour
/// cycle and interpolated by [`ColorRamp`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtmosphereConfig {
    /// Real-world seconds per in-game day. 0 = time is paused.
    pub day_length_seconds: f32,
    /// Starting hour (0-24).
    pub start_time: f32,
    /// Whether time advancement is paused.
    pub time_paused: bool,
    /// Latitude in degrees (affects sun altitude). Default 45.0.
    pub latitude: f32,
    /// Angular size of the sun disc (radians). Cosmetic.
    pub sun_size: f32,

    // -- Color ramps keyed by hour (0-24) ---------------------------------

    /// Sun color (linear RGB) over the day.
    pub sun_color_ramp: ColorRamp<[f32; 3]>,
    /// Sun intensity multiplier over the day.
    pub sun_intensity_ramp: ColorRamp<f32>,
    /// Ambient light color (linear RGB) over the day.
    pub ambient_color_ramp: ColorRamp<[f32; 3]>,
    /// Ambient intensity multiplier over the day.
    pub ambient_intensity_ramp: ColorRamp<f32>,
    /// Sky zenith color over the day.
    pub sky_zenith_ramp: ColorRamp<[f32; 3]>,
    /// Sky horizon color over the day.
    pub sky_horizon_ramp: ColorRamp<[f32; 3]>,
    /// Fog base color over the day.
    pub fog_color_ramp: ColorRamp<[f32; 3]>,

    // -- Sub-configs -------------------------------------------------------

    /// Fog parameters.
    pub fog: FogConfig,
    /// Moon definitions (up to 4).
    pub moons: Vec<MoonConfig>,
    /// Weather parameters.
    pub weather: WeatherConfig,
    /// Wind parameters.
    pub wind: WindConfig,
    /// Cloud parameters.
    pub clouds: CloudConfig,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            day_length_seconds: 1200.0, // 20 minutes per day
            start_time: 10.0,
            time_paused: true, // Paused by default so existing behavior is preserved
            latitude: 45.0,
            sun_size: 0.02,

            // ----- Sun color ramp -----
            // Night: dim blue-ish, Dawn: warm orange, Day: near-white, Dusk: warm red
            sun_color_ramp: ColorRamp::new(vec![
                (0.0, [0.1, 0.1, 0.2]),     // midnight - very dim blue
                (5.0, [0.15, 0.1, 0.2]),     // pre-dawn
                (5.5, [0.9, 0.4, 0.2]),      // early dawn - deep orange
                (6.5, [1.0, 0.65, 0.35]),    // dawn - orange
                (7.5, [1.0, 0.85, 0.65]),    // late dawn - warm gold
                (9.0, [1.0, 0.95, 0.88]),    // mid-morning
                (10.0, [1.0, 0.98, 0.95]),   // CRITICAL: match current hardcoded value
                (12.0, [1.0, 0.98, 0.95]),   // noon - same as 10
                (15.0, [1.0, 0.97, 0.92]),   // afternoon
                (17.0, [1.0, 0.85, 0.65]),   // pre-dusk
                (18.0, [1.0, 0.55, 0.25]),   // dusk - deep orange
                (19.0, [0.9, 0.3, 0.15]),    // late dusk
                (19.5, [0.3, 0.15, 0.2]),    // twilight
                (20.5, [0.1, 0.1, 0.2]),     // night
            ]),

            // ----- Sun intensity ramp -----
            // Aligned with sun arc: sunrise at 6:00, sunset at 18:00
            sun_intensity_ramp: ColorRamp::new(vec![
                (0.0, 0.0),     // midnight
                (5.0, 0.0),     // pre-dawn
                (5.5, 0.02),    // civil twilight (sun ~7° below horizon)
                (6.0, 0.1),     // sunrise - sun at horizon
                (7.0, 0.5),     // early morning
                (8.0, 1.0),     // morning
                (10.0, 1.5),    // CRITICAL: match current hardcoded value
                (12.0, 1.5),    // noon peak
                (14.0, 1.5),    // early afternoon
                (16.0, 1.0),    // afternoon
                (17.0, 0.5),    // late afternoon
                (18.0, 0.1),    // sunset - sun at horizon
                (18.5, 0.02),   // civil twilight
                (19.0, 0.0),    // night
            ]),

            // ----- Ambient color ramp -----
            // Matches current hardcoded [0.03, 0.04, 0.05] at time 10
            ambient_color_ramp: ColorRamp::new(vec![
                (0.0, [0.005, 0.005, 0.015]),  // midnight - very dark blue
                (5.0, [0.005, 0.005, 0.015]),   // pre-dawn
                (6.0, [0.02, 0.015, 0.025]),    // dawn - slight purple
                (7.5, [0.025, 0.03, 0.04]),     // morning
                (10.0, [0.03, 0.04, 0.05]),     // CRITICAL: match current hardcoded value
                (12.0, [0.03, 0.04, 0.05]),     // noon
                (15.0, [0.03, 0.04, 0.05]),     // afternoon
                (17.5, [0.025, 0.025, 0.035]),  // pre-dusk
                (19.0, [0.015, 0.01, 0.02]),    // dusk
                (20.0, [0.005, 0.005, 0.015]),  // night
            ]),

            // ----- Ambient intensity ramp -----
            ambient_intensity_ramp: ColorRamp::new(vec![
                (0.0, 0.3),     // night - some moonlight
                (5.0, 0.3),     // pre-dawn
                (6.0, 0.5),     // dawn
                (8.0, 0.8),     // morning
                (10.0, 1.0),    // full day
                (14.0, 1.0),    // afternoon
                (17.0, 0.8),    // pre-dusk
                (19.0, 0.5),    // dusk
                (20.0, 0.3),    // night
            ]),

            // ----- Sky zenith (top of sky dome) -----
            sky_zenith_ramp: ColorRamp::new(vec![
                (0.0, [0.0, 0.0, 0.02]),       // midnight - near black
                (5.0, [0.0, 0.0, 0.03]),        // pre-dawn
                (6.0, [0.05, 0.05, 0.15]),      // dawn
                (7.0, [0.1, 0.2, 0.5]),         // morning
                (10.0, [0.15, 0.35, 0.65]),     // day
                (12.0, [0.15, 0.35, 0.65]),     // noon
                (16.0, [0.15, 0.3, 0.6]),       // afternoon
                (18.0, [0.1, 0.1, 0.3]),        // dusk
                (19.5, [0.02, 0.02, 0.06]),     // twilight
                (20.5, [0.0, 0.0, 0.02]),       // night
            ]),

            // ----- Sky horizon -----
            sky_horizon_ramp: ColorRamp::new(vec![
                (0.0, [0.01, 0.01, 0.02]),      // midnight
                (5.0, [0.02, 0.02, 0.04]),      // pre-dawn
                (5.5, [0.2, 0.1, 0.05]),        // dawn glow
                (6.5, [0.6, 0.3, 0.15]),        // sunrise
                (8.0, [0.5, 0.6, 0.7]),         // morning haze
                (10.0, [0.45, 0.55, 0.7]),      // day
                (12.0, [0.4, 0.55, 0.7]),       // noon
                (16.0, [0.45, 0.5, 0.6]),       // afternoon
                (17.5, [0.6, 0.35, 0.15]),      // pre-dusk
                (18.5, [0.5, 0.2, 0.1]),        // sunset
                (19.5, [0.1, 0.05, 0.05]),      // twilight
                (20.5, [0.01, 0.01, 0.02]),     // night
            ]),

            // ----- Fog color -----
            fog_color_ramp: ColorRamp::new(vec![
                (0.0, [0.02, 0.02, 0.04]),      // night fog - dark blue
                (6.0, [0.3, 0.25, 0.2]),        // dawn fog - warm
                (10.0, [0.5, 0.55, 0.6]),       // day fog
                (12.0, [0.55, 0.6, 0.65]),      // noon fog
                (17.0, [0.5, 0.45, 0.4]),       // pre-dusk
                (18.5, [0.3, 0.2, 0.15]),       // dusk fog - warm
                (20.0, [0.02, 0.02, 0.04]),     // night fog
            ]),

            fog: FogConfig::default(),

            moons: vec![MoonConfig::default()],

            weather: WeatherConfig::default(),
            wind: WindConfig::default(),
            clouds: CloudConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fog config
// ---------------------------------------------------------------------------

/// Configuration for height-based and distance-based fog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FogConfig {
    pub enabled: bool,
    pub height_fog_density: f32,
    pub height_fog_falloff: f32,
    pub height_fog_base: f32,
    pub distance_fog_density: f32,
    pub distance_fog_start: f32,
    pub distance_fog_end: f32,
    pub inscattering_intensity: f32,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            height_fog_density: 0.0,
            height_fog_falloff: 0.05,
            height_fog_base: 0.0,
            distance_fog_density: 0.0,
            distance_fog_start: 100.0,
            distance_fog_end: 500.0,
            inscattering_intensity: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Moon config
// ---------------------------------------------------------------------------

/// Configuration for a single moon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoonConfig {
    pub name: String,
    pub color: [f32; 3],
    pub size: f32,
    pub orbit_period_days: f32,
    pub orbit_inclination: f32,
    pub phase_offset: f32,
    pub brightness: f32,
}

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            name: "Luna".to_string(),
            color: [0.9, 0.9, 1.0],
            size: 0.998, // cos-angle threshold: 1.0 = tiny dot, 0.99 = ~8° disc
            orbit_period_days: 29.5,
            orbit_inclination: 5.14,
            phase_offset: 14.75, // Start near full moon (0.5 phase at day 0)
            brightness: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Weather config + preset
// ---------------------------------------------------------------------------

/// Weather configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherConfig {
    pub initial_preset: WeatherPreset,
    pub transition_duration: f32,
    pub auto_weather: bool,
    pub auto_change_interval: f32,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            initial_preset: WeatherPreset::Clear,
            transition_duration: 30.0,
            auto_weather: false,
            auto_change_interval: 300.0,
        }
    }
}

/// Named weather presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherPreset {
    Clear,
    PartlyCloudy,
    Overcast,
    Foggy,
    Rain,
    Snow,
    Storm,
}

impl Default for WeatherPreset {
    fn default() -> Self {
        Self::Clear
    }
}

// ---------------------------------------------------------------------------
// Wind config
// ---------------------------------------------------------------------------

/// Configuration for wind (affects cloud movement and future particle systems).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindConfig {
    /// Normalized XZ direction (default: [1.0, 0.0] = east).
    pub base_direction: [f32; 2],
    /// World units per second (default: 2.0).
    pub base_speed: f32,
    /// Max gust multiplier above base (default: 0.3).
    pub gust_strength: f32,
    /// Gust oscillation Hz (default: 0.1).
    pub gust_frequency: f32,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self {
            base_direction: [1.0, 0.0],
            base_speed: 2.0,
            gust_strength: 0.3,
            gust_frequency: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Cloud config
// ---------------------------------------------------------------------------

/// Configuration for dynamic shader-based clouds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Whether clouds are enabled.
    pub enabled: bool,
    /// Cloud layer base altitude (default: 80.0).
    pub altitude: f32,
    /// Cloud layer thickness (default: 20.0).
    pub thickness: f32,
    /// Base coverage before weather modulation, 0-1 (default: 0.3).
    pub coverage: f32,
    /// Cloud optical density (default: 0.8).
    pub density: f32,
    /// Noise frequency (default: 0.02).
    pub noise_scale: f32,
    /// Detail noise frequency (default: 0.1).
    pub detail_scale: f32,
    /// Base cloud color (default: white).
    pub cloud_color: [f32; 3],
    /// Cloud shadow/underside color (default: [0.4, 0.4, 0.5]).
    pub shadow_color: [f32; 3],
    /// How blocky/sharp cloud edges are (default: 3.0, higher=blockier).
    pub edge_sharpness: f32,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            altitude: 80.0,
            thickness: 20.0,
            coverage: 0.3,
            density: 0.8,
            noise_scale: 0.02,
            detail_scale: 0.1,
            cloud_color: [1.0, 1.0, 1.0],
            shadow_color: [0.4, 0.4, 0.5],
            edge_sharpness: 3.0,
        }
    }
}
