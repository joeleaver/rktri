//! Unified atmosphere and weather system.
//!
//! Provides time-of-day driven lighting, sky colors, fog, moon phases, and
//! weather effects. The main entry point is [`AtmosphereSystem`] which is
//! updated each frame and produces an [`AtmosphereState`] (CPU-side) and
//! [`AtmosphereUniform`] (GPU-ready buffer).

pub mod color_ramp;
pub mod config;
pub mod fog;
pub mod moon;
pub mod state;
pub mod sun;
pub mod time;
pub mod weather;

// Re-exports
pub use color_ramp::ColorRamp;
pub use config::{AtmosphereConfig, CloudConfig, FogConfig, MoonConfig, WeatherConfig, WeatherPreset, WindConfig};
pub use state::{AtmosphereState, AtmosphereUniform, WindState};
pub use time::TimeOfDay;
pub use weather::{WeatherModifiers, WeatherStateMachine};

use moon::{compute_moon_direction, compute_moon_phase};
use sun::compute_sun_direction;

// ---------------------------------------------------------------------------
// AtmosphereSystem
// ---------------------------------------------------------------------------

/// Main atmosphere system. Call [`update`](Self::update) each frame, then read
/// the resulting [`state`](Self::state) or [`uniform`](Self::uniform).
pub struct AtmosphereSystem {
    config: AtmosphereConfig,
    time: TimeOfDay,
    weather: WeatherStateMachine,
    state: AtmosphereState,
}

impl AtmosphereSystem {
    /// Create a new atmosphere system from the given configuration.
    pub fn new(config: AtmosphereConfig) -> Self {
        let time = TimeOfDay::new(config.start_time);
        let weather_sm = WeatherStateMachine::new(config.weather.initial_preset);

        let mut sys = Self {
            config,
            time,
            weather: weather_sm,
            state: AtmosphereState::default(),
        };
        sys.recompute_state();
        sys
    }

    /// Advance time by `dt` real seconds and recompute all state.
    pub fn update(&mut self, dt: f32) {
        // Advance time (if not paused)
        if !self.config.time_paused {
            self.time.advance(dt, self.config.day_length_seconds);
        }

        // Advance weather transitions
        self.weather.update(dt);

        // Preserve wind accumulated offset across recomputes
        let prev_offset = self.state.wind.accumulated_offset;

        self.recompute_state();

        // Wind computation
        let wind_config = &self.config.wind;
        let weather_mods = self.weather.current_modifiers();
        let base_speed = wind_config.base_speed * weather_mods.wind_speed_multiplier;
        let gust = (self.time.hour() * wind_config.gust_frequency * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        let gust_factor = gust * wind_config.gust_strength * weather_mods.wind_gust_multiplier;
        let speed = base_speed * (1.0 + gust_factor);

        let dir_2d = glam::Vec2::from(wind_config.base_direction).normalize_or_zero();
        let direction = glam::Vec3::new(dir_2d.x, 0.0, dir_2d.y);

        self.state.wind.direction = direction.to_array();
        self.state.wind.speed = speed;
        self.state.wind.gust_factor = gust_factor;
        // Accumulate offset for cloud movement
        self.state.wind.accumulated_offset = [
            prev_offset[0] + direction.x * speed * dt,
            prev_offset[1], // No vertical wind for clouds
            prev_offset[2] + direction.z * speed * dt,
        ];
    }

    /// Current atmosphere state (CPU-side).
    #[inline]
    pub fn state(&self) -> &AtmosphereState {
        &self.state
    }

    /// Build a GPU-ready uniform from current state.
    pub fn uniform(&self) -> AtmosphereUniform {
        AtmosphereUniform::from(&self.state)
    }

    /// Immutable reference to the configuration.
    #[inline]
    pub fn config(&self) -> &AtmosphereConfig {
        &self.config
    }

    /// Mutable reference to the configuration.
    ///
    /// After modifying, call [`update`](Self::update) (even with dt=0) to
    /// recompute state.
    #[inline]
    pub fn config_mut(&mut self) -> &mut AtmosphereConfig {
        &mut self.config
    }

    /// Set time of day and immediately recompute state.
    pub fn set_time(&mut self, hour: f32) {
        self.time.set(hour);
        self.recompute_state();
    }

    /// Begin transitioning to a new weather preset.
    pub fn set_weather(&mut self, preset: WeatherPreset) {
        self.weather.set_target(preset, self.config.weather.transition_duration);
    }

    /// Get the current weather preset.
    #[inline]
    pub fn current_preset(&self) -> WeatherPreset {
        self.weather.current_preset()
    }

    /// Current sun direction (normalized, world-space).
    #[inline]
    pub fn sun_direction(&self) -> glam::Vec3 {
        glam::Vec3::from(self.state.sun_direction)
    }

    /// Whether the sun intensity is effectively zero (nighttime).
    #[inline]
    pub fn is_night(&self) -> bool {
        self.state.sun_intensity < 0.05
    }

    /// Whether the current time is in the dawn range.
    #[inline]
    pub fn is_dawn(&self) -> bool {
        self.time.is_dawn()
    }

    /// Whether the current time is in the dusk range.
    #[inline]
    pub fn is_dusk(&self) -> bool {
        self.time.is_dusk()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Recompute all state from current time, config, and weather.
    fn recompute_state(&mut self) {
        let hour = self.time.hour();
        let weather_mods = self.weather.current_modifiers();

        // Sun
        let sun_dir = compute_sun_direction(hour, self.config.latitude);
        let sun_color = self.config.sun_color_ramp.sample(hour);
        let ramp_intensity =
            self.config.sun_intensity_ramp.sample(hour) * weather_mods.sun_intensity_multiplier;
        // Zero out sun intensity when below the horizon, with smooth fade near horizon
        let horizon_factor = if sun_dir.y <= 0.0 {
            0.0
        } else {
            (sun_dir.y * 10.0).min(1.0) // smooth fade in first ~6° above horizon
        };
        let sun_intensity = ramp_intensity * horizon_factor;

        // Ambient
        let ambient_color = self.config.ambient_color_ramp.sample(hour);
        let ambient_intensity =
            self.config.ambient_intensity_ramp.sample(hour) * weather_mods.ambient_intensity_multiplier;

        // Sky
        let sky_zenith = self.config.sky_zenith_ramp.sample(hour);
        let sky_horizon = self.config.sky_horizon_ramp.sample(hour);

        // Fog
        let fog_color = self.config.fog_color_ramp.sample(hour);
        let fog_density = if self.config.fog.enabled {
            self.config.fog.distance_fog_density * weather_mods.fog_density_multiplier
        } else {
            0.0
        };

        // Moons
        let mut moon_dirs = [[0.0_f32; 3]; 4];
        let mut moon_colors = [[0.0_f32; 3]; 4];
        let mut moon_phases = [0.0_f32; 4];
        let mut moon_sizes = [0.0_f32; 4];
        let moon_count = self.config.moons.len().min(4) as u32;

        for (i, mcfg) in self.config.moons.iter().take(4).enumerate() {
            let dir = compute_moon_direction(hour, self.time.day_count(), mcfg);
            moon_dirs[i] = dir.to_array();
            moon_colors[i] = mcfg.color;
            moon_phases[i] = compute_moon_phase(self.time.day_count(), mcfg);
            moon_sizes[i] = mcfg.size;
        }

        // Primary directional light: pick sun or moon (whichever is brighter)
        // Moon light comes from the first moon only; secondary moons are cosmetic
        let (primary_light_dir, primary_light_col, primary_light_int) = if moon_count > 0 {
            let mcfg = &self.config.moons[0];
            let moon_dir = glam::Vec3::from(moon_dirs[0]);
            let phase = moon_phases[0];

            // Phase illumination: 0 at new moon (phase=0), 1 at full moon (phase=0.5)
            let phase_illumination =
                0.5 - 0.5 * (phase * 2.0 * std::f32::consts::PI).cos();

            // Moon only provides light when above the horizon
            let moon_horizon = if moon_dir.y <= 0.0 {
                0.0
            } else {
                (moon_dir.y * 10.0).min(1.0)
            };

            let moon_intensity = mcfg.brightness * phase_illumination * moon_horizon;

            if sun_intensity > moon_intensity {
                (sun_dir.to_array(), sun_color, sun_intensity)
            } else {
                (moon_dirs[0], mcfg.color, moon_intensity)
            }
        } else {
            (sun_dir.to_array(), sun_color, sun_intensity)
        };

        self.state = AtmosphereState {
            time_of_day: hour,
            day_count: self.time.day_count(),
            sun_direction: sun_dir.to_array(),
            sun_color,
            sun_intensity,
            ambient_color,
            ambient_intensity,
            sky_zenith_color: sky_zenith,
            sky_horizon_color: sky_horizon,
            sky_intensity: 1.0,
            fog_color,
            fog_density,
            fog_height_falloff: self.config.fog.height_fog_falloff,
            fog_height_base: self.config.fog.height_fog_base,
            fog_inscattering: self.config.fog.inscattering_intensity,
            primary_light_direction: primary_light_dir,
            primary_light_color: primary_light_col,
            primary_light_intensity: primary_light_int,
            moon_directions: moon_dirs,
            moon_colors,
            moon_phases,
            moon_sizes,
            moon_count,
            cloud_coverage: weather_mods.cloud_coverage,
            cloud_density: weather_mods.cloud_coverage,
            precipitation_intensity: weather_mods.precipitation_intensity,
            weather_wetness: weather_mods.wetness,
            wind: WindState::default(),
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_at_time_10() {
        let sys = AtmosphereSystem::new(AtmosphereConfig::default());
        let s = sys.state();

        // Must closely match current hardcoded values
        assert!(
            (s.sun_color[0] - 1.0).abs() < 0.01,
            "sun_color[0] = {} expected ~1.0",
            s.sun_color[0]
        );
        assert!(
            (s.sun_color[1] - 0.98).abs() < 0.01,
            "sun_color[1] = {} expected ~0.98",
            s.sun_color[1]
        );
        assert!(
            (s.sun_color[2] - 0.95).abs() < 0.01,
            "sun_color[2] = {} expected ~0.95",
            s.sun_color[2]
        );

        assert!(
            (s.sun_intensity - 1.5).abs() < 0.01,
            "sun_intensity = {} expected ~1.5",
            s.sun_intensity
        );

        assert!(
            (s.ambient_color[0] - 0.03).abs() < 0.005,
            "ambient[0] = {} expected ~0.03",
            s.ambient_color[0]
        );
        assert!(
            (s.ambient_color[1] - 0.04).abs() < 0.005,
            "ambient[1] = {} expected ~0.04",
            s.ambient_color[1]
        );
        assert!(
            (s.ambient_color[2] - 0.05).abs() < 0.005,
            "ambient[2] = {} expected ~0.05",
            s.ambient_color[2]
        );
    }

    #[test]
    fn test_sun_direction_at_default() {
        let sys = AtmosphereSystem::new(AtmosphereConfig::default());
        let dir = sys.sun_direction();

        // At time=10.0, sun should be high in the sky (sinusoidal arc)
        // hour_angle = (10-12) * 15° = -30°, altitude = sin((10-6)/12 * PI) * 90°
        let hour_angle = (10.0_f32 - 12.0) * 15.0_f32.to_radians();
        let day_angle = (10.0_f32 - 6.0) * std::f32::consts::PI / 12.0;
        let altitude = (day_angle.sin() * 90.0_f32).to_radians();
        let reference = glam::Vec3::new(
            hour_angle.sin() * altitude.cos(),
            altitude.sin(),
            hour_angle.cos() * altitude.cos(),
        )
        .normalize();

        assert!(
            (dir - reference).length() < 1e-5,
            "Sun dir {dir:?} != reference {reference:?}"
        );
    }

    #[test]
    fn test_set_time_recomputes() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        sys.set_time(6.0);
        assert!((sys.state().time_of_day - 6.0).abs() < 1e-4);

        // Dawn should have lower sun intensity
        assert!(sys.state().sun_intensity < 1.0);
    }

    #[test]
    fn test_is_night_at_midnight() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        sys.set_time(0.0);
        assert!(sys.is_night());
    }

    #[test]
    fn test_is_dawn() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        sys.set_time(6.0);
        assert!(sys.is_dawn());
        assert!(!sys.is_dusk());
    }

    #[test]
    fn test_is_dusk() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        sys.set_time(18.0);
        assert!(sys.is_dusk());
        assert!(!sys.is_dawn());
    }

    #[test]
    fn test_weather_affects_state() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        let clear_intensity = sys.state().sun_intensity;

        sys.set_weather(WeatherPreset::Overcast);
        // Advance past transition
        for _ in 0..100 {
            sys.update(1.0);
        }
        let overcast_intensity = sys.state().sun_intensity;

        assert!(
            overcast_intensity < clear_intensity,
            "Overcast ({overcast_intensity}) should be dimmer than clear ({clear_intensity})"
        );
    }

    #[test]
    fn test_uniform_from_system() {
        let sys = AtmosphereSystem::new(AtmosphereConfig::default());
        let u = sys.uniform();
        assert_eq!(u.sun_intensity, sys.state().sun_intensity);
        assert_eq!(u.sun_color, sys.state().sun_color);
    }

    #[test]
    fn test_moon_count() {
        let sys = AtmosphereSystem::new(AtmosphereConfig::default());
        assert_eq!(sys.state().moon_count, 1);
    }

    #[test]
    fn test_time_paused_no_advance() {
        let mut sys = AtmosphereSystem::new(AtmosphereConfig::default());
        let t0 = sys.state().time_of_day;
        sys.update(10.0);
        let t1 = sys.state().time_of_day;
        assert!(
            (t0 - t1).abs() < 1e-6,
            "Time should not advance when paused: {t0} vs {t1}"
        );
    }

    #[test]
    fn test_time_advances_when_unpaused() {
        let mut config = AtmosphereConfig::default();
        config.time_paused = false;
        let mut sys = AtmosphereSystem::new(config);
        let t0 = sys.state().time_of_day;
        sys.update(10.0);
        let t1 = sys.state().time_of_day;
        assert!(
            t1 > t0,
            "Time should advance when unpaused: {t0} vs {t1}"
        );
    }
}
