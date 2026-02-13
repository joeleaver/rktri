//! Weather state machine (stub).
//!
//! Manages transitions between weather presets with linear blending.

use serde::{Deserialize, Serialize};

use crate::atmosphere::config::WeatherPreset;

/// Multipliers applied to atmosphere parameters based on weather.
#[derive(Clone, Debug)]
pub struct WeatherModifiers {
    /// Cloud coverage `[0.0, 1.0]`.
    pub cloud_coverage: f32,
    /// Multiplier on fog density.
    pub fog_density_multiplier: f32,
    /// Multiplier on ambient intensity.
    pub ambient_intensity_multiplier: f32,
    /// Multiplier on sun intensity (0.0 = fully blocked).
    pub sun_intensity_multiplier: f32,
    /// Precipitation intensity `[0.0, 1.0]`.
    pub precipitation_intensity: f32,
    /// Surface wetness `[0.0, 1.0]`.
    pub wetness: f32,
    /// Applied to WindConfig.base_speed.
    pub wind_speed_multiplier: f32,
    /// Applied to WindConfig.gust_strength.
    pub wind_gust_multiplier: f32,
}

impl Default for WeatherModifiers {
    fn default() -> Self {
        Self {
            cloud_coverage: 0.0,
            fog_density_multiplier: 1.0,
            ambient_intensity_multiplier: 1.0,
            sun_intensity_multiplier: 1.0,
            precipitation_intensity: 0.0,
            wetness: 0.0,
            wind_speed_multiplier: 1.0,
            wind_gust_multiplier: 1.0,
        }
    }
}

impl WeatherModifiers {
    /// Linearly interpolate between two modifier sets.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            cloud_coverage: self.cloud_coverage
                + (other.cloud_coverage - self.cloud_coverage) * t,
            fog_density_multiplier: self.fog_density_multiplier
                + (other.fog_density_multiplier - self.fog_density_multiplier) * t,
            ambient_intensity_multiplier: self.ambient_intensity_multiplier
                + (other.ambient_intensity_multiplier - self.ambient_intensity_multiplier) * t,
            sun_intensity_multiplier: self.sun_intensity_multiplier
                + (other.sun_intensity_multiplier - self.sun_intensity_multiplier) * t,
            precipitation_intensity: self.precipitation_intensity
                + (other.precipitation_intensity - self.precipitation_intensity) * t,
            wetness: self.wetness + (other.wetness - self.wetness) * t,
            wind_speed_multiplier: self.wind_speed_multiplier
                + (other.wind_speed_multiplier - self.wind_speed_multiplier) * t,
            wind_gust_multiplier: self.wind_gust_multiplier
                + (other.wind_gust_multiplier - self.wind_gust_multiplier) * t,
        }
    }
}

/// Maps a [`WeatherPreset`] to its target modifier values.
fn preset_modifiers(preset: WeatherPreset) -> WeatherModifiers {
    match preset {
        WeatherPreset::Clear => WeatherModifiers {
            cloud_coverage: 0.0,
            fog_density_multiplier: 1.0,
            ambient_intensity_multiplier: 1.0,
            sun_intensity_multiplier: 1.0,
            precipitation_intensity: 0.0,
            wetness: 0.0,
            wind_speed_multiplier: 1.0,
            wind_gust_multiplier: 1.0,
        },
        WeatherPreset::PartlyCloudy => WeatherModifiers {
            cloud_coverage: 0.3,
            fog_density_multiplier: 1.0,
            ambient_intensity_multiplier: 0.9,
            sun_intensity_multiplier: 0.9,
            precipitation_intensity: 0.0,
            wetness: 0.0,
            wind_speed_multiplier: 1.2,
            wind_gust_multiplier: 1.2,
        },
        WeatherPreset::Overcast => WeatherModifiers {
            cloud_coverage: 0.8,
            fog_density_multiplier: 1.2,
            ambient_intensity_multiplier: 0.6,
            sun_intensity_multiplier: 0.3,
            precipitation_intensity: 0.0,
            wetness: 0.0,
            wind_speed_multiplier: 1.5,
            wind_gust_multiplier: 1.5,
        },
        WeatherPreset::Foggy => WeatherModifiers {
            cloud_coverage: 0.5,
            fog_density_multiplier: 3.0,
            ambient_intensity_multiplier: 0.7,
            sun_intensity_multiplier: 0.5,
            precipitation_intensity: 0.0,
            wetness: 0.2,
            wind_speed_multiplier: 0.5,
            wind_gust_multiplier: 0.5,
        },
        WeatherPreset::Rain => WeatherModifiers {
            cloud_coverage: 0.9,
            fog_density_multiplier: 1.5,
            ambient_intensity_multiplier: 0.5,
            sun_intensity_multiplier: 0.2,
            precipitation_intensity: 0.7,
            wetness: 0.8,
            wind_speed_multiplier: 2.0,
            wind_gust_multiplier: 2.0,
        },
        WeatherPreset::Snow => WeatherModifiers {
            cloud_coverage: 0.85,
            fog_density_multiplier: 1.3,
            ambient_intensity_multiplier: 0.6,
            sun_intensity_multiplier: 0.3,
            precipitation_intensity: 0.5,
            wetness: 0.3,
            wind_speed_multiplier: 1.5,
            wind_gust_multiplier: 1.0,
        },
        WeatherPreset::Storm => WeatherModifiers {
            cloud_coverage: 1.0,
            fog_density_multiplier: 2.0,
            ambient_intensity_multiplier: 0.3,
            sun_intensity_multiplier: 0.1,
            precipitation_intensity: 1.0,
            wetness: 1.0,
            wind_speed_multiplier: 4.0,
            wind_gust_multiplier: 3.0,
        },
    }
}

/// State machine that blends between weather presets over time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherStateMachine {
    current_preset: WeatherPreset,
    target_preset: WeatherPreset,
    /// Progress of the current transition, 0.0 to 1.0.
    blend_factor: f32,
    /// Duration of the current transition in seconds.
    transition_duration: f32,
}

impl WeatherStateMachine {
    /// Create a new weather state machine starting at the given preset.
    pub fn new(initial: WeatherPreset) -> Self {
        Self {
            current_preset: initial,
            target_preset: initial,
            blend_factor: 1.0, // fully arrived
            transition_duration: 0.0,
        }
    }

    /// Begin transitioning to a new preset over the given duration.
    pub fn set_target(&mut self, preset: WeatherPreset, transition_seconds: f32) {
        if preset != self.target_preset {
            self.current_preset = self.target_preset;
            self.target_preset = preset;
            self.blend_factor = 0.0;
            self.transition_duration = transition_seconds;
        }
    }

    /// Advance the transition by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if self.blend_factor < 1.0 {
            let step = if self.transition_duration > 0.0 {
                dt / self.transition_duration
            } else {
                1.0
            };
            self.blend_factor = (self.blend_factor + step).min(1.0);
        }
    }

    /// Get the current blended modifiers.
    pub fn current_modifiers(&self) -> WeatherModifiers {
        if self.blend_factor >= 1.0 {
            return preset_modifiers(self.target_preset);
        }
        let from = preset_modifiers(self.current_preset);
        let to = preset_modifiers(self.target_preset);
        from.lerp(&to, self.blend_factor)
    }

    /// Current (target) preset.
    pub fn current_preset(&self) -> WeatherPreset {
        self.target_preset
    }

    /// Whether we are mid-transition.
    pub fn is_transitioning(&self) -> bool {
        self.blend_factor < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_preset_returns_expected_modifiers() {
        let wsm = WeatherStateMachine::new(WeatherPreset::Clear);
        let m = wsm.current_modifiers();
        assert!((m.cloud_coverage - 0.0).abs() < 1e-6);
        assert!((m.fog_density_multiplier - 1.0).abs() < 1e-6);
        assert!((m.ambient_intensity_multiplier - 1.0).abs() < 1e-6);
        assert!((m.sun_intensity_multiplier - 1.0).abs() < 1e-6);
        assert!((m.precipitation_intensity - 0.0).abs() < 1e-6);
        assert!((m.wetness - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_storm_preset_returns_expected_modifiers() {
        let wsm = WeatherStateMachine::new(WeatherPreset::Storm);
        let m = wsm.current_modifiers();
        assert!((m.cloud_coverage - 1.0).abs() < 1e-6);
        assert!((m.fog_density_multiplier - 2.0).abs() < 1e-6);
        assert!((m.ambient_intensity_multiplier - 0.3).abs() < 1e-6);
        assert!((m.sun_intensity_multiplier - 0.1).abs() < 1e-6);
        assert!((m.precipitation_intensity - 1.0).abs() < 1e-6);
        assert!((m.wetness - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transition_clear_to_rain_blends_correctly() {
        let mut wsm = WeatherStateMachine::new(WeatherPreset::Clear);
        wsm.set_target(WeatherPreset::Rain, 10.0);

        // At t=0, should still be Clear
        let m0 = wsm.current_modifiers();
        assert!((m0.sun_intensity_multiplier - 1.0).abs() < 1e-4);
        assert!((m0.precipitation_intensity - 0.0).abs() < 1e-4);

        // Advance halfway
        wsm.update(5.0);
        let m_half = wsm.current_modifiers();
        // Should be halfway between Clear (1.0) and Rain (0.2)
        assert!((m_half.sun_intensity_multiplier - 0.6).abs() < 1e-4);
        // Should be halfway between Clear (0.0) and Rain (0.7)
        assert!((m_half.precipitation_intensity - 0.35).abs() < 1e-4);

        // Advance to completion
        wsm.update(5.0);
        let m_end = wsm.current_modifiers();
        assert!((m_end.sun_intensity_multiplier - 0.2).abs() < 1e-4);
        assert!((m_end.precipitation_intensity - 0.7).abs() < 1e-4);
    }

    #[test]
    fn test_transition_completes_after_duration() {
        let mut wsm = WeatherStateMachine::new(WeatherPreset::Clear);
        wsm.set_target(WeatherPreset::Overcast, 10.0);

        assert!(wsm.is_transitioning());

        // Advance past the duration
        wsm.update(10.0);
        assert!(!wsm.is_transitioning());
        assert_eq!(wsm.current_preset(), WeatherPreset::Overcast);

        // Verify we're at the target modifiers
        let m = wsm.current_modifiers();
        assert!((m.sun_intensity_multiplier - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_rapid_transitions() {
        let mut wsm = WeatherStateMachine::new(WeatherPreset::Clear);

        // Start transition to Rain
        wsm.set_target(WeatherPreset::Rain, 10.0);
        wsm.update(3.0); // 30% through

        assert!(wsm.is_transitioning());

        // Interrupt with transition to Storm
        wsm.set_target(WeatherPreset::Storm, 10.0);

        // Should restart from current state
        let m2 = wsm.current_modifiers();
        // At t=0 of new transition, we should be at the previous target (Rain)
        assert!((m2.sun_intensity_multiplier - 0.2).abs() < 1e-4);

        // Advance the Storm transition halfway
        wsm.update(5.0);
        let m3 = wsm.current_modifiers();
        // Should be between Rain (0.2) and Storm (0.1)
        assert!(m3.sun_intensity_multiplier < 0.2);
        assert!(m3.sun_intensity_multiplier > 0.1);

        // Complete the transition
        wsm.update(5.0);
        assert!(!wsm.is_transitioning());
        assert_eq!(wsm.current_preset(), WeatherPreset::Storm);
    }

    #[test]
    fn test_no_transition_when_same_preset() {
        let mut wsm = WeatherStateMachine::new(WeatherPreset::Rain);
        wsm.set_target(WeatherPreset::Rain, 10.0);
        assert!(!wsm.is_transitioning());
    }
}
