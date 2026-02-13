//! Procedural shader grass system.
//!
//! Generates grass blades on-the-fly during SVO raytracing. Grass placement
//! is driven by `MaskOctree<GrassCell>` per chunk, with per-profile
//! rendering parameters (height, width, density, color, sway).

pub mod config;
pub mod params;
pub mod profile;

pub use config::GrassConfig;
pub use params::GrassParams;
pub use profile::{GrassProfile, GrassCell, GpuGrassProfile, GrassProfileDef, GrassProfileTable};

use crate::atmosphere::state::WindState;

/// Manages grass configuration and builds per-frame GPU params.
pub struct GrassSystem {
    config: GrassConfig,
    profile_table: GrassProfileTable,
}

impl GrassSystem {
    pub fn new(config: GrassConfig) -> Self {
        Self {
            config,
            profile_table: GrassProfileTable::default(),
        }
    }

    pub fn config(&self) -> &GrassConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut GrassConfig {
        &mut self.config
    }

    pub fn profile_table(&self) -> &GrassProfileTable {
        &self.profile_table
    }

    pub fn profile_table_mut(&mut self) -> &mut GrassProfileTable {
        &mut self.profile_table
    }

    /// Build GPU-ready params from current config, wind state, and elapsed time.
    pub fn build_params(&self, wind: &WindState, time: f32) -> GrassParams {
        GrassParams {
            enabled: u32::from(self.config.enabled),
            max_distance: self.config.max_distance,
            fade_start: self.config.fade_start,
            time,
            wind_direction: wind.direction,
            wind_speed: wind.speed,
            profile_count: self.profile_table.len() as u32,
            _pad: [0.0; 3],
        }
    }

    /// Produce GPU profile data for upload.
    pub fn profile_table_gpu_data(&self) -> Vec<GpuGrassProfile> {
        self.profile_table.gpu_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = GrassConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.max_distance > cfg.fade_start);
    }

    #[test]
    fn test_build_params() {
        let sys = GrassSystem::new(GrassConfig::default());
        let wind = WindState {
            direction: [1.0, 0.0, 0.0],
            speed: 3.0,
            gust_factor: 0.5,
            accumulated_offset: [0.0; 3],
        };
        let params = sys.build_params(&wind, 1.5);
        assert_eq!(params.enabled, 1);
        assert_eq!(params.wind_speed, 3.0);
        assert_eq!(params.time, 1.5);
        assert!(params.profile_count >= 6);
    }

    #[test]
    fn test_disabled() {
        let mut sys = GrassSystem::new(GrassConfig::default());
        sys.config_mut().enabled = false;
        let wind = WindState::default();
        let params = sys.build_params(&wind, 0.0);
        assert_eq!(params.enabled, 0);
    }

    #[test]
    fn test_profile_table_access() {
        let sys = GrassSystem::new(GrassConfig::default());
        assert_eq!(sys.profile_table().len(), 6);
        let gpu_data = sys.profile_table_gpu_data();
        assert_eq!(gpu_data.len(), 6);
    }
}
