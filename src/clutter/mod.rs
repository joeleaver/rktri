//! Procedural ground clutter system.
//!
//! Manages ground clutter (rocks, sticks, fallen logs, etc.) using SVO masks.
//! Clutter is placed during world generation based on biome and terrain properties.

pub mod config;
pub mod params;
pub mod profile;
pub mod library;

pub use config::ClutterConfig;
pub use params::ClutterParams;
pub use profile::{ClutterProfile, ClutterCell, ClutterObject, ClutterProfileTable};
pub use library::{ClutterLibrary, ClutterData};

/// Manages clutter configuration.
pub struct ClutterSystem {
    config: ClutterConfig,
}

impl ClutterSystem {
    pub fn new(config: ClutterConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ClutterConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ClutterConfig {
        &mut self.config
    }

    /// Build GPU-ready params from current config.
    pub fn build_params(&self) -> ClutterParams {
        ClutterParams::from_config(
            self.config.enabled,
            self.config.max_distance,
            self.config.fade_start,
        )
    }
}

impl Default for ClutterSystem {
    fn default() -> Self {
        Self::new(ClutterConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let sys = ClutterSystem::default();
        assert!(sys.config().enabled);
        assert!(sys.config().max_distance > sys.config().fade_start);
    }

    #[test]
    fn test_build_params() {
        let sys = ClutterSystem::default();
        let params = sys.build_params();
        assert_eq!(params.enabled, 1);
    }

    #[test]
    fn test_disabled() {
        let mut sys = ClutterSystem::default();
        sys.config_mut().enabled = false;
        let params = sys.build_params();
        assert_eq!(params.enabled, 0);
    }
}
