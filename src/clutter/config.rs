//! Clutter system configuration.

use crate::clutter::profile::ClutterProfileTable;

/// Configuration for the clutter system.
#[derive(Clone, Debug)]
pub struct ClutterConfig {
    /// Whether clutter rendering is enabled
    pub enabled: bool,
    /// Maximum distance to render clutter
    pub max_distance: f32,
    /// Distance at which clutter starts to fade
    pub fade_start: f32,
    /// Clutter profile table
    pub profile_table: ClutterProfileTable,
}

impl Default for ClutterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_distance: 60.0,
            fade_start: 40.0,
            profile_table: ClutterProfileTable::default(),
        }
    }
}

impl ClutterConfig {
    /// Create a new clutter config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a reference to the profile table.
    pub fn profile_table(&self) -> &ClutterProfileTable {
        &self.profile_table
    }

    /// Get a mutable reference to the profile table.
    pub fn profile_table_mut(&mut self) -> &mut ClutterProfileTable {
        &mut self.profile_table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ClutterConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.max_distance > cfg.fade_start);
    }

    #[test]
    fn test_config_disabled() {
        let mut cfg = ClutterConfig::default();
        cfg.enabled = false;
        assert!(!cfg.enabled);
    }
}
