//! Grass configuration (user-facing global settings).
//!
//! Per-profile settings (density, height, width, color, sway, spacing, coverage)
//! are managed through `GrassProfileTable` in `profile.rs`.

/// User-facing grass configuration (global settings only).
#[derive(Clone, Debug)]
pub struct GrassConfig {
    /// Master on/off.
    pub enabled: bool,
    /// Maximum render distance in meters.
    pub max_distance: f32,
    /// Distance at which grass begins fading.
    pub fade_start: f32,
}

impl Default for GrassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_distance: 80.0,
            fade_start: 50.0,
        }
    }
}
