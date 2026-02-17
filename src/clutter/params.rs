//! Clutter GPU parameters.

use bytemuck::{Pod, Zeroable};

/// GPU uniform parameters for clutter rendering.
/// Must match `ClutterParams` in WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClutterParams {
    /// Whether clutter is enabled (0 or 1)
    pub enabled: u32,
    /// Maximum render distance
    pub max_distance: f32,
    /// Distance at which clutter starts to fade
    pub fade_start: f32,
    /// Padding for alignment
    pub _pad: f32,
}

impl Default for ClutterParams {
    fn default() -> Self {
        Self {
            enabled: 1,
            max_distance: 60.0,
            fade_start: 40.0,
            _pad: 0.0,
        }
    }
}

impl ClutterParams {
    /// Create params from config.
    pub fn from_config(enabled: bool, max_distance: f32, fade_start: f32) -> Self {
        Self {
            enabled: u32::from(enabled),
            max_distance,
            fade_start,
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clutter_params_size() {
        assert_eq!(std::mem::size_of::<ClutterParams>(), 16);
    }

    #[test]
    fn test_clutter_params_alignment() {
        assert_eq!(std::mem::size_of::<ClutterParams>() % 16, 0);
    }

    #[test]
    fn test_default_params() {
        let params = ClutterParams::default();
        assert_eq!(params.enabled, 1);
        assert_eq!(params.max_distance, 60.0);
    }
}
