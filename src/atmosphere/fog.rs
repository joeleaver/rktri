//! Fog calculation (stub).

use crate::atmosphere::config::FogConfig;

/// Compute the combined fog factor for a point at the given distance and height.
///
/// Returns a value in `[0.0, 1.0]` where 0 = no fog and 1 = fully fogged.
/// When fog is disabled, always returns 0.0.
pub fn compute_fog_factor(distance: f32, height: f32, config: &FogConfig) -> f32 {
    if !config.enabled {
        return 0.0;
    }

    // Distance-based exponential fog
    let distance_fog = if config.distance_fog_density > 0.0 {
        let d = (distance - config.distance_fog_start).max(0.0);
        1.0 - (-d * config.distance_fog_density).exp()
    } else {
        0.0
    };

    // Height-based exponential fog
    let height_fog = if config.height_fog_density > 0.0 {
        let h = (config.height_fog_base - height).max(0.0);
        let falloff = (-h * config.height_fog_falloff).exp();
        config.height_fog_density * (1.0 - falloff)
    } else {
        0.0
    };

    // Combine: max of both fog types, clamped
    (distance_fog + height_fog).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disabled_fog() -> FogConfig {
        FogConfig::default()
    }

    #[test]
    fn test_disabled_fog_returns_zero() {
        let cfg = disabled_fog();
        assert_eq!(compute_fog_factor(100.0, 50.0, &cfg), 0.0);
    }

    #[test]
    fn test_distance_fog_increases_with_distance() {
        let cfg = FogConfig {
            enabled: true,
            distance_fog_density: 0.01,
            distance_fog_start: 10.0,
            ..Default::default()
        };
        let f_near = compute_fog_factor(20.0, 0.0, &cfg);
        let f_far = compute_fog_factor(200.0, 0.0, &cfg);
        assert!(f_far > f_near);
    }

    #[test]
    fn test_fog_clamps_to_one() {
        let cfg = FogConfig {
            enabled: true,
            distance_fog_density: 1.0,
            distance_fog_start: 0.0,
            height_fog_density: 1.0,
            height_fog_falloff: 0.01,
            height_fog_base: 100.0,
            ..Default::default()
        };
        let f = compute_fog_factor(10000.0, 0.0, &cfg);
        assert!(f <= 1.0);
    }
}
