//! Moon position and phase calculation (stub).

use crate::atmosphere::config::MoonConfig;

/// Compute the direction to a moon given the current hour, day, and config.
///
/// The moon traces an arc opposite the sun: rises at ~18:00, peaks at midnight,
/// sets at ~6:00. The orbital period causes the rise/set times to drift slowly
/// over the month, and inclination tilts the orbit plane slightly.
pub fn compute_moon_direction(hour: f32, day: u32, config: &MoonConfig) -> glam::Vec3 {
    // Base arc: 12 hours offset from sun (rises at sunset, peaks at midnight)
    let hour_angle = hour * 15.0_f32.to_radians();
    let day_angle = (hour - 18.0) * std::f32::consts::PI / 12.0;
    let altitude = (day_angle.sin() * 80.0_f32).to_radians(); // max 80° (not quite zenith)

    // Orbital drift: the moon's position shifts slightly each day
    let orbit_fraction = (day as f32 + config.phase_offset) / config.orbit_period_days;
    let incl = config.orbit_inclination.to_radians();
    let orbit_tilt = (orbit_fraction * std::f32::consts::TAU).sin() * incl;

    glam::Vec3::new(
        hour_angle.sin() * altitude.cos(),
        altitude.sin() + orbit_tilt.sin() * 0.1,
        hour_angle.cos() * altitude.cos(),
    )
    .normalize()
}

/// Compute the moon phase (0.0 = new, 0.5 = full, 1.0 = new again).
pub fn compute_moon_phase(day: u32, config: &MoonConfig) -> f32 {
    let cycle = (day as f32 + config.phase_offset) / config.orbit_period_days;
    cycle.fract()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_moon() -> MoonConfig {
        MoonConfig::default()
    }

    #[test]
    fn test_phase_cycles() {
        let cfg = default_moon();
        // With phase_offset=14.75, day 0 gives (14.75/29.5) = 0.5 (full moon)
        let p0 = compute_moon_phase(0, &cfg);
        assert!((p0 - 0.5).abs() < 1e-4, "Day 0 phase = {p0}, expected ~0.5 (full moon)");

        // Half period later = new moon
        let p_half = compute_moon_phase(15, &cfg);
        // (15 + 14.75) / 29.5 = 29.75/29.5 ≈ 1.008 → fract ≈ 0.008 (near new moon)
        assert!(p_half < 0.05 || p_half > 0.95, "Day 15 phase = {p_half}, expected near 0/1 (new moon)");
    }

    #[test]
    fn test_direction_is_normalized() {
        let cfg = default_moon();
        for day in 0..30 {
            for hour in [0.0, 6.0, 12.0, 18.0, 23.0] {
                let dir = compute_moon_direction(hour, day, &cfg);
                assert!((dir.length() - 1.0).abs() < 1e-3,
                    "Moon dir not normalized at day {day} hour {hour}: len={}",
                    dir.length());
            }
        }
    }

    #[test]
    fn test_moon_above_horizon_at_midnight() {
        let cfg = default_moon();
        let dir = compute_moon_direction(0.0, 0, &cfg);
        assert!(dir.y > 0.5, "Moon should be high at midnight, Y={}", dir.y);
    }

    #[test]
    fn test_moon_below_horizon_at_noon() {
        let cfg = default_moon();
        let dir = compute_moon_direction(12.0, 0, &cfg);
        assert!(dir.y < 0.0, "Moon should be below horizon at noon, Y={}", dir.y);
    }
}
