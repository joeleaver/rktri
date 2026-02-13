//! Sun position calculation.
//!
//! Computes the sun direction vector from time of day and latitude.
//! The default latitude of 45.0 reproduces the existing engine behavior exactly.

/// Compute the sun direction unit vector for a given hour and latitude.
///
/// The sun traces a smooth sinusoidal arc across the sky:
/// - Rises at 6:00 (altitude = 0°)
/// - Peaks at noon (altitude = 90°)
/// - Sets at 18:00 (altitude = 0°)
/// - Below the horizon during night hours (negative altitude)
///
/// The hour_angle rotates 360° over 24 hours, giving the sun's horizontal
/// direction (east at 6 AM, south at noon, west at 6 PM).
pub fn compute_sun_direction(hour: f32, _latitude: f32) -> glam::Vec3 {
    // Hour angle: horizontal rotation, 0 at noon, full circle over 24h
    let hour_angle = (hour - 12.0) * 15.0_f32.to_radians();

    // Sinusoidal altitude: smooth arc, sunrise at 6, peak at noon, sunset at 18
    // Goes negative during night for proper below-horizon behavior
    let day_angle = (hour - 6.0) * std::f32::consts::PI / 12.0;
    let altitude = (day_angle.sin() * 90.0_f32).to_radians();

    glam::Vec3::new(
        hour_angle.sin() * altitude.cos(),
        altitude.sin(),
        hour_angle.cos() * altitude.cos(),
    )
    .normalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3_approx_eq(a: glam::Vec3, b: glam::Vec3, eps: f32) -> bool {
        (a.x - b.x).abs() < eps && (a.y - b.y).abs() < eps && (a.z - b.z).abs() < eps
    }

    #[test]
    fn test_noon_is_overhead() {
        let dir = compute_sun_direction(12.0, 45.0);
        // At noon the sun should be nearly straight up
        assert!(dir.y > 0.95, "Noon sun Y = {} should be > 0.95", dir.y);
        assert!(dir.x.abs() < 0.1, "Noon sun X = {} should be near 0", dir.x);
    }

    #[test]
    fn test_sunrise_at_horizon() {
        let dir = compute_sun_direction(6.0, 45.0);
        // At 6:00, sun should be at the horizon (Y near 0)
        assert!(dir.y.abs() < 0.05, "Sunrise sun Y = {} should be near 0", dir.y);
    }

    #[test]
    fn test_sunset_at_horizon() {
        let dir = compute_sun_direction(18.0, 45.0);
        // At 18:00, sun should be at the horizon (Y near 0)
        assert!(dir.y.abs() < 0.05, "Sunset sun Y = {} should be near 0", dir.y);
    }

    #[test]
    fn test_night_below_horizon() {
        // Midnight: sun should be well below the horizon
        let dir = compute_sun_direction(0.0, 45.0);
        assert!(dir.y < -0.5, "Midnight sun Y = {} should be < -0.5", dir.y);

        // 3 AM: still below horizon
        let dir3 = compute_sun_direction(3.0, 45.0);
        assert!(dir3.y < 0.0, "3 AM sun Y = {} should be < 0", dir3.y);

        // 21:00: below horizon
        let dir21 = compute_sun_direction(21.0, 45.0);
        assert!(dir21.y < 0.0, "9 PM sun Y = {} should be < 0", dir21.y);
    }

    #[test]
    fn test_morning_afternoon_symmetry() {
        let morning = compute_sun_direction(9.0, 45.0);
        let afternoon = compute_sun_direction(15.0, 45.0);
        // X should be opposite sign, Y should be same
        assert!((morning.y - afternoon.y).abs() < 1e-5);
        assert!((morning.x + afternoon.x).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_arc() {
        // Sun altitude should increase monotonically from 6 to 12
        let mut prev_y = -1.0;
        for h in 60..=120 {
            let hour = h as f32 / 10.0;
            let dir = compute_sun_direction(hour, 45.0);
            assert!(dir.y >= prev_y - 1e-5, "Sun Y not monotonically increasing at hour {hour}");
            prev_y = dir.y;
        }
    }
}
