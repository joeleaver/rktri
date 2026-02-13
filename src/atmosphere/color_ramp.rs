//! Generic keyframe interpolation for atmosphere parameters.
//!
//! [`ColorRamp`] provides time-based interpolation over a 24-hour cycle with
//! proper wrapping around midnight. Used for sun color, ambient light, sky
//! colors, fog color, and any other parameter that varies with time of day.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Lerp trait
// ---------------------------------------------------------------------------

/// Trait for types that can be linearly interpolated.
pub trait Lerp: Clone {
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for [f32; 3] {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0] + (other[0] - self[0]) * t,
            self[1] + (other[1] - self[1]) * t,
            self[2] + (other[2] - self[2]) * t,
        ]
    }
}

impl Lerp for [f32; 4] {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0] + (other[0] - self[0]) * t,
            self[1] + (other[1] - self[1]) * t,
            self[2] + (other[2] - self[2]) * t,
            self[3] + (other[3] - self[3]) * t,
        ]
    }
}

// ---------------------------------------------------------------------------
// ColorRamp
// ---------------------------------------------------------------------------

/// Keyframe-based value ramp with wrapping support for a 0-24 hour cycle.
///
/// Keys are `(time, value)` pairs sorted by time. Sampling at any time `t`
/// returns a linearly interpolated value between the surrounding keys,
/// wrapping correctly around midnight (24.0 == 0.0).
#[derive(Clone, Debug)]
pub struct ColorRamp<T: Lerp> {
    keys: Vec<(f32, T)>,
}

impl<T: Lerp> ColorRamp<T> {
    /// Create a new ramp from unsorted keys. Keys are sorted by time.
    pub fn new(mut keys: Vec<(f32, T)>) -> Self {
        keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { keys }
    }

    /// Create a constant ramp that always returns the same value.
    pub fn constant(value: T) -> Self {
        Self {
            keys: vec![(0.0, value)],
        }
    }

    /// Sample the ramp at time `t` (0.0 to 24.0), with wrapping.
    pub fn sample(&self, t: f32) -> T {
        assert!(!self.keys.is_empty(), "ColorRamp must have at least one key");

        if self.keys.len() == 1 {
            return self.keys[0].1.clone();
        }

        // Wrap t into [0, 24)
        let t = ((t % 24.0) + 24.0) % 24.0;

        // Find the two keys that bracket t
        // keys are sorted by time ascending
        let n = self.keys.len();

        // Find first key with time > t
        let upper_idx = self.keys.iter().position(|k| k.0 > t);

        match upper_idx {
            Some(0) => {
                // t is before the first key -> wrap: interpolate between last key and first key
                let (t_a, ref v_a) = self.keys[n - 1];
                let (t_b, ref v_b) = self.keys[0];
                let span = (t_b + 24.0) - t_a;
                if span < 1e-6 {
                    return v_a.clone();
                }
                let frac = (t - t_a + 24.0) / span;
                v_a.lerp(v_b, frac)
            }
            Some(idx) => {
                // Normal case: t is between keys[idx-1] and keys[idx]
                let (t_a, ref v_a) = self.keys[idx - 1];
                let (t_b, ref v_b) = self.keys[idx];
                let span = t_b - t_a;
                if span < 1e-6 {
                    return v_a.clone();
                }
                let frac = (t - t_a) / span;
                v_a.lerp(v_b, frac)
            }
            None => {
                // t is >= all keys -> wrap: interpolate between last key and first key
                let (t_a, ref v_a) = self.keys[n - 1];
                let (t_b, ref v_b) = self.keys[0];
                let span = (t_b + 24.0) - t_a;
                if span < 1e-6 {
                    return v_a.clone();
                }
                let frac = (t - t_a) / span;
                v_a.lerp(v_b, frac)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Serde support
// ---------------------------------------------------------------------------

impl<T: Lerp + Serialize> Serialize for ColorRamp<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.keys.serialize(serializer)
    }
}

impl<'de, T: Lerp + Deserialize<'de>> Deserialize<'de> for ColorRamp<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let keys = Vec::<(f32, T)>::deserialize(deserializer)?;
        Ok(Self::new(keys))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_eq_3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    #[test]
    fn test_single_key_returns_constant() {
        let ramp = ColorRamp::constant(0.5_f32);
        assert!(approx_eq_f32(ramp.sample(0.0), 0.5, 1e-6));
        assert!(approx_eq_f32(ramp.sample(12.0), 0.5, 1e-6));
        assert!(approx_eq_f32(ramp.sample(23.9), 0.5, 1e-6));
    }

    #[test]
    fn test_basic_interpolation_f32() {
        let ramp = ColorRamp::new(vec![(0.0, 0.0_f32), (24.0, 24.0)]);
        // At midpoint
        assert!(approx_eq_f32(ramp.sample(12.0), 12.0, 1e-4));
        // At quarter
        assert!(approx_eq_f32(ramp.sample(6.0), 6.0, 1e-4));
    }

    #[test]
    fn test_basic_interpolation_vec3() {
        let ramp = ColorRamp::new(vec![
            (0.0, [0.0_f32, 0.0, 0.0]),
            (12.0, [1.0, 1.0, 1.0]),
        ]);
        let mid = ramp.sample(6.0);
        assert!(approx_eq_3(mid, [0.5, 0.5, 0.5], 1e-4));
    }

    #[test]
    fn test_wrapping_around_midnight() {
        // Key at 22:00 = 0.0, key at 4:00 = 1.0
        // The span across midnight is 6 hours (22->24->4)
        let ramp = ColorRamp::new(vec![(4.0, 1.0_f32), (22.0, 0.0)]);

        // At 22:00 -> should be 0.0 (at key)
        assert!(approx_eq_f32(ramp.sample(22.0), 0.0, 1e-4));
        // At 4:00 -> should be 1.0 (at key)
        assert!(approx_eq_f32(ramp.sample(4.0), 1.0, 1e-4));
        // At 1:00 -> 3 hours into the 6 hour span from 22->4, so t=0.5
        assert!(approx_eq_f32(ramp.sample(1.0), 0.5, 1e-4));
        // At 23:00 -> 1 hour into the 6 hour span, t=1/6
        let expected = 1.0 / 6.0;
        assert!(approx_eq_f32(ramp.sample(23.0), expected, 1e-4));
    }

    #[test]
    fn test_multi_key_ramp() {
        let ramp = ColorRamp::new(vec![
            (0.0, 0.0_f32),
            (6.0, 0.5),
            (12.0, 1.0),
            (18.0, 0.5),
        ]);
        // Exact key values
        assert!(approx_eq_f32(ramp.sample(0.0), 0.0, 1e-4));
        assert!(approx_eq_f32(ramp.sample(6.0), 0.5, 1e-4));
        assert!(approx_eq_f32(ramp.sample(12.0), 1.0, 1e-4));
        assert!(approx_eq_f32(ramp.sample(18.0), 0.5, 1e-4));
        // Midpoints
        assert!(approx_eq_f32(ramp.sample(3.0), 0.25, 1e-4));
        assert!(approx_eq_f32(ramp.sample(9.0), 0.75, 1e-4));
    }

    #[test]
    fn test_wrapping_vec3() {
        let ramp = ColorRamp::new(vec![
            (22.0, [1.0_f32, 0.0, 0.0]),
            (2.0, [0.0, 0.0, 1.0]),
        ]);
        // Midpoint across midnight at 0:00 (2 hours into 4-hour span)
        let mid = ramp.sample(0.0);
        assert!(approx_eq_3(mid, [0.5, 0.0, 0.5], 1e-4));
    }

    #[test]
    fn test_negative_time_wraps() {
        let ramp = ColorRamp::new(vec![(0.0, 0.0_f32), (12.0, 1.0)]);
        // -1.0 should wrap to 23.0
        let val = ramp.sample(-1.0);
        // At 23.0, wrapping segment from 12->0 (span 12), at offset 11 => t = 11/12
        let expected = 1.0 + (0.0 - 1.0) * (11.0 / 12.0); // 1 - 11/12 = 1/12
        assert!(approx_eq_f32(val, expected, 1e-4));
    }

    #[test]
    fn test_sample_at_exact_keys() {
        let ramp = ColorRamp::new(vec![
            (6.0, 1.0_f32),
            (12.0, 2.0),
            (18.0, 3.0),
        ]);
        assert!(approx_eq_f32(ramp.sample(6.0), 1.0, 1e-6));
        assert!(approx_eq_f32(ramp.sample(12.0), 2.0, 1e-6));
        assert!(approx_eq_f32(ramp.sample(18.0), 3.0, 1e-6));
    }
}
