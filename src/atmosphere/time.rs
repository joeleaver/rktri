//! Time of day tracking with day counting.

/// Tracks the current hour within a 24-hour cycle plus elapsed day count.
#[derive(Clone, Debug)]
pub struct TimeOfDay {
    /// Current hour, in the range `[0.0, 24.0)`.
    hour: f32,
    /// Number of full days that have elapsed.
    day_count: u32,
}

impl TimeOfDay {
    /// Create a new time starting at the given hour.
    pub fn new(start_hour: f32) -> Self {
        Self {
            hour: start_hour.clamp(0.0, 24.0),
            day_count: 0,
        }
    }

    /// Advance real time by `dt_seconds`, converting via `day_length_seconds`
    /// (i.e. how many real seconds constitute one full in-game day).
    pub fn advance(&mut self, dt_seconds: f32, day_length_seconds: f32) {
        if day_length_seconds <= 0.0 {
            return;
        }
        let hours_per_second = 24.0 / day_length_seconds;
        self.hour += dt_seconds * hours_per_second;

        // Handle wrapping
        while self.hour >= 24.0 {
            self.hour -= 24.0;
            self.day_count += 1;
        }
        while self.hour < 0.0 {
            self.hour += 24.0;
            self.day_count = self.day_count.saturating_sub(1);
        }
    }

    /// Set the hour directly, clamping to `[0.0, 24.0]`.
    pub fn set(&mut self, hour: f32) {
        self.hour = hour.clamp(0.0, 24.0);
        // Normalize 24.0 to 0.0
        if self.hour >= 24.0 {
            self.hour = 0.0;
        }
    }

    /// Current hour in the range `[0.0, 24.0)`.
    #[inline]
    pub fn hour(&self) -> f32 {
        self.hour
    }

    /// Number of full days that have passed.
    #[inline]
    pub fn day_count(&self) -> u32 {
        self.day_count
    }

    /// Whether it is currently nighttime (sun fully below horizon).
    #[inline]
    pub fn is_night(&self) -> bool {
        self.hour < 5.5 || self.hour > 19.5
    }

    /// Whether it is currently dawn.
    #[inline]
    pub fn is_dawn(&self) -> bool {
        (5.0..7.5).contains(&self.hour)
    }

    /// Whether it is currently dusk.
    #[inline]
    pub fn is_dusk(&self) -> bool {
        (17.0..19.5).contains(&self.hour)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clamps() {
        let t = TimeOfDay::new(30.0);
        assert_eq!(t.hour(), 24.0_f32.min(24.0)); // clamped
        let t2 = TimeOfDay::new(-5.0);
        assert_eq!(t2.hour(), 0.0);
    }

    #[test]
    fn test_advance_basic() {
        let mut t = TimeOfDay::new(10.0);
        // day_length = 120s means 24h in 120s => 0.2h per second
        t.advance(1.0, 120.0);
        let expected = 10.0 + 24.0 / 120.0;
        assert!((t.hour() - expected).abs() < 1e-4);
        assert_eq!(t.day_count(), 0);
    }

    #[test]
    fn test_advance_wraps_day() {
        let mut t = TimeOfDay::new(23.0);
        // Advance 2 hours worth: 2.0 / (24.0/120.0) = 10 seconds at 120s day
        t.advance(10.0, 120.0); // 10 * 0.2 = 2 hours
        assert!((t.hour() - 1.0).abs() < 1e-4);
        assert_eq!(t.day_count(), 1);
    }

    #[test]
    fn test_advance_multiple_days() {
        let mut t = TimeOfDay::new(0.0);
        // 120s day, advance 360s = 3 full days
        t.advance(360.0, 120.0);
        assert!((t.hour() - 0.0).abs() < 1e-3);
        assert_eq!(t.day_count(), 3);
    }

    #[test]
    fn test_set() {
        let mut t = TimeOfDay::new(0.0);
        t.set(15.5);
        assert!((t.hour() - 15.5).abs() < 1e-6);
        t.set(25.0); // clamped -> 24.0 -> normalized to 0.0
        assert!((t.hour() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_periods() {
        let mut t = TimeOfDay::new(3.0);
        assert!(t.is_night());
        assert!(!t.is_dawn());
        assert!(!t.is_dusk());

        t.set(6.0);
        assert!(!t.is_night());
        assert!(t.is_dawn());

        t.set(12.0);
        assert!(!t.is_night());
        assert!(!t.is_dawn());
        assert!(!t.is_dusk());

        t.set(18.0);
        assert!(t.is_dusk());

        t.set(21.0);
        assert!(t.is_night());
    }

    #[test]
    fn test_zero_day_length_no_advance() {
        let mut t = TimeOfDay::new(10.0);
        t.advance(100.0, 0.0);
        assert!((t.hour() - 10.0).abs() < 1e-6);
    }
}
