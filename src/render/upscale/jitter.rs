//! Jitter patterns for temporal anti-aliasing
//!
//! Implements Halton sequence generation for stable, low-discrepancy jitter patterns
//! used in temporal anti-aliasing and upscaling techniques.

use glam::Mat4;

/// Halton sequence generator for stable jitter patterns
///
/// Generates low-discrepancy 2D points using Halton sequences with bases 2 and 3.
/// This provides better temporal stability than random jitter while maintaining
/// good spatial coverage.
pub struct HaltonSequence {
    index: u32,
    current_x: f32,
    current_y: f32,
}

impl HaltonSequence {
    /// Create a new Halton sequence starting at index 0
    pub fn new() -> Self {
        let mut seq = Self {
            index: 0,
            current_x: 0.0,
            current_y: 0.0,
        };
        seq.compute_current();
        seq
    }

    /// Compute the current Halton values
    fn compute_current(&mut self) {
        // Use index + 1 to avoid (0,0) as first sample
        self.current_x = halton(self.index + 1, 2);
        self.current_y = halton(self.index + 1, 3);
    }

    /// Get the current jitter value without advancing
    ///
    /// Returns (x, y) in the range [0, 1]
    pub fn current(&self) -> (f32, f32) {
        (self.current_x, self.current_y)
    }

    /// Advance to the next sample in the sequence
    ///
    /// Returns the new (x, y) jitter values in the range [0, 1]
    pub fn next(&mut self) -> (f32, f32) {
        self.index += 1;
        self.compute_current();
        self.current()
    }

    /// Reset the sequence to the beginning
    pub fn reset(&mut self) {
        self.index = 0;
        self.compute_current();
    }

    /// Get the current sequence index
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl Default for HaltonSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a Halton number for the given index and base
///
/// The Halton sequence is a low-discrepancy sequence used for quasi-Monte Carlo methods.
/// It provides better spatial coverage than random sampling while being deterministic.
///
/// # Arguments
/// * `index` - Sequence index (0-based)
/// * `base` - Prime number base (typically 2, 3, 5, 7, etc.)
///
/// # Returns
/// A value in the range [0, 1]
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f32;

    while index > 0 {
        result += f * (index % base) as f32;
        index /= base;
        f /= base as f32;
    }

    result
}

/// Convert normalized jitter to pixel offsets
///
/// # Arguments
/// * `jitter` - Normalized jitter (x, y) in range [0, 1] or [-0.5, 0.5]
/// * `width` - Render width in pixels
/// * `height` - Render height in pixels
///
/// # Returns
/// Pixel offsets (x, y)
pub fn jitter_to_pixels(jitter: (f32, f32), _width: u32, _height: u32) -> (f32, f32) {
    (jitter.0, jitter.1)
}

/// Apply jitter to a projection matrix
///
/// Modifies the projection matrix to shift the image by the given jitter offset.
/// This is used for temporal anti-aliasing by rendering each frame with a slight
/// sub-pixel offset.
///
/// # Arguments
/// * `proj` - Original projection matrix
/// * `jitter_pixels` - Jitter offset in pixels (typically -0.5 to +0.5)
/// * `width` - Render width in pixels
/// * `height` - Render height in pixels
///
/// # Returns
/// Modified projection matrix with jitter applied
pub fn apply_jitter_to_projection(
    mut proj: Mat4,
    jitter_pixels: (f32, f32),
    width: u32,
    height: u32,
) -> Mat4 {
    // Convert pixel offset to NDC space
    let jitter_ndc_x = (jitter_pixels.0 * 2.0) / width as f32;
    let jitter_ndc_y = (jitter_pixels.1 * 2.0) / height as f32;

    // Apply offset to projection matrix
    // This shifts the projection by modifying the translation components
    proj.w_axis.x += jitter_ndc_x;
    proj.w_axis.y += jitter_ndc_y;

    proj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_base2() {
        // Base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
        assert_eq!(halton(4, 2), 0.125);
    }

    #[test]
    fn test_halton_base3() {
        // Base 3: 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9, ...
        let h1 = halton(1, 3);
        let h2 = halton(2, 3);
        assert!((h1 - 1.0/3.0).abs() < 1e-6);
        assert!((h2 - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_halton_range() {
        // All values should be in [0, 1]
        for i in 0..100 {
            let h2 = halton(i, 2);
            let h3 = halton(i, 3);
            assert!(h2 >= 0.0 && h2 <= 1.0);
            assert!(h3 >= 0.0 && h3 <= 1.0);
        }
    }

    #[test]
    fn test_halton_sequence_new() {
        let seq = HaltonSequence::new();
        let (x, y) = seq.current();

        // First value should be halton(1, 2) and halton(1, 3)
        assert_eq!(x, 0.5);
        assert!((y - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_halton_sequence_advance() {
        let mut seq = HaltonSequence::new();

        let first = seq.current();
        let second = seq.next();
        let third = seq.next();

        assert_ne!(first, second);
        assert_ne!(second, third);
        assert_ne!(first, third);
    }

    #[test]
    fn test_halton_sequence_reset() {
        let mut seq = HaltonSequence::new();
        let initial = seq.current();

        seq.next();
        seq.next();
        seq.next();

        seq.reset();
        assert_eq!(seq.current(), initial);
        assert_eq!(seq.index(), 0);
    }

    #[test]
    fn test_halton_sequence_index() {
        let mut seq = HaltonSequence::new();
        assert_eq!(seq.index(), 0);

        seq.next();
        assert_eq!(seq.index(), 1);

        seq.next();
        assert_eq!(seq.index(), 2);
    }

    #[test]
    fn test_jitter_to_pixels() {
        let jitter = (0.5, -0.5);
        let pixels = jitter_to_pixels(jitter, 1920, 1080);

        // Currently just passes through
        assert_eq!(pixels, (0.5, -0.5));
    }

    #[test]
    fn test_apply_jitter_to_projection() {
        let proj = Mat4::IDENTITY;
        let jittered = apply_jitter_to_projection(proj, (0.5, 0.5), 1920, 1080);

        // Should modify the translation components
        assert_ne!(jittered.w_axis.x, 0.0);
        assert_ne!(jittered.w_axis.y, 0.0);

        // Other components should be unchanged
        assert_eq!(jittered.x_axis, proj.x_axis);
        assert_eq!(jittered.y_axis, proj.y_axis);
        assert_eq!(jittered.z_axis, proj.z_axis);
    }

    #[test]
    fn test_jitter_ndc_conversion() {
        let proj = Mat4::IDENTITY;

        // 0.5 pixel offset on 1920 width should be 0.5 * 2 / 1920 in NDC
        let jittered = apply_jitter_to_projection(proj, (0.5, 0.0), 1920, 1080);
        let expected_x = (0.5 * 2.0) / 1920.0;

        assert!((jittered.w_axis.x - expected_x).abs() < 1e-6);
        assert_eq!(jittered.w_axis.y, 0.0);
    }

    #[test]
    fn test_halton_sequence_coverage() {
        let mut seq = HaltonSequence::new();
        let mut points = Vec::new();

        // Generate 16 samples (common TAA pattern length)
        for _ in 0..16 {
            points.push(seq.current());
            seq.next();
        }

        // All points should be unique
        for i in 0..points.len() {
            for j in (i+1)..points.len() {
                assert_ne!(points[i], points[j]);
            }
        }

        // All points should be in [0, 1] range
        for (x, y) in &points {
            assert!(*x >= 0.0 && *x <= 1.0);
            assert!(*y >= 0.0 && *y <= 1.0);
        }
    }
}
