//! FSR 2.0 upscaling integration
//!
//! Provides a wrapper around AMD FidelityFX Super Resolution 2.0 for temporal upscaling.
//! Currently implemented as a placeholder with bilinear fallback until full FSR integration.

use super::jitter::HaltonSequence;

/// FSR 2.0 quality presets
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsrQuality {
    /// Quality mode - 1.5x upscale (66.7% render resolution)
    Quality,
    /// Balanced mode - 1.7x upscale (58.8% render resolution)
    Balanced,
    /// Performance mode - 2.0x upscale (50% render resolution)
    Performance,
    /// Ultra Performance mode - 3.0x upscale (33.3% render resolution)
    UltraPerf,
}

impl FsrQuality {
    /// Get the render scale factor (render_res / display_res)
    pub fn render_scale(&self) -> f32 {
        match self {
            Self::Quality => 1.0 / 1.5,
            Self::Balanced => 1.0 / 1.7,
            Self::Performance => 0.5,
            Self::UltraPerf => 1.0 / 3.0,
        }
    }

    /// Get a descriptive name for this quality level
    pub fn name(&self) -> &'static str {
        match self {
            Self::Quality => "Quality",
            Self::Balanced => "Balanced",
            Self::Performance => "Performance",
            Self::UltraPerf => "Ultra Performance",
        }
    }
}

/// FSR 2.0 upscaler (placeholder implementation)
///
/// This is a placeholder that manages render resolution and jitter patterns.
/// Actual FSR 2.0 integration would hook into AMD's FidelityFX SDK here.
pub struct FsrUpscaler {
    quality: FsrQuality,
    render_width: u32,
    render_height: u32,
    display_width: u32,
    display_height: u32,
    jitter_sequence: HaltonSequence,
}

impl FsrUpscaler {
    /// Create a new FSR upscaler
    ///
    /// # Arguments
    /// * `display_width` - Target display width in pixels
    /// * `display_height` - Target display height in pixels
    /// * `quality` - FSR quality preset
    pub fn new(display_width: u32, display_height: u32, quality: FsrQuality) -> Self {
        let scale = quality.render_scale();
        let render_width = ((display_width as f32 * scale) as u32).max(1);
        let render_height = ((display_height as f32 * scale) as u32).max(1);

        Self {
            quality,
            render_width,
            render_height,
            display_width,
            display_height,
            jitter_sequence: HaltonSequence::new(),
        }
    }

    /// Get the render resolution (before upscaling)
    pub fn render_size(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }

    /// Get the display resolution (after upscaling)
    pub fn display_size(&self) -> (u32, u32) {
        (self.display_width, self.display_height)
    }

    /// Get the current quality preset
    pub fn quality(&self) -> FsrQuality {
        self.quality
    }

    /// Get the current jitter offset in pixels (centered around 0)
    ///
    /// Returns (x_offset, y_offset) in pixels, ranging from -0.5 to +0.5
    pub fn jitter(&self) -> (f32, f32) {
        let (jx, jy) = self.jitter_sequence.current();
        // Center jitter around 0 (-0.5 to +0.5 range)
        (jx - 0.5, jy - 0.5)
    }

    /// Advance to the next frame and update jitter
    pub fn next_frame(&mut self) {
        self.jitter_sequence.next();
    }

    /// Check if FSR 2.0 is actually available (currently always false)
    pub fn is_available() -> bool {
        // TODO: Check for FSR library availability
        false
    }

    /// Reset the jitter sequence (useful when switching scenes or resetting temporal data)
    pub fn reset_jitter(&mut self) {
        self.jitter_sequence.reset();
    }

    /// Get the current jitter index
    pub fn jitter_index(&self) -> u32 {
        self.jitter_sequence.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsr_quality_scales() {
        assert_eq!(FsrQuality::Quality.render_scale(), 1.0 / 1.5);
        assert_eq!(FsrQuality::Balanced.render_scale(), 1.0 / 1.7);
        assert_eq!(FsrQuality::Performance.render_scale(), 0.5);
        assert_eq!(FsrQuality::UltraPerf.render_scale(), 1.0 / 3.0);
    }

    #[test]
    fn test_fsr_quality_names() {
        assert_eq!(FsrQuality::Quality.name(), "Quality");
        assert_eq!(FsrQuality::Performance.name(), "Performance");
    }

    #[test]
    fn test_fsr_upscaler_resolution() {
        let upscaler = FsrUpscaler::new(1920, 1080, FsrQuality::Performance);
        assert_eq!(upscaler.render_size(), (960, 540));
        assert_eq!(upscaler.display_size(), (1920, 1080));
    }

    #[test]
    fn test_fsr_upscaler_quality_preset() {
        let upscaler = FsrUpscaler::new(3840, 2160, FsrQuality::Quality);
        let (rw, rh) = upscaler.render_size();

        // Should be approximately 66.7% of display resolution
        assert!((rw as f32 / 3840.0 - 2.0/3.0).abs() < 0.01);
        assert!((rh as f32 / 2160.0 - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_fsr_jitter_range() {
        let mut upscaler = FsrUpscaler::new(1920, 1080, FsrQuality::Performance);

        // Test several jitter values
        for _ in 0..16 {
            let (jx, jy) = upscaler.jitter();
            assert!(jx >= -0.5 && jx <= 0.5);
            assert!(jy >= -0.5 && jy <= 0.5);
            upscaler.next_frame();
        }
    }

    #[test]
    fn test_fsr_jitter_reset() {
        let mut upscaler = FsrUpscaler::new(1920, 1080, FsrQuality::Performance);

        let initial = upscaler.jitter();
        upscaler.next_frame();
        upscaler.next_frame();

        let after_advance = upscaler.jitter();
        assert_ne!(initial, after_advance);

        upscaler.reset_jitter();
        let after_reset = upscaler.jitter();
        assert_eq!(initial, after_reset);
    }

    #[test]
    fn test_fsr_not_available() {
        assert!(!FsrUpscaler::is_available());
    }
}
