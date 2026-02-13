//! Upscaling and temporal anti-aliasing

pub mod fsr2;
pub mod jitter;

#[cfg(feature = "dlss")]
pub mod dlss;

pub use fsr2::{FsrQuality, FsrUpscaler};
pub use jitter::{apply_jitter_to_projection, halton, jitter_to_pixels, HaltonSequence};

#[cfg(feature = "dlss")]
pub use dlss::DlssUpscaler;
