//! Error types for the Rktri engine

use thiserror::Error;

/// Main error type for the engine
#[derive(Debug, Error)]
pub enum Error {
    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Window error: {0}")]
    Window(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Voxel error: {0}")]
    Voxel(String),

    #[error("Animation error: {0}")]
    Animation(String),

    #[error("Streaming error: {0}")]
    Streaming(String),
}
