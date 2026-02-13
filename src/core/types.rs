//! Core type aliases and re-exports

pub use glam::{
    Vec2, Vec3, Vec4,
    Mat3, Mat4,
    Quat,
    IVec3, UVec3,
};

/// Standard Result type for the engine
pub type Result<T> = std::result::Result<T, crate::core::error::Error>;
