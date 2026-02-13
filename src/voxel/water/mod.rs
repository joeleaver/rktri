//! Water system for oceans, rivers, and underwater rendering.

pub mod volume;
pub mod system;

pub use volume::{WaterBody, WaterBodyType, WaterProperties, WaterSurface};
pub use system::{WaterSystem, MATERIAL_WATER};
