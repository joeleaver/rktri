//! Brush system for procedural voxel generation
//!
//! Allows painting voxels at different octree levels using primitive shapes.

pub mod primitive;
pub mod stroke;
pub mod session;
pub mod builder;

// Re-exports
pub use primitive::{BrushPrimitive, Axis};
pub use stroke::{BrushStroke, BlendMode};
pub use session::BrushSession;
pub use builder::BrushOctreeBuilder;
