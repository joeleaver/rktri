//! True 3D region classification for octree building.
//!
//! This replaces the height-based VoxelEvaluator trait with proper
//! 3D classification that works for caves, floating islands, and
//! arbitrary volumetric content.

use crate::core::types::Vec3;
use crate::voxel::voxel::Voxel;
use crate::math::aabb::Aabb;

/// Hint for octree builder about region content.
/// Used for early-out optimization during octree construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegionHint {
    /// Region is entirely empty (air, void) - can skip subdivision
    Empty,
    /// Region is entirely solid with uniform material
    Solid { material: u8, color: u16 },
    /// Region has mixed content - must subdivide further
    Mixed,
    /// Classification is expensive or unknown - just evaluate voxels directly
    Unknown,
}

/// Trait for 3D region classification.
///
/// Unlike the old VoxelEvaluator with height_at(), this trait has NO
/// assumptions about terrain being a heightfield. It works equally well
/// for caves (solid above, air below), floating islands, and any 3D structure.
pub trait RegionClassifier: Send + Sync {
    /// Classify a 3D region (AABB) for early-out optimization.
    ///
    /// Returns a hint about the region's content:
    /// - Empty: Skip this region entirely
    /// - Solid: Fill with uniform material
    /// - Mixed: Must subdivide and evaluate
    /// - Unknown: Classification too expensive, evaluate directly
    fn classify_region(&self, aabb: &Aabb) -> RegionHint;

    /// Evaluate single voxel at world position.
    ///
    /// This is called for leaf nodes or when classify_region returns Mixed/Unknown.
    fn evaluate(&self, pos: Vec3) -> Voxel;
}

impl RegionHint {
    /// Returns true if this region can be skipped (Empty or Solid)
    pub fn is_terminal(&self) -> bool {
        matches!(self, RegionHint::Empty | RegionHint::Solid { .. })
    }

    /// Returns true if this region needs subdivision
    pub fn needs_subdivision(&self) -> bool {
        matches!(self, RegionHint::Mixed | RegionHint::Unknown)
    }
}
