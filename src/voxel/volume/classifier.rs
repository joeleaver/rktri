//! Volume classifier for arbitrary 3D shapes.
//!
//! Works with the VolumetricGrid spatial index to classify
//! regions containing volumetric objects like trees.

use glam::Vec3;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::math::aabb::Aabb;
use crate::voxel::svo::volumetric::VolumetricGrid;
use crate::voxel::voxel::Voxel;

/// Classifier for volumetric objects (trees, caves, etc.)
///
/// Uses spatial index to find objects that intersect the query region.
/// This is true 3D classification - no heightfield assumptions.
pub struct VolumeClassifier<'a> {
    /// Spatial index of volumetric objects
    grid: &'a VolumetricGrid,
    /// Background hint when no objects present
    background: RegionHint,
}

impl<'a> VolumeClassifier<'a> {
    /// Create a new volume classifier.
    pub fn new(grid: &'a VolumetricGrid) -> Self {
        Self {
            grid,
            background: RegionHint::Empty,
        }
    }

    /// Create with a custom background hint.
    pub fn with_background(grid: &'a VolumetricGrid, background: RegionHint) -> Self {
        Self { grid, background }
    }
}

impl<'a> RegionClassifier for VolumeClassifier<'a> {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        // Query spatial index for objects intersecting this region
        let objects = self.grid.query_aabb(aabb);

        if objects.is_empty() {
            return self.background;
        }

        // If any object intersects, region has mixed content
        // (We could optimize further by checking if object fully contains region
        // and is uniform, but Mixed is always safe)
        RegionHint::Mixed
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        // Query objects at this exact position
        let objects = self.grid.query_point(pos);

        for obj in objects {
            if let Some(voxel) = obj.sample_at(pos) {
                if !voxel.is_empty() {
                    return voxel;
                }
            }
        }

        Voxel::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_grid_returns_background() {
        let grid = VolumetricGrid::new(4.0);
        let classifier = VolumeClassifier::new(&grid);

        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_custom_background() {
        let grid = VolumetricGrid::new(4.0);
        let classifier = VolumeClassifier::with_background(
            &grid,
            RegionHint::Solid { material: 1, color: 0 }
        );

        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        match classifier.classify_region(&aabb) {
            RegionHint::Solid { .. } => {}
            other => panic!("Expected Solid background, got {:?}", other),
        }
    }
}
