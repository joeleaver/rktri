//! Edit delta representation.

use crate::core::types::Vec3;
use crate::math::aabb::Aabb;
use crate::voxel::voxel::Voxel;
use crate::voxel::chunk::ChunkCoord;

/// Type of edit operation.
#[derive(Clone, Debug)]
pub enum EditOp {
    /// Set a single voxel to a specific value
    SetVoxel {
        position: Vec3,
        voxel: Voxel,
    },
    /// Clear a single voxel (make empty)
    ClearVoxel {
        position: Vec3,
    },
    /// Fill a region with a material
    FillRegion {
        region: Aabb,
        voxel: Voxel,
    },
    /// Clear a region (make all empty)
    ClearRegion {
        region: Aabb,
    },
}

impl EditOp {
    /// Get the affected region of this edit.
    pub fn affected_region(&self) -> Aabb {
        match self {
            EditOp::SetVoxel { position, .. } | EditOp::ClearVoxel { position } => {
                // Single voxel - tiny AABB
                let half = Vec3::splat(0.004); // ~half voxel
                Aabb::new(*position - half, *position + half)
            }
            EditOp::FillRegion { region, .. } | EditOp::ClearRegion { region } => {
                *region
            }
        }
    }
}

/// A single edit operation with metadata.
#[derive(Clone, Debug)]
pub struct EditDelta {
    /// Unique identifier
    pub id: u64,
    /// Timestamp (frame number when applied)
    pub frame: u32,
    /// The edit operation
    pub op: EditOp,
    /// Affected chunks (computed from region)
    pub affected_chunks: Vec<ChunkCoord>,
}

impl EditDelta {
    /// Create a new edit delta.
    pub fn new(id: u64, frame: u32, op: EditOp) -> Self {
        let affected_chunks = Self::compute_affected_chunks(&op);
        Self {
            id,
            frame,
            op,
            affected_chunks,
        }
    }

    /// Compute which chunks are affected by this edit.
    fn compute_affected_chunks(op: &EditOp) -> Vec<ChunkCoord> {
        let region = op.affected_region();
        let chunk_size = 4.0; // CHUNK_SIZE_METERS

        let min_cx = (region.min.x / chunk_size).floor() as i32;
        let min_cy = (region.min.y / chunk_size).floor() as i32;
        let min_cz = (region.min.z / chunk_size).floor() as i32;
        let max_cx = (region.max.x / chunk_size).floor() as i32;
        let max_cy = (region.max.y / chunk_size).floor() as i32;
        let max_cz = (region.max.z / chunk_size).floor() as i32;

        let mut chunks = Vec::new();
        for x in min_cx..=max_cx {
            for y in min_cy..=max_cy {
                for z in min_cz..=max_cz {
                    chunks.push(ChunkCoord::new(x, y, z));
                }
            }
        }
        chunks
    }

    /// Get the affected region.
    pub fn affected_region(&self) -> Aabb {
        self.op.affected_region()
    }

    /// Apply this edit to evaluate a voxel at position.
    /// Returns Some(voxel) if the edit affects this position, None otherwise.
    pub fn evaluate_at(&self, pos: Vec3) -> Option<Voxel> {
        match &self.op {
            EditOp::SetVoxel { position, voxel } => {
                let diff = (*position - pos).abs();
                if diff.x < 0.004 && diff.y < 0.004 && diff.z < 0.004 {
                    Some(*voxel)
                } else {
                    None
                }
            }
            EditOp::ClearVoxel { position } => {
                let diff = (*position - pos).abs();
                if diff.x < 0.004 && diff.y < 0.004 && diff.z < 0.004 {
                    Some(Voxel::EMPTY)
                } else {
                    None
                }
            }
            EditOp::FillRegion { region, voxel } => {
                if region.contains_point(pos) {
                    Some(*voxel)
                } else {
                    None
                }
            }
            EditOp::ClearRegion { region } => {
                if region.contains_point(pos) {
                    Some(Voxel::EMPTY)
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_voxel_delta() {
        let delta = EditDelta::new(
            1, 0,
            EditOp::SetVoxel {
                position: Vec3::new(5.0, 5.0, 5.0),
                voxel: Voxel::from_rgb565(0x1234, 1),
            },
        );

        // Should affect chunk (1, 1, 1) at position 5,5,5 with chunk_size=4
        assert!(delta.affected_chunks.contains(&ChunkCoord::new(1, 1, 1)));

        // Should return the voxel at the exact position
        let result = delta.evaluate_at(Vec3::new(5.0, 5.0, 5.0));
        assert!(result.is_some());
        assert_eq!(result.unwrap().material_id, 1);

        // Should return None for other positions
        assert!(delta.evaluate_at(Vec3::new(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn test_fill_region_delta() {
        let delta = EditDelta::new(
            2, 0,
            EditOp::FillRegion {
                region: Aabb::new(Vec3::ZERO, Vec3::new(8.0, 4.0, 4.0)),
                voxel: Voxel::from_rgb565(0, 5),
            },
        );

        // Should affect chunks (0,0,0) and (1,0,0) since region spans 8m in X
        assert!(delta.affected_chunks.contains(&ChunkCoord::new(0, 0, 0)));
        assert!(delta.affected_chunks.contains(&ChunkCoord::new(1, 0, 0)));
    }

    #[test]
    fn test_clear_region_delta() {
        let delta = EditDelta::new(
            3, 0,
            EditOp::ClearRegion {
                region: Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0)),
            },
        );

        let result = delta.evaluate_at(Vec3::new(2.0, 2.0, 2.0));
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }
}
