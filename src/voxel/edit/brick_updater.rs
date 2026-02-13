use crate::voxel::brick_handle::BrickId;
use crate::voxel::brick::VoxelBrick;
use crate::core::types::Vec3;
use super::delta::EditDelta;

/// Updates individual bricks by applying edits to base data.
pub struct BrickUpdater;

impl BrickUpdater {
    pub fn new() -> Self {
        Self
    }

    /// Update a single brick by applying edits.
    /// `brick_origin` is the world-space origin of this brick.
    /// `voxel_size` is the size of each voxel in the brick.
    /// Edits are applied in order (by id - lowest first).
    pub fn update_brick(
        &self,
        base: &VoxelBrick,
        edits: &[&EditDelta],
        brick_origin: Vec3,
        voxel_size: f32,
    ) -> VoxelBrick {
        let mut result = *base;

        // Sort edits by id (ascending = chronological)
        let mut sorted_edits: Vec<&EditDelta> = edits.to_vec();
        sorted_edits.sort_by_key(|e| e.id);

        // Apply each edit to the brick's 2x2x2 voxels
        for edit in &sorted_edits {
            for z in 0..2u8 {
                for y in 0..2u8 {
                    for x in 0..2u8 {
                        let voxel_pos = brick_origin
                            + Vec3::new(
                                x as f32 * voxel_size + voxel_size * 0.5,
                                y as f32 * voxel_size + voxel_size * 0.5,
                                z as f32 * voxel_size + voxel_size * 0.5,
                            );

                        if let Some(new_voxel) = edit.evaluate_at(voxel_pos) {
                            let idx = (z as usize * 4) + (y as usize * 2) + x as usize;
                            result.voxels[idx] = new_voxel;
                        }
                    }
                }
            }
        }

        result
    }

    /// Batch update multiple bricks.
    pub fn update_bricks(
        &self,
        updates: &[(BrickId, &VoxelBrick, Vec<&EditDelta>, Vec3, f32)],
    ) -> Vec<(BrickId, VoxelBrick)> {
        updates
            .iter()
            .map(|(id, base, edits, origin, voxel_size)| {
                (*id, self.update_brick(base, edits, *origin, *voxel_size))
            })
            .collect()
    }
}

impl Default for BrickUpdater {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::voxel::Voxel;

    #[test]
    fn test_empty_edits() {
        let updater = BrickUpdater::new();
        let base = VoxelBrick {
            voxels: [Voxel::EMPTY; 8],
        };
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let voxel_size = 1.0;

        let result = updater.update_brick(&base, &[], origin, voxel_size);

        assert_eq!(result.voxels, base.voxels);
    }

    #[test]
    fn test_set_voxel_edit() {
        let updater = BrickUpdater::new();
        let mut base_voxels = [Voxel::EMPTY; 8];
        base_voxels[0] = Voxel::from_rgb565(0xF800, 0); // Red voxel
        let base = VoxelBrick {
            voxels: base_voxels,
        };

        let origin = Vec3::new(0.0, 0.0, 0.0);
        let voxel_size = 1.0;

        // Create a mock edit that sets a voxel to blue
        let edit = EditDelta::new(
            1,
            0,
            crate::voxel::edit::delta::EditOp::SetVoxel {
                position: Vec3::new(0.5, 0.5, 0.5),
                voxel: crate::voxel::voxel::Voxel::from_rgb565(0x001F, 1), // Blue voxel
            },
        );

        let result = updater.update_brick(&base, &[&edit], origin, voxel_size);

        // The voxel at index 0 should be updated to blue
        assert_eq!(result.voxels[0], Voxel::from_rgb565(0x001F, 1));
    }
}
