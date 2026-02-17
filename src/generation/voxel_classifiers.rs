//! Direct voxel classifiers — generate rock/tree octrees from masks.
//!
//! Masks are 2D probability maps sampled at terrain height during mask creation.
//! The classifier checks mask at (x,z) and places rocks/trees at terrain surface.

use glam::Vec3;
use crate::math::Aabb;
use crate::mask::MaskOctree;
use crate::terrain::generator::TerrainGenerator;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::voxel::Voxel;

use super::terrain_gen::materials::{ROCK, WOOD, LEAVES};
use super::{RockCell, TreeCell};

/// Rock classifier: checks mask for rock presence, places at terrain surface.
pub struct RockMaskExtractor<'a> {
    pub rock_mask: &'a MaskOctree<RockCell>,
    pub terrain: &'a TerrainGenerator,
    pub chunk_origin: Vec3,
}

impl<'a> RegionClassifier for RockMaskExtractor<'a> {
    fn classify_region(&self, _aabb: &Aabb) -> RegionHint {
        // No optimization - just evaluate voxels directly
        RegionHint::Unknown
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        // Get terrain height at this position
        let terrain_height = self.terrain.height_at(pos.x, pos.z);

        // Only place rocks AT terrain surface (within small range)
        let depth = terrain_height - pos.y;
        if depth < -0.5 || depth > 1.0 {
            return Voxel::EMPTY;
        }

        // Check mask at this position
        let local_pos = pos - self.chunk_origin;
        let rock_cell = self.rock_mask.sample(self.chunk_origin, local_pos);

        if rock_cell.0 > 0.2 {
            let dx_enc = 128u8;
            let dz_enc = 128u8;
            let gradient_color = (dz_enc as u16) << 8 | dx_enc as u16;

            Voxel {
                color: gradient_color,
                material_id: ROCK,
                flags: 255,
            }
        } else {
            Voxel::EMPTY
        }
    }
}

/// Tree classifier: checks mask for tree presence, places wood/leaves at terrain.
pub struct TreeMaskExtractor<'a> {
    pub tree_mask: &'a MaskOctree<TreeCell>,
    pub terrain: &'a TerrainGenerator,
    pub chunk_origin: Vec3,
}

impl<'a> TreeMaskExtractor<'a> {
    fn get_material(&self, world_y: f32, terrain_y: f32) -> Option<u8> {
        let relative_y = world_y - terrain_y;
        if relative_y >= 0.0 && relative_y < 4.0 {
            Some(WOOD)
        } else if relative_y >= 3.0 && relative_y < 8.0 {
            Some(LEAVES)
        } else {
            None
        }
    }
}

impl<'a> RegionClassifier for TreeMaskExtractor<'a> {
    fn classify_region(&self, _aabb: &Aabb) -> RegionHint {
        // No optimization - just evaluate voxels directly
        RegionHint::Unknown
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        // Get terrain height
        let terrain_height = self.terrain.height_at(pos.x, pos.z);

        // Only place trees starting from terrain surface upward
        let relative_y = pos.y - terrain_height;
        if relative_y < 0.0 || relative_y > 10.0 {
            return Voxel::EMPTY;
        }

        // Check mask
        let local_pos = pos - self.chunk_origin;
        let tree_cell = self.tree_mask.sample(self.chunk_origin, local_pos);

        if tree_cell.0 > 0.2 {
            if let Some(material_id) = self.get_material(pos.y, terrain_height) {
                let dx_enc = 128u8;
                let dz_enc = 128u8;
                let gradient_color = (dz_enc as u16) << 8 | dx_enc as u16;

                return Voxel {
                    color: gradient_color,
                    material_id,
                    flags: 255,
                };
            }
        }

        Voxel::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::CHUNK_SIZE;

    #[test]
    fn test_rock_classifier_uses_terrain() {
        let rock_mask: MaskOctree<RockCell> = MaskOctree::new(CHUNK_SIZE as f32, 3);
        let terrain = TerrainGenerator::new(Default::default());

        let classifier = RockMaskExtractor {
            rock_mask: &rock_mask,
            terrain: &terrain,
            chunk_origin: Vec3::ZERO,
        };

        // Above terrain should be empty
        let voxel = classifier.evaluate(Vec3::new(10.0, 100.0, 10.0));
        assert!(voxel.is_empty());
    }
}
