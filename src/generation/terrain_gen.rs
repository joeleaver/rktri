//! Mask-driven terrain classifier — reads biome from a MaskOctree<BiomeId>
//! instead of computing from noise on every voxel evaluation.

use glam::Vec3;
use crate::math::Aabb;
use crate::mask::{BiomeId, MaskOctree};
use crate::terrain::biome::Biome;
use crate::terrain::generator::TerrainGenerator;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::voxel::Voxel;

/// Material IDs for different surface types.
pub mod materials {
    /// Terrain surface materials (biome-based, IDs 1-14)
    pub const DIRT: u8 = 1;
    pub const GRASS: u8 = 2;
    pub const SAND: u8 = 3;
    pub const STONE: u8 = 4;
    pub const SNOW: u8 = 5;
    pub const WATER: u8 = 6;

    /// Rock/stone materials (IDs 20-29)
    pub const ROCK: u8 = 20;

    /// Vegetation materials (IDs 30-39)
    pub const WOOD: u8 = 31;
    pub const LEAVES: u8 = 32;
}

/// Terrain classifier that reads biome data from a pre-built mask octree.
///
/// Uses height-based shell logic with biome lookups from the mask.
pub struct MaskDrivenTerrainClassifier<'a> {
    terrain: &'a TerrainGenerator,
    biome_mask: &'a MaskOctree<BiomeId>,
    chunk_origin: Vec3,
    voxel_size: f32,
}

impl<'a> MaskDrivenTerrainClassifier<'a> {
    pub fn new(
        terrain: &'a TerrainGenerator,
        biome_mask: &'a MaskOctree<BiomeId>,
        chunk_origin: Vec3,
        voxel_size: f32,
    ) -> Self {
        Self {
            terrain,
            biome_mask,
            chunk_origin,
            voxel_size,
        }
    }
}

impl<'a> RegionClassifier for MaskDrivenTerrainClassifier<'a> {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        let region_size = aabb.max.x - aabb.min.x;

        // Single center height sample — cheap and effective at all scales.
        let cx = (aabb.min.x + aabb.max.x) * 0.5;
        let cz = (aabb.min.z + aabb.max.z) * 0.5;
        let h = self.terrain.height_at(cx, cz);

        // Margin accounts for height variation within this region.
        let margin = (region_size * 0.3).max(0.1);

        // Region entirely above terrain → empty
        if aabb.min.y > h + margin {
            return RegionHint::Empty;
        }

        // Region entirely below terrain surface → empty (shell 1.0m thick)
        if aabb.max.y < h - margin - 1.0 {
            return RegionHint::Empty;
        }

        // Straddles the surface → must subdivide
        RegionHint::Mixed
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        let height = self.terrain.height_at(pos.x, pos.z);
        if pos.y > height {
            return Voxel::EMPTY;
        }
        let depth = height - pos.y;
        if depth >= 1.0 {
            return Voxel::EMPTY;
        }

        // Get biome from mask
        let biome_id = self.biome_mask.sample(self.chunk_origin, pos);
        let biome = Biome::from_id(biome_id);

        // Material based on biome
        let material_id = biome.surface_color().material_id;

        // Compute terrain gradient via finite differences for smooth normals.
        let eps = self.voxel_size;
        let dh_dx = (self.terrain.height_at(pos.x + eps, pos.z)
                   - self.terrain.height_at(pos.x - eps, pos.z)) / (2.0 * eps);
        let dh_dz = (self.terrain.height_at(pos.x, pos.z + eps)
                   - self.terrain.height_at(pos.x, pos.z - eps)) / (2.0 * eps);

        // Encode gradient in color field: each maps [-4, 4] → [0, 255]
        let dx_enc = (((dh_dx + 4.0) / 8.0).clamp(0.0, 1.0) * 255.0) as u8;
        let dz_enc = (((dh_dz + 4.0) / 8.0).clamp(0.0, 1.0) * 255.0) as u8;
        let gradient_color = (dz_enc as u16) << 8 | dx_enc as u16;

        // Height fraction in flags byte (1-255)
        let voxel_bottom = pos.y - self.voxel_size * 0.5;
        let h_frac = ((height - voxel_bottom) / self.voxel_size).clamp(0.0, 1.0);
        let flags = ((h_frac * 254.0) as u8).max(1);

        Voxel {
            color: gradient_color,
            material_id,
            flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::MaskBuilder;
    use crate::terrain::biome::BiomeMap;
    use crate::terrain::generator::TerrainParams;
    use crate::generation::biome_gen::BiomeNoiseGenerator;
    use crate::voxel::svo::adaptive::AdaptiveOctreeBuilder;
    use crate::voxel::chunk::CHUNK_SIZE;

    #[test]
    fn test_mask_classifier_empty_above_terrain() {
        let params = TerrainParams::default();
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);
        let mask = MaskBuilder::new(3).build(&biome_gen, Vec3::ZERO, CHUNK_SIZE as f32);

        let classifier = MaskDrivenTerrainClassifier::new(
            &terrain, &mask, Vec3::ZERO, CHUNK_SIZE as f32 / 128.0,
        );

        // Region far above max terrain height
        let aabb = Aabb::new(Vec3::new(0.0, 200.0, 0.0), Vec3::new(4.0, 204.0, 4.0));
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_mask_classifier_builds_octree() {
        let params = TerrainParams::default();
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let origin = Vec3::ZERO;
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);
        let mask = MaskBuilder::new(3).build(&biome_gen, origin, CHUNK_SIZE as f32);

        let classifier = MaskDrivenTerrainClassifier::new(
            &terrain, &mask, origin, CHUNK_SIZE as f32 / 128.0,
        );

        let builder = AdaptiveOctreeBuilder::new(128);
        let octree = builder.build(&classifier, origin, CHUNK_SIZE as f32);

        assert!(octree.node_count() > 0);
    }
}
