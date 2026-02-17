//! Mask-driven terrain classifier — reads biome from a MaskOctree<BiomeId>
//! instead of computing from noise on every voxel evaluation.

use glam::Vec3;
use crate::math::Aabb;
use crate::mask::{BiomeId, MaskOctree};
use crate::terrain::biome::Biome;
use crate::terrain::generator::TerrainGenerator;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::sdf::{encode_gradient, GRADIENT_RANGE};
use crate::voxel::voxel::Voxel;

/// Terrain classifier that reads biome data from a pre-built mask octree.
///
/// Replaces `BiomeTerrainClassifier` — same height-based shell logic,
/// but biome lookups come from the mask instead of raw noise.
pub struct MaskDrivenTerrainClassifier<'a> {
    terrain: &'a TerrainGenerator,
    biome_mask: &'a MaskOctree<BiomeId>,
    /// World-space origin of the chunk (for local→world conversion in mask lookups).
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

        // Get biome from mask (world pos used directly since mask origin = chunk origin)
        let biome_id = self.biome_mask.sample(self.chunk_origin, pos);
        let biome = Biome::from_id(biome_id);
        let base = biome.surface_color();

        // Compute gradient via finite differences for smooth shading.
        // Uses the SDF module's gradient range constant.
        let eps = self.voxel_size;
        let dh_dx = (self.terrain.height_at(pos.x + eps, pos.z)
                   - self.terrain.height_at(pos.x - eps, pos.z)) / (2.0 * eps);
        let dh_dz = (self.terrain.height_at(pos.x, pos.z + eps)
                   - self.terrain.height_at(pos.x, pos.z - eps)) / (2.0 * eps);

        // Clamp gradient to valid range for encoding
        let dh_dx_clamped = dh_dx.clamp(-GRADIENT_RANGE, GRADIENT_RANGE);
        let dh_dz_clamped = dh_dz.clamp(-GRADIENT_RANGE, GRADIENT_RANGE);

        // Encode gradient in color field
        let color = encode_gradient(dh_dx_clamped, dh_dz_clamped);

        // Store height fraction in flags for sub-voxel refinement
        let voxel_bottom = pos.y - self.voxel_size * 0.5;
        let h_frac = ((height - voxel_bottom) / self.voxel_size).clamp(0.0, 1.0);
        let flags = ((h_frac * 254.0) as u8).max(1);

        Voxel {
            color,
            material_id: base.material_id,
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
    fn test_mask_classifier_empty_below_terrain() {
        let params = TerrainParams {
            height_scale: 64.0,
            ..Default::default()
        };
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);
        let mask = MaskBuilder::new(3).build(&biome_gen, Vec3::ZERO, CHUNK_SIZE as f32);

        let classifier = MaskDrivenTerrainClassifier::new(
            &terrain, &mask, Vec3::ZERO, CHUNK_SIZE as f32 / 128.0,
        );

        // Region deep underground
        let aabb = Aabb::new(Vec3::new(0.0, -100.0, 0.0), Vec3::new(4.0, -50.0, 4.0));
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_mask_classifier_evaluate_above() {
        let params = TerrainParams::default();
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);
        let mask = MaskBuilder::new(3).build(&biome_gen, Vec3::ZERO, CHUNK_SIZE as f32);

        let classifier = MaskDrivenTerrainClassifier::new(
            &terrain, &mask, Vec3::ZERO, CHUNK_SIZE as f32 / 128.0,
        );

        // Above terrain should be empty
        let voxel = classifier.evaluate(Vec3::new(2.0, 200.0, 2.0));
        assert!(voxel.is_empty());
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

        // Should be able to build a terrain octree from the mask-driven classifier
        let builder = AdaptiveOctreeBuilder::new(128);
        let octree = builder.build(&classifier, origin, CHUNK_SIZE as f32);

        // The octree should have some content (this chunk is at y=0, terrain ~0-64)
        assert!(octree.node_count() > 0);
    }

    #[test]
    fn test_mask_vs_direct_brick_count() {
        // Compare MaskDrivenTerrainClassifier output against BiomeTerrainClassifier
        use crate::terrain::generator::BiomeTerrainClassifier;

        let params = TerrainParams {
            scale: 150.0,
            height_scale: 80.0,
            octaves: 5,
            sea_level: 20.0,
            ..Default::default()
        };
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let origin = Vec3::ZERO;
        let voxel_size = CHUNK_SIZE as f32 / 128.0;

        // Direct classifier (existing)
        let direct_classifier = BiomeTerrainClassifier {
            terrain: &terrain,
            biome_map: &biome_map,
            voxel_size,
        };
        let builder = AdaptiveOctreeBuilder::new(128);
        let direct_octree = builder.build(&direct_classifier, origin, CHUNK_SIZE as f32);

        // Mask-driven classifier (new)
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);
        let mask = MaskBuilder::new(3).build(&biome_gen, origin, CHUNK_SIZE as f32);
        let mask_classifier = MaskDrivenTerrainClassifier::new(
            &terrain, &mask, origin, voxel_size,
        );
        let mask_octree = builder.build(&mask_classifier, origin, CHUNK_SIZE as f32);

        // Brick counts should be identical or very close (mask resolution may cause
        // slight biome boundary shifts, but overall structure should match)
        let direct_bricks = direct_octree.brick_count();
        let mask_bricks = mask_octree.brick_count();

        let diff = (direct_bricks as i64 - mask_bricks as i64).unsigned_abs();
        let tolerance = (direct_bricks.max(1) as f64 * 0.15) as u64; // 15% tolerance
        assert!(
            diff <= tolerance,
            "Brick count mismatch: direct={}, mask={}, diff={} (tolerance={})",
            direct_bricks, mask_bricks, diff, tolerance
        );
    }
}
