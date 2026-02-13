//! Biome noise generator — wraps BiomeMap as a MaskGenerator<BiomeId>.

use glam::Vec3;
use crate::math::Aabb;
use crate::mask::{BiomeId, MaskGenerator, MaskHint};
use crate::terrain::biome::BiomeMap;
use crate::terrain::generator::TerrainGenerator;

/// Procedural biome generator that adapts the existing BiomeMap noise
/// into the MaskGenerator interface for building biome mask octrees.
pub struct BiomeNoiseGenerator<'a> {
    biome_map: &'a BiomeMap,
    terrain: &'a TerrainGenerator,
    sea_level: f32,
}

impl<'a> BiomeNoiseGenerator<'a> {
    pub fn new(biome_map: &'a BiomeMap, terrain: &'a TerrainGenerator, sea_level: f32) -> Self {
        Self {
            biome_map,
            terrain,
            sea_level,
        }
    }
}

impl<'a> MaskGenerator<BiomeId> for BiomeNoiseGenerator<'a> {
    fn classify_region(&self, aabb: &Aabb) -> MaskHint<BiomeId> {
        // Sample biome at corners + center of the XZ projection.
        // Y doesn't matter for biome classification (biomes are column-based).
        let samples = [
            (aabb.min.x, aabb.min.z),
            (aabb.max.x, aabb.min.z),
            (aabb.min.x, aabb.max.z),
            (aabb.max.x, aabb.max.z),
            ((aabb.min.x + aabb.max.x) * 0.5, (aabb.min.z + aabb.max.z) * 0.5),
        ];

        let mut first: Option<BiomeId> = None;
        for (x, z) in samples {
            let h = self.terrain.height_at(x, z);
            let biome = self.biome_map.biome_at(x, z, h, self.sea_level);
            let id = biome.to_id();
            match first {
                None => first = Some(id),
                Some(prev) if prev != id => return MaskHint::Mixed,
                _ => {}
            }
        }

        match first {
            Some(id) => MaskHint::Uniform(id),
            None => MaskHint::Uniform(BiomeId::default()),
        }
    }

    fn evaluate(&self, pos: Vec3) -> BiomeId {
        let h = self.terrain.height_at(pos.x, pos.z);
        self.biome_map.biome_at(pos.x, pos.z, h, self.sea_level).to_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::generator::TerrainParams;
    use crate::mask::MaskBuilder;

    #[test]
    fn test_biome_noise_evaluate() {
        let params = TerrainParams::default();
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);

        // Should return a valid BiomeId at any position
        let id = biome_gen.evaluate(Vec3::new(100.0, 0.0, 100.0));
        assert!(id.0 <= 8, "BiomeId {} out of range", id.0);
    }

    #[test]
    fn test_biome_noise_classify() {
        let params = TerrainParams::default();
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);

        // Small region should likely be uniform
        let small = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5));
        let hint = biome_gen.classify_region(&small);
        match hint {
            MaskHint::Uniform(id) => assert!(id.0 <= 8),
            MaskHint::Mixed => {} // Also valid
        }
    }

    #[test]
    fn test_biome_mask_matches_biome_map() {
        let params = TerrainParams {
            scale: 150.0,
            height_scale: 80.0,
            octaves: 5,
            sea_level: 20.0,
            ..Default::default()
        };
        let terrain = TerrainGenerator::new(params.clone());
        let biome_map = BiomeMap::new(12345);
        let biome_gen = BiomeNoiseGenerator::new(&biome_map, &terrain, params.sea_level);

        // Build a biome mask for a chunk at origin
        let builder = MaskBuilder::new(3);
        let origin = Vec3::ZERO;
        let mask = builder.build(&biome_gen, origin, 4.0);

        // Compare mask.sample() vs biome_map.biome_at() at several positions
        let test_positions = [
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 1.5),
            Vec3::new(2.5, 2.5, 2.5),
            Vec3::new(3.5, 3.5, 3.5),
        ];

        for pos in &test_positions {
            let world_pos = origin + *pos;
            let mask_biome = mask.sample(origin, world_pos);

            let h = terrain.height_at(world_pos.x, world_pos.z);
            let direct_biome = biome_map.biome_at(world_pos.x, world_pos.z, h, params.sea_level).to_id();

            assert_eq!(
                mask_biome, direct_biome,
                "Mismatch at {:?}: mask={:?}, direct={:?}",
                pos, mask_biome, direct_biome
            );
        }
    }
}
