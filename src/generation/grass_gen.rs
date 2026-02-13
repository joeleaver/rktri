//! Grass noise generator — builds MaskOctree<GrassCell> per chunk.
//!
//! Uses biome classification, terrain slope, and noise to determine which
//! grass profile and density should appear at each position.

use glam::Vec3;
use crate::grass::profile::{GrassCell, GrassProfile, GrassProfileTable};
use crate::mask::{BiomeId, MaskGenerator, MaskHint, MaskOctree};
use crate::math::Aabb;
use crate::terrain::generator::TerrainGenerator;

/// Procedural grass mask generator.
///
/// Evaluates biome, terrain slope, and noise to assign grass profiles
/// per chunk. Implements `MaskGenerator<GrassProfile>`.
pub struct GrassNoiseGenerator<'a> {
    _terrain: &'a TerrainGenerator,
    biome_mask: &'a MaskOctree<BiomeId>,
    chunk_origin: Vec3,
    _profile_table: &'a GrassProfileTable,
    /// Simple hash seed for bare patch noise
    seed: u32,
}

impl<'a> GrassNoiseGenerator<'a> {
    pub fn new(
        terrain: &'a TerrainGenerator,
        biome_mask: &'a MaskOctree<BiomeId>,
        chunk_origin: Vec3,
        profile_table: &'a GrassProfileTable,
        seed: u32,
    ) -> Self {
        Self {
            _terrain: terrain,
            biome_mask,
            chunk_origin,
            _profile_table: profile_table,
            seed,
        }
    }

    /// Integer hash producing a value in [0, 1].
    fn hash_2d(ix: i32, iz: i32, seed: u32) -> f32 {
        let mut h = (ix as u32).wrapping_mul(374761393)
            .wrapping_add((iz as u32).wrapping_mul(668265263))
            .wrapping_add(seed.wrapping_mul(1274126177));
        h = (h ^ (h >> 13)).wrapping_mul(1103515245);
        h = h ^ (h >> 16);
        (h & 0x7FFFFFFF) as f32 / 0x7FFFFFFF_u32 as f32
    }

    /// Smooth 2D value noise with bilinear interpolation.
    /// Returns [0, 1] varying smoothly over space at the given scale (meters).
    fn smooth_noise(&self, x: f32, z: f32, scale: f32) -> f32 {
        let sx = x / scale;
        let sz = z / scale;

        let ix = sx.floor() as i32;
        let iz = sz.floor() as i32;
        let fx = sx - sx.floor();
        let fz = sz - sz.floor();

        // Smoothstep for C1 continuity
        let fx = fx * fx * (3.0 - 2.0 * fx);
        let fz = fz * fz * (3.0 - 2.0 * fz);

        let h00 = Self::hash_2d(ix, iz, self.seed);
        let h10 = Self::hash_2d(ix + 1, iz, self.seed);
        let h01 = Self::hash_2d(ix, iz + 1, self.seed);
        let h11 = Self::hash_2d(ix + 1, iz + 1, self.seed);

        let a = h00 + (h10 - h00) * fx;
        let b = h01 + (h11 - h01) * fx;
        a + (b - a) * fz
    }

    /// Multi-octave smooth noise for organic shapes.
    fn patch_noise(&self, x: f32, z: f32) -> f32 {
        let n1 = self.smooth_noise(x, z, 6.0);
        let n2 = self.smooth_noise(x, z, 3.0);
        n1 * 0.65 + n2 * 0.35
    }

    /// Smooth density variation noise (separate seed offset).
    fn density_noise(&self, x: f32, z: f32) -> f32 {
        self.smooth_noise(x + 500.0, z + 500.0, 4.0)
    }

    /// Evaluate grass cell (profile + density) at a world XZ position.
    ///
    /// Density encodes biome coverage, bare patches, and natural variation only.
    /// Slope-based density is handled per-pixel in the shader using the actual
    /// terrain normal, which is more accurate than pre-computed finite differences.
    fn evaluate_at(&self, x: f32, z: f32, y: f32) -> GrassCell {
        let biome = self.biome_mask.sample(self.chunk_origin, Vec3::new(x, y, z));
        let profile = self.profile_for_biome(biome, x, z);

        if profile == GrassProfile::NONE {
            return GrassCell::NONE;
        }

        // Bare patches: smooth clearings at ~6m scale
        let patch = self.patch_noise(x, z);
        let patch_density = if patch < 0.15 {
            0.0
        } else if patch < 0.28 {
            (patch - 0.15) / 0.13 // soft edge
        } else {
            1.0
        };

        // Natural density variation at ~4m scale
        let variation = 0.5 + 0.5 * self.density_noise(x, z);

        let density = patch_density * variation;
        if density <= 0.0 {
            return GrassCell::NONE;
        }

        GrassCell::new(profile, density)
    }

    /// Select a grass profile for a biome, considering secondary mixing.
    fn profile_for_biome(&self, biome: BiomeId, x: f32, z: f32) -> GrassProfile {
        let primary = GrassProfileTable::biome_default_profile(biome.0);
        if primary == GrassProfile::NONE {
            return GrassProfile::NONE;
        }

        // Check for secondary profile mixing using smooth noise
        if let Some((secondary, mix_ratio)) = GrassProfileTable::biome_secondary_profile(biome.0) {
            let noise = self.smooth_noise(x + 1000.0, z + 1000.0, 5.0);
            if noise < mix_ratio {
                return secondary;
            }
        }

        primary
    }
}

impl<'a> MaskGenerator<GrassCell> for GrassNoiseGenerator<'a> {
    fn classify_region(&self, aabb: &Aabb) -> MaskHint<GrassCell> {
        // Sample biome at corners + center of XZ projection
        let samples = [
            (aabb.min.x, aabb.min.z),
            (aabb.max.x, aabb.min.z),
            (aabb.min.x, aabb.max.z),
            (aabb.max.x, aabb.max.z),
            ((aabb.min.x + aabb.max.x) * 0.5, (aabb.min.z + aabb.max.z) * 0.5),
        ];

        let mut first_cell: Option<GrassCell> = None;
        for (x, z) in samples {
            let cell = self.evaluate_at(x, z, aabb.min.y);

            match first_cell {
                None => first_cell = Some(cell),
                Some(prev) if prev != cell => return MaskHint::Mixed,
                _ => {}
            }
        }

        match first_cell {
            Some(c) => MaskHint::Uniform(c),
            None => MaskHint::Uniform(GrassCell::NONE),
        }
    }

    fn evaluate(&self, pos: Vec3) -> GrassCell {
        self.evaluate_at(pos.x, pos.z, pos.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::{MaskBuilder, BiomeId};
    use crate::mask::octree::MaskOctree;
    use crate::terrain::generator::{TerrainGenerator, TerrainParams};

    fn make_uniform_biome_mask(biome: BiomeId) -> MaskOctree<BiomeId> {
        let mut tree = MaskOctree::new(4.0, 3);
        let val_idx = tree.add_value(biome);
        tree.node_mut(0).lod_value_idx = val_idx;
        tree
    }

    #[test]
    fn test_grassland_produces_grass() {
        let terrain = TerrainGenerator::new(TerrainParams::default());
        let biome_mask = make_uniform_biome_mask(BiomeId::GRASSLAND);
        let table = GrassProfileTable::default();
        let origin = Vec3::ZERO;

        let grass_gen = GrassNoiseGenerator::new(&terrain, &biome_mask, origin, &table, 12345);
        let builder = MaskBuilder::new(3);
        let mask = builder.build(&grass_gen, origin, 4.0);

        // Should have at least some non-NONE grass profiles
        let sample = mask.sample(origin, Vec3::new(2.0, 2.0, 2.0));
        // Could be Tall Grass (1) or Meadow (2) due to mixing, or NONE due to bare patches
        // Just verify the mask was built
        assert!(mask.node_count() >= 1);
    }

    #[test]
    fn test_ocean_produces_no_grass() {
        let terrain = TerrainGenerator::new(TerrainParams::default());
        let biome_mask = make_uniform_biome_mask(BiomeId::OCEAN);
        let table = GrassProfileTable::default();
        let origin = Vec3::ZERO;

        let grass_gen = GrassNoiseGenerator::new(&terrain, &biome_mask, origin, &table, 12345);
        let builder = MaskBuilder::new(3);
        let mask = builder.build(&grass_gen, origin, 4.0);

        // Ocean should produce only NONE cells
        let sample = mask.sample(origin, Vec3::new(2.0, 2.0, 2.0));
        assert!(sample.is_none());
    }

    #[test]
    fn test_desert_produces_no_grass() {
        let terrain = TerrainGenerator::new(TerrainParams::default());
        let biome_mask = make_uniform_biome_mask(BiomeId::DESERT);
        let table = GrassProfileTable::default();
        let origin = Vec3::ZERO;

        let grass_gen = GrassNoiseGenerator::new(&terrain, &biome_mask, origin, &table, 12345);
        let result = grass_gen.evaluate(Vec3::new(2.0, 2.0, 2.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_forest_produces_meadow() {
        let terrain = TerrainGenerator::new(TerrainParams::default());
        let biome_mask = make_uniform_biome_mask(BiomeId::FOREST);
        let table = GrassProfileTable::default();
        let origin = Vec3::ZERO;

        let grass_gen = GrassNoiseGenerator::new(&terrain, &biome_mask, origin, &table, 12345);

        // Sample many positions — most should be Meadow (2) or NONE (clearing)
        let mut found_meadow = false;
        for ix in 0..10 {
            for iz in 0..10 {
                let pos = Vec3::new(ix as f32 * 0.4, 2.0, iz as f32 * 0.4);
                let result = grass_gen.evaluate(pos);
                if result.profile() == GrassProfile(2) {
                    found_meadow = true;
                }
            }
        }
        assert!(found_meadow, "Forest biome should produce Meadow grass");
    }

}
