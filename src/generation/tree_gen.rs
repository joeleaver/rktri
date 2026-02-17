//! Tree noise generator — builds tree placement mask per chunk.
//!
//! Uses biome, terrain height, slope, and noise to determine tree probability.
//! Trees spawn on surface in appropriate biomes (Forest, Grassland).

use glam::Vec3;
use crate::mask::{BiomeId, MaskGenerator, MaskHint, MaskOctree, MaskValue};
use crate::math::Aabb;
use crate::terrain::generator::TerrainGenerator;

/// Tree probability cell: [0, 1] indicating tree probability.
/// 0 = no tree, 1 = solid tree location.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeCell(pub f32);

impl TreeCell {
    pub const NONE: Self = Self(0.0);
    pub fn is_none(self) -> bool {
        self.0 <= 0.0
    }
}

impl MaskValue for TreeCell {}

/// Procedural tree distribution generator.
///
/// Evaluates biome, terrain height, slope, and noise to assign tree probability.
pub struct TreeNoiseGenerator<'a> {
    terrain: &'a TerrainGenerator,
    biome_mask: &'a MaskOctree<BiomeId>,
    chunk_origin: Vec3,
    seed: u32,
}

impl<'a> TreeNoiseGenerator<'a> {
    pub fn new(
        terrain: &'a TerrainGenerator,
        biome_mask: &'a MaskOctree<BiomeId>,
        chunk_origin: Vec3,
        seed: u32,
    ) -> Self {
        Self {
            terrain,
            biome_mask,
            chunk_origin,
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

    /// Smooth 2D value noise.
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

    /// Multi-octave noise for tree distribution.
    fn tree_noise(&self, x: f32, z: f32) -> f32 {
        let n1 = self.smooth_noise(x, z, 10.0);
        let n2 = self.smooth_noise(x + 100.0, z + 100.0, 5.0);
        let n3 = self.smooth_noise(x + 200.0, z + 200.0, 2.5);
        n1 * 0.5 + n2 * 0.35 + n3 * 0.15
    }

    /// Estimate slope at position using finite differences.
    fn slope_at(&self, x: f32, z: f32) -> f32 {
        let eps = 0.5;
        let h_xp = self.terrain.height_at(x + eps, z);
        let h_xn = self.terrain.height_at(x - eps, z);
        let h_zp = self.terrain.height_at(x, z + eps);
        let h_zn = self.terrain.height_at(x, z - eps);

        let dx = (h_xp - h_xn) / (2.0 * eps);
        let dz = (h_zp - h_zn) / (2.0 * eps);
        (dx * dx + dz * dz).sqrt()
    }

    /// Evaluate tree probability at a world position.
    fn evaluate_at(&self, x: f32, z: f32) -> TreeCell {
        let height = self.terrain.height_at(x, z);

        // Sample biome
        let biome_id = self.biome_mask.sample(self.chunk_origin, Vec3::new(x, height, z));

        // Biomes that can have trees: Forest, Grassland (both need trees)
        // Forest has more trees, Grassland has fewer but we want more for this demo
        let (biome_has_trees, tree_multiplier) = match biome_id {
            BiomeId(4) => (true, 1.0),   // Forest - max trees
            BiomeId(3) => (true, 0.8),   // Grassland - good tree coverage
            _ => (false, 0.0),
        };

        if !biome_has_trees {
            return TreeCell::NONE;
        }

        // Check slope - trees prefer gentle terrain
        let slope = self.slope_at(x, z);
        let slope_factor = 1.0 - (slope / 1.5).min(1.0); // Trees don't like steep slopes

        // Base tree probability from noise
        let noise = self.tree_noise(x, z);

        // Tree threshold: gentle slopes + high noise = tree
        // Forest: higher density, Grassland: moderate density
        let base_threshold = if matches!(biome_id, BiomeId(4)) { 0.55 } else { 0.5 };
        let threshold = base_threshold - (1.0 - slope_factor) * 0.25; // Lower threshold on flat ground

        let tree_prob = if noise > threshold {
            ((noise - threshold) / (1.0 - threshold)).min(1.0)
        } else {
            0.0
        };

        TreeCell(tree_prob * tree_multiplier * slope_factor)
    }
}

impl<'a> MaskGenerator<TreeCell> for TreeNoiseGenerator<'a> {
    fn classify_region(&self, aabb: &Aabb) -> MaskHint<TreeCell> {
        // Sample at XZ corners + center
        let samples = [
            (aabb.min.x, aabb.min.z),
            (aabb.max.x, aabb.min.z),
            (aabb.min.x, aabb.max.z),
            (aabb.max.x, aabb.max.z),
            ((aabb.min.x + aabb.max.x) * 0.5, (aabb.min.z + aabb.max.z) * 0.5),
        ];

        let mut first_cell: Option<TreeCell> = None;
        for (x, z) in samples {
            let cell = self.evaluate_at(x, z);
            match first_cell {
                None => first_cell = Some(cell),
                Some(prev) if (prev.0 > 0.0) != (cell.0 > 0.0) => return MaskHint::Mixed,
                _ => {}
            }
        }

        match first_cell {
            Some(cell) => MaskHint::Uniform(cell),
            None => MaskHint::Uniform(TreeCell::NONE),
        }
    }

    fn evaluate(&self, pos: Vec3) -> TreeCell {
        self.evaluate_at(pos.x, pos.z)
    }
}
