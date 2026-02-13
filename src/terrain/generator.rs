//! Noise-based procedural terrain generation

use glam::Vec3;

use super::biome::BiomeMap;
use crate::math::Aabb;
use crate::voxel::chunk::{Chunk, ChunkCoord, CHUNK_SIZE};
use crate::voxel::svo::adaptive::AdaptiveOctreeBuilder;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::voxel::Voxel;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// Parameters controlling terrain generation
#[derive(Clone, Debug)]
pub struct TerrainParams {
    pub seed: u32,
    pub scale: f32,        // Horizontal scale (larger = smoother)
    pub height_scale: f32, // Vertical scale (max height)
    pub octaves: u32,      // FBM octaves (detail levels)
    pub persistence: f32,  // FBM persistence (0.5 typical)
    pub lacunarity: f32,   // FBM lacunarity (2.0 typical)
    pub sea_level: f32,    // Height below which is "water"
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 12345,
            scale: 100.0,
            height_scale: 64.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            sea_level: 32.0,
        }
    }
}

/// Procedural terrain generator using fractal Brownian motion (FBM)
pub struct TerrainGenerator {
    params: TerrainParams,
    noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    /// Create a new terrain generator with the given parameters
    pub fn new(params: TerrainParams) -> Self {
        let noise = Fbm::<Perlin>::new(params.seed)
            .set_octaves(params.octaves as usize)
            .set_persistence(params.persistence as f64)
            .set_lacunarity(params.lacunarity as f64);

        Self { params, noise }
    }

    /// Get terrain parameters
    pub fn params(&self) -> &TerrainParams {
        &self.params
    }

    /// Get terrain height at world position (x, z)
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        // Sample noise in normalized coordinates
        let nx = (x / self.params.scale) as f64;
        let nz = (z / self.params.scale) as f64;

        // Get noise value in range [-1, 1]
        let noise_value = self.noise.get([nx, nz]);

        // Map to height range [0, height_scale]
        let normalized = (noise_value + 1.0) / 2.0;
        (normalized * self.params.height_scale as f64) as f32
    }

    /// Get min/max height bounds in an XZ region for adaptive octree building
    pub fn height_bounds(&self, min_x: f32, max_x: f32, min_z: f32, max_z: f32) -> (f32, f32) {
        let heights = [
            self.height_at(min_x, min_z),
            self.height_at(max_x, min_z),
            self.height_at(min_x, max_z),
            self.height_at(max_x, max_z),
            self.height_at((min_x + max_x) / 2.0, (min_z + max_z) / 2.0),
        ];

        let min_h = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_h = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        (min_h, max_h)
    }

    /// Generate a chunk using adaptive top-down octree building.
    /// No dense array allocation — empty space is free.
    pub fn generate_chunk_adaptive(&self, coord: ChunkCoord, biome_map: &BiomeMap) -> Chunk {
        let origin = coord.world_origin();
        const TERRAIN_VOXELS: u32 = 128;
        let voxel_size = CHUNK_SIZE as f32 / TERRAIN_VOXELS as f32;

        let classifier = BiomeTerrainClassifier {
            terrain: self,
            biome_map,
            voxel_size,
        };
        let builder = AdaptiveOctreeBuilder::new(TERRAIN_VOXELS);
        let octree = builder.build(&classifier, origin, CHUNK_SIZE as f32);

        let mut chunk = Chunk::from_octree(coord, octree);
        chunk.modified = true;
        chunk
    }
}

/// Wraps TerrainGenerator + BiomeMap as a RegionClassifier for adaptive octree building.
///
/// Kept for backward compatibility; new code should prefer
/// `generation::MaskDrivenTerrainClassifier` which reads biome from a mask octree.
pub struct BiomeTerrainClassifier<'a> {
    pub(crate) terrain: &'a TerrainGenerator,
    pub(crate) biome_map: &'a BiomeMap,
    pub(crate) voxel_size: f32,
}

impl<'a> RegionClassifier for BiomeTerrainClassifier<'a> {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        let region_size = aabb.max.x - aabb.min.x;

        let cx = (aabb.min.x + aabb.max.x) * 0.5;
        let cz = (aabb.min.z + aabb.max.z) * 0.5;
        let h = self.terrain.height_at(cx, cz);

        let margin = (region_size * 0.3).max(0.1);

        if aabb.min.y > h + margin {
            return RegionHint::Empty;
        }

        if aabb.max.y < h - margin - 1.0 {
            return RegionHint::Empty;
        }

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

        let biome = self.biome_map.biome_at(pos.x, pos.z, height, self.terrain.params().sea_level);
        let base = biome.surface_color();

        let eps = self.voxel_size;
        let dh_dx = (self.terrain.height_at(pos.x + eps, pos.z)
                   - self.terrain.height_at(pos.x - eps, pos.z)) / (2.0 * eps);
        let dh_dz = (self.terrain.height_at(pos.x, pos.z + eps)
                   - self.terrain.height_at(pos.x, pos.z - eps)) / (2.0 * eps);

        let dx_enc = (((dh_dx + 4.0) / 8.0).clamp(0.0, 1.0) * 255.0) as u8;
        let dz_enc = (((dh_dz + 4.0) / 8.0).clamp(0.0, 1.0) * 255.0) as u8;
        let gradient_color = (dz_enc as u16) << 8 | dx_enc as u16;

        let voxel_bottom = pos.y - self.voxel_size * 0.5;
        let h_frac = ((height - voxel_bottom) / self.voxel_size).clamp(0.0, 1.0);
        let flags = ((h_frac * 254.0) as u8).max(1);

        Voxel {
            color: gradient_color,
            material_id: base.material_id,
            flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_terrain_params_default() {
        let params = TerrainParams::default();
        assert_eq!(params.seed, 12345);
        assert_eq!(params.scale, 100.0);
        assert_eq!(params.height_scale, 64.0);
        assert_eq!(params.octaves, 4);
        assert_eq!(params.persistence, 0.5);
        assert_eq!(params.lacunarity, 2.0);
        assert_eq!(params.sea_level, 32.0);
    }

    #[test]
    fn test_terrain_generator_new() {
        let params = TerrainParams::default();
        let generator = TerrainGenerator::new(params);
        assert_eq!(generator.params.seed, 12345);
    }

    #[test]
    fn test_height_at() {
        let generator = TerrainGenerator::new(TerrainParams::default());

        let height = generator.height_at(0.0, 0.0);
        assert!(height >= 0.0);
        assert!(height <= 64.0);

        let height2 = generator.height_at(0.0, 0.0);
        assert_eq!(height, height2);

        let height3 = generator.height_at(100.0, 100.0);
        assert!(height3 >= 0.0);
        assert!(height3 <= 64.0);
    }

    #[test]
    fn test_height_at_consistency() {
        let generator = TerrainGenerator::new(TerrainParams::default());
        let positions = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0), (-50.0, -50.0)];

        for (x, z) in positions {
            let h1 = generator.height_at(x, z);
            let h2 = generator.height_at(x, z);
            assert_eq!(h1, h2, "Height should be consistent at ({}, {})", x, z);
        }
    }

    #[test]
    fn test_generate_chunk_different_seeds() {
        let gen1 = TerrainGenerator::new(TerrainParams { seed: 1, ..Default::default() });
        let gen2 = TerrainGenerator::new(TerrainParams { seed: 2, ..Default::default() });

        let h1 = gen1.height_at(50.0, 50.0);
        let h2 = gen2.height_at(50.0, 50.0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_height_bounds() {
        let generator = TerrainGenerator::new(TerrainParams::default());

        let (min_h, max_h) = generator.height_bounds(0.0, 10.0, 0.0, 10.0);
        assert!(min_h <= max_h);
        assert!(min_h >= 0.0);
        assert!(max_h <= 64.0);

        let h1 = generator.height_at(0.0, 0.0);
        let h2 = generator.height_at(10.0, 10.0);
        let h3 = generator.height_at(5.0, 5.0);

        assert!(h1 >= min_h && h1 <= max_h);
        assert!(h2 >= min_h && h2 <= max_h);
        assert!(h3 >= min_h && h3 <= max_h);
    }

    #[test]
    fn test_biome_classifier_empty_above_terrain() {
        let generator = TerrainGenerator::new(TerrainParams::default());
        let biome_map = BiomeMap::new(12345);
        let classifier = BiomeTerrainClassifier {
            terrain: &generator,
            biome_map: &biome_map,
            voxel_size: 4.0 / 128.0,
        };

        let aabb = Aabb::new(Vec3::new(0.0, 200.0, 0.0), Vec3::new(64.0, 264.0, 64.0));
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_biome_classifier_empty_below_terrain() {
        let generator = TerrainGenerator::new(TerrainParams {
            height_scale: 64.0,
            ..Default::default()
        });
        let biome_map = BiomeMap::new(12345);
        let classifier = BiomeTerrainClassifier {
            terrain: &generator,
            biome_map: &biome_map,
            voxel_size: 4.0 / 128.0,
        };

        let aabb = Aabb::new(Vec3::new(0.0, -100.0, 0.0), Vec3::new(64.0, -50.0, 64.0));
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_biome_classifier_mixed_at_surface() {
        let generator = TerrainGenerator::new(TerrainParams::default());
        let biome_map = BiomeMap::new(12345);
        let classifier = BiomeTerrainClassifier {
            terrain: &generator,
            biome_map: &biome_map,
            voxel_size: 4.0 / 128.0,
        };

        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(64.0, 64.0, 64.0));
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Mixed);
    }

    #[test]
    fn test_biome_classifier_evaluate() {
        let generator = TerrainGenerator::new(TerrainParams::default());
        let biome_map = BiomeMap::new(12345);
        let classifier = BiomeTerrainClassifier {
            terrain: &generator,
            biome_map: &biome_map,
            voxel_size: 4.0 / 128.0,
        };

        let voxel = classifier.evaluate(Vec3::new(0.0, 200.0, 0.0));
        assert!(voxel.is_empty());

        // Deep underground should be empty (thin shell only)
        let height = generator.height_at(0.0, 0.0);
        let voxel = classifier.evaluate(Vec3::new(0.0, height - 10.0, 0.0));
        assert!(voxel.is_empty());
    }

    #[test]
    fn test_generate_chunk_adaptive() {
        let generator = TerrainGenerator::new(TerrainParams::default());
        let biome_map = BiomeMap::new(12345);

        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = generator.generate_chunk_adaptive(coord, &biome_map);

        assert_eq!(chunk.coord, coord);
        assert!(chunk.modified);
        assert!(chunk.octree.node_count() > 0);
    }

    #[test]
    fn test_generate_chunk_adaptive_empty() {
        let generator = TerrainGenerator::new(TerrainParams {
            height_scale: 10.0,
            ..Default::default()
        });
        let biome_map = BiomeMap::new(12345);

        let coord = ChunkCoord::new(0, 5, 0);
        let chunk = generator.generate_chunk_adaptive(coord, &biome_map);

        assert_eq!(chunk.coord, coord);
        assert!(chunk.modified);
        assert_eq!(chunk.octree.brick_count(), 0);
    }
}
