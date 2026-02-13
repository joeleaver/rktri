//! World generation pipeline — builds terrain chunks using mask layers.
//!
//! The pipeline orchestrates:
//! 1. Biome mask construction (MaskOctree<BiomeId>)
//! 2. Terrain octree building (via MaskDrivenTerrainClassifier)
//! 3. Grass mask construction (MaskOctree<GrassCell>)

pub mod config;
pub mod biome_gen;
pub mod terrain_gen;
pub mod grass_gen;

pub use config::GenerationConfig;
pub use biome_gen::BiomeNoiseGenerator;
pub use terrain_gen::MaskDrivenTerrainClassifier;
pub use grass_gen::GrassNoiseGenerator;

use glam::Vec3;
use rayon::prelude::*;
use crate::grass::profile::{GrassCell, GrassProfileTable};
use crate::mask::{MaskBuilder, MaskOctree};
use crate::terrain::biome::BiomeMap;
use crate::terrain::generator::TerrainGenerator;
use crate::voxel::chunk::{Chunk, ChunkCoord, CHUNK_SIZE};
use crate::voxel::svo::adaptive::AdaptiveOctreeBuilder;

/// Result of generating a single chunk: terrain octree + grass mask.
pub struct GeneratedChunk {
    pub chunk: Chunk,
    pub grass_mask: MaskOctree<GrassCell>,
}

/// Orchestrates chunk generation: biome mask → terrain octree → grass mask.
pub struct GenerationPipeline {
    terrain: TerrainGenerator,
    biome_map: BiomeMap,
    biome_mask_depth: u8,
    grass_mask_depth: u8,
    sea_level: f32,
    profile_table: GrassProfileTable,
    seed: u32,
}

impl GenerationPipeline {
    /// Create a new pipeline from configuration.
    pub fn new(config: &GenerationConfig) -> Self {
        let terrain = TerrainGenerator::new(config.terrain_params.clone());
        let biome_map = BiomeMap::new(config.seed);

        Self {
            terrain,
            biome_map,
            biome_mask_depth: config.biome_mask_depth,
            grass_mask_depth: config.grass_mask_depth,
            sea_level: config.terrain_params.sea_level,
            profile_table: GrassProfileTable::default(),
            seed: config.seed,
        }
    }

    /// Generate a single chunk with biome-aware terrain (no grass mask).
    ///
    /// Use `generate_chunk_with_grass` for full generation including grass.
    pub fn generate_chunk(&self, coord: ChunkCoord) -> Chunk {
        self.generate_chunk_with_grass(coord).chunk
    }

    /// Generate a single chunk with biome-aware terrain and grass mask.
    pub fn generate_chunk_with_grass(&self, coord: ChunkCoord) -> GeneratedChunk {
        let origin = coord.world_origin();
        let chunk_size = CHUNK_SIZE as f32;

        // 1. Build biome mask for this chunk
        let biome_gen = BiomeNoiseGenerator::new(&self.biome_map, &self.terrain, self.sea_level);
        let biome_mask = MaskBuilder::new(self.biome_mask_depth)
            .build(&biome_gen, origin, chunk_size);

        // 2. Build terrain octree reading biome from mask
        const TERRAIN_VOXELS: u32 = 128;
        let voxel_size = chunk_size / TERRAIN_VOXELS as f32;
        let classifier = MaskDrivenTerrainClassifier::new(
            &self.terrain,
            &biome_mask,
            origin,
            voxel_size,
        );
        let builder = AdaptiveOctreeBuilder::new(TERRAIN_VOXELS);
        let octree = builder.build(&classifier, origin, chunk_size);

        // 3. Build grass mask from biome mask + slope
        let grass_gen = GrassNoiseGenerator::new(
            &self.terrain, &biome_mask, origin, &self.profile_table, self.seed,
        );
        let grass_mask = MaskBuilder::new(self.grass_mask_depth)
            .build(&grass_gen, origin, chunk_size);

        let mut chunk = Chunk::from_octree(coord, octree);
        chunk.modified = true;

        GeneratedChunk { chunk, grass_mask }
    }

    /// Get terrain height at a world position (delegates to TerrainGenerator).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.terrain.height_at(x, z)
    }

    /// Get a reference to the terrain generator.
    pub fn terrain(&self) -> &TerrainGenerator {
        &self.terrain
    }

    /// Get a reference to the biome map.
    pub fn biome_map(&self) -> &BiomeMap {
        &self.biome_map
    }

    /// Get a reference to the profile table.
    pub fn profile_table(&self) -> &GrassProfileTable {
        &self.profile_table
    }

    /// Generate chunks around a center position, skipping already-existing coords.
    ///
    /// Uses height-guided Y-level filtering to only generate chunks near the
    /// terrain surface (typically 2-3 Y levels per XZ column).
    /// Parallelized with rayon for large worlds.
    ///
    /// Returns a list of (coord, chunk) pairs for the caller to insert.
    pub fn generate_chunks_around(
        &self,
        center: Vec3,
        radius: f32,
        existing: &dyn Fn(ChunkCoord) -> bool,
    ) -> Vec<(ChunkCoord, Chunk)> {
        let chunk_f = CHUNK_SIZE as f32;
        let chunk_radius = (radius / chunk_f).floor() as i32;
        let center_coord = ChunkCoord::from_world_pos(center);

        // Phase 1: Collect coords to generate (sequential — needs `existing` callback)
        let mut coords_to_generate = Vec::new();
        for dx in -chunk_radius..=chunk_radius {
            for dz in -chunk_radius..=chunk_radius {
                let cx = (center_coord.x + dx) as f32 * chunk_f + chunk_f * 0.5;
                let cz = (center_coord.z + dz) as f32 * chunk_f + chunk_f * 0.5;
                let h = self.terrain.height_at(cx, cz);

                let min_y = ((h - chunk_f) / chunk_f).floor().max(0.0) as i32;
                let max_y = ((h + chunk_f) / chunk_f).ceil() as i32;

                for dy in min_y..=max_y {
                    let coord = ChunkCoord::new(
                        center_coord.x + dx,
                        dy,
                        center_coord.z + dz,
                    );
                    if !existing(coord) {
                        coords_to_generate.push(coord);
                    }
                }
            }
        }

        log::info!("Generating {} candidate chunks ({} chunk radius, {}m)...",
            coords_to_generate.len(), chunk_radius, radius * 2.0);

        // Phase 2: Generate chunks in parallel
        let start = std::time::Instant::now();
        let results: Vec<_> = coords_to_generate
            .par_iter()
            .filter_map(|&coord| {
                let chunk = self.generate_chunk(coord);
                if chunk.octree.brick_count() > 0 {
                    Some((coord, chunk))
                } else {
                    None
                }
            })
            .collect();

        let elapsed = start.elapsed();
        log::info!("Generated {} chunks with geometry in {:.1}s ({:.0} chunks/sec)",
            results.len(), elapsed.as_secs_f64(),
            results.len() as f64 / elapsed.as_secs_f64());

        results
    }

    /// Generate chunks with grass masks around a center position.
    ///
    /// Like `generate_chunks_around` but returns `GeneratedChunk` with grass masks.
    pub fn generate_chunks_with_grass_around(
        &self,
        center: Vec3,
        radius: f32,
        existing: &dyn Fn(ChunkCoord) -> bool,
    ) -> Vec<(ChunkCoord, GeneratedChunk)> {
        let chunk_f = CHUNK_SIZE as f32;
        let chunk_radius = (radius / chunk_f).floor() as i32;
        let center_coord = ChunkCoord::from_world_pos(center);

        let mut coords_to_generate = Vec::new();
        for dx in -chunk_radius..=chunk_radius {
            for dz in -chunk_radius..=chunk_radius {
                let cx = (center_coord.x + dx) as f32 * chunk_f + chunk_f * 0.5;
                let cz = (center_coord.z + dz) as f32 * chunk_f + chunk_f * 0.5;
                let h = self.terrain.height_at(cx, cz);

                let min_y = ((h - chunk_f) / chunk_f).floor().max(0.0) as i32;
                let max_y = ((h + chunk_f) / chunk_f).ceil() as i32;

                for dy in min_y..=max_y {
                    let coord = ChunkCoord::new(
                        center_coord.x + dx,
                        dy,
                        center_coord.z + dz,
                    );
                    if !existing(coord) {
                        coords_to_generate.push(coord);
                    }
                }
            }
        }

        log::info!("Generating {} candidate chunks with grass ({} chunk radius, {}m)...",
            coords_to_generate.len(), chunk_radius, radius * 2.0);

        let start = std::time::Instant::now();
        let results: Vec<_> = coords_to_generate
            .par_iter()
            .filter_map(|&coord| {
                let generated = self.generate_chunk_with_grass(coord);
                if generated.chunk.octree.brick_count() > 0 {
                    Some((coord, generated))
                } else {
                    None
                }
            })
            .collect();

        let elapsed = start.elapsed();
        log::info!("Generated {} chunks with geometry + grass in {:.1}s ({:.0} chunks/sec)",
            results.len(), elapsed.as_secs_f64(),
            results.len() as f64 / elapsed.as_secs_f64());

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::generator::TerrainParams;

    fn test_config() -> GenerationConfig {
        GenerationConfig {
            seed: 12345,
            terrain_params: TerrainParams {
                scale: 150.0,
                height_scale: 80.0,
                octaves: 5,
                sea_level: 20.0,
                ..Default::default()
            },
            biome_mask_depth: 3,
            grass_mask_depth: 5,
        }
    }

    #[test]
    fn test_pipeline_create() {
        let config = test_config();
        let _pipeline = GenerationPipeline::new(&config);
    }

    #[test]
    fn test_pipeline_generate_chunk() {
        let config = test_config();
        let pipeline = GenerationPipeline::new(&config);

        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = pipeline.generate_chunk(coord);

        assert_eq!(chunk.coord, coord);
        assert!(chunk.modified);
    }

    #[test]
    fn test_pipeline_generate_chunk_with_grass() {
        let config = test_config();
        let pipeline = GenerationPipeline::new(&config);

        let coord = ChunkCoord::new(0, 0, 0);
        let generated = pipeline.generate_chunk_with_grass(coord);

        assert_eq!(generated.chunk.coord, coord);
        assert!(generated.chunk.modified);
        // Grass mask should exist (even if empty)
        assert!(generated.grass_mask.node_count() >= 1);
    }

    #[test]
    fn test_pipeline_generate_chunk_at_surface() {
        let config = test_config();
        let pipeline = GenerationPipeline::new(&config);

        // Find the Y level where terrain exists at x=0, z=0
        let h = pipeline.height_at(2.0, 2.0);
        let y_level = (h / CHUNK_SIZE as f32).floor() as i32;

        let coord = ChunkCoord::new(0, y_level, 0);
        let chunk = pipeline.generate_chunk(coord);

        // Chunk at terrain surface should have geometry
        assert!(chunk.octree.brick_count() > 0,
            "Expected geometry at surface level y={} (height={})",
            y_level, h);
    }

    #[test]
    fn test_pipeline_height_at() {
        let config = test_config();
        let pipeline = GenerationPipeline::new(&config);

        let h = pipeline.height_at(0.0, 0.0);
        assert!(h >= 0.0 && h <= 80.0);
    }

    #[test]
    fn test_pipeline_generate_chunks_around() {
        let config = GenerationConfig {
            seed: 12345,
            terrain_params: TerrainParams {
                scale: 150.0,
                height_scale: 80.0,
                octaves: 5,
                sea_level: 20.0,
                ..Default::default()
            },
            biome_mask_depth: 3,
            grass_mask_depth: 5,
        };
        let pipeline = GenerationPipeline::new(&config);

        // Generate a small radius of chunks
        let center = Vec3::new(2.0, 55.0, 2.0);
        let results = pipeline.generate_chunks_around(center, 4.0, &|_| false);

        // Should produce at least some chunks
        assert!(!results.is_empty(), "Expected at least one chunk near terrain");

        // All chunks should have geometry
        for (coord, chunk) in &results {
            assert!(chunk.octree.brick_count() > 0,
                "Chunk {:?} should have geometry", coord);
        }
    }
}
