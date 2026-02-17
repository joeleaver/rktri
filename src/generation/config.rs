//! Generation configuration extracted from SceneConfig.

use crate::terrain::generator::TerrainParams;

/// Configuration for the terrain generation pipeline.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Random seed for terrain and biome generation.
    pub seed: u32,
    /// Terrain noise parameters.
    pub terrain_params: TerrainParams,
    /// Max depth for biome mask octree (depth 3 = 8 cells/side = 0.5m resolution).
    pub biome_mask_depth: u8,
    /// Max depth for grass mask octree (depth 5 = 32 cells/side = 0.125m resolution).
    pub grass_mask_depth: u8,
    /// Max depth for rock mask octree (depth 4 = 16 cells/side = 0.25m resolution).
    pub rock_mask_depth: u8,
    /// Max depth for tree mask octree (depth 4 = 16 cells/side = 0.25m resolution).
    pub tree_mask_depth: u8,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            seed: 12345,
            terrain_params: TerrainParams::default(),
            biome_mask_depth: 3,
            grass_mask_depth: 5,
            rock_mask_depth: 4,
            tree_mask_depth: 4,
        }
    }
}

impl GenerationConfig {
    /// Create from scene-level terrain params and seed.
    pub fn from_terrain(seed: u32, terrain_params: TerrainParams) -> Self {
        Self {
            seed,
            terrain_params,
            biome_mask_depth: 3,
            grass_mask_depth: 5,
            rock_mask_depth: 4,
            tree_mask_depth: 4,
        }
    }
}
