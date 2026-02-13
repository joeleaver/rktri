//! Scene manager for generating and managing test scenes

use std::collections::HashMap;

use glam::Vec3;
use crate::generation::{GenerationConfig, GenerationPipeline};
use crate::voxel::{
    chunk::{ChunkCoord, CHUNK_SIZE},
    svo::Octree,
    layer::LayerId,
};
use super::config::SceneConfig;
use super::flatten::FlatChunkEntry;
use super::graph::SceneGraph;
use super::node::{NodeContent, SceneNodeId};

/// Manages the test scene: terrain generation with biome-aware coloring.
pub struct SceneManager {
    config: SceneConfig,
    pipeline: GenerationPipeline,
    scene_graph: SceneGraph,
    terrain_node: SceneNodeId,
}

impl SceneManager {
    /// Create a new scene manager from configuration
    pub fn new(config: SceneConfig) -> Self {
        let gen_config = GenerationConfig::from_terrain(config.seed, config.terrain_params.clone());
        let pipeline = GenerationPipeline::new(&gen_config);

        let mut scene_graph = SceneGraph::new();
        let root = scene_graph.root();
        let terrain_node = scene_graph.add_child(
            root,
            "terrain",
            LayerId::TERRAIN,
            NodeContent::ChunkedRegion { chunks: HashMap::new() },
        );

        Self {
            config,
            pipeline,
            scene_graph,
            terrain_node,
        }
    }

    /// Generate a terrain-only chunk with biome-aware coloring.
    pub fn generate_chunk(&self, coord: ChunkCoord) -> crate::voxel::chunk::Chunk {
        self.pipeline.generate_chunk(coord)
    }

    /// Insert a generated chunk's octree into the scene graph's terrain layer.
    pub fn insert_chunk(&mut self, coord: ChunkCoord, octree: Octree) {
        if let Some(node) = self.scene_graph.get_mut(self.terrain_node) {
            if let NodeContent::ChunkedRegion { ref mut chunks } = node.content {
                chunks.insert(coord, octree);
            }
        }
    }

    /// Generate chunks around a center position within view distance.
    ///
    /// For each XZ column, samples the terrain height and only generates Y levels
    /// near the surface (typically 2-3 instead of all 21). Empty chunks are not
    /// inserted into the scene graph.
    pub fn generate_chunks_around(&mut self, center: Vec3, radius: f32) {
        let center_coord = ChunkCoord::from_world_pos(center);
        let chunk_radius = (radius / CHUNK_SIZE as f32).floor() as i32;

        log::info!(
            "Generating chunks: center={:?}, radius={}, chunk_radius={}",
            center_coord, radius, chunk_radius
        );

        let results = self.pipeline.generate_chunks_around(
            center,
            radius,
            &|coord| self.has_chunk(coord),
        );

        let generated = results.len() as u32;
        for (coord, chunk) in results {
            self.insert_chunk(coord, chunk.octree);
        }

        log::info!(
            "Chunk generation complete: {} with geometry inserted",
            generated
        );
    }

    /// Check whether a chunk coordinate is already loaded in the terrain layer.
    fn has_chunk(&self, coord: ChunkCoord) -> bool {
        if let Some(node) = self.scene_graph.get(self.terrain_node) {
            if let NodeContent::ChunkedRegion { ref chunks } = node.content {
                return chunks.contains_key(&coord);
            }
        }
        false
    }

    /// Get all loaded chunks as (coord, octree) pairs for multi-chunk GPU upload.
    ///
    /// This is the backward-compatible bridge — main.rs sees the same API.
    pub fn get_chunks(&self) -> Vec<(ChunkCoord, Octree)> {
        if let Some(node) = self.scene_graph.get(self.terrain_node) {
            if let NodeContent::ChunkedRegion { ref chunks } = node.content {
                return chunks.iter().map(|(c, o)| (*c, o.clone())).collect();
            }
        }
        Vec::new()
    }

    /// Flatten the entire scene graph into GPU-ready entries.
    pub fn flatten(&mut self) -> Vec<FlatChunkEntry> {
        self.scene_graph.flatten()
    }

    /// Get a reference to the scene graph.
    pub fn scene_graph(&self) -> &SceneGraph {
        &self.scene_graph
    }

    /// Get a mutable reference to the scene graph.
    pub fn scene_graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.scene_graph
    }

    /// Get the terrain node ID.
    pub fn terrain_node(&self) -> SceneNodeId {
        self.terrain_node
    }

    /// Get the scene configuration
    pub fn config(&self) -> &SceneConfig {
        &self.config
    }

    /// Get a reference to the generation pipeline.
    pub fn pipeline(&self) -> &GenerationPipeline {
        &self.pipeline
    }

    /// Get chunk count (terrain layer only)
    pub fn chunk_count(&self) -> usize {
        if let Some(node) = self.scene_graph.get(self.terrain_node) {
            if let NodeContent::ChunkedRegion { ref chunks } = node.content {
                return chunks.len();
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::svo::Octree;
    use crate::voxel::chunk::CHUNK_SIZE;

    /// Create a minimal mock octree for tests that don't need real terrain.
    /// Avoids the 512MB dense array allocation from generate_chunk_with_biome.
    fn mock_octree() -> Octree {
        Octree::new(CHUNK_SIZE as f32, 3)
    }

    #[test]
    fn test_scene_manager_new() {
        let config = SceneConfig::default();
        let manager = SceneManager::new(config.clone());

        assert_eq!(manager.chunk_count(), 0);
        assert_eq!(manager.config().seed, config.seed);
    }

    #[test]
    fn test_insert_multiple_chunks() {
        let config = SceneConfig::default();
        let mut manager = SceneManager::new(config);

        let coords = [
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(1, 0, 0),
            ChunkCoord::new(0, 0, 1),
        ];
        for coord in &coords {
            manager.insert_chunk(*coord, mock_octree());
        }
        assert_eq!(manager.chunk_count(), 3);
    }

    #[test]
    fn test_no_duplicate_chunks() {
        let config = SceneConfig::default();
        let mut manager = SceneManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);

        manager.insert_chunk(coord, mock_octree());
        assert_eq!(manager.chunk_count(), 1);

        // Insert at same coord again — should replace, not duplicate
        manager.insert_chunk(coord, mock_octree());
        assert_eq!(manager.chunk_count(), 1);
    }

    #[test]
    fn test_config_access() {
        let config = SceneConfig {
            seed: 99999,
            ..Default::default()
        };
        let manager = SceneManager::new(config.clone());

        assert_eq!(manager.config().seed, 99999);
    }

    #[test]
    fn test_get_chunks_from_scene_graph() {
        let config = SceneConfig::default();
        let mut manager = SceneManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);
        manager.insert_chunk(coord, mock_octree());

        let chunks = manager.get_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, coord);
    }

    #[test]
    fn test_flatten_terrain() {
        let config = SceneConfig::default();
        let mut manager = SceneManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);
        manager.insert_chunk(coord, mock_octree());

        let entries = manager.flatten();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].layer_id, LayerId::TERRAIN);
        assert_eq!(entries[0].coord, coord);
    }

    #[test]
    fn test_scene_graph_access() {
        let config = SceneConfig::default();
        let manager = SceneManager::new(config);

        // Root + terrain node
        assert_eq!(manager.scene_graph().node_count(), 2);
    }

    // --- Heavy integration tests ---
    // Run with: cargo test -p rktri -- scene::manager --ignored

    #[test]
    #[ignore]
    fn test_generate_chunk() {
        let config = SceneConfig::default();
        let manager = SceneManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = manager.generate_chunk(coord);
        assert_eq!(chunk.coord, coord);
        assert!(chunk.octree.root_size() > 0.0);
    }

    #[test]
    #[ignore]
    fn test_flatten_with_terrain() {
        let config = SceneConfig::default();
        let manager = SceneManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = manager.generate_chunk(coord);
        let mut manager2 = SceneManager::new(SceneConfig::default());
        manager2.insert_chunk(chunk.coord, chunk.octree);

        let entries = manager2.flatten();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].layer_id, LayerId::TERRAIN);
    }
}
