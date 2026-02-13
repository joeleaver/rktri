//! WorldIndex - spatial lookup for chunks and bricks.

use std::collections::HashMap;
use glam::Vec3;

use crate::voxel::chunk::ChunkCoord;
use crate::voxel::chunk_handle::ChunkHandle;
use crate::voxel::super_chunk::{SuperChunk, SuperChunkCoord};
use crate::voxel::hierarchy::{world_to_chunk, world_to_super_chunk, CHUNK_SIZE_METERS};
use crate::math::Aabb;

/// Spatial index for efficient chunk and brick lookup.
pub struct WorldIndex {
    /// SuperChunk lookup (sparse for large worlds)
    super_chunks: HashMap<SuperChunkCoord, SuperChunk>,
    /// Direct chunk lookup (for fast access)
    chunks: HashMap<ChunkCoord, ChunkHandle>,
    /// Chunks near camera (for dense iteration)
    local_cache: Vec<ChunkCoord>,
    /// Camera position for local cache
    cache_center: Vec3,
    /// Local cache radius in chunks
    cache_radius: i32,
}

impl WorldIndex {
    /// Create a new world index.
    pub fn new() -> Self {
        Self {
            super_chunks: HashMap::new(),
            chunks: HashMap::new(),
            local_cache: Vec::new(),
            cache_center: Vec3::ZERO,
            cache_radius: 8, // 8 chunks = 32m
        }
    }

    /// Add a chunk to the index.
    pub fn add_chunk(&mut self, handle: ChunkHandle) {
        let coord = handle.coord;

        // Add to chunk map
        self.chunks.insert(coord, handle);

        // Update SuperChunk
        let (sx, sy, sz) = world_to_super_chunk(Vec3::new(
            coord.x as f32 * CHUNK_SIZE_METERS,
            coord.y as f32 * CHUNK_SIZE_METERS,
            coord.z as f32 * CHUNK_SIZE_METERS,
        ));
        let super_coord = SuperChunkCoord::new(sx, sy, sz);

        let super_chunk = self.super_chunks
            .entry(super_coord)
            .or_insert_with(|| SuperChunk::new(super_coord));

        // Calculate local position within SuperChunk
        let local_x = (coord.x - sx * 16) as u32;
        let local_y = (coord.y - sy * 16) as u32;
        let local_z = (coord.z - sz * 16) as u32;
        super_chunk.chunk_mask.set_present(local_x, local_y, local_z, true);
    }

    /// Remove a chunk from the index.
    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<ChunkHandle> {
        self.chunks.remove(&coord)
    }

    /// Get a chunk by coordinate.
    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&ChunkHandle> {
        self.chunks.get(&coord)
    }

    /// Get mutable chunk by coordinate.
    pub fn get_chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut ChunkHandle> {
        self.chunks.get_mut(&coord)
    }

    /// Get chunk at world position.
    pub fn chunk_at_position(&self, pos: Vec3) -> Option<&ChunkHandle> {
        let (x, y, z) = world_to_chunk(pos);
        self.get_chunk(ChunkCoord::new(x, y, z))
    }

    /// Update local cache around camera.
    pub fn update_local_cache(&mut self, camera_pos: Vec3) {
        self.cache_center = camera_pos;
        self.local_cache.clear();

        let (cx, cy, cz) = world_to_chunk(camera_pos);
        let r = self.cache_radius;

        for x in (cx - r)..=(cx + r) {
            for y in (cy - r)..=(cy + r) {
                for z in (cz - r)..=(cz + r) {
                    let coord = ChunkCoord::new(x, y, z);
                    if self.chunks.contains_key(&coord) {
                        self.local_cache.push(coord);
                    }
                }
            }
        }
    }

    /// Get chunks in local cache (near camera).
    pub fn local_chunks(&self) -> impl Iterator<Item = &ChunkHandle> {
        self.local_cache.iter().filter_map(|c| self.chunks.get(c))
    }

    /// Get all chunk coordinates.
    pub fn all_chunk_coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.chunks.keys()
    }

    /// Total number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total number of SuperChunks.
    pub fn super_chunk_count(&self) -> usize {
        self.super_chunks.len()
    }

    /// Iterate chunks in AABB.
    pub fn chunks_in_aabb(&self, aabb: &Aabb) -> Vec<&ChunkHandle> {
        let min_coord = world_to_chunk(aabb.min);
        let max_coord = world_to_chunk(aabb.max);

        let mut result = Vec::new();
        for x in min_coord.0..=max_coord.0 {
            for y in min_coord.1..=max_coord.1 {
                for z in min_coord.2..=max_coord.2 {
                    if let Some(handle) = self.get_chunk(ChunkCoord::new(x, y, z)) {
                        result.push(handle);
                    }
                }
            }
        }
        result
    }

    /// Get SuperChunk at coordinate.
    pub fn get_super_chunk(&self, coord: SuperChunkCoord) -> Option<&SuperChunk> {
        self.super_chunks.get(&coord)
    }
}

impl Default for WorldIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_chunk() {
        let mut index = WorldIndex::new();
        let coord = ChunkCoord::new(1, 2, 3);
        let handle = ChunkHandle::new(coord);

        index.add_chunk(handle);

        assert!(index.get_chunk(coord).is_some());
        assert_eq!(index.chunk_count(), 1);
    }

    #[test]
    fn test_chunk_at_position() {
        let mut index = WorldIndex::new();
        let coord = ChunkCoord::new(0, 0, 0);
        index.add_chunk(ChunkHandle::new(coord));

        // Position inside chunk (0,0,0) which spans 0-4m
        let handle = index.chunk_at_position(Vec3::new(2.0, 2.0, 2.0));
        assert!(handle.is_some());

        // Position in different chunk
        let handle = index.chunk_at_position(Vec3::new(10.0, 0.0, 0.0));
        assert!(handle.is_none());
    }

    #[test]
    fn test_local_cache() {
        let mut index = WorldIndex::new();

        // Add some chunks
        for x in -2..=2 {
            for z in -2..=2 {
                index.add_chunk(ChunkHandle::new(ChunkCoord::new(x, 0, z)));
            }
        }

        index.update_local_cache(Vec3::ZERO);

        // All 25 chunks should be in local cache (within radius 8)
        assert_eq!(index.local_chunks().count(), 25);
    }
}
