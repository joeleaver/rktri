//! Chunk invalidation tracking for dirty voxel regions after edits.

use std::collections::{HashMap, HashSet};
use crate::math::Aabb;
use crate::voxel::chunk::ChunkCoord;
use crate::voxel::brick_handle::BrickId;

const CHUNK_SIZE_METERS: f32 = 4.0;

/// Tracks which chunks and bricks need rebuild after edits.
///
/// When voxels are edited, affected chunks must be marked for rebuild.
/// This structure tracks dirty state at two granularities:
/// - Whole chunks (full rebuild)
/// - Individual bricks (sub-chunk updates)
#[derive(Debug)]
pub struct ChunkInvalidator {
    /// Chunks needing full rebuild
    dirty_chunks: HashSet<ChunkCoord>,
    /// Bricks needing update (sub-chunk granularity)
    dirty_bricks: HashSet<BrickId>,
    /// Generation counters for cache invalidation
    generations: HashMap<ChunkCoord, u32>,
}

impl ChunkInvalidator {
    /// Create a new chunk invalidator with empty state.
    pub fn new() -> Self {
        Self {
            dirty_chunks: HashSet::new(),
            dirty_bricks: HashSet::new(),
            generations: HashMap::new(),
        }
    }

    /// Mark a region (AABB) as dirty - computes affected chunks and bricks.
    ///
    /// Calculates which chunk coordinates overlap the given region,
    /// marks them dirty, and increments their generation counters.
    pub fn mark_dirty(&mut self, region: &Aabb) {
        let chunk_size = CHUNK_SIZE_METERS;

        // Calculate chunk range from AABB bounds
        let min_cx = (region.min.x / chunk_size).floor() as i32;
        let min_cy = (region.min.y / chunk_size).floor() as i32;
        let min_cz = (region.min.z / chunk_size).floor() as i32;
        let max_cx = (region.max.x / chunk_size).floor() as i32;
        let max_cy = (region.max.y / chunk_size).floor() as i32;
        let max_cz = (region.max.z / chunk_size).floor() as i32;

        // Mark all affected chunks
        for x in min_cx..=max_cx {
            for y in min_cy..=max_cy {
                for z in min_cz..=max_cz {
                    let coord = ChunkCoord::new(x, y, z);
                    self.dirty_chunks.insert(coord);
                    // Increment generation counter for cache invalidation
                    let generation = self.generations.entry(coord).or_insert(0);
                    *generation = generation.wrapping_add(1);
                }
            }
        }
    }

    /// Mark a specific chunk as dirty and increment its generation.
    pub fn mark_chunk_dirty(&mut self, coord: ChunkCoord) {
        self.dirty_chunks.insert(coord);
        let generation = self.generations.entry(coord).or_insert(0);
        *generation = generation.wrapping_add(1);
    }

    /// Mark a specific brick as dirty.
    pub fn mark_brick_dirty(&mut self, brick_id: BrickId) {
        self.dirty_bricks.insert(brick_id);
        // Also mark the containing chunk dirty
        self.mark_chunk_dirty(brick_id.chunk);
    }

    /// Take all dirty chunks and clear the dirty list.
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkCoord> {
        self.dirty_chunks.drain().collect()
    }

    /// Take all dirty bricks and clear the dirty list.
    pub fn take_dirty_bricks(&mut self) -> Vec<BrickId> {
        self.dirty_bricks.drain().collect()
    }

    /// Get the current generation counter for a chunk.
    ///
    /// Generation counter increments each time a chunk is marked dirty.
    /// Useful for cache invalidation and version checking.
    pub fn generation(&self, coord: &ChunkCoord) -> u32 {
        self.generations.get(coord).copied().unwrap_or(0)
    }

    /// Check if there are any dirty chunks or bricks.
    pub fn has_dirty(&self) -> bool {
        !self.dirty_chunks.is_empty() || !self.dirty_bricks.is_empty()
    }

    /// Check if a specific chunk is marked dirty.
    pub fn is_chunk_dirty(&self, coord: &ChunkCoord) -> bool {
        self.dirty_chunks.contains(coord)
    }

    /// Check if a specific brick is marked dirty.
    pub fn is_brick_dirty(&self, brick_id: &BrickId) -> bool {
        self.dirty_bricks.contains(brick_id)
    }

    /// Clear all dirty state and reset generation counters.
    pub fn clear(&mut self) {
        self.dirty_chunks.clear();
        self.dirty_bricks.clear();
        self.generations.clear();
    }

    /// Get count of dirty chunks (for diagnostics).
    pub fn dirty_chunk_count(&self) -> usize {
        self.dirty_chunks.len()
    }

    /// Get count of dirty bricks (for diagnostics).
    pub fn dirty_brick_count(&self) -> usize {
        self.dirty_bricks.len()
    }
}

impl Default for ChunkInvalidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Vec3;

    #[test]
    fn test_new() {
        let invalidator = ChunkInvalidator::new();
        assert!(!invalidator.has_dirty());
        assert_eq!(invalidator.dirty_chunk_count(), 0);
        assert_eq!(invalidator.dirty_brick_count(), 0);
    }

    #[test]
    fn test_mark_chunk_dirty() {
        let mut invalidator = ChunkInvalidator::new();
        let coord = ChunkCoord::new(0, 0, 0);

        invalidator.mark_chunk_dirty(coord);
        assert!(invalidator.has_dirty());
        assert!(invalidator.is_chunk_dirty(&coord));
        assert_eq!(invalidator.generation(&coord), 1);
    }

    #[test]
    fn test_mark_chunk_dirty_multiple_times() {
        let mut invalidator = ChunkInvalidator::new();
        let coord = ChunkCoord::new(1, 2, 3);

        invalidator.mark_chunk_dirty(coord);
        assert_eq!(invalidator.generation(&coord), 1);

        invalidator.mark_chunk_dirty(coord);
        assert_eq!(invalidator.generation(&coord), 2);

        invalidator.mark_chunk_dirty(coord);
        assert_eq!(invalidator.generation(&coord), 3);
    }

    #[test]
    fn test_take_dirty_chunks() {
        let mut invalidator = ChunkInvalidator::new();
        let coord1 = ChunkCoord::new(0, 0, 0);
        let coord2 = ChunkCoord::new(1, 1, 1);

        invalidator.mark_chunk_dirty(coord1);
        invalidator.mark_chunk_dirty(coord2);

        assert_eq!(invalidator.dirty_chunk_count(), 2);
        let dirty = invalidator.take_dirty_chunks();
        assert_eq!(dirty.len(), 2);
        assert!(!invalidator.has_dirty());
    }

    #[test]
    fn test_mark_dirty_region_single_chunk() {
        let mut invalidator = ChunkInvalidator::new();

        // Region entirely within one chunk (0,0,0)
        let region = Aabb {
            min: Vec3::new(0.5, 0.5, 0.5),
            max: Vec3::new(1.5, 1.5, 1.5),
        };

        invalidator.mark_dirty(&region);
        assert_eq!(invalidator.dirty_chunk_count(), 1);

        let chunks = invalidator.take_dirty_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], ChunkCoord::new(0, 0, 0));
    }

    #[test]
    fn test_mark_dirty_region_multiple_chunks() {
        let mut invalidator = ChunkInvalidator::new();

        // Region spanning 2x2x2 chunks
        let region = Aabb {
            min: Vec3::new(2.0, 2.0, 2.0),
            max: Vec3::new(10.0, 10.0, 10.0),
        };

        invalidator.mark_dirty(&region);
        // Should span from chunk (0,0,0) to chunk (2,2,2) = 3x3x3 = 27 chunks
        assert_eq!(invalidator.dirty_chunk_count(), 27);

        let chunks = invalidator.take_dirty_chunks();
        assert_eq!(chunks.len(), 27);
    }

    #[test]
    fn test_mark_dirty_generation_increments() {
        let mut invalidator = ChunkInvalidator::new();

        let region1 = Aabb {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(2.0, 2.0, 2.0),
        };

        invalidator.mark_dirty(&region1);
        let gen1 = invalidator.generation(&ChunkCoord::new(0, 0, 0));
        assert_eq!(gen1, 1);

        // Mark overlapping region again
        let region2 = Aabb {
            min: Vec3::new(1.0, 1.0, 1.0),
            max: Vec3::new(3.0, 3.0, 3.0),
        };

        invalidator.mark_dirty(&region2);
        let gen2 = invalidator.generation(&ChunkCoord::new(0, 0, 0));
        assert_eq!(gen2, 2);
    }

    #[test]
    fn test_mark_brick_dirty() {
        let mut invalidator = ChunkInvalidator::new();
        let chunk_coord = ChunkCoord::new(5, 5, 5);
        let brick_id = BrickId {
            chunk: chunk_coord,
            local_index: 42,
        };

        invalidator.mark_brick_dirty(brick_id);
        assert!(invalidator.is_brick_dirty(&brick_id));
        assert!(invalidator.is_chunk_dirty(&chunk_coord));
        assert_eq!(invalidator.generation(&chunk_coord), 1);
    }

    #[test]
    fn test_take_dirty_bricks() {
        let mut invalidator = ChunkInvalidator::new();
        let coord = ChunkCoord::new(0, 0, 0);
        let brick1 = BrickId {
            chunk: coord,
            local_index: 0,
        };
        let brick2 = BrickId {
            chunk: coord,
            local_index: 1,
        };

        invalidator.mark_brick_dirty(brick1);
        invalidator.mark_brick_dirty(brick2);

        assert_eq!(invalidator.dirty_brick_count(), 2);
        let bricks = invalidator.take_dirty_bricks();
        assert_eq!(bricks.len(), 2);
        assert_eq!(invalidator.dirty_brick_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut invalidator = ChunkInvalidator::new();

        let region = Aabb {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(8.0, 8.0, 8.0),
        };

        invalidator.mark_dirty(&region);
        assert!(invalidator.has_dirty());

        let coord = ChunkCoord::new(0, 0, 0);
        assert_eq!(invalidator.generation(&coord), 1);

        invalidator.clear();
        assert!(!invalidator.has_dirty());
        assert_eq!(invalidator.generation(&coord), 0);
    }

    #[test]
    fn test_default() {
        let invalidator = ChunkInvalidator::default();
        assert!(!invalidator.has_dirty());
    }
}
