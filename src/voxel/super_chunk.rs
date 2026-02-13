//! SuperChunk - coarse-grained spatial container.
//!
//! A SuperChunk contains a 16x16x16 grid of regular chunks and provides
//! LOD data for distant rendering.

use std::time::Instant;
use glam::Vec3;

use crate::voxel::hierarchy::{CHUNKS_PER_SUPER_CHUNK, SUPER_CHUNK_SIZE_METERS};

/// Coordinate of a SuperChunk in the world grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SuperChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl SuperChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Get the world-space origin of this SuperChunk.
    pub fn world_origin(&self) -> Vec3 {
        Vec3::new(
            self.x as f32 * SUPER_CHUNK_SIZE_METERS,
            self.y as f32 * SUPER_CHUNK_SIZE_METERS,
            self.z as f32 * SUPER_CHUNK_SIZE_METERS,
        )
    }

    /// Get the world-space center of this SuperChunk.
    pub fn world_center(&self) -> Vec3 {
        self.world_origin() + Vec3::splat(SUPER_CHUNK_SIZE_METERS / 2.0)
    }

    /// Convert from world position.
    pub fn from_world_pos(pos: Vec3) -> Self {
        Self {
            x: (pos.x / SUPER_CHUNK_SIZE_METERS).floor() as i32,
            y: (pos.y / SUPER_CHUNK_SIZE_METERS).floor() as i32,
            z: (pos.z / SUPER_CHUNK_SIZE_METERS).floor() as i32,
        }
    }
}

/// Bitmap tracking which chunks are present in a SuperChunk.
/// 16x16x16 = 4096 bits = 512 bytes
#[derive(Clone)]
pub struct ChunkPresenceMask {
    bits: [u64; 64], // 64 * 64 = 4096 bits
}

impl ChunkPresenceMask {
    pub fn new() -> Self {
        Self { bits: [0; 64] }
    }

    /// Check if a chunk is present.
    pub fn is_present(&self, x: u32, y: u32, z: u32) -> bool {
        debug_assert!(x < CHUNKS_PER_SUPER_CHUNK);
        debug_assert!(y < CHUNKS_PER_SUPER_CHUNK);
        debug_assert!(z < CHUNKS_PER_SUPER_CHUNK);

        let index = (z * CHUNKS_PER_SUPER_CHUNK * CHUNKS_PER_SUPER_CHUNK
                   + y * CHUNKS_PER_SUPER_CHUNK + x) as usize;
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] & (1 << bit)) != 0
    }

    /// Mark a chunk as present.
    pub fn set_present(&mut self, x: u32, y: u32, z: u32, present: bool) {
        debug_assert!(x < CHUNKS_PER_SUPER_CHUNK);
        debug_assert!(y < CHUNKS_PER_SUPER_CHUNK);
        debug_assert!(z < CHUNKS_PER_SUPER_CHUNK);

        let index = (z * CHUNKS_PER_SUPER_CHUNK * CHUNKS_PER_SUPER_CHUNK
                   + y * CHUNKS_PER_SUPER_CHUNK + x) as usize;
        let word = index / 64;
        let bit = index % 64;

        if present {
            self.bits[word] |= 1 << bit;
        } else {
            self.bits[word] &= !(1 << bit);
        }
    }

    /// Count present chunks.
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Check if all chunks are absent.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }
}

impl Default for ChunkPresenceMask {
    fn default() -> Self {
        Self::new()
    }
}

/// A SuperChunk containing a 16x16x16 grid of chunks.
pub struct SuperChunk {
    /// Coordinate in the SuperChunk grid
    pub coord: SuperChunkCoord,
    /// Bitmap of which chunks are present
    pub chunk_mask: ChunkPresenceMask,
    /// Last time this SuperChunk was accessed
    pub last_access: Instant,
    /// Cached distance to camera (for streaming priority)
    pub distance_to_camera: f32,
    /// Whether this SuperChunk has any modified chunks
    pub is_dirty: bool,
}

impl SuperChunk {
    /// Create a new empty SuperChunk.
    pub fn new(coord: SuperChunkCoord) -> Self {
        Self {
            coord,
            chunk_mask: ChunkPresenceMask::new(),
            last_access: Instant::now(),
            distance_to_camera: f32::MAX,
            is_dirty: false,
        }
    }

    /// Update distance to camera.
    pub fn update_distance(&mut self, camera_pos: Vec3) {
        self.distance_to_camera = self.coord.world_center().distance(camera_pos);
    }

    /// Mark as accessed.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Get chunk count.
    pub fn chunk_count(&self) -> u32 {
        self.chunk_mask.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_chunk_coord() {
        let coord = SuperChunkCoord::new(1, 2, 3);
        assert_eq!(coord.world_origin(), Vec3::new(64.0, 128.0, 192.0));
    }

    #[test]
    fn test_chunk_presence_mask() {
        let mut mask = ChunkPresenceMask::new();
        assert!(!mask.is_present(5, 5, 5));

        mask.set_present(5, 5, 5, true);
        assert!(mask.is_present(5, 5, 5));
        assert_eq!(mask.count(), 1);

        mask.set_present(5, 5, 5, false);
        assert!(!mask.is_present(5, 5, 5));
        assert!(mask.is_empty());
    }
}
