//! Chunk system for managing cubic regions of voxel space

use crate::math::Aabb;
use crate::voxel::svo::Octree;
use glam::Vec3;
use std::time::Instant;

/// Size of a chunk in world units (meters)
/// Smaller chunks allow higher voxel resolution without memory explosion
pub const CHUNK_SIZE: u32 = 4;

/// Voxels per meter - controls resolution (higher = finer detail)
/// 128 voxels/meter = ~0.78cm base voxels
/// With brick-level detail (2x2x2), effective resolution is ~1.56cm
/// CHUNK_VOXELS must be power of 2 for octree builder
pub const VOXELS_PER_METER: u32 = 128;

/// Number of voxels per chunk side
/// 4m chunk * 128 voxels/m = 512 voxels per side (power of 2)
pub const CHUNK_VOXELS: u32 = CHUNK_SIZE * VOXELS_PER_METER;

/// Size of a single voxel in meters
pub const VOXEL_SIZE: f32 = 1.0 / VOXELS_PER_METER as f32; // ~0.0078m (0.78cm) for 128 voxels/m

/// Integer coordinate identifying a chunk in the world grid
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    /// Create a new chunk coordinate
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert world position to chunk coordinate
    pub fn from_world_pos(pos: Vec3) -> Self {
        Self {
            x: (pos.x / CHUNK_SIZE as f32).floor() as i32,
            y: (pos.y / CHUNK_SIZE as f32).floor() as i32,
            z: (pos.z / CHUNK_SIZE as f32).floor() as i32,
        }
    }

    /// Get the world-space origin (minimum corner) of this chunk
    pub fn world_origin(&self) -> Vec3 {
        Vec3::new(
            self.x as f32 * CHUNK_SIZE as f32,
            self.y as f32 * CHUNK_SIZE as f32,
            self.z as f32 * CHUNK_SIZE as f32,
        )
    }
}

/// A single chunk containing a 64m³ region of voxel data
pub struct Chunk {
    /// Coordinate of this chunk in the world grid
    pub coord: ChunkCoord,
    /// Sparse voxel octree containing the voxel data
    pub octree: Octree,
    /// Whether this chunk has been modified since last save
    pub modified: bool,
    /// Last time this chunk was accessed (for LRU eviction)
    pub last_access: Instant,
}

impl Chunk {
    /// Create a new empty chunk at the given coordinate
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            octree: Octree::new(CHUNK_SIZE as f32, 10), // 10 levels = 1cm voxels (4m / 2^10)
            modified: false,
            last_access: Instant::now(),
        }
    }

    /// Create a chunk from an existing octree
    pub fn from_octree(coord: ChunkCoord, octree: Octree) -> Self {
        Self {
            coord,
            octree,
            modified: false,
            last_access: Instant::now(),
        }
    }

    /// Get the world-space bounding box for this chunk
    pub fn world_bounds(&self) -> Aabb {
        let origin = self.coord.world_origin();
        Aabb::new(origin, origin + Vec3::splat(CHUNK_SIZE as f32))
    }

    /// Mark this chunk as recently accessed (updates last_access time)
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_coord_new() {
        let coord = ChunkCoord::new(1, 2, 3);
        assert_eq!(coord.x, 1);
        assert_eq!(coord.y, 2);
        assert_eq!(coord.z, 3);
    }

    #[test]
    fn test_from_world_pos() {
        let cs = CHUNK_SIZE as f32;

        // Position in chunk (0,0,0) - center of first chunk
        let coord = ChunkCoord::from_world_pos(Vec3::new(cs / 2.0, cs / 2.0, cs / 2.0));
        assert_eq!(coord, ChunkCoord::new(0, 0, 0));

        // Position in chunk (1,0,0) - start of second chunk
        let coord = ChunkCoord::from_world_pos(Vec3::new(cs, 0.0, 0.0));
        assert_eq!(coord, ChunkCoord::new(1, 0, 0));

        // Position in chunk (1,2,3)
        let coord = ChunkCoord::from_world_pos(Vec3::new(cs * 1.5, cs * 2.5, cs * 3.5));
        assert_eq!(coord, ChunkCoord::new(1, 2, 3));

        // Negative coordinates
        let coord = ChunkCoord::from_world_pos(Vec3::new(-10.0, -20.0, -30.0));
        assert_eq!(coord, ChunkCoord::new(-3, -5, -8));

        // Edge case: exactly on boundary
        let coord = ChunkCoord::from_world_pos(Vec3::new(cs * 2.0, cs, 0.0));
        assert_eq!(coord, ChunkCoord::new(2, 1, 0));
    }

    #[test]
    fn test_world_origin() {
        let cs = CHUNK_SIZE as f32;

        let coord = ChunkCoord::new(0, 0, 0);
        assert_eq!(coord.world_origin(), Vec3::ZERO);

        let coord = ChunkCoord::new(1, 2, 3);
        assert_eq!(coord.world_origin(), Vec3::new(cs, cs * 2.0, cs * 3.0));

        let coord = ChunkCoord::new(-1, -1, -1);
        assert_eq!(coord.world_origin(), Vec3::new(-cs, -cs, -cs));
    }

    #[test]
    fn test_world_origin_round_trip() {
        // Test that converting chunk coord to world pos and back gives same coord
        let original = ChunkCoord::new(5, -3, 10);
        let world_pos = original.world_origin() + Vec3::splat(CHUNK_SIZE as f32 / 2.0); // Center of chunk
        let recovered = ChunkCoord::from_world_pos(world_pos);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_chunk_new() {
        let coord = ChunkCoord::new(1, 2, 3);
        let chunk = Chunk::new(coord);

        assert_eq!(chunk.coord, coord);
        assert_eq!(chunk.octree.root_size(), CHUNK_SIZE as f32);
        assert_eq!(chunk.octree.max_depth(), 10);
        assert!(!chunk.modified);
    }

    #[test]
    fn test_chunk_world_bounds() {
        let cs = CHUNK_SIZE as f32;
        let chunk = Chunk::new(ChunkCoord::new(1, 2, 3));
        let bounds = chunk.world_bounds();

        assert_eq!(bounds.min, Vec3::new(cs, cs * 2.0, cs * 3.0));
        assert_eq!(bounds.max, Vec3::new(cs * 2.0, cs * 3.0, cs * 4.0));
        assert_eq!(bounds.size(), Vec3::splat(cs));
    }

    #[test]
    fn test_chunk_touch() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        let first_access = chunk.last_access;

        std::thread::sleep(std::time::Duration::from_millis(10));
        chunk.touch();

        assert!(chunk.last_access > first_access);
    }

    #[test]
    fn test_chunk_from_octree() {
        let coord = ChunkCoord::new(5, 6, 7);
        let octree = Octree::new(64.0, 10);
        let chunk = Chunk::from_octree(coord, octree);

        assert_eq!(chunk.coord, coord);
        assert_eq!(chunk.octree.root_size(), 64.0);
        assert_eq!(chunk.octree.max_depth(), 10);
    }
}
