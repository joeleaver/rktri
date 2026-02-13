//! Voxel hierarchy constants and utilities.
//!
//! Defines the three-tier spatial hierarchy:
//! - SuperChunk (64m): Coarse-grained streaming unit
//! - Chunk (4m): Standard octree unit
//! - Brick (2x2x2): Leaf voxel storage

/// SuperChunk size in meters (64m x 64m x 64m)
pub const SUPER_CHUNK_SIZE_METERS: f32 = 64.0;

/// Number of chunks per SuperChunk dimension (16)
pub const CHUNKS_PER_SUPER_CHUNK: u32 = 16;

/// Chunk size in meters (4m x 4m x 4m) - matches existing CHUNK_SIZE
pub const CHUNK_SIZE_METERS: f32 = 4.0;

/// Brick size in voxels (2x2x2)
pub const BRICK_SIZE_VOXELS: u32 = 2;

/// Number of voxels per brick (8)
pub const VOXELS_PER_BRICK: u32 = BRICK_SIZE_VOXELS * BRICK_SIZE_VOXELS * BRICK_SIZE_VOXELS;

/// Voxels per meter (128 = ~7.8mm per voxel ≈ 1cm)
pub const VOXELS_PER_METER: u32 = 128;

/// Voxel size in meters
pub const VOXEL_SIZE_METERS: f32 = 1.0 / VOXELS_PER_METER as f32;

/// Voxels per chunk dimension (512)
pub const VOXELS_PER_CHUNK: u32 = (CHUNK_SIZE_METERS * VOXELS_PER_METER as f32) as u32;

/// Bricks per chunk dimension (256)
pub const BRICKS_PER_CHUNK: u32 = VOXELS_PER_CHUNK / BRICK_SIZE_VOXELS;

/// Total bricks per chunk (256^3 = 16,777,216)
pub const TOTAL_BRICKS_PER_CHUNK: u32 = BRICKS_PER_CHUNK * BRICKS_PER_CHUNK * BRICKS_PER_CHUNK;

/// Convert world position to SuperChunk coordinate
pub fn world_to_super_chunk(pos: glam::Vec3) -> (i32, i32, i32) {
    (
        (pos.x / SUPER_CHUNK_SIZE_METERS).floor() as i32,
        (pos.y / SUPER_CHUNK_SIZE_METERS).floor() as i32,
        (pos.z / SUPER_CHUNK_SIZE_METERS).floor() as i32,
    )
}

/// Convert world position to chunk coordinate
pub fn world_to_chunk(pos: glam::Vec3) -> (i32, i32, i32) {
    (
        (pos.x / CHUNK_SIZE_METERS).floor() as i32,
        (pos.y / CHUNK_SIZE_METERS).floor() as i32,
        (pos.z / CHUNK_SIZE_METERS).floor() as i32,
    )
}

/// Convert chunk coordinate to world position (chunk origin)
pub fn chunk_to_world(chunk: (i32, i32, i32)) -> glam::Vec3 {
    glam::Vec3::new(
        chunk.0 as f32 * CHUNK_SIZE_METERS,
        chunk.1 as f32 * CHUNK_SIZE_METERS,
        chunk.2 as f32 * CHUNK_SIZE_METERS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_hierarchy_relationships() {
        // 16 chunks fit in a super chunk
        assert_eq!(
            (SUPER_CHUNK_SIZE_METERS / CHUNK_SIZE_METERS) as u32,
            CHUNKS_PER_SUPER_CHUNK
        );

        // 512 voxels per chunk
        assert_eq!(VOXELS_PER_CHUNK, 512);

        // 256 bricks per chunk dimension
        assert_eq!(BRICKS_PER_CHUNK, 256);
    }

    #[test]
    fn test_world_to_chunk() {
        assert_eq!(world_to_chunk(Vec3::new(0.0, 0.0, 0.0)), (0, 0, 0));
        assert_eq!(world_to_chunk(Vec3::new(4.5, 0.0, 0.0)), (1, 0, 0));
        assert_eq!(world_to_chunk(Vec3::new(-0.1, 0.0, 0.0)), (-1, 0, 0));
    }

    #[test]
    fn test_chunk_to_world() {
        assert_eq!(chunk_to_world((0, 0, 0)), Vec3::ZERO);
        assert_eq!(chunk_to_world((1, 2, 3)), Vec3::new(4.0, 8.0, 12.0));
    }
}
