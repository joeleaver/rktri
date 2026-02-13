//! Flattened scene graph output for GPU upload.
//!
//! `FlatChunkEntry` is the result of walking the scene graph — one entry per visible
//! chunk/instance, ready to be packed into `GpuChunkInfo[]`.

use glam::Vec3;

use crate::voxel::chunk::ChunkCoord;
use crate::voxel::layer::LayerId;
use crate::voxel::svo::Octree;

/// One entry in the flattened visible set, ready for GPU upload.
#[derive(Clone, Debug)]
pub struct FlatChunkEntry {
    /// Chunk coordinate (real for terrain, synthetic for instances).
    pub coord: ChunkCoord,
    /// The octree data for this entry.
    pub octree: Octree,
    /// World-space minimum corner after transform.
    pub world_min: Vec3,
    /// Root size of the octree in world units.
    pub root_size: f32,
    /// Which layer this entry belongs to.
    pub layer_id: LayerId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_chunk_entry_creation() {
        let entry = FlatChunkEntry {
            coord: ChunkCoord::new(1, 0, 2),
            octree: Octree::new(4.0, 8),
            world_min: Vec3::new(4.0, 0.0, 8.0),
            root_size: 4.0,
            layer_id: LayerId::TERRAIN,
        };
        assert_eq!(entry.coord, ChunkCoord::new(1, 0, 2));
        assert_eq!(entry.root_size, 4.0);
        assert_eq!(entry.layer_id, LayerId::TERRAIN);
    }

    #[test]
    fn test_flat_chunk_entry_clone() {
        let entry = FlatChunkEntry {
            coord: ChunkCoord::new(0, 0, 0),
            octree: Octree::new(4.0, 8),
            world_min: Vec3::ZERO,
            root_size: 4.0,
            layer_id: LayerId::STATIC_OBJECTS,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.coord, entry.coord);
        assert_eq!(cloned.layer_id, entry.layer_id);
    }
}
