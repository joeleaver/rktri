//! Chunk serialization and disk I/O

use crate::grass::profile::GrassCell;
use crate::mask::{MaskOctree, MaskNode};
use crate::voxel::svo::{Octree, node::OctreeNode};
use crate::voxel::brick::VoxelBrick;
use rkyv::{Archive, Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::io;

/// Chunk coordinate in world space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// Serializable chunk data
///
/// Since OctreeNode and VoxelBrick already implement Pod/Zeroable (bytemuck),
/// we can serialize them directly as byte arrays for efficient storage.
#[derive(Archive, Deserialize, Serialize)]
pub struct ChunkData {
    pub coord_x: i32,
    pub coord_y: i32,
    pub coord_z: i32,
    pub root_size: f32,
    pub max_depth: u8,
    /// Serialized OctreeNode data (already Pod-compatible)
    pub nodes: Vec<OctreeNode>,
    /// Serialized VoxelBrick data (already Pod-compatible)
    pub bricks: Vec<VoxelBrick>,
}

/// Runtime chunk containing octree and metadata
pub struct Chunk {
    pub coord: ChunkCoord,
    pub octree: Octree,
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunk")
            .field("coord", &self.coord)
            .field("octree", &"<Octree>")
            .finish()
    }
}

impl Chunk {
    /// Create a new chunk with default octree
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            octree: Octree::default(),
        }
    }

    /// Create chunk from octree
    pub fn from_octree(coord: ChunkCoord, octree: Octree) -> Self {
        Self { coord, octree }
    }

    /// Get chunk coordinate
    pub fn coord(&self) -> ChunkCoord {
        self.coord
    }

    /// Get octree reference
    pub fn octree(&self) -> &Octree {
        &self.octree
    }

    /// Get mutable octree reference
    pub fn octree_mut(&mut self) -> &mut Octree {
        &mut self.octree
    }
}

/// Serialize a chunk to bytes (uncompressed)
pub fn serialize_chunk(chunk: &Chunk) -> Result<Vec<u8>, io::Error> {
    let data = ChunkData {
        coord_x: chunk.coord.x,
        coord_y: chunk.coord.y,
        coord_z: chunk.coord.z,
        root_size: chunk.octree.root_size(),
        max_depth: chunk.octree.max_depth(),
        nodes: chunk.octree.nodes_slice().to_vec(),
        bricks: chunk.octree.bricks_slice().to_vec(),
    };

    // Use rkyv's zero-copy serialization
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok(bytes.to_vec())
}

/// Deserialize a chunk from bytes (uncompressed)
pub fn deserialize_chunk(data: &[u8]) -> Result<Chunk, io::Error> {
    // Deserialize using rkyv
    let archived = rkyv::access::<ArchivedChunkData, rkyv::rancor::Error>(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let chunk_data: ChunkData = rkyv::deserialize::<ChunkData, rkyv::rancor::Error>(archived)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // Reconstruct octree from serialized data using from_serialized
    let octree = Octree::from_serialized(
        chunk_data.root_size,
        chunk_data.max_depth,
        chunk_data.nodes,
        chunk_data.bricks,
    );

    let coord = ChunkCoord::new(chunk_data.coord_x, chunk_data.coord_y, chunk_data.coord_z);

    Ok(Chunk::from_octree(coord, octree))
}

/// Compress a serialized chunk using LZ4
pub fn compress_chunk(chunk: &Chunk) -> Result<Vec<u8>, io::Error> {
    let serialized = serialize_chunk(chunk)?;
    Ok(lz4_flex::compress_prepend_size(&serialized))
}

/// Decompress and deserialize a chunk
pub fn decompress_chunk(data: &[u8]) -> Result<Chunk, io::Error> {
    let decompressed = lz4_flex::decompress_size_prepended(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", e)))?;
    deserialize_chunk(&decompressed)
}

/// Get the file path for a chunk
pub fn chunk_path(base_dir: &Path, coord: ChunkCoord) -> PathBuf {
    // Organize chunks in subdirectories by Y coordinate to avoid too many files in one dir
    // Format: base_dir/y_{coord.y}/chunk_{x}_{y}_{z}.rkc
    base_dir
        .join(format!("y_{}", coord.y))
        .join(format!("chunk_{}_{}_{}.rkc", coord.x, coord.y, coord.z))
}

/// Save a chunk to disk (compressed)
pub async fn save_chunk(base_dir: &Path, chunk: &Chunk) -> Result<(), io::Error> {
    let path = chunk_path(base_dir, chunk.coord);

    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let compressed = compress_chunk(chunk)?;
    tokio::fs::write(&path, compressed).await?;

    Ok(())
}

/// Load a chunk from disk (if it exists)
pub async fn load_chunk(base_dir: &Path, coord: ChunkCoord) -> Result<Option<Chunk>, io::Error> {
    let path = chunk_path(base_dir, coord);

    // Check if file exists
    if !path.exists() {
        return Ok(None);
    }

    let compressed = tokio::fs::read(&path).await?;
    let chunk = decompress_chunk(&compressed)?;

    Ok(Some(chunk))
}

/// Delete a chunk from disk
pub async fn delete_chunk(base_dir: &Path, coord: ChunkCoord) -> Result<(), io::Error> {
    let path = chunk_path(base_dir, coord);

    if path.exists() {
        tokio::fs::remove_file(&path).await?;
    }

    Ok(())
}

/// Check if a chunk exists on disk
pub async fn chunk_exists(base_dir: &Path, coord: ChunkCoord) -> bool {
    chunk_path(base_dir, coord).exists()
}

// --- Grass mask serialization ---

/// Serializable grass mask node data.
#[derive(Archive, Deserialize, Serialize)]
struct GrassMaskNodeData {
    child_mask: u8,
    leaf_mask: u8,
    flags: u16,
    child_offset: u32,
    value_offset: u32,
    lod_value_idx: u32,
}

/// Serializable grass mask data.
#[derive(Archive, Deserialize, Serialize)]
pub struct GrassMaskData {
    pub coord_x: i32,
    pub coord_y: i32,
    pub coord_z: i32,
    pub max_depth: u8,
    pub root_size: f32,
    nodes: Vec<GrassMaskNodeData>,
    values: Vec<u16>,
}

/// Serialize a grass mask octree to bytes (uncompressed).
pub fn serialize_grass_mask(coord: ChunkCoord, mask: &MaskOctree<GrassCell>) -> Result<Vec<u8>, io::Error> {
    let mut nodes = Vec::with_capacity(mask.node_count());
    for i in 0..mask.node_count() {
        let node = mask.node(i as u32);
        nodes.push(GrassMaskNodeData {
            child_mask: node.child_mask,
            leaf_mask: node.leaf_mask,
            flags: node.flags,
            child_offset: node.child_offset,
            value_offset: node.value_offset,
            lod_value_idx: node.lod_value_idx,
        });
    }

    let mut values = Vec::with_capacity(mask.value_count());
    for i in 0..mask.value_count() {
        values.push(mask.value(i as u32).0);
    }

    let data = GrassMaskData {
        coord_x: coord.x,
        coord_y: coord.y,
        coord_z: coord.z,
        max_depth: mask.max_depth(),
        root_size: mask.root_size(),
        nodes,
        values,
    };

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    Ok(bytes.to_vec())
}

/// Deserialize a grass mask octree from bytes (uncompressed).
pub fn deserialize_grass_mask(data: &[u8]) -> Result<(ChunkCoord, MaskOctree<GrassCell>), io::Error> {
    let archived = rkyv::access::<ArchivedGrassMaskData, rkyv::rancor::Error>(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let mask_data: GrassMaskData = rkyv::deserialize::<GrassMaskData, rkyv::rancor::Error>(archived)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let coord = ChunkCoord::new(mask_data.coord_x, mask_data.coord_y, mask_data.coord_z);

    let mut mask = MaskOctree::with_capacity(
        mask_data.root_size,
        mask_data.max_depth,
        mask_data.nodes.len(),
        mask_data.values.len(),
    );

    // The first node (root) is already created by with_capacity, so update it
    if let Some(first_node) = mask_data.nodes.first() {
        let root = mask.node_mut(0);
        root.child_mask = first_node.child_mask;
        root.leaf_mask = first_node.leaf_mask;
        root.flags = first_node.flags;
        root.child_offset = first_node.child_offset;
        root.value_offset = first_node.value_offset;
        root.lod_value_idx = first_node.lod_value_idx;
    }

    // Add remaining nodes
    for node_data in mask_data.nodes.iter().skip(1) {
        mask.add_node(MaskNode {
            child_mask: node_data.child_mask,
            leaf_mask: node_data.leaf_mask,
            flags: node_data.flags,
            child_offset: node_data.child_offset,
            value_offset: node_data.value_offset,
            lod_value_idx: node_data.lod_value_idx,
        });
    }

    // Add values
    for &v in &mask_data.values {
        mask.add_value(GrassCell(v));
    }

    Ok((coord, mask))
}

/// Compress a serialized grass mask using LZ4.
pub fn compress_grass_mask(coord: ChunkCoord, mask: &MaskOctree<GrassCell>) -> Result<Vec<u8>, io::Error> {
    let serialized = serialize_grass_mask(coord, mask)?;
    Ok(lz4_flex::compress_prepend_size(&serialized))
}

/// Decompress and deserialize a grass mask.
pub fn decompress_grass_mask(data: &[u8]) -> Result<(ChunkCoord, MaskOctree<GrassCell>), io::Error> {
    let decompressed = lz4_flex::decompress_size_prepended(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", e)))?;
    deserialize_grass_mask(&decompressed)
}

/// Get the file path for a grass mask chunk.
pub fn grass_mask_path(base_dir: &Path, coord: ChunkCoord) -> PathBuf {
    base_dir.join(format!("chunk_{}_{}_{}.rkm", coord.x, coord.y, coord.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_coord() {
        let coord = ChunkCoord::new(1, 2, 3);
        assert_eq!(coord.x, 1);
        assert_eq!(coord.y, 2);
        assert_eq!(coord.z, 3);
    }

    #[test]
    fn test_chunk_path() {
        let base = Path::new("/tmp/chunks");
        let coord = ChunkCoord::new(5, 10, -3);
        let path = chunk_path(base, coord);

        assert_eq!(
            path,
            PathBuf::from("/tmp/chunks/y_10/chunk_5_10_-3.rkc")
        );
    }

    #[test]
    fn test_serialize_deserialize_empty_chunk() {
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = Chunk::new(coord);

        // Serialize
        let serialized = serialize_chunk(&chunk).expect("serialization failed");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized = deserialize_chunk(&serialized).expect("deserialization failed");
        assert_eq!(deserialized.coord, coord);
    }

    #[test]
    fn test_compress_decompress_chunk() {
        let coord = ChunkCoord::new(1, 2, 3);
        let chunk = Chunk::new(coord);

        // Compress
        let compressed = compress_chunk(&chunk).expect("compression failed");
        assert!(!compressed.is_empty());

        // Decompress
        let decompressed = decompress_chunk(&compressed).expect("decompression failed");
        assert_eq!(decompressed.coord, coord);
    }

    #[test]
    fn test_compression_ratio() {
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = Chunk::new(coord);

        let uncompressed = serialize_chunk(&chunk).expect("serialization failed");
        let compressed = compress_chunk(&chunk).expect("compression failed");

        // Empty chunks should compress well
        println!(
            "Compression: {} bytes -> {} bytes ({:.1}% reduction)",
            uncompressed.len(),
            compressed.len(),
            100.0 * (1.0 - compressed.len() as f64 / uncompressed.len() as f64)
        );

        // Compressed should be smaller for repetitive data
        assert!(compressed.len() <= uncompressed.len());
    }

    // TODO: Enable tokio test support in Cargo.toml
    // #[tokio::test]
    // async fn test_save_and_load_chunk() {
    //     let temp_dir = std::env::temp_dir().join("rktri_test_chunks");
    //     let coord = ChunkCoord::new(5, 10, -3);
    //     let chunk = Chunk::new(coord);

    //     // Save
    //     save_chunk(&temp_dir, &chunk).await.expect("save failed");

    //     // Verify file exists
    //     assert!(chunk_exists(&temp_dir, coord).await);

    //     // Load
    //     let loaded = load_chunk(&temp_dir, coord)
    //         .await
    //         .expect("load failed")
    //         .expect("chunk not found");

    //     assert_eq!(loaded.coord, coord);

    //     // Cleanup
    //     delete_chunk(&temp_dir, coord).await.expect("delete failed");
    //     tokio::fs::remove_dir_all(&temp_dir).await.ok();
    // }

    // #[tokio::test]
    // async fn test_load_nonexistent_chunk() {
    //     let temp_dir = std::env::temp_dir().join("rktri_test_chunks_nonexistent");
    //     let coord = ChunkCoord::new(999, 999, 999);

    //     let result = load_chunk(&temp_dir, coord).await.expect("load should not error");
    //     assert!(result.is_none());
    // }

    #[test]
    fn test_grass_mask_serialize_roundtrip() {
        use glam::Vec3;
        use crate::mask::MaskBuilder;
        use crate::mask::BiomeId;
        use crate::grass::profile::GrassProfileTable;
        use crate::generation::GrassNoiseGenerator;
        use crate::terrain::generator::{TerrainGenerator, TerrainParams};

        // Build a simple grass mask
        let terrain = TerrainGenerator::new(TerrainParams::default());
        let mut biome_mask = MaskOctree::new(4.0, 3);
        let val_idx = biome_mask.add_value(BiomeId::GRASSLAND);
        biome_mask.node_mut(0).lod_value_idx = val_idx;

        let table = GrassProfileTable::default();
        let origin = Vec3::ZERO;
        let grass_gen = GrassNoiseGenerator::new(&terrain, &biome_mask, origin, &table, 12345);
        let mask = MaskBuilder::new(3).build(&grass_gen, origin, 4.0);

        let coord = ChunkCoord::new(1, 5, 3);

        // Serialize + decompress roundtrip
        let compressed = compress_grass_mask(coord, &mask).expect("compression failed");
        let (rt_coord, rt_mask) = decompress_grass_mask(&compressed).expect("decompression failed");

        assert_eq!(rt_coord, coord);
        assert_eq!(rt_mask.node_count(), mask.node_count());
        assert_eq!(rt_mask.value_count(), mask.value_count());
        assert_eq!(rt_mask.max_depth(), mask.max_depth());

        // Sample at the same position should return the same value
        let sample_pos = Vec3::new(2.0, 2.0, 2.0);
        let original = mask.sample(origin, sample_pos);
        let roundtrip = rt_mask.sample(origin, sample_pos);
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_grass_mask_path() {
        let base = Path::new("/tmp/grass");
        let coord = ChunkCoord::new(5, 10, -3);
        let path = grass_mask_path(base, coord);
        assert_eq!(path, PathBuf::from("/tmp/grass/chunk_5_10_-3.rkm"));
    }
}
