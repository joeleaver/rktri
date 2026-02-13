//! Tree data serialization and storage

use rkyv::{Archive, Deserialize, Serialize};
use std::io;
use std::path::Path;

use crate::voxel::svo::{Octree, node::OctreeNode};
use crate::voxel::brick::VoxelBrick;
use crate::voxel::procgen::TreeStyle;

/// Current version of tree data format
pub const TREE_DATA_VERSION: u32 = 2;

/// File extension for tree data files
pub const TREE_FILE_EXTENSION: &str = "rkt";

/// Serializable tree data with metadata
#[derive(Archive, Deserialize, Serialize)]
pub struct TreeData {
    /// Format version for compatibility
    pub version: u32,
    /// Tree style (stored as u8 for rkyv compatibility)
    pub style: u8,
    /// Generation seed for reproducibility
    pub seed: u64,
    /// Bounding box min corner (relative to tree origin)
    pub bounds_min: [f32; 3],
    /// Bounding box max corner (relative to tree origin)
    pub bounds_max: [f32; 3],
    /// Octree root size in world units
    pub root_size: f32,
    /// Octree maximum depth
    pub max_depth: u8,
    /// Octree nodes
    pub nodes: Vec<OctreeNode>,
    /// Voxel bricks
    pub bricks: Vec<VoxelBrick>,
}

impl TreeData {
    /// Create TreeData from a generated octree with metadata
    pub fn from_octree(octree: &Octree, style: TreeStyle, seed: u64) -> Self {
        // Calculate bounding box from octree (simplified - uses root size)
        let half = octree.root_size() / 2.0;

        Self {
            version: TREE_DATA_VERSION,
            style: style as u8,
            seed,
            bounds_min: [-half, 0.0, -half],
            bounds_max: [half, octree.root_size(), half],
            root_size: octree.root_size(),
            max_depth: octree.max_depth(),
            nodes: octree.nodes_slice().to_vec(),
            bricks: octree.bricks_slice().to_vec(),
        }
    }

    /// Reconstruct an Octree from this TreeData
    pub fn to_octree(&self) -> Octree {
        // Tree octrees are compacted to packed indexing during generation,
        // so dense_children defaults to false (correct for packed layout)
        Octree::from_serialized(
            self.root_size,
            self.max_depth,
            self.nodes.clone(),
            self.bricks.clone(),
        )
    }

    /// Get the tree style
    pub fn style(&self) -> TreeStyle {
        match self.style {
            0 => TreeStyle::Oak,
            1 => TreeStyle::Willow,
            2 => TreeStyle::Elm,
            _ => TreeStyle::Oak, // Default fallback
        }
    }

    /// Get bounding box dimensions
    pub fn bounds_size(&self) -> [f32; 3] {
        [
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        ]
    }

    /// Serialize to compressed bytes (rkyv + LZ4)
    pub fn to_bytes(&self) -> Result<Vec<u8>, io::Error> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        Ok(lz4_flex::compress_prepend_size(&bytes))
    }

    /// Deserialize from compressed bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, io::Error> {
        let decompressed = lz4_flex::decompress_size_prepended(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", e)))?;

        let archived = rkyv::access::<ArchivedTreeData, rkyv::rancor::Error>(&decompressed)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let tree_data: TreeData = rkyv::deserialize::<TreeData, rkyv::rancor::Error>(archived)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        // Version check
        if tree_data.version != TREE_DATA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Tree data version mismatch: expected {}, got {}", TREE_DATA_VERSION, tree_data.version),
            ));
        }

        Ok(tree_data)
    }

    /// Save to file (async)
    pub async fn save(&self, path: &Path) -> Result<(), io::Error> {
        let bytes = self.to_bytes()?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(path, bytes).await
    }

    /// Load from file (async)
    pub async fn load(path: &Path) -> Result<Self, io::Error> {
        let bytes = tokio::fs::read(path).await?;
        Self::from_bytes(&bytes)
    }

    /// Save to file (sync)
    pub fn save_sync(&self, path: &Path) -> Result<(), io::Error> {
        let bytes = self.to_bytes()?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, bytes)
    }

    /// Load from file (sync)
    pub fn load_sync(path: &Path) -> Result<Self, io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.nodes.len() * std::mem::size_of::<OctreeNode>()
            + self.bricks.len() * std::mem::size_of::<VoxelBrick>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::procgen::TreeGenerator;

    #[test]
    fn test_tree_data_round_trip() {
        // Generate a tree
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let octree = generator.generate(8.0, 5);

        // Convert to TreeData
        let tree_data = TreeData::from_octree(&octree, TreeStyle::Oak, 42);

        // Serialize and deserialize
        let bytes = tree_data.to_bytes().expect("serialization failed");
        let restored = TreeData::from_bytes(&bytes).expect("deserialization failed");

        // Verify
        assert_eq!(restored.version, TREE_DATA_VERSION);
        assert_eq!(restored.style, TreeStyle::Oak as u8);
        assert_eq!(restored.seed, 42);
        assert_eq!(restored.nodes.len(), tree_data.nodes.len());
        assert_eq!(restored.bricks.len(), tree_data.bricks.len());
    }

    #[test]
    fn test_tree_data_to_octree() {
        let mut generator = TreeGenerator::from_style(123, TreeStyle::Elm);
        let original = generator.generate(8.0, 5);

        let tree_data = TreeData::from_octree(&original, TreeStyle::Elm, 123);
        let restored = tree_data.to_octree();

        assert_eq!(restored.node_count(), original.node_count());
        assert_eq!(restored.brick_count(), original.brick_count());
        assert_eq!(restored.root_size(), original.root_size());
        assert_eq!(restored.max_depth(), original.max_depth());
    }

    #[test]
    fn test_compression_ratio() {
        let mut generator = TreeGenerator::from_style(999, TreeStyle::Willow);
        let octree = generator.generate(8.0, 5);
        let tree_data = TreeData::from_octree(&octree, TreeStyle::Willow, 999);

        let compressed = tree_data.to_bytes().expect("compression failed");
        let uncompressed_size = tree_data.memory_usage();

        println!(
            "Tree compression: {} bytes -> {} bytes ({:.1}% of original)",
            uncompressed_size,
            compressed.len(),
            100.0 * compressed.len() as f64 / uncompressed_size as f64
        );

        // Should achieve some compression
        assert!(compressed.len() < uncompressed_size);
    }

    #[test]
    fn test_style_conversion() {
        let tree_data = TreeData {
            version: TREE_DATA_VERSION,
            style: TreeStyle::Elm as u8,
            seed: 0,
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
            root_size: 16.0,
            max_depth: 7,
            nodes: vec![],
            bricks: vec![],
        };

        assert_eq!(tree_data.style(), TreeStyle::Elm);
    }
}
