//! HashDAG - Content-addressable DAG for maximum voxel compression
//!
//! A HashDAG stores nodes per-level where identical subtrees share the same storage
//! regardless of position. This provides maximum deduplication since nodes are
//! identified by their content hash rather than their position in the tree.

use std::collections::HashMap;
use super::{Octree, OctreeNode};
use crate::voxel::brick::VoxelBrick;

/// A content hash for DAG nodes
pub type NodeHash = u64;

/// Node data in the HashDAG
#[derive(Clone, Debug)]
pub struct HashDagNode {
    /// Node flags (valid mask + leaf mask)
    pub flags: u32,
    /// Child hashes (for internal children) - index into level above
    pub children: Vec<NodeHash>,
    /// Brick hashes (for leaf children) - index into brick pool
    pub brick_hashes: Vec<NodeHash>,
    /// LOD color data
    pub lod_color: u16,
    /// LOD material data
    pub lod_material: u16,
}

/// Content-addressable DAG for maximum voxel compression.
///
/// Nodes are stored per-level (level 0 = root, level N = leaves).
/// Identical subtrees share the same node regardless of position.
pub struct HashDag {
    /// Level-wise node storage (level 0 = root)
    levels: Vec<HashMap<NodeHash, HashDagNode>>,
    /// Brick storage (deduplicated)
    bricks: HashMap<NodeHash, VoxelBrick>,
    /// Root hash
    root: NodeHash,
    /// Root size in meters
    root_size: f32,
    /// Maximum depth
    max_depth: usize,
    /// Total unique nodes
    total_nodes: usize,
}

impl HashDag {
    /// Build a HashDAG from an existing Octree.
    pub fn from_octree(octree: &Octree) -> Self {
        let max_depth = octree.max_depth() as usize;

        // Initialize level storage
        let mut levels: Vec<HashMap<NodeHash, HashDagNode>> =
            (0..=max_depth).map(|_| HashMap::new()).collect();
        let mut bricks = HashMap::new();

        // Phase 1: Hash all bricks using FNV-1a
        let old_bricks = octree.bricks_slice();
        let mut brick_hashes = Vec::with_capacity(old_bricks.len());

        for brick in old_bricks {
            let hash = Self::hash_brick(brick);
            brick_hashes.push(hash);
            bricks.entry(hash).or_insert(*brick);
        }

        // Phase 2: Process octree bottom-up, level by level
        // Build mapping from old node indices to their hashes at each level
        let mut node_to_hash: Vec<Option<NodeHash>> = vec![None; octree.node_count()];
        let mut node_to_level: Vec<usize> = vec![0; octree.node_count()];

        // First pass: determine depth of each node
        Self::compute_node_depths(octree, 0, 0, &mut node_to_level);

        // Second pass: hash nodes bottom-up
        let root_hash = Self::hash_octree_recursive(
            octree,
            0,
            0,
            &brick_hashes,
            &node_to_level,
            &mut node_to_hash,
            &mut levels,
        );

        let total_nodes = levels.iter().map(|level| level.len()).sum();

        Self {
            levels,
            bricks,
            root: root_hash,
            root_size: octree.root_size(),
            max_depth,
            total_nodes,
        }
    }

    /// Compute depth of each node in the tree
    fn compute_node_depths(
        octree: &Octree,
        node_idx: u32,
        depth: usize,
        node_to_level: &mut [usize],
    ) {
        node_to_level[node_idx as usize] = depth;

        let node = octree.node(node_idx);
        let valid_mask = node.child_valid_mask();
        let leaf_mask = node.child_leaf_mask();

        if valid_mask == 0 {
            return;
        }

        let mut internal_count = 0u32;
        for child_idx in 0u8..8 {
            let child_mask = 1u8 << child_idx;
            if valid_mask & child_mask == 0 {
                continue;
            }

            // Only recurse into internal children
            if leaf_mask & child_mask == 0 {
                let child_node_idx = node.child_offset + internal_count;
                Self::compute_node_depths(octree, child_node_idx, depth + 1, node_to_level);
                internal_count += 1;
            }
        }
    }

    /// Hash a node recursively, processing children first (bottom-up)
    fn hash_octree_recursive(
        octree: &Octree,
        node_idx: u32,
        level: usize,
        brick_hashes: &[NodeHash],
        node_to_level: &[usize],
        node_to_hash: &mut [Option<NodeHash>],
        levels: &mut [HashMap<NodeHash, HashDagNode>],
    ) -> NodeHash {
        // Check if already hashed
        if let Some(hash) = node_to_hash[node_idx as usize] {
            return hash;
        }

        let node = octree.node(node_idx);
        let valid_mask = node.child_valid_mask();
        let leaf_mask = node.child_leaf_mask();

        let mut children = Vec::new();
        let mut brick_hashes_vec = Vec::new();

        if valid_mask != 0 {
            let mut internal_count = 0u32;
            let mut leaf_count = 0u32;

            for child_idx in 0u8..8 {
                let child_mask = 1u8 << child_idx;
                if valid_mask & child_mask == 0 {
                    continue;
                }

                if leaf_mask & child_mask != 0 {
                    // Leaf child - get brick hash
                    let brick_idx = node.brick_offset + leaf_count;
                    let brick_hash = brick_hashes[brick_idx as usize];
                    brick_hashes_vec.push(brick_hash);
                    leaf_count += 1;
                } else {
                    // Internal child - recurse
                    let child_node_idx = node.child_offset + internal_count;
                    let child_hash = Self::hash_octree_recursive(
                        octree,
                        child_node_idx,
                        level + 1,
                        brick_hashes,
                        node_to_level,
                        node_to_hash,
                        levels,
                    );
                    children.push(child_hash);
                    internal_count += 1;
                }
            }
        }

        // Compute hash for this node
        let hash = Self::hash_node(node.flags, &children, &brick_hashes_vec, node.lod_color, node.lod_material);

        // Store node at its level
        let dag_node = HashDagNode {
            flags: node.flags,
            children,
            brick_hashes: brick_hashes_vec,
            lod_color: node.lod_color,
            lod_material: node.lod_material,
        };

        levels[level].entry(hash).or_insert(dag_node);
        node_to_hash[node_idx as usize] = Some(hash);

        hash
    }

    /// Hash a voxel brick using FNV-1a
    fn hash_brick(brick: &VoxelBrick) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for voxel in &brick.voxels {
            hash ^= voxel.color as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= voxel.material_id as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= voxel.flags as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Hash a node based on its content
    fn hash_node(
        flags: u32,
        children: &[NodeHash],
        brick_hashes: &[NodeHash],
        lod_color: u16,
        lod_material: u16,
    ) -> NodeHash {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;

        // Hash flags
        hash ^= flags as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Hash child hashes
        for &child_hash in children {
            hash ^= child_hash;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Hash brick hashes
        for &brick_hash in brick_hashes {
            hash ^= brick_hash;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Hash LOD data
        hash ^= lod_color as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= lod_material as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        hash
    }

    /// Get compression statistics.
    pub fn stats(&self) -> HashDagStats {
        let unique_nodes = self.total_nodes;
        let unique_bricks = self.bricks.len();

        // Rough memory estimate
        let node_memory = unique_nodes * (
            std::mem::size_of::<u32>() + // flags
            std::mem::size_of::<u16>() * 2 + // lod_color, lod_material
            std::mem::size_of::<Vec<NodeHash>>() * 2 + // children, brick_hashes
            std::mem::size_of::<u64>() // hash key
        );
        let brick_memory = unique_bricks * (
            std::mem::size_of::<VoxelBrick>() +
            std::mem::size_of::<u64>() // hash key
        );

        HashDagStats {
            levels: self.levels.len(),
            unique_nodes,
            unique_bricks,
            total_memory_bytes: node_memory + brick_memory,
        }
    }

    /// Total unique nodes across all levels.
    pub fn unique_node_count(&self) -> usize {
        self.total_nodes
    }

    /// Total unique bricks.
    pub fn unique_brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Get root hash.
    pub fn root_hash(&self) -> NodeHash {
        self.root
    }

    /// Get node at a specific level by hash.
    pub fn node_at(&self, level: usize, hash: NodeHash) -> Option<&HashDagNode> {
        if level < self.levels.len() {
            self.levels[level].get(&hash)
        } else {
            None
        }
    }

    /// Convert back to a standard Octree (for GPU upload).
    pub fn to_octree(&self) -> Octree {
        let mut octree = Octree::with_capacity(
            self.root_size,
            self.max_depth as u8,
            self.total_nodes,
            self.bricks.len(),
        );

        // Build mapping from hash to new indices
        let mut hash_to_node_idx: HashMap<NodeHash, u32> = HashMap::new();
        let mut hash_to_brick_idx: HashMap<NodeHash, u32> = HashMap::new();

        // Add all bricks first
        for (hash, brick) in &self.bricks {
            let idx = octree.add_brick(*brick);
            hash_to_brick_idx.insert(*hash, idx);
        }

        // Process nodes top-down from root
        if let Some(root_node) = self.levels[0].get(&self.root) {
            Self::rebuild_octree_recursive(
                root_node,
                0,
                &self.levels,
                &self.bricks,
                &mut hash_to_node_idx,
                &hash_to_brick_idx,
                &mut octree,
            );
        }

        octree
    }

    /// Recursively rebuild octree from HashDAG
    fn rebuild_octree_recursive(
        dag_node: &HashDagNode,
        level: usize,
        levels: &[HashMap<NodeHash, HashDagNode>],
        _bricks: &HashMap<NodeHash, VoxelBrick>,
        hash_to_node_idx: &mut HashMap<NodeHash, u32>,
        hash_to_brick_idx: &HashMap<NodeHash, u32>,
        octree: &mut Octree,
    ) -> u32 {
        // Create octree node
        let mut node = OctreeNode::empty();
        node.flags = dag_node.flags;
        node.lod_color = dag_node.lod_color;
        node.lod_material = dag_node.lod_material;

        let valid_mask = (dag_node.flags & 0xFF) as u8;
        let leaf_mask = ((dag_node.flags >> 8) & 0xFF) as u8;

        // Process children
        if valid_mask != 0 {
            let mut child_indices = Vec::new();
            let mut brick_indices = Vec::new();

            let mut child_iter = dag_node.children.iter();
            let mut brick_iter = dag_node.brick_hashes.iter();

            for child_idx in 0u8..8 {
                let child_mask = 1u8 << child_idx;
                if valid_mask & child_mask == 0 {
                    continue;
                }

                if leaf_mask & child_mask != 0 {
                    // Leaf child - get brick index
                    if let Some(&brick_hash) = brick_iter.next() {
                        if let Some(&brick_idx) = hash_to_brick_idx.get(&brick_hash) {
                            brick_indices.push(brick_idx);
                        }
                    }
                } else {
                    // Internal child - recurse
                    if let Some(&child_hash) = child_iter.next() {
                        // Check if already processed
                        let child_node_idx = if let Some(&existing_idx) = hash_to_node_idx.get(&child_hash) {
                            existing_idx
                        } else {
                            // Process child
                            if let Some(child_dag_node) = levels.get(level + 1).and_then(|l| l.get(&child_hash)) {
                                let idx = Self::rebuild_octree_recursive(
                                    child_dag_node,
                                    level + 1,
                                    levels,
                                    _bricks,
                                    hash_to_node_idx,
                                    hash_to_brick_idx,
                                    octree,
                                );
                                hash_to_node_idx.insert(child_hash, idx);
                                idx
                            } else {
                                0 // Shouldn't happen
                            }
                        };
                        child_indices.push(child_node_idx);
                    }
                }
            }

            // Set offsets
            if !child_indices.is_empty() {
                node.child_offset = child_indices[0];
            }
            if !brick_indices.is_empty() {
                node.brick_offset = brick_indices[0];
            }
        }

        octree.add_node(node)
    }

    /// Merge another HashDAG into this one (for combining base + edits).
    /// Returns a new HashDAG with shared structure.
    pub fn merge(&self, other: &HashDag) -> HashDag {
        let max_depth = self.max_depth.max(other.max_depth);

        // Union of all bricks
        let mut merged_bricks = self.bricks.clone();
        for (hash, brick) in &other.bricks {
            merged_bricks.entry(*hash).or_insert(*brick);
        }

        // Union of nodes at each level
        let mut merged_levels: Vec<HashMap<NodeHash, HashDagNode>> =
            (0..=max_depth).map(|_| HashMap::new()).collect();

        for (level_idx, level) in self.levels.iter().enumerate() {
            for (hash, node) in level {
                merged_levels[level_idx].entry(*hash).or_insert_with(|| node.clone());
            }
        }

        for (level_idx, level) in other.levels.iter().enumerate() {
            for (hash, node) in level {
                merged_levels[level_idx].entry(*hash).or_insert_with(|| node.clone());
            }
        }

        // Use other's root (edits override base)
        let root = other.root;
        let root_size = other.root_size;

        let total_nodes = merged_levels.iter().map(|level| level.len()).sum();

        HashDag {
            levels: merged_levels,
            bricks: merged_bricks,
            root,
            root_size,
            max_depth,
            total_nodes,
        }
    }
}

#[derive(Debug)]
pub struct HashDagStats {
    pub levels: usize,
    pub unique_nodes: usize,
    pub unique_bricks: usize,
    pub total_memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::svo::builder::{OctreeBuilder, create_test_sphere};

    #[test]
    fn test_from_octree_basic() {
        // Create a simple octree
        let size = 32u32;
        let voxels = create_test_sphere(size, 14.0);
        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        // Convert to HashDAG
        let hashdag = HashDag::from_octree(&octree);

        // Should have some nodes and bricks
        assert!(hashdag.unique_node_count() > 0);
        assert!(hashdag.unique_brick_count() > 0);

        let stats = hashdag.stats();
        println!("HashDAG stats: {:?}", stats);
    }

    #[test]
    fn test_compression_ratio() {
        // Create a symmetric scene - should compress well
        let size = 64u32;
        let voxels = create_test_sphere(size, 28.0);
        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        let original_nodes = octree.node_count();
        let original_bricks = octree.brick_count();

        let hashdag = HashDag::from_octree(&octree);
        let unique_nodes = hashdag.unique_node_count();
        let unique_bricks = hashdag.unique_brick_count();

        println!(
            "Compression: nodes {} -> {} ({:.2}x), bricks {} -> {} ({:.2}x)",
            original_nodes, unique_nodes,
            original_nodes as f32 / unique_nodes.max(1) as f32,
            original_bricks, unique_bricks,
            original_bricks as f32 / unique_bricks.max(1) as f32
        );

        // Should achieve some compression on symmetric data
        assert!(unique_nodes <= original_nodes);
        assert!(unique_bricks <= original_bricks);
    }

    #[test]
    fn test_to_octree_roundtrip() {
        // Create octree
        let size = 32u32;
        let voxels = create_test_sphere(size, 14.0);
        let builder = OctreeBuilder::new(size);
        let original = builder.build(&voxels, size as f32);

        // Convert to HashDAG and back
        let hashdag = HashDag::from_octree(&original);
        let reconstructed = hashdag.to_octree();

        // Should have same root size and max depth
        assert_eq!(reconstructed.root_size(), original.root_size());
        assert_eq!(reconstructed.max_depth(), original.max_depth());

        // Should preserve structure (may have different indices due to deduplication)
        assert!(reconstructed.node_count() > 0);
        assert!(reconstructed.brick_count() > 0);
    }
}
