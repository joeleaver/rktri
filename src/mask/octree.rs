//! Sparse octree for typed mask values.
//!
//! Stores typed values in a sparse octree with separate node and value arrays.
//! The node array is non-generic (fixed 16 bytes per node), while leaf values
//! are stored in a separate `Vec<T>`.

use glam::Vec3;
use crate::math::Aabb;
use super::MaskValue;

/// A single node in the mask octree.
///
/// Internal nodes have children (via `child_mask`); leaf children store
/// values directly (via `leaf_mask` + `value_offset`).
#[derive(Clone, Debug)]
#[repr(C)]
pub struct MaskNode {
    /// Which of 8 children exist (bit i = child i present)
    pub child_mask: u8,
    /// Which existing children are leaves storing values (subset of child_mask)
    pub leaf_mask: u8,
    /// Reserved flags
    pub flags: u16,
    /// Index of first internal (non-leaf) child in nodes[]
    pub child_offset: u32,
    /// Index of first leaf value in values[]
    pub value_offset: u32,
    /// Index into values[] for LOD summary value of this node
    pub lod_value_idx: u32,
}

impl MaskNode {
    pub fn empty() -> Self {
        Self {
            child_mask: 0,
            leaf_mask: 0,
            flags: 0,
            child_offset: 0,
            value_offset: 0,
            lod_value_idx: u32::MAX,
        }
    }

    /// Number of internal (non-leaf) children.
    pub fn internal_child_count(&self) -> u32 {
        (self.child_mask & !self.leaf_mask).count_ones()
    }

    /// Number of leaf children.
    pub fn leaf_child_count(&self) -> u32 {
        (self.child_mask & self.leaf_mask).count_ones()
    }

    /// Returns true if this node has no children at all.
    pub fn is_empty(&self) -> bool {
        self.child_mask == 0
    }
}

/// Sparse octree storing typed mask values.
///
/// `root_size` is typically the chunk size (4.0m). `max_depth` controls
/// resolution: depth 3 = 8 cells/side = 0.5m, depth 5 = 32 cells/side = 0.125m.
pub struct MaskOctree<T: MaskValue> {
    nodes: Vec<MaskNode>,
    values: Vec<T>,
    root_size: f32,
    max_depth: u8,
}

impl<T: MaskValue> MaskOctree<T> {
    /// Create an empty mask octree.
    pub fn new(root_size: f32, max_depth: u8) -> Self {
        Self {
            nodes: vec![MaskNode::empty()],
            values: Vec::new(),
            root_size,
            max_depth,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(root_size: f32, max_depth: u8, node_cap: usize, value_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(node_cap);
        nodes.push(MaskNode::empty());
        Self {
            nodes,
            values: Vec::with_capacity(value_cap),
            root_size,
            max_depth,
        }
    }

    /// Returns true if the octree contains only default values.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1 && self.nodes[0].is_empty() && self.values.is_empty()
    }

    /// Size of the smallest cell at max depth.
    pub fn voxel_size(&self) -> f32 {
        self.root_size / (1u32 << self.max_depth) as f32
    }

    /// Root size in world units.
    pub fn root_size(&self) -> f32 {
        self.root_size
    }

    /// Max depth of the octree.
    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of stored values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Slice of all nodes (for GPU upload).
    pub fn nodes_slice(&self) -> &[MaskNode] {
        &self.nodes
    }

    /// Slice of all values (for GPU upload).
    pub fn values_slice(&self) -> &[T] {
        &self.values
    }

    /// Add a node, returning its index.
    pub fn add_node(&mut self, node: MaskNode) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
    }

    /// Add a value, returning its index.
    pub fn add_value(&mut self, value: T) -> u32 {
        let idx = self.values.len() as u32;
        self.values.push(value);
        idx
    }

    /// Get mutable reference to a node.
    pub fn node_mut(&mut self, index: u32) -> &mut MaskNode {
        &mut self.nodes[index as usize]
    }

    /// Get reference to a node.
    pub fn node(&self, index: u32) -> &MaskNode {
        &self.nodes[index as usize]
    }

    /// Get a stored value by index.
    pub fn value(&self, index: u32) -> T {
        self.values[index as usize]
    }

    /// Sample the mask value at a local position (relative to octree origin).
    ///
    /// Traverses the tree from root to the deepest available node,
    /// returning the most specific value found.
    pub fn sample(&self, origin: Vec3, local_pos: Vec3) -> T {
        self.sample_node(0, origin, self.root_size, local_pos)
    }

    fn sample_node(&self, node_idx: u32, origin: Vec3, size: f32, pos: Vec3) -> T {
        let node = &self.nodes[node_idx as usize];

        if node.is_empty() {
            // Leaf or empty node — return LOD value if available
            if node.lod_value_idx != u32::MAX {
                return self.values[node.lod_value_idx as usize];
            }
            return T::default();
        }

        // Determine which child octant contains this position
        let half = size * 0.5;
        let center = origin + Vec3::splat(half);
        let child_idx = ((if pos.x >= center.x { 1u8 } else { 0 })
            | (if pos.y >= center.y { 2 } else { 0 })
            | (if pos.z >= center.z { 4 } else { 0 })) as u8;

        // Check if this child exists
        if node.child_mask & (1 << child_idx) == 0 {
            // Child doesn't exist — return this node's LOD value
            if node.lod_value_idx != u32::MAX {
                return self.values[node.lod_value_idx as usize];
            }
            return T::default();
        }

        let child_origin = Vec3::new(
            if child_idx & 1 != 0 { origin.x + half } else { origin.x },
            if child_idx & 2 != 0 { origin.y + half } else { origin.y },
            if child_idx & 4 != 0 { origin.z + half } else { origin.z },
        );

        // Is this child a leaf?
        if node.leaf_mask & (1 << child_idx) != 0 {
            // Count leaf children before this one to find value index
            let leaf_rank = (node.leaf_mask & ((1 << child_idx) - 1)).count_ones();
            let value_idx = node.value_offset + leaf_rank;
            return self.values[value_idx as usize];
        }

        // Internal child — recurse
        let internal_rank = ((node.child_mask & !node.leaf_mask) & ((1 << child_idx) - 1)).count_ones();
        let child_node_idx = node.child_offset + internal_rank;
        self.sample_node(child_node_idx, child_origin, half, pos)
    }

    /// Classify a region: returns `Some(val)` if uniform, `None` if mixed.
    ///
    /// Checks if the AABB falls entirely within a single leaf or uniform subtree.
    pub fn classify_region(&self, origin: Vec3, aabb: &Aabb) -> Option<T> {
        self.classify_node(0, origin, self.root_size, aabb)
    }

    fn classify_node(&self, node_idx: u32, origin: Vec3, size: f32, aabb: &Aabb) -> Option<T> {
        let node = &self.nodes[node_idx as usize];

        if node.is_empty() {
            // Uniform node
            if node.lod_value_idx != u32::MAX {
                return Some(self.values[node.lod_value_idx as usize]);
            }
            return Some(T::default());
        }

        let half = size * 0.5;

        // Check which children the AABB overlaps
        let min_child = (
            if aabb.min.x >= origin.x + half { 1u8 } else { 0 }
                | if aabb.min.y >= origin.y + half { 2 } else { 0 }
                | if aabb.min.z >= origin.z + half { 4 } else { 0 },
        );
        let max_child = (
            if aabb.max.x > origin.x + half { 1u8 } else { 0 }
                | if aabb.max.y > origin.y + half { 2 } else { 0 }
                | if aabb.max.z > origin.z + half { 4 } else { 0 },
        );

        // If min and max fall in the same octant, recurse into that child
        if min_child.0 == max_child.0 {
            let child_idx = min_child.0;

            if node.child_mask & (1 << child_idx) == 0 {
                // Child doesn't exist — uniform default
                if node.lod_value_idx != u32::MAX {
                    return Some(self.values[node.lod_value_idx as usize]);
                }
                return Some(T::default());
            }

            if node.leaf_mask & (1 << child_idx) != 0 {
                let leaf_rank = (node.leaf_mask & ((1 << child_idx) - 1)).count_ones();
                return Some(self.values[(node.value_offset + leaf_rank) as usize]);
            }

            let internal_rank = ((node.child_mask & !node.leaf_mask) & ((1 << child_idx) - 1)).count_ones();
            let child_node_idx = node.child_offset + internal_rank;
            let child_origin = Vec3::new(
                if child_idx & 1 != 0 { origin.x + half } else { origin.x },
                if child_idx & 2 != 0 { origin.y + half } else { origin.y },
                if child_idx & 4 != 0 { origin.z + half } else { origin.z },
            );
            return self.classify_node(child_node_idx, child_origin, half, aabb);
        }

        // AABB spans multiple children — check if all overlapped children have same value
        // For simplicity, return None (mixed) when spanning multiple children
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::BiomeId;

    #[test]
    fn test_empty_octree() {
        let tree: MaskOctree<f32> = MaskOctree::new(4.0, 3);
        assert!(tree.is_empty());
        assert_eq!(tree.voxel_size(), 0.5);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.value_count(), 0);
    }

    #[test]
    fn test_sample_empty_returns_default() {
        let tree: MaskOctree<BiomeId> = MaskOctree::new(4.0, 3);
        let val = tree.sample(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(val, BiomeId::default());
    }

    #[test]
    fn test_uniform_octree() {
        let mut tree: MaskOctree<BiomeId> = MaskOctree::new(4.0, 3);
        // Store a single LOD value on root = uniform forest
        let val_idx = tree.add_value(BiomeId::FOREST);
        tree.node_mut(0).lod_value_idx = val_idx;

        let sampled = tree.sample(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(sampled, BiomeId::FOREST);

        // classify_region should return uniform
        let aabb = Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(3.5, 3.5, 3.5));
        assert_eq!(tree.classify_region(Vec3::ZERO, &aabb), Some(BiomeId::FOREST));
    }

    #[test]
    fn test_voxel_size_computation() {
        let tree: MaskOctree<f32> = MaskOctree::new(4.0, 3);
        assert!((tree.voxel_size() - 0.5).abs() < 1e-6);

        let tree2: MaskOctree<f32> = MaskOctree::new(4.0, 5);
        assert!((tree2.voxel_size() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_node_counts() {
        let node = MaskNode {
            child_mask: 0b1010_1010,  // children 1,3,5,7
            leaf_mask:  0b0000_1010,  // children 1,3 are leaves
            flags: 0,
            child_offset: 0,
            value_offset: 0,
            lod_value_idx: u32::MAX,
        };
        assert_eq!(node.internal_child_count(), 2); // children 5,7
        assert_eq!(node.leaf_child_count(), 2);      // children 1,3
    }
}
