//! Sparse Voxel Octree container

use glam::Vec3;

use super::node::OctreeNode;
use crate::voxel::brick::VoxelBrick;
use crate::voxel::voxel::Voxel;

/// Sparse Voxel Octree data structure
#[derive(Debug, Clone)]
pub struct Octree {
    /// All octree nodes (root is at index 0)
    nodes: Vec<OctreeNode>,
    /// Voxel bricks referenced by leaf nodes
    bricks: Vec<VoxelBrick>,
    /// World-space size of the root node
    root_size: f32,
    /// Maximum tree depth (affects minimum voxel size)
    max_depth: u8,
    /// If true, child nodes use dense indexing (child_offset + child_idx)
    /// instead of packed indexing (child_offset + count_of_valid_internal_before).
    /// BrushOctreeBuilder uses dense; AdaptiveOctreeBuilder uses packed.
    dense_children: bool,
}

impl Octree {
    /// Create a new empty octree
    pub fn new(root_size: f32, max_depth: u8) -> Self {
        Self {
            nodes: vec![OctreeNode::empty()],
            bricks: Vec::new(),
            root_size,
            max_depth,
            dense_children: false,
        }
    }

    /// Create octree with pre-allocated capacity
    pub fn with_capacity(root_size: f32, max_depth: u8, node_capacity: usize, brick_capacity: usize) -> Self {
        Self {
            nodes: {
                let mut v = Vec::with_capacity(node_capacity);
                v.push(OctreeNode::empty());
                v
            },
            bricks: Vec::with_capacity(brick_capacity),
            root_size,
            max_depth,
            dense_children: false,
        }
    }

    /// Reconstruct octree from serialized node and brick data
    pub fn from_serialized(
        root_size: f32,
        max_depth: u8,
        nodes: Vec<OctreeNode>,
        bricks: Vec<VoxelBrick>,
    ) -> Self {
        Self {
            nodes,
            bricks,
            root_size,
            max_depth,
            dense_children: false,
        }
    }

    /// Mark this octree as using dense child indexing
    /// (child_offset + child_idx instead of packed counting)
    pub fn set_dense_children(&mut self, dense: bool) {
        self.dense_children = dense;
    }

    /// Get root node
    pub fn root(&self) -> &OctreeNode {
        &self.nodes[0]
    }

    /// Get mutable root node
    pub fn root_mut(&mut self) -> &mut OctreeNode {
        &mut self.nodes[0]
    }

    /// Get node by index
    pub fn node(&self, index: u32) -> &OctreeNode {
        &self.nodes[index as usize]
    }

    /// Get mutable node by index
    pub fn node_mut(&mut self, index: u32) -> &mut OctreeNode {
        &mut self.nodes[index as usize]
    }

    /// Get brick by index
    pub fn brick(&self, index: u32) -> &VoxelBrick {
        &self.bricks[index as usize]
    }

    /// Get mutable brick by index
    pub fn brick_mut(&mut self, index: u32) -> &mut VoxelBrick {
        &mut self.bricks[index as usize]
    }

    /// Get all nodes as slice (for GPU upload)
    pub fn nodes_slice(&self) -> &[OctreeNode] {
        &self.nodes
    }

    /// Get all bricks as slice (for GPU upload)
    pub fn bricks_slice(&self) -> &[VoxelBrick] {
        &self.bricks
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of bricks
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Get root size in world units
    pub fn root_size(&self) -> f32 {
        self.root_size
    }

    /// Get maximum depth
    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Calculate voxel size at maximum depth
    pub fn voxel_size(&self) -> f32 {
        self.root_size / (1 << self.max_depth) as f32
    }

    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<OctreeNode>() * self.nodes.len()
            + std::mem::size_of::<VoxelBrick>() * self.bricks.len()
    }

    /// Add a node and return its index
    pub fn add_node(&mut self, node: OctreeNode) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(node);
        index
    }

    /// Add a brick and return its index
    pub fn add_brick(&mut self, brick: VoxelBrick) -> u32 {
        let index = self.bricks.len() as u32;
        self.bricks.push(brick);
        index
    }

    /// Check if octree is empty (only has empty root)
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1 && self.nodes[0].is_empty() && self.bricks.is_empty()
    }

    /// Sample voxel at a local position within the octree bounds.
    /// The octree is centered at origin, so valid positions are in range [-root_size/2, root_size/2].
    /// Returns Voxel::EMPTY if position is outside bounds or in empty region.
    pub fn sample_voxel(&self, local_pos: Vec3) -> Voxel {
        let half = self.root_size / 2.0;

        // Check bounds
        if local_pos.x < -half
            || local_pos.x >= half
            || local_pos.y < -half
            || local_pos.y >= half
            || local_pos.z < -half
            || local_pos.z >= half
        {
            return Voxel::EMPTY;
        }

        // Traverse octree to find voxel
        self.sample_voxel_recursive(0, Vec3::ZERO, self.root_size, local_pos)
    }

    fn sample_voxel_recursive(
        &self,
        node_idx: u32,
        center: Vec3,
        size: f32,
        target: Vec3,
    ) -> Voxel {
        let node = &self.nodes[node_idx as usize];

        if node.is_empty() {
            return Voxel::EMPTY;
        }

        // Terminal leaf: single brick contains 2x2x2 voxels for this node
        if node.is_terminal_leaf() {
            let brick = &self.bricks[node.brick_offset as usize];
            let bx = if target.x >= center.x { 1 } else { 0 };
            let by = if target.y >= center.y { 1 } else { 0 };
            let bz = if target.z >= center.z { 1 } else { 0 };
            return *brick.get(bx, by, bz);
        }

        // Determine which octant the target falls in
        let child_idx = ((if target.x >= center.x { 1 } else { 0 })
            | (if target.y >= center.y { 2 } else { 0 })
            | (if target.z >= center.z { 4 } else { 0 })) as u8;

        // Check if this child exists
        let child_mask = 1u8 << child_idx;
        if node.child_valid_mask() & child_mask == 0 {
            return Voxel::EMPTY;
        }

        // Calculate child center
        let quarter = size / 4.0;
        let child_center = center
            + Vec3::new(
                if child_idx & 1 != 0 { quarter } else { -quarter },
                if child_idx & 2 != 0 { quarter } else { -quarter },
                if child_idx & 4 != 0 { quarter } else { -quarter },
            );

        // Check if child is a leaf (brick)
        if node.child_leaf_mask() & child_mask != 0 {
            // Count how many leaf children come before this one to get brick index
            let leaf_mask = node.child_leaf_mask();
            let valid_mask = node.child_valid_mask();
            let mut brick_count = 0u32;
            for i in 0..child_idx {
                let m = 1u8 << i;
                if valid_mask & m != 0 && leaf_mask & m != 0 {
                    brick_count += 1;
                }
            }
            let brick_idx = node.brick_offset + brick_count;
            let brick = &self.bricks[brick_idx as usize];

            // Calculate position within brick (2x2x2)
            // Child covers size/2, brick has 2 voxels per axis, so each voxel is size/4
            let voxel_size = size / 4.0;
            let rel = target - child_center + Vec3::splat(voxel_size); // offset to corner
            let bx = ((rel.x / voxel_size).floor() as i32).clamp(0, 1) as u8;
            let by = ((rel.y / voxel_size).floor() as i32).clamp(0, 1) as u8;
            let bz = ((rel.z / voxel_size).floor() as i32).clamp(0, 1) as u8;

            return *brick.get(bx, by, bz);
        }

        // Descend to child node
        let child_node_idx = if self.dense_children {
            // Dense: all 8 children allocated consecutively
            node.child_offset + child_idx as u32
        } else {
            // Packed: only valid internal children are consecutive
            let leaf_mask = node.child_leaf_mask();
            let valid_mask = node.child_valid_mask();
            let mut internal_count = 0u32;
            for i in 0..child_idx {
                let m = 1u8 << i;
                if valid_mask & m != 0 && leaf_mask & m == 0 {
                    internal_count += 1;
                }
            }
            node.child_offset + internal_count
        };
        self.sample_voxel_recursive(child_node_idx, child_center, size / 2.0, target)
    }

    /// Iterate all non-empty voxels, calling the callback with (local_position, voxel).
    /// Positions are relative to octree center (range [-root_size/2, root_size/2]).
    pub fn iterate_voxels<F: FnMut(Vec3, Voxel)>(&self, mut callback: F) {
        if self.nodes.is_empty() || self.root().is_empty() {
            return;
        }
        self.iterate_voxels_recursive(0, Vec3::ZERO, self.root_size, &mut callback);
    }

    fn iterate_voxels_recursive<F: FnMut(Vec3, Voxel)>(
        &self,
        node_idx: u32,
        center: Vec3,
        size: f32,
        callback: &mut F,
    ) {
        let node = &self.nodes[node_idx as usize];

        if node.is_empty() {
            return;
        }

        // Terminal leaf: single brick contains 2x2x2 voxels
        if node.is_terminal_leaf() {
            let brick = &self.bricks[node.brick_offset as usize];
            let half_voxel = size / 4.0;
            for bz in 0..2u8 {
                for by in 0..2u8 {
                    for bx in 0..2u8 {
                        let voxel = brick.get(bx, by, bz);
                        if !voxel.is_empty() {
                            let voxel_center = center
                                + Vec3::new(
                                    if bx == 1 { half_voxel } else { -half_voxel },
                                    if by == 1 { half_voxel } else { -half_voxel },
                                    if bz == 1 { half_voxel } else { -half_voxel },
                                );
                            callback(voxel_center, *voxel);
                        }
                    }
                }
            }
            return;
        }

        let quarter = size / 4.0;
        let valid_mask = node.child_valid_mask();
        let leaf_mask = node.child_leaf_mask();

        let mut internal_count = 0u32;
        let mut leaf_count = 0u32;

        for child_idx in 0u8..8 {
            let child_mask = 1u8 << child_idx;
            if valid_mask & child_mask == 0 {
                continue;
            }

            let child_center = center
                + Vec3::new(
                    if child_idx & 1 != 0 { quarter } else { -quarter },
                    if child_idx & 2 != 0 { quarter } else { -quarter },
                    if child_idx & 4 != 0 { quarter } else { -quarter },
                );

            if leaf_mask & child_mask != 0 {
                // Leaf node - iterate brick voxels
                let brick_idx = node.brick_offset + leaf_count;
                let brick = &self.bricks[brick_idx as usize];
                let voxel_size = size / 4.0; // Each brick voxel is size/4

                for bz in 0..2u8 {
                    for by in 0..2u8 {
                        for bx in 0..2u8 {
                            let voxel = brick.get(bx, by, bz);
                            if !voxel.is_empty() {
                                let voxel_center = child_center
                                    + Vec3::new(
                                        (bx as f32 - 0.5) * voxel_size,
                                        (by as f32 - 0.5) * voxel_size,
                                        (bz as f32 - 0.5) * voxel_size,
                                    );
                                callback(voxel_center, *voxel);
                            }
                        }
                    }
                }
                leaf_count += 1;
            } else {
                // Internal node - recurse
                let child_node_idx = if self.dense_children {
                    node.child_offset + child_idx as u32
                } else {
                    node.child_offset + internal_count
                };
                self.iterate_voxels_recursive(child_node_idx, child_center, size / 2.0, callback);
                internal_count += 1;
            }
        }
    }
}

impl Octree {
    /// Compact a dense-children octree into packed indexing, removing empty nodes.
    ///
    /// BrushOctreeBuilder allocates all 8 children per node (dense). This method
    /// produces a new octree with only non-empty nodes, using packed indexing
    /// (like AdaptiveOctreeBuilder). Typically reduces node count by 80-90% for
    /// sparse structures like trees.
    pub fn compact_from_dense(&self) -> Octree {
        assert!(self.dense_children, "compact_from_dense requires dense_children octree");

        let mut new_octree = Octree::with_capacity(
            self.root_size,
            self.max_depth,
            self.nodes.len() / 4, // estimate: ~25% of nodes are non-empty
            self.bricks.len(),
        );
        // New octree uses packed indexing
        // dense_children defaults to false

        // Root is already at index 0 in new_octree
        self.compact_node_recursive(&mut new_octree, 0, 0);

        new_octree
    }

    /// Recursively compact a node from the dense tree into the new packed tree.
    /// Returns true if the node is empty (caller can skip it).
    fn compact_node_recursive(
        &self,
        new_octree: &mut Octree,
        old_idx: u32,
        new_idx: u32,
    ) -> bool {
        let old_node = &self.nodes[old_idx as usize];

        // Empty node
        if old_node.is_empty() {
            return true;
        }

        // Terminal leaf: copy brick, set up terminal leaf in new tree
        if old_node.is_terminal_leaf() {
            let new_brick_idx = new_octree.add_brick(self.bricks[old_node.brick_offset as usize]);
            let new_node = new_octree.node_mut(new_idx);
            new_node.lod_color = old_node.lod_color;
            new_node.lod_material = old_node.lod_material;
            new_node.set_lod_level(old_node.lod_level());
            new_node.brick_offset = new_brick_idx;
            // child_valid_mask = 0, child_leaf_mask = 0 (terminal leaf)
            return false;
        }

        // Internal node with children
        let valid_mask = old_node.child_valid_mask();

        // Collect non-empty children (dense indexing: child_offset + child_idx)
        let mut non_empty_children: Vec<u8> = Vec::new();
        for child_idx in 0..8u8 {
            if valid_mask & (1 << child_idx) != 0 {
                non_empty_children.push(child_idx);
            }
        }

        if non_empty_children.is_empty() {
            return true;
        }

        // Pre-allocate nodes for non-empty children (they must be consecutive for packed indexing)
        let first_new_child = new_octree.node_count() as u32;
        for _ in 0..non_empty_children.len() {
            new_octree.add_node(OctreeNode::empty());
        }

        // Recursively compact each non-empty child
        let mut new_valid_mask = 0u8;
        let mut packed_idx = 0u32;

        for &child_idx in &non_empty_children {
            let old_child_idx = old_node.child_offset + child_idx as u32;
            let new_child_idx = first_new_child + packed_idx;

            let is_empty = self.compact_node_recursive(new_octree, old_child_idx, new_child_idx);
            if !is_empty {
                new_valid_mask |= 1 << child_idx;
            }

            packed_idx += 1;
        }

        // Update parent node
        let new_node = new_octree.node_mut(new_idx);
        new_node.lod_color = old_node.lod_color;
        new_node.lod_material = old_node.lod_material;
        new_node.set_lod_level(old_node.lod_level());
        new_node.set_child_valid_mask(new_valid_mask);
        new_node.set_child_leaf_mask(0); // Brush trees don't use leaf children on parent
        new_node.child_offset = if new_valid_mask != 0 { first_new_child } else { 0 };

        new_valid_mask == 0
    }
}

impl Octree {
    /// Prune empty subtrees from a packed-indexing octree.
    ///
    /// The AdaptiveOctreeBuilder pre-allocates child nodes before recursing, so
    /// regions classified as Mixed that turn out empty still leave nodes in the tree.
    /// This rebuilds the tree keeping only subtrees that contain bricks.
    pub fn prune(&self) -> Octree {
        if self.is_empty() {
            return Octree::new(self.root_size, self.max_depth);
        }

        // Phase 1: Mark which nodes have content (any descendant with bricks)
        let mut has_content = vec![false; self.nodes.len()];
        self.prune_mark_content(0, &mut has_content);

        if !has_content[0] {
            return Octree::new(self.root_size, self.max_depth);
        }

        // Phase 2: Rebuild with only content-bearing subtrees
        let content_nodes = has_content.iter().filter(|&&c| c).count();
        let mut new_octree = Octree::with_capacity(
            self.root_size,
            self.max_depth,
            content_nodes,
            self.bricks.len(),
        );

        self.prune_rebuild(0, 0, &has_content, &mut new_octree);

        new_octree
    }

    /// Phase 1: Recursively mark nodes that have content (bricks in their subtree).
    fn prune_mark_content(&self, node_idx: u32, has_content: &mut [bool]) -> bool {
        let node = &self.nodes[node_idx as usize];

        if node.is_empty() {
            return false;
        }

        // Terminal leaf always has content (a brick)
        if node.is_terminal_leaf() {
            has_content[node_idx as usize] = true;
            return true;
        }

        let valid = node.child_valid_mask();
        let leaf = node.child_leaf_mask();
        let mut any_content = false;

        // Leaf children always have content (they reference bricks)
        if valid & leaf != 0 {
            any_content = true;
        }

        // Check internal children recursively
        let mut internal_idx = 0u32;
        for bit in 0..8u8 {
            if valid & (1 << bit) == 0 {
                continue;
            }
            if leaf & (1 << bit) != 0 {
                continue; // leaf, already counted above
            }

            let child_node_idx = node.child_offset + internal_idx;
            if self.prune_mark_content(child_node_idx, has_content) {
                any_content = true;
            }
            internal_idx += 1;
        }

        has_content[node_idx as usize] = any_content;
        any_content
    }

    /// Phase 2: Rebuild tree keeping only content-bearing subtrees.
    fn prune_rebuild(
        &self,
        old_idx: u32,
        new_idx: u32,
        has_content: &[bool],
        new_octree: &mut Octree,
    ) {
        let old_node = &self.nodes[old_idx as usize];

        // Terminal leaf: copy brick
        if old_node.is_terminal_leaf() {
            let new_brick = new_octree.add_brick(self.bricks[old_node.brick_offset as usize]);
            let n = new_octree.node_mut(new_idx);
            n.lod_color = old_node.lod_color;
            n.lod_material = old_node.lod_material;
            n.set_lod_level(old_node.lod_level());
            n.brick_offset = new_brick;
            return;
        }

        let valid = old_node.child_valid_mask();
        let leaf = old_node.child_leaf_mask();

        // Determine which children to keep
        let mut new_valid: u8 = 0;
        let mut new_leaf: u8 = 0;
        let mut old_internal_idx = 0u32;

        for bit in 0..8u8 {
            if valid & (1 << bit) == 0 {
                continue;
            }
            if leaf & (1 << bit) != 0 {
                // Leaf child: always keep (has brick)
                new_valid |= 1 << bit;
                new_leaf |= 1 << bit;
            } else {
                // Internal child: keep only if subtree has content
                let child_node_idx = old_node.child_offset + old_internal_idx;
                if has_content[child_node_idx as usize] {
                    new_valid |= 1 << bit;
                }
                old_internal_idx += 1;
            }
        }

        if new_valid == 0 {
            return; // Node stays empty
        }

        // Pre-allocate nodes for kept internal children
        let new_internal_count = (new_valid & !new_leaf).count_ones();
        let first_new_child = if new_internal_count > 0 {
            let idx = new_octree.node_count() as u32;
            for _ in 0..new_internal_count {
                new_octree.add_node(OctreeNode::empty());
            }
            idx
        } else {
            0
        };

        // Detect old-format solid nodes: all children are leaves but only 1 brick allocated.
        // Convert these to terminal leaves (single brick, no children).
        let num_leaves = (valid & leaf).count_ones() as u32;
        let max_old_brick = self.bricks.len() as u32;
        if num_leaves > 0 && old_node.brick_offset + num_leaves > max_old_brick {
            // Shared-brick solid node: convert to terminal leaf
            let new_brick = new_octree.add_brick(self.bricks[old_node.brick_offset as usize]);
            let n = new_octree.node_mut(new_idx);
            n.lod_color = old_node.lod_color;
            n.lod_material = old_node.lod_material;
            n.set_lod_level(old_node.lod_level());
            n.brick_offset = new_brick;
            // child_valid_mask=0 (terminal leaf) - already default
            return;
        }

        // Copy leaf bricks in bit order
        let first_new_brick = new_octree.brick_count() as u32;
        let mut old_leaf_count = 0u32;
        for bit in 0..8u8 {
            if valid & (1 << bit) == 0 {
                continue;
            }
            if leaf & (1 << bit) != 0 {
                let old_brick_idx = old_node.brick_offset + old_leaf_count;
                new_octree.add_brick(self.bricks[old_brick_idx as usize]);
                old_leaf_count += 1;
            }
        }

        // Recurse into kept internal children
        let mut new_int_idx = 0u32;
        let mut old_int_idx = 0u32;
        for bit in 0..8u8 {
            if valid & (1 << bit) == 0 {
                continue;
            }
            if leaf & (1 << bit) != 0 {
                continue;
            }

            let old_child_idx = old_node.child_offset + old_int_idx;
            if has_content[old_child_idx as usize] {
                let new_child_idx = first_new_child + new_int_idx;
                self.prune_rebuild(old_child_idx, new_child_idx, has_content, new_octree);
                new_int_idx += 1;
            }
            old_int_idx += 1;
        }

        // Update this node
        let n = new_octree.node_mut(new_idx);
        n.lod_color = old_node.lod_color;
        n.lod_material = old_node.lod_material;
        n.set_lod_level(old_node.lod_level());
        n.set_child_valid_mask(new_valid);
        n.set_child_leaf_mask(new_leaf);
        n.child_offset = first_new_child;
        n.brick_offset = first_new_brick;
    }
}

impl Default for Octree {
    fn default() -> Self {
        Self::new(64.0, 13) // 64m root, ~1cm voxels at max depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let octree = Octree::new(64.0, 10);
        assert_eq!(octree.node_count(), 1);
        assert_eq!(octree.brick_count(), 0);
        assert_eq!(octree.root_size(), 64.0);
        assert_eq!(octree.max_depth(), 10);
    }

    #[test]
    fn test_voxel_size() {
        let octree = Octree::new(64.0, 13); // 64m / 2^13 = ~0.0078m ≈ 0.8cm
        assert!((octree.voxel_size() - 0.0078125).abs() < 0.0001);
    }

    #[test]
    fn test_add_node_and_brick() {
        let mut octree = Octree::new(64.0, 10);

        let node_idx = octree.add_node(OctreeNode::empty());
        assert_eq!(node_idx, 1); // Root is 0

        let brick_idx = octree.add_brick(VoxelBrick::EMPTY);
        assert_eq!(brick_idx, 0);
    }

    #[test]
    fn test_memory_usage() {
        let mut octree = Octree::new(64.0, 10);
        let initial = octree.memory_usage();

        octree.add_brick(VoxelBrick::EMPTY);
        assert_eq!(octree.memory_usage(), initial + 32); // VoxelBrick is 32 bytes
    }
}
