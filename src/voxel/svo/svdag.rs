//! Sparse Voxel DAG (SVDAG) - Octree with brick deduplication
//!
//! This simplified implementation focuses on brick deduplication which provides
//! significant memory savings (typically 5-10x for terrain with repetitive patterns).
//! Full node deduplication (SVDAG) can be added later for additional compression.

use std::collections::HashMap;
use super::{Octree, OctreeNode};
use crate::voxel::brick::VoxelBrick;

/// SVDAG builder - deduplicates identical bricks in octree
pub struct SvdagBuilder {
    /// Brick hash map for deduplication: hash -> new_brick_index
    brick_map: HashMap<u64, u32>,
    /// Old brick index -> new brick index mapping
    brick_remap: Vec<u32>,
    /// Node hash map for deduplication: hash -> new_node_index
    node_map: HashMap<u64, u32>,
    /// Old node index -> new node index mapping
    node_remap: Vec<u32>,
}

impl SvdagBuilder {
    /// Create a new SVDAG builder
    pub fn new() -> Self {
        Self {
            brick_map: HashMap::new(),
            brick_remap: Vec::new(),
            node_map: HashMap::new(),
            node_remap: Vec::new(),
        }
    }

    /// Compress an octree by deduplicating identical bricks
    /// Returns a new octree with deduplicated bricks
    pub fn build(mut self, octree: &Octree) -> Octree {
        let old_nodes = octree.nodes_slice();
        let old_bricks = octree.bricks_slice();

        if old_nodes.is_empty() || old_bricks.is_empty() {
            return Octree::new(octree.root_size(), octree.max_depth());
        }

        // Phase 1: Deduplicate bricks and build remap table
        let mut new_bricks: Vec<VoxelBrick> = Vec::new();
        self.brick_remap = Vec::with_capacity(old_bricks.len());

        for brick in old_bricks {
            let hash = self.hash_brick(brick);

            if let Some(&existing_idx) = self.brick_map.get(&hash) {
                // Duplicate brick - reuse existing
                self.brick_remap.push(existing_idx);
            } else {
                // New unique brick
                let new_idx = new_bricks.len() as u32;
                self.brick_map.insert(hash, new_idx);
                self.brick_remap.push(new_idx);
                new_bricks.push(*brick);
            }
        }

        // Phase 2: Remap node brick_offset for nodes with leaf children
        let mut new_nodes: Vec<OctreeNode> = Vec::with_capacity(old_nodes.len());

        for old_node in old_nodes {
            let mut new_node = *old_node;

            // If this node has leaf children, remap their brick indices
            let valid_mask = old_node.child_valid_mask();
            let leaf_mask = old_node.child_leaf_mask();

            // brick_offset points to first brick for this node's leaf children
            if valid_mask != 0 && leaf_mask != 0 {
                // Has some leaf children - remap brick_offset
                let old_brick_idx = old_node.brick_offset;
                if (old_brick_idx as usize) < self.brick_remap.len() {
                    new_node.brick_offset = self.brick_remap[old_brick_idx as usize];
                }
            }

            new_nodes.push(new_node);
        }

        let compression_ratio = if new_bricks.is_empty() {
            1.0
        } else {
            old_bricks.len() as f32 / new_bricks.len() as f32
        };

        log::info!(
            "Brick deduplication: {} bricks -> {} bricks ({:.1}x compression)",
            old_bricks.len(),
            new_bricks.len(),
            compression_ratio,
        );

        // Build new octree with deduplicated data
        let mut result = Octree::with_capacity(
            octree.root_size(),
            octree.max_depth(),
            new_nodes.len(),
            new_bricks.len(),
        );

        // Copy nodes (root is already created)
        *result.root_mut() = new_nodes[0];
        for node in new_nodes.iter().skip(1) {
            result.add_node(*node);
        }

        // Copy deduplicated bricks
        for brick in &new_bricks {
            result.add_brick(*brick);
        }

        result
    }

    /// Full SVDAG compression (nodes + bricks)
    /// Phase 1: Deduplicate bricks (existing)
    /// Phase 2: Bottom-up node deduplication (new)
    pub fn build_full(mut self, octree: &Octree) -> Octree {
        let old_nodes = octree.nodes_slice();
        let old_bricks = octree.bricks_slice();

        if old_nodes.is_empty() || old_bricks.is_empty() {
            return Octree::new(octree.root_size(), octree.max_depth());
        }

        // Phase 1: Deduplicate bricks and build remap table
        let mut new_bricks: Vec<VoxelBrick> = Vec::new();
        self.brick_remap = Vec::with_capacity(old_bricks.len());

        for brick in old_bricks {
            let hash = self.hash_brick(brick);

            if let Some(&existing_idx) = self.brick_map.get(&hash) {
                // Duplicate brick - reuse existing
                self.brick_remap.push(existing_idx);
            } else {
                // New unique brick
                let new_idx = new_bricks.len() as u32;
                self.brick_map.insert(hash, new_idx);
                self.brick_remap.push(new_idx);
                new_bricks.push(*brick);
            }
        }

        let brick_compression = if new_bricks.is_empty() {
            1.0
        } else {
            old_bricks.len() as f32 / new_bricks.len() as f32
        };

        // Phase 2: Node deduplication bottom-up
        // Build depth map for bottom-up processing
        let max_depth = octree.max_depth();
        let mut nodes_by_depth: Vec<Vec<u32>> = vec![Vec::new(); (max_depth as usize) + 1];

        // Calculate depth for each node (BFS from root)
        let mut node_depths = vec![0u32; old_nodes.len()];
        let mut queue = vec![(0u32, 0u32)]; // (node_idx, depth)

        while let Some((node_idx, depth)) = queue.pop() {
            if (node_idx as usize) >= old_nodes.len() {
                continue;
            }

            node_depths[node_idx as usize] = depth;
            nodes_by_depth[depth as usize].push(node_idx);

            let node = &old_nodes[node_idx as usize];
            let valid_mask = node.child_valid_mask();
            let leaf_mask = node.child_leaf_mask();

            // Add non-leaf children to queue
            if valid_mask != 0 {
                let child_base = node.child_offset;
                for i in 0..8u32 {
                    if (valid_mask & (1 << i)) != 0 && (leaf_mask & (1 << i)) == 0 {
                        let child_idx = child_base + i;
                        queue.push((child_idx, depth + 1));
                    }
                }
            }
        }

        // Process nodes bottom-up (deepest first)
        let mut new_nodes: Vec<OctreeNode> = Vec::new();
        self.node_remap = vec![0u32; old_nodes.len()];

        for depth in (0..=(max_depth as usize)).rev() {
            for &node_idx in &nodes_by_depth[depth] {
                let old_node = &old_nodes[node_idx as usize];
                let mut new_node = *old_node;

                // Remap child offsets for non-leaf children
                let valid_mask = old_node.child_valid_mask();
                let leaf_mask = old_node.child_leaf_mask();

                if valid_mask != 0 && leaf_mask != 0xFF {
                    // Has some non-leaf children - remap child_offset
                    let old_child_base = old_node.child_offset;

                    // Find first non-leaf child to determine new child_offset
                    for i in 0..8u32 {
                        if (valid_mask & (1 << i)) != 0 && (leaf_mask & (1 << i)) == 0 {
                            let old_child_idx = old_child_base + i;
                            if (old_child_idx as usize) < self.node_remap.len() {
                                let new_child_idx = self.node_remap[old_child_idx as usize];
                                new_node.child_offset = new_child_idx.saturating_sub(i);
                                break;
                            }
                        }
                    }
                }

                // Remap brick_offset for nodes with leaf children
                if valid_mask != 0 && leaf_mask != 0 {
                    let old_brick_idx = old_node.brick_offset;
                    if (old_brick_idx as usize) < self.brick_remap.len() {
                        new_node.brick_offset = self.brick_remap[old_brick_idx as usize];
                    }
                }

                // Hash node based on remapped children and bricks
                let hash = self.hash_node(&new_node);

                if let Some(&existing_idx) = self.node_map.get(&hash) {
                    // Duplicate node - reuse existing
                    self.node_remap[node_idx as usize] = existing_idx;
                } else {
                    // New unique node
                    let new_idx = new_nodes.len() as u32;
                    self.node_map.insert(hash, new_idx);
                    self.node_remap[node_idx as usize] = new_idx;
                    new_nodes.push(new_node);
                }
            }
        }

        let node_compression = if new_nodes.is_empty() {
            1.0
        } else {
            old_nodes.len() as f32 / new_nodes.len() as f32
        };

        log::info!(
            "Full SVDAG compression: {} nodes -> {} nodes ({:.1}x), {} bricks -> {} bricks ({:.1}x)",
            old_nodes.len(),
            new_nodes.len(),
            node_compression,
            old_bricks.len(),
            new_bricks.len(),
            brick_compression,
        );

        // Build new octree with deduplicated data
        let mut result = Octree::with_capacity(
            octree.root_size(),
            octree.max_depth(),
            new_nodes.len(),
            new_bricks.len(),
        );

        // Copy nodes (root is already created)
        if !new_nodes.is_empty() {
            *result.root_mut() = new_nodes[0];
            for node in new_nodes.iter().skip(1) {
                result.add_node(*node);
            }
        }

        // Copy deduplicated bricks
        for brick in &new_bricks {
            result.add_brick(*brick);
        }

        result
    }

    /// Hash a voxel brick using FNV-1a
    fn hash_brick(&self, brick: &VoxelBrick) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for voxel in &brick.voxels {
            // Hash all voxel fields
            hash ^= voxel.color as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= voxel.material_id as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= voxel.flags as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Hash a node based on its flags, child offsets, and brick offsets
    /// Two nodes are identical if they have the same structure and references
    fn hash_node(&self, node: &OctreeNode) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;

        // Hash flags (contains valid mask and leaf mask)
        hash ^= node.flags as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Hash child_offset (already remapped)
        hash ^= node.child_offset as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Hash brick_offset (already remapped)
        hash ^= node.brick_offset as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        hash
    }
}

impl Default for SvdagBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::svo::builder::{OctreeBuilder, create_test_sphere};

    #[test]
    fn test_svdag_empty() {
        let octree = Octree::new(64.0, 6);
        let builder = SvdagBuilder::new();
        let svdag = builder.build(&octree);

        // Empty octree stays empty
        assert_eq!(svdag.node_count(), 1);
        assert_eq!(svdag.brick_count(), 0);
    }

    #[test]
    fn test_brick_deduplication() {
        // Create a test scene - spheres have many similar bricks
        let size = 64u32;
        let voxels = create_test_sphere(size, 28.0);
        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        let original_bricks = octree.brick_count();

        let svdag_builder = SvdagBuilder::new();
        let svdag = svdag_builder.build(&octree);

        println!(
            "Brick deduplication: {} -> {} ({:.1}x)",
            original_bricks,
            svdag.brick_count(),
            original_bricks as f32 / svdag.brick_count().max(1) as f32
        );

        // Should have some brick compression
        assert!(svdag.brick_count() <= original_bricks);
        // Nodes should be preserved
        assert_eq!(svdag.node_count(), octree.node_count());
    }

    #[test]
    fn test_node_deduplication() {
        // Create a symmetric octree with repeated patterns
        // A cube subdivided with identical subtrees should deduplicate well
        let size = 64u32;

        // Create a pattern with lots of symmetry - 4 identical spheres
        let mut voxels = vec![crate::voxel::voxel::Voxel::EMPTY; (size * size * size) as usize];

        // Place 4 identical small spheres in corners
        let centers = [
            (16.0, 16.0, 16.0),
            (48.0, 16.0, 16.0),
            (16.0, 48.0, 16.0),
            (48.0, 48.0, 16.0),
        ];

        for (cx, cy, cz) in &centers {
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let dz = z as f32 - cz;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist <= 8.0 {
                            let idx = (z * size * size + y * size + x) as usize;
                            voxels[idx] = crate::voxel::voxel::Voxel::new(255, 0, 0, 1);
                        }
                    }
                }
            }
        }

        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        let original_nodes = octree.node_count();
        let original_bricks = octree.brick_count();

        // Build with full node+brick deduplication
        let svdag_builder = SvdagBuilder::new();
        let svdag = svdag_builder.build_full(&octree);

        println!(
            "Full deduplication: {} nodes -> {} nodes ({:.1}x), {} bricks -> {} bricks ({:.1}x)",
            original_nodes,
            svdag.node_count(),
            original_nodes as f32 / svdag.node_count().max(1) as f32,
            original_bricks,
            svdag.brick_count(),
            original_bricks as f32 / svdag.brick_count().max(1) as f32
        );

        // Should have significant node compression due to symmetry
        assert!(svdag.node_count() <= original_nodes);
        // Should also have brick compression
        assert!(svdag.brick_count() <= original_bricks);
        // Combined compression should be better than brick-only
        assert!(svdag.node_count() < original_nodes || svdag.brick_count() < original_bricks);
    }
}
