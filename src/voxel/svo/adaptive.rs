//! Adaptive octree construction with hierarchical subdivision
//!
//! Builds sparse octrees on-demand from a voxel evaluator function.
//! No dense voxel arrays needed — evaluates voxels at leaf level only.
//! Empty regions are detected by sampling corners, avoiding unnecessary subdivision.

use super::{Octree, OctreeNode};
use super::classifier::{RegionClassifier, RegionHint};
use crate::voxel::brick::VoxelBrick;
use crate::voxel::voxel::{rgb_to_565, rgb565_to_rgb, Voxel};
use crate::math::Aabb;
use glam::Vec3;

/// Builder for constructing octrees adaptively without dense allocation
pub struct AdaptiveOctreeBuilder {
    /// Voxels per side (e.g., 512)
    size: u32,
    /// Maximum octree depth
    max_depth: u8,
}

impl AdaptiveOctreeBuilder {
    /// Create a new adaptive builder for given grid size
    /// Size must be power of 2
    pub fn new(size: u32) -> Self {
        assert!(size.is_power_of_two(), "Size must be power of 2");
        let max_depth = (size as f32).log2() as u8;
        Self {
            size,
            max_depth,
        }
    }

    /// Build octree from a simple voxel evaluator function.
    ///
    /// No classifier abstraction needed — just pass a function that returns
    /// the voxel at any world position. Empty regions are detected by sampling
    /// corners of each octant. The result is automatically pruned.
    pub fn build_simple<F>(&self, evaluator: &F, origin: Vec3, chunk_size: f32) -> Octree
    where
        F: Fn(Vec3) -> Voxel,
    {
        let voxel_size = chunk_size / self.size as f32;

        // Don't early-out at chunk level — corner sampling is too coarse
        // for heightfield terrain that barely enters a chunk. Let recursive
        // subdivision handle empty detection at finer granularity.
        let mut octree = Octree::with_capacity(chunk_size, self.max_depth, 1024, 512);
        self.build_node_simple(&mut octree, evaluator, origin, self.size, voxel_size, 0);

        // Remove any pre-allocated but empty subtrees
        octree.prune()
    }

    /// Recursively build octree node from evaluator
    fn build_node_simple<F>(
        &self,
        octree: &mut Octree,
        evaluator: &F,
        origin: Vec3,
        size_voxels: u32,
        voxel_size: f32,
        node_index: u32,
    ) -> (u16, u8, bool)
    where
        F: Fn(Vec3) -> Voxel,
    {
        // Base case: 2x2x2 brick level
        if size_voxels == 2 {
            return self.build_leaf_simple(octree, evaluator, origin, voxel_size);
        }

        // Quick empty check via corner sampling — but only for very small regions
        // where 9-point sampling is dense enough to catch thin surface shells.
        // For a shell of thickness T and voxel size V, need region ≤ T/V voxels.
        // At 4 voxels (~0.03m), this reliably catches shells ≥ 0.04m thick.
        if size_voxels <= 4 {
            let region_size = size_voxels as f32 * voxel_size;
            if self.is_region_likely_empty(evaluator, origin, region_size) {
                return (0, 0, true);
            }
        }

        // Subdivide into 8 children
        let half = size_voxels / 2;
        let half_world = half as f32 * voxel_size;

        let mut child_valid_mask: u8 = 0;
        let mut child_leaf_mask: u8 = 0;
        let mut child_indices: [u32; 8] = [0; 8];
        let mut child_colors: [(u16, u8); 8] = [(0, 0); 8];

        // First pass: check children and allocate nodes
        for child_idx in 0..8u8 {
            let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
            let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
            let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
            let child_origin = Vec3::new(cx, cy, cz);

            // Sample corners to skip empty regions (only at fine granularity)
            if half <= 4 {
                let child_size = half as f32 * voxel_size;
                if self.is_region_likely_empty(evaluator, child_origin, child_size) {
                    continue;
                }
            }

            child_valid_mask |= 1 << child_idx;

            if half == 2 {
                // Next level is leaf (brick) level
                child_leaf_mask |= 1 << child_idx;
            } else {
                // Pre-allocate internal child node
                let child_node_idx = octree.add_node(OctreeNode::empty());
                child_indices[child_idx as usize] = child_node_idx;
            }
        }

        if child_valid_mask == 0 {
            return (0, 0, true);
        }

        // Second pass: build children
        let (first_child_offset, first_brick_offset) = if child_leaf_mask == child_valid_mask {
            // All children are leaves
            let first_brick = octree.brick_count() as u32;
            let mut actual_valid: u8 = 0;
            let mut actual_leaf: u8 = 0;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) != 0 {
                    let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
                    let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
                    let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
                    let child_origin = Vec3::new(cx, cy, cz);

                    let (color, mat, is_empty) = self.build_leaf_simple(octree, evaluator, child_origin, voxel_size);
                    if !is_empty {
                        actual_valid |= 1 << child_idx;
                        actual_leaf |= 1 << child_idx;
                        child_colors[child_idx as usize] = (color, mat);
                    }
                }
            }

            child_valid_mask = actual_valid;
            child_leaf_mask = actual_leaf;
            (0, first_brick)
        } else {
            // Mixed: some internal nodes, some leaves
            let mut first_child = u32::MAX;
            let first_brick = octree.brick_count() as u32;
            let mut actual_leaf: u8 = 0;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) == 0 {
                    continue;
                }

                let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
                let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
                let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
                let child_origin = Vec3::new(cx, cy, cz);

                if child_leaf_mask & (1 << child_idx) != 0 {
                    // Leaf child
                    let (color, mat, is_empty) = self.build_leaf_simple(octree, evaluator, child_origin, voxel_size);
                    if !is_empty {
                        actual_leaf |= 1 << child_idx;
                        child_colors[child_idx as usize] = (color, mat);
                    } else {
                        child_valid_mask &= !(1 << child_idx);
                    }
                } else {
                    // Internal child - recurse
                    let child_node_idx = child_indices[child_idx as usize];
                    if first_child == u32::MAX {
                        first_child = child_node_idx;
                    }
                    let (color, mat, _is_empty) = self.build_node_simple(
                        octree, evaluator, child_origin, half, voxel_size, child_node_idx,
                    );
                    child_colors[child_idx as usize] = (color, mat);
                }
            }

            child_leaf_mask = actual_leaf;
            (first_child, first_brick)
        };

        if child_valid_mask == 0 {
            return (0, 0, true);
        }

        let (lod_color, lod_material) = Self::average_child_colors(&child_colors, child_valid_mask);

        let node = octree.node_mut(node_index);
        node.set_child_valid_mask(child_valid_mask);
        node.set_child_leaf_mask(child_leaf_mask);
        node.child_offset = first_child_offset;
        node.brick_offset = first_brick_offset;
        node.lod_color = lod_color;
        node.lod_material = lod_material as u16;

        (lod_color, lod_material, false)
    }

    /// Build a 2x2x2 leaf brick from evaluator
    fn build_leaf_simple<F>(
        &self,
        octree: &mut Octree,
        evaluator: &F,
        origin: Vec3,
        voxel_size: f32,
    ) -> (u16, u8, bool)
    where
        F: Fn(Vec3) -> Voxel,
    {
        let mut brick = VoxelBrick::EMPTY;
        let mut all_empty = true;

        for dz in 0..2u32 {
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let pos = origin + Vec3::new(
                        dx as f32 * voxel_size + voxel_size * 0.5,
                        dy as f32 * voxel_size + voxel_size * 0.5,
                        dz as f32 * voxel_size + voxel_size * 0.5,
                    );
                    let voxel = evaluator(pos);
                    if !voxel.is_empty() {
                        all_empty = false;
                    }
                    brick.set(dx as u8, dy as u8, dz as u8, voxel);
                }
            }
        }

        if all_empty {
            return (0, 0, true);
        }

        let lod_color = brick.average_color();
        let lod_material = brick.average_material();
        octree.add_brick(brick);

        (lod_color, lod_material, false)
    }

    /// Check if a region is likely empty by sampling corners and center.
    /// Returns true if all 9 sample points are empty.
    fn is_region_likely_empty<F>(&self, evaluator: &F, origin: Vec3, size: f32) -> bool
    where
        F: Fn(Vec3) -> Voxel,
    {
        let half = size * 0.5;
        let samples = [
            origin,
            origin + Vec3::new(size, 0.0, 0.0),
            origin + Vec3::new(0.0, size, 0.0),
            origin + Vec3::new(size, size, 0.0),
            origin + Vec3::new(0.0, 0.0, size),
            origin + Vec3::new(size, 0.0, size),
            origin + Vec3::new(0.0, size, size),
            origin + Vec3::new(size, size, size),
            origin + Vec3::splat(half), // center
        ];

        samples.iter().all(|&pos| evaluator(pos).is_empty())
    }

    /// Classify a chunk using the RegionClassifier
    pub fn classify_chunk<C: RegionClassifier>(
        &self,
        classifier: &C,
        chunk_origin: Vec3,
        chunk_size: f32,
    ) -> RegionHint {
        let aabb = Aabb::new(chunk_origin, chunk_origin + Vec3::splat(chunk_size));
        classifier.classify_region(&aabb)
    }

    /// Build an octree adaptively from a RegionClassifier
    ///
    /// Uses hierarchical classification to skip uniform regions.
    pub fn build<C: RegionClassifier>(
        &self,
        classifier: &C,
        chunk_origin: Vec3,
        chunk_size: f32,
    ) -> Octree {
        let hint = self.classify_chunk(classifier, chunk_origin, chunk_size);

        match hint {
            RegionHint::Empty => {
                // Return empty octree
                Octree::new(chunk_size, self.max_depth)
            }
            RegionHint::Solid { material, color } => {
                // Return octree with single solid root
                self.build_solid_octree(classifier, chunk_origin, chunk_size, material, color)
            }
            RegionHint::Mixed | RegionHint::Unknown => {
                // Build adaptively with recursive subdivision
                self.build_mixed_octree(classifier, chunk_origin, chunk_size)
            }
        }
    }

    /// Build a solid octree (uniform material throughout)
    fn build_solid_octree<C: RegionClassifier>(
        &self,
        classifier: &C,
        chunk_origin: Vec3,
        chunk_size: f32,
        material: u8,
        color: u16,
    ) -> Octree {
        // Sample center voxel to verify material/color
        let center = chunk_origin + Vec3::splat(chunk_size * 0.5);
        let voxel = classifier.evaluate(center);

        let mut octree = Octree::with_capacity(chunk_size, self.max_depth, 1, 2);
        // Reserve brick index 0 as padding — brick_offset=0 is the "no brick"
        // sentinel in OctreeNode::is_empty(), so real bricks must start at index 1.
        octree.add_brick(VoxelBrick::EMPTY);

        if voxel.is_empty() {
            return octree;
        }

        // Use the provided material/color or fall back to evaluated voxel
        let final_color = if color != 0 { color } else { voxel.color };
        let final_material = if material != 0 { material } else { voxel.material_id };

        // Create a single brick with uniform voxels
        let uniform_voxel = Voxel::from_rgb565(final_color, final_material);
        let brick = VoxelBrick::new([uniform_voxel; 8]);
        let brick_idx = octree.add_brick(brick);

        // Terminal leaf: single brick represents entire solid region
        let root = octree.root_mut();
        root.set_child_valid_mask(0);
        root.set_child_leaf_mask(0);
        root.brick_offset = brick_idx;
        root.lod_color = final_color;
        root.lod_material = final_material as u16;

        octree
    }

    /// Build octree for a mixed region with adaptive subdivision
    fn build_mixed_octree<C: RegionClassifier>(
        &self,
        classifier: &C,
        chunk_origin: Vec3,
        chunk_size: f32,
    ) -> Octree {
        let voxel_size = chunk_size / self.size as f32;
        let mut octree = Octree::with_capacity(chunk_size, self.max_depth, 1024, 513);
        // Reserve brick index 0 as padding — brick_offset=0 is the "no brick"
        // sentinel in OctreeNode::is_empty(), so real bricks must start at index 1.
        octree.add_brick(VoxelBrick::EMPTY);

        // Build recursively starting from root
        self.build_node(
            &mut octree,
            classifier,
            chunk_origin,
            self.size,
            voxel_size,
            0,
        );

        octree
    }

    /// Classify a region using the RegionClassifier
    fn classify_region<C: RegionClassifier>(
        &self,
        classifier: &C,
        origin: Vec3,
        size_voxels: u32,
        voxel_size: f32,
    ) -> RegionHint {
        let region_size = size_voxels as f32 * voxel_size;
        let aabb = Aabb::new(origin, origin + Vec3::splat(region_size));
        classifier.classify_region(&aabb)
    }

    /// Recursively build octree node
    /// Returns (lod_color, lod_material, is_empty)
    fn build_node<C: RegionClassifier>(
        &self,
        octree: &mut Octree,
        classifier: &C,
        origin: Vec3,
        size_voxels: u32,
        voxel_size: f32,
        node_index: u32,
    ) -> (u16, u8, bool) {
        // Base case: 2x2x2 brick level
        if size_voxels == 2 {
            return self.build_leaf(octree, classifier, origin, voxel_size, node_index);
        }

        // Classify this region
        let hint = self.classify_region(classifier, origin, size_voxels, voxel_size);

        match hint {
            RegionHint::Empty => {
                // Leave node empty
                (0, 0, true)
            }
            RegionHint::Solid { material, color } => {
                // Create uniform leaf
                self.build_solid_node(octree, color, material, node_index)
            }
            RegionHint::Mixed | RegionHint::Unknown => {
                // Subdivide into 8 children
                self.build_mixed_node(octree, classifier, origin, size_voxels, voxel_size, node_index)
            }
        }
    }

    /// Build a solid (uniform) node as a terminal leaf
    fn build_solid_node(
        &self,
        octree: &mut Octree,
        color: u16,
        material: u8,
        node_index: u32,
    ) -> (u16, u8, bool) {
        if color == 0 && material == 0 {
            return (0, 0, true);
        }

        // Create a single brick with uniform voxels (terminal leaf)
        let voxel = Voxel::from_rgb565(color, material);
        let brick = VoxelBrick::new([voxel; 8]);
        let brick_idx = octree.add_brick(brick);

        // Terminal leaf: child_valid_mask=0, brick_offset!=0
        let node = octree.node_mut(node_index);
        node.set_child_valid_mask(0);
        node.set_child_leaf_mask(0);
        node.brick_offset = brick_idx;
        node.lod_color = color;
        node.lod_material = material as u16;

        (color, material, false)
    }

    /// Build a mixed node with 8 children
    fn build_mixed_node<C: RegionClassifier>(
        &self,
        octree: &mut Octree,
        classifier: &C,
        origin: Vec3,
        size_voxels: u32,
        voxel_size: f32,
        node_index: u32,
    ) -> (u16, u8, bool) {
        let half = size_voxels / 2;
        let half_world = half as f32 * voxel_size;

        let mut child_valid_mask: u8 = 0;
        let mut child_leaf_mask: u8 = 0;
        let mut child_indices: [u32; 8] = [0; 8];
        let mut child_colors: [(u16, u8); 8] = [(0, 0); 8];
        let mut all_empty = true;

        // First pass: classify children and allocate nodes
        for child_idx in 0..8u8 {
            let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
            let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
            let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
            let child_origin = Vec3::new(cx, cy, cz);

            // Quick classification
            let hint = self.classify_region(classifier, child_origin, half, voxel_size);

            match hint {
                RegionHint::Empty => {
                    // Skip empty children
                    continue;
                }
                RegionHint::Solid { .. } | RegionHint::Mixed | RegionHint::Unknown => {
                    all_empty = false;
                    child_valid_mask |= 1 << child_idx;

                    if half == 2 {
                        // Next level is leaf level
                        child_leaf_mask |= 1 << child_idx;
                    } else {
                        // Create child node
                        let child_node_idx = octree.add_node(OctreeNode::empty());
                        child_indices[child_idx as usize] = child_node_idx;
                    }
                }
            }
        }

        if all_empty {
            return (0, 0, true);
        }

        // Second pass: build children
        // Update valid/leaf masks based on actual evaluation (some regions
        // classified as Mixed may turn out to be entirely empty at leaf level)
        let (first_child_offset, first_brick_offset) = if child_leaf_mask == child_valid_mask {
            // All children are leaves - only need brick offset
            let first_brick = octree.brick_count() as u32;
            let mut actual_valid: u8 = 0;
            let mut actual_leaf: u8 = 0;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) != 0 {
                    let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
                    let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
                    let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
                    let child_origin = Vec3::new(cx, cy, cz);

                    let (color, mat, is_empty) = self.build_leaf(octree, classifier, child_origin, voxel_size, u32::MAX);
                    if !is_empty {
                        actual_valid |= 1 << child_idx;
                        actual_leaf |= 1 << child_idx;
                        child_colors[child_idx as usize] = (color, mat);
                    }
                }
            }

            child_valid_mask = actual_valid;
            child_leaf_mask = actual_leaf;

            (0, first_brick)
        } else {
            // Mixed: some internal nodes, some leaves
            let mut first_child = u32::MAX;
            let first_brick = octree.brick_count() as u32;
            // Track actual leaf children (bricks are added on-demand, so empty ones can be skipped)
            let mut actual_leaf: u8 = 0;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) == 0 {
                    continue;
                }

                let cx = origin.x + if child_idx & 1 != 0 { half_world } else { 0.0 };
                let cy = origin.y + if child_idx & 2 != 0 { half_world } else { 0.0 };
                let cz = origin.z + if child_idx & 4 != 0 { half_world } else { 0.0 };
                let child_origin = Vec3::new(cx, cy, cz);

                if child_leaf_mask & (1 << child_idx) != 0 {
                    // Leaf child - create brick (skip if empty)
                    let (color, mat, is_empty) = self.build_leaf(octree, classifier, child_origin, voxel_size, u32::MAX);
                    if !is_empty {
                        actual_leaf |= 1 << child_idx;
                        child_colors[child_idx as usize] = (color, mat);
                    } else {
                        // Remove empty leaf from valid_mask (no brick was allocated)
                        child_valid_mask &= !(1 << child_idx);
                    }
                } else {
                    // Internal child - recurse (keep in valid_mask since node was pre-allocated)
                    let child_node_idx = child_indices[child_idx as usize];
                    if first_child == u32::MAX {
                        first_child = child_node_idx;
                    }
                    let (color, mat, _is_empty) = self.build_node(
                        octree,
                        classifier,
                        child_origin,
                        half,
                        voxel_size,
                        child_node_idx,
                    );
                    child_colors[child_idx as usize] = (color, mat);
                }
            }

            child_leaf_mask = actual_leaf;

            (first_child, first_brick)
        };

        // If all children turned out empty after evaluation, this node is empty
        if child_valid_mask == 0 {
            return (0, 0, true);
        }

        // Calculate LOD color as average of children
        let (lod_color, lod_material) = Self::average_child_colors(&child_colors, child_valid_mask);

        // Update current node
        let node = octree.node_mut(node_index);
        node.set_child_valid_mask(child_valid_mask);
        node.set_child_leaf_mask(child_leaf_mask);
        node.child_offset = first_child_offset;
        node.brick_offset = first_brick_offset;
        node.lod_color = lod_color;
        node.lod_material = lod_material as u16;

        (lod_color, lod_material, false)
    }

    /// Build a 2x2x2 leaf brick
    ///
    /// Only adds the brick to the octree if it contains non-empty voxels.
    /// The caller must check the `is_empty` return value and exclude empty
    /// children from `valid_mask` to keep brick indexing consistent.
    fn build_leaf<C: RegionClassifier>(
        &self,
        octree: &mut Octree,
        classifier: &C,
        origin: Vec3,
        voxel_size: f32,
        _node_index: u32,
    ) -> (u16, u8, bool) {
        let mut brick = VoxelBrick::EMPTY;
        let mut all_empty = true;

        for dz in 0..2u32 {
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let pos = origin + Vec3::new(
                        dx as f32 * voxel_size + voxel_size * 0.5,
                        dy as f32 * voxel_size + voxel_size * 0.5,
                        dz as f32 * voxel_size + voxel_size * 0.5,
                    );
                    let voxel = classifier.evaluate(pos);
                    if !voxel.is_empty() {
                        all_empty = false;
                    }
                    brick.set(dx as u8, dy as u8, dz as u8, voxel);
                }
            }
        }

        if all_empty {
            return (0, 0, true);
        }

        let lod_color = brick.average_color();
        let lod_material = brick.average_material();
        octree.add_brick(brick);

        (lod_color, lod_material, false)
    }

    /// Average child colors for LOD
    fn average_child_colors(colors: &[(u16, u8); 8], valid_mask: u8) -> (u16, u8) {
        let mut r_sum: u32 = 0;
        let mut g_sum: u32 = 0;
        let mut b_sum: u32 = 0;
        let mut count: u32 = 0;
        let mut mat_counts = [0u8; 256];

        for i in 0..8 {
            if valid_mask & (1 << i) != 0 && colors[i].0 != 0 {
                let (r, g, b) = rgb565_to_rgb(colors[i].0);
                r_sum += r as u32;
                g_sum += g as u32;
                b_sum += b as u32;
                mat_counts[colors[i].1 as usize] = mat_counts[colors[i].1 as usize].saturating_add(1);
                count += 1;
            }
        }

        if count == 0 {
            return (0, 0);
        }

        let lod_color = rgb_to_565(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        );

        let lod_material = mat_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        (lod_color, lod_material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple flat terrain classifier for testing
    struct FlatTerrainClassifier {
        height: f32,
        material: u8,
        color: (u8, u8, u8),
    }

    impl FlatTerrainClassifier {
        fn new(height: f32) -> Self {
            Self {
                height,
                material: 1,
                color: (139, 90, 43), // Brown/dirt color
            }
        }
    }

    impl RegionClassifier for FlatTerrainClassifier {
        fn classify_region(&self, aabb: &Aabb) -> RegionHint {
            // Region entirely above terrain
            if aabb.min.y > self.height {
                return RegionHint::Empty;
            }
            // Region entirely below terrain (solid)
            if aabb.max.y < self.height - 3.0 {
                let color = rgb_to_565(self.color.0, self.color.1, self.color.2);
                return RegionHint::Solid { material: self.material, color };
            }
            // Region intersects terrain surface
            RegionHint::Mixed
        }

        fn evaluate(&self, pos: Vec3) -> Voxel {
            if pos.y < self.height {
                Voxel::new(self.color.0, self.color.1, self.color.2, self.material)
            } else {
                Voxel::EMPTY
            }
        }
    }

    /// Sinusoidal terrain classifier for testing
    struct HillyTerrainClassifier {
        base_height: f32,
        amplitude: f32,
        frequency: f32,
    }

    impl HillyTerrainClassifier {
        fn new(base_height: f32, amplitude: f32, frequency: f32) -> Self {
            Self {
                base_height,
                amplitude,
                frequency,
            }
        }

        fn height_at(&self, x: f32, z: f32) -> f32 {
            self.base_height
                + self.amplitude * (x * self.frequency).sin()
                + self.amplitude * 0.5 * (z * self.frequency * 1.3).cos()
        }
    }

    impl RegionClassifier for HillyTerrainClassifier {
        fn classify_region(&self, aabb: &Aabb) -> RegionHint {
            // Sample heights at corners and center to determine region type
            let sample_points = [
                (aabb.min.x, aabb.min.z),
                (aabb.max.x, aabb.min.z),
                (aabb.min.x, aabb.max.z),
                (aabb.max.x, aabb.max.z),
                (aabb.center().x, aabb.center().z),
            ];

            let mut min_height = f32::MAX;
            let mut max_height = f32::MIN;

            for (x, z) in sample_points {
                let h = self.height_at(x, z);
                min_height = min_height.min(h);
                max_height = max_height.max(h);
            }

            // Region entirely above terrain
            if aabb.min.y > max_height + 1.0 {
                return RegionHint::Empty;
            }

            // Region intersects or is below - need to subdivide
            RegionHint::Mixed
        }

        fn evaluate(&self, pos: Vec3) -> Voxel {
            let height = self.height_at(pos.x, pos.z);
            if pos.y < height {
                let depth = height - pos.y;
                if depth > 1.0 {
                    Voxel::new(128, 128, 128, 2) // Stone
                } else {
                    Voxel::new(139, 90, 43, 1) // Dirt
                }
            } else {
                Voxel::EMPTY
            }
        }
    }

    #[test]
    fn test_classify_empty_chunk() {
        let classifier = FlatTerrainClassifier::new(0.0);
        let builder = AdaptiveOctreeBuilder::new(8);

        // Chunk above terrain
        let hint = builder.classify_chunk(&classifier, Vec3::new(0.0, 10.0, 0.0), 4.0);
        assert_eq!(hint, RegionHint::Empty);
    }

    #[test]
    fn test_classify_solid_chunk() {
        let classifier = FlatTerrainClassifier::new(100.0);
        let builder = AdaptiveOctreeBuilder::new(8);

        // Chunk deep below terrain
        let hint = builder.classify_chunk(&classifier, Vec3::new(0.0, 0.0, 0.0), 4.0);
        match hint {
            RegionHint::Solid { .. } => {}
            _ => panic!("Expected Solid classification"),
        }
    }

    #[test]
    fn test_classify_mixed_chunk() {
        let classifier = FlatTerrainClassifier::new(2.0);
        let builder = AdaptiveOctreeBuilder::new(8);

        // Chunk intersecting terrain
        let hint = builder.classify_chunk(&classifier, Vec3::new(0.0, 0.0, 0.0), 4.0);
        assert_eq!(hint, RegionHint::Mixed);
    }

    #[test]
    fn test_build_empty_chunk() {
        let classifier = FlatTerrainClassifier::new(0.0);
        let builder = AdaptiveOctreeBuilder::new(8);

        let octree = builder.build(&classifier, Vec3::new(0.0, 10.0, 0.0), 4.0);
        assert!(octree.is_empty());
        assert_eq!(octree.brick_count(), 0);
    }

    #[test]
    fn test_build_surface_chunk() {
        let classifier = FlatTerrainClassifier::new(2.0);
        let builder = AdaptiveOctreeBuilder::new(16);

        let octree = builder.build(&classifier, Vec3::new(0.0, 0.0, 0.0), 4.0);
        assert!(!octree.is_empty());
        assert!(octree.brick_count() > 0);

        println!(
            "Surface chunk: {} nodes, {} bricks",
            octree.node_count(),
            octree.brick_count()
        );
    }

    #[test]
    fn test_build_hilly_terrain() {
        let classifier = HillyTerrainClassifier::new(2.0, 1.0, 0.5);
        let builder = AdaptiveOctreeBuilder::new(32);

        let octree = builder.build(&classifier, Vec3::new(0.0, 0.0, 0.0), 4.0);
        assert!(!octree.is_empty());

        println!(
            "Hilly terrain: {} nodes, {} bricks, {} bytes",
            octree.node_count(),
            octree.brick_count(),
            octree.memory_usage()
        );
    }

    #[test]
    fn test_adaptive_compression() {
        // Compare adaptive build with theoretical dense size
        let classifier = FlatTerrainClassifier::new(2.0);
        let size = 64u32;
        let builder = AdaptiveOctreeBuilder::new(size);

        let octree = builder.build(&classifier, Vec3::new(0.0, 0.0, 0.0), 4.0);

        let dense_size = size * size * size * 4; // 4 bytes per voxel
        let adaptive_size = octree.memory_usage();

        println!(
            "Compression: {} -> {} bytes ({:.1}x)",
            dense_size,
            adaptive_size,
            dense_size as f32 / adaptive_size as f32
        );

        // Adaptive should be smaller than dense for terrain with ~50% solid
        assert!(adaptive_size < dense_size as usize);
    }
}
