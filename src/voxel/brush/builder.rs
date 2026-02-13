//! Octree builder from brush strokes

use glam::Vec3;
use crate::math::Aabb;
use crate::voxel::svo::octree::Octree;
use crate::voxel::svo::node::OctreeNode;
use crate::voxel::brick::VoxelBrick;
use crate::voxel::voxel::{Voxel, rgb565_to_rgb, rgb_to_565};
use super::{BrushSession, BrushStroke, BlendMode};

/// Builder for constructing octrees from brush strokes
pub struct BrushOctreeBuilder {
    /// Root size in world units
    root_size: f32,
    /// Maximum tree depth
    max_depth: u8,
}

impl BrushOctreeBuilder {
    /// Create a new builder
    pub fn new(root_size: f32, max_depth: u8) -> Self {
        Self {
            root_size,
            max_depth,
        }
    }

    /// Build octree from brush session
    pub fn build(&self, session: &BrushSession) -> Octree {
        let mut octree = Octree::with_capacity(
            self.root_size,
            self.max_depth,
            1024,  // Estimated node count
            512,   // Estimated brick count
        );
        // BrushOctreeBuilder pre-allocates all 8 children densely
        octree.set_dense_children(true);

        // Build from root
        let root_center = Vec3::ZERO;
        let stroke_refs: Vec<&BrushStroke> = session.strokes().iter().collect();
        self.build_recursive(
            &mut octree,
            0, // Root node index
            root_center,
            self.root_size,
            0, // Current depth
            &stroke_refs,
        );

        octree
    }

    /// Recursive octree construction - returns (lod_color, lod_material, is_empty)
    fn build_recursive(
        &self,
        octree: &mut Octree,
        node_index: u32,
        center: Vec3,
        size: f32,
        depth: u8,
        strokes: &[&BrushStroke],
    ) -> (u16, u8, bool) {
        let half_size = size * 0.5;
        let node_bounds = Aabb::from_center_half_extent(center, Vec3::splat(half_size));

        // Filter strokes that intersect this node
        let relevant_strokes: Vec<&BrushStroke> = strokes
            .iter()
            .filter(|stroke| stroke.intersects_aabb(&node_bounds))
            .copied()
            .collect();

        // If no strokes intersect, node stays empty
        if relevant_strokes.is_empty() {
            return (0, 0, true);
        }

        // At maximum depth, create leaf brick
        if depth >= self.max_depth {
            return self.create_leaf_brick(octree, node_index, center, size, &relevant_strokes);
        }

        // Check for early termination: if all strokes want to paint at this level or coarser,
        // and the node is fully inside a stroke, we can fill it as solid without subdividing
        if let Some((color, material)) = self.check_early_termination(&relevant_strokes, &node_bounds, depth) {
            if color == 0 && material == 0 {
                // Carving - node becomes empty
                return (0, 0, true);
            }
            // Create a terminal leaf with uniform voxels
            let voxel = Voxel::from_rgb565(color, material);
            let brick = VoxelBrick::new([voxel; 8]);
            let brick_index = octree.add_brick(brick);
            let node = octree.node_mut(node_index);
            node.lod_color = color;
            node.lod_material = material as u16;
            node.set_lod_level(depth);
            node.brick_offset = brick_index;
            node.set_child_valid_mask(0); // Terminal leaf
            node.set_child_leaf_mask(0);
            return (color, material, false);
        }

        // Otherwise, subdivide and recurse
        let child_size = size * 0.5;
        let quarter_size = size * 0.25;

        let mut child_valid_mask = 0u8;
        let child_leaf_mask = 0u8;
        let first_child_index = octree.node_count() as u32;
        let mut child_colors: [(u16, u8); 8] = [(0, 0); 8];

        // Pre-allocate 8 child nodes
        for _ in 0..8 {
            octree.add_node(OctreeNode::empty());
        }

        // Process each octant
        for child_idx in 0..8u8 {
            let child_center = center + child_offset(child_idx) * quarter_size;

            // Recursively build child
            let child_node_index = first_child_index + child_idx as u32;
            let (child_lod_color, child_lod_mat, child_empty) = self.build_recursive(
                octree,
                child_node_index,
                child_center,
                child_size,
                depth + 1,
                &relevant_strokes,
            );

            child_colors[child_idx as usize] = (child_lod_color, child_lod_mat);

            // Check if child is non-empty
            if !child_empty {
                child_valid_mask |= 1 << child_idx;
                // Don't set child_leaf_mask here. sample_voxel uses packed indexing
                // for leaf bricks (parent.brick_offset + count), but we store bricks
                // on child nodes directly. Leaving child_leaf_mask = 0 forces
                // sample_voxel to descend into child nodes for correct traversal.
            }
        }

        // Calculate LOD color as average of children
        let (lod_color, lod_material) = Self::average_child_colors(&child_colors, child_valid_mask);

        // Update parent node with child information
        let node = octree.node_mut(node_index);
        node.set_child_valid_mask(child_valid_mask);
        node.set_child_leaf_mask(child_leaf_mask);
        node.child_offset = if child_valid_mask != 0 {
            first_child_index
        } else {
            0
        };
        node.lod_color = lod_color;
        node.lod_material = lod_material as u16;
        node.set_lod_level(depth);

        (lod_color, lod_material, child_valid_mask == 0)
    }

    /// Check if we can terminate early at this level
    /// Returns Some((color, material)) if the node can be filled as solid
    fn check_early_termination(
        &self,
        strokes: &[&BrushStroke],
        node_bounds: &Aabb,
        depth: u8,
    ) -> Option<(u16, u8)> {
        // Find strokes that want to paint at this level or coarser
        let coarse_strokes: Vec<_> = strokes
            .iter()
            .filter(|s| s.target_level <= depth)
            .collect();

        if coarse_strokes.is_empty() {
            return None;
        }

        // Check if any coarse stroke fully contains this node
        // We test all 8 corners of the node bounds
        let corners = [
            Vec3::new(node_bounds.min.x, node_bounds.min.y, node_bounds.min.z),
            Vec3::new(node_bounds.max.x, node_bounds.min.y, node_bounds.min.z),
            Vec3::new(node_bounds.min.x, node_bounds.max.y, node_bounds.min.z),
            Vec3::new(node_bounds.max.x, node_bounds.max.y, node_bounds.min.z),
            Vec3::new(node_bounds.min.x, node_bounds.min.y, node_bounds.max.z),
            Vec3::new(node_bounds.max.x, node_bounds.min.y, node_bounds.max.z),
            Vec3::new(node_bounds.min.x, node_bounds.max.y, node_bounds.max.z),
            Vec3::new(node_bounds.max.x, node_bounds.max.y, node_bounds.max.z),
        ];

        // Check if all corners are inside the same stroke with the same result
        for stroke in coarse_strokes.iter() {
            let all_inside = corners.iter().all(|&c| stroke.contains_point(c));
            if all_inside {
                // This stroke fully covers the node
                match stroke.blend_mode {
                    BlendMode::Replace | BlendMode::Add => {
                        return Some((stroke.voxel.color, stroke.voxel.material_id));
                    }
                    BlendMode::Subtract => {
                        // Carving - node becomes empty
                        return Some((0, 0));
                    }
                }
            }
        }

        // Check if there are any strokes that want finer detail
        let wants_finer = strokes.iter().any(|s| s.target_level > depth);
        if !wants_finer {
            // All strokes are at this level or coarser, but none fully contain the node
            // We still need to subdivide to get the boundaries right
            return None;
        }

        None
    }

    /// Average child colors for LOD propagation
    fn average_child_colors(colors: &[(u16, u8); 8], valid_mask: u8) -> (u16, u8) {
        let mut r_sum: u32 = 0;
        let mut g_sum: u32 = 0;
        let mut b_sum: u32 = 0;
        let mut count: u32 = 0;
        let mut mat_counts = [0u8; 256];

        for i in 0..8 {
            if valid_mask & (1 << i) != 0 {
                let (color, mat) = colors[i];
                let (r, g, b) = rgb565_to_rgb(color);
                r_sum += r as u32;
                g_sum += g as u32;
                b_sum += b as u32;
                count += 1;
                mat_counts[mat as usize] = mat_counts[mat as usize].saturating_add(1);
            }
        }

        if count == 0 {
            return (0, 0);
        }

        let avg_r = (r_sum / count) as u8;
        let avg_g = (g_sum / count) as u8;
        let avg_b = (b_sum / count) as u8;
        let avg_color = rgb_to_565(avg_r, avg_g, avg_b);

        // Most common material
        let avg_mat = mat_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        (avg_color, avg_mat)
    }

    /// Create a leaf brick by evaluating strokes at voxel centers
    /// Returns (lod_color, lod_material, is_empty)
    fn create_leaf_brick(
        &self,
        octree: &mut Octree,
        node_index: u32,
        center: Vec3,
        size: f32,
        strokes: &[&BrushStroke],
    ) -> (u16, u8, bool) {
        let quarter_size = size * 0.25;
        let mut voxels = [Voxel::EMPTY; 8];

        // Evaluate each voxel in the 2x2x2 brick
        for voxel_idx in 0..8 {
            let offset = brick_voxel_offset(voxel_idx);
            let voxel_center = center + offset * quarter_size;

            // Evaluate all strokes at this point
            voxels[voxel_idx] = self.evaluate_strokes(voxel_center, strokes);
        }

        // Only create brick if it contains any non-empty voxels
        let brick = VoxelBrick::new(voxels);
        if !brick.is_empty() {
            let brick_index = octree.add_brick(brick);

            // Get LOD values before updating node
            let lod_color = brick.average_color();
            let lod_material = brick.average_material();

            // Update node as terminal leaf (single brick, no children)
            // sample_voxel handles this via is_terminal_leaf() check
            let node = octree.node_mut(node_index);
            node.brick_offset = brick_index;
            node.set_child_valid_mask(0); // No children - terminal leaf
            node.set_child_leaf_mask(0);
            node.lod_color = lod_color;
            node.lod_material = lod_material as u16;

            (lod_color, lod_material, false)
        } else {
            (0, 0, true)
        }
    }

    /// Evaluate all strokes at a point and return the resulting voxel
    fn evaluate_strokes(&self, point: Vec3, strokes: &[&BrushStroke]) -> Voxel {
        let mut result = Voxel::EMPTY;

        for stroke in strokes {
            if !stroke.contains_point(point) {
                continue;
            }

            // Apply blend mode
            match stroke.blend_mode {
                BlendMode::Replace => {
                    result = stroke.voxel;
                }
                BlendMode::Add => {
                    // Only fill if currently empty
                    if result.is_empty() {
                        result = stroke.voxel;
                    }
                }
                BlendMode::Subtract => {
                    // Carve out: make empty
                    result = Voxel::EMPTY;
                }
            }
        }

        result
    }
}

/// Get child octant offset (-1 or +1 for each axis)
fn child_offset(child_idx: u8) -> Vec3 {
    debug_assert!(child_idx < 8);
    Vec3::new(
        if child_idx & 1 != 0 { 1.0 } else { -1.0 },
        if child_idx & 2 != 0 { 1.0 } else { -1.0 },
        if child_idx & 4 != 0 { 1.0 } else { -1.0 },
    )
}

/// Get brick voxel offset within octant (-0.5 or +0.5 for each axis)
fn brick_voxel_offset(voxel_idx: usize) -> Vec3 {
    debug_assert!(voxel_idx < 8);
    Vec3::new(
        if voxel_idx & 1 != 0 { 0.5 } else { -0.5 },
        if voxel_idx & 2 != 0 { 0.5 } else { -0.5 },
        if voxel_idx & 4 != 0 { 0.5 } else { -0.5 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_session() {
        let builder = BrushOctreeBuilder::new(64.0, 5);
        let session = BrushSession::new();
        let octree = builder.build(&session);

        assert!(octree.is_empty());
        assert_eq!(octree.node_count(), 1); // Just root
        assert_eq!(octree.brick_count(), 0);
    }

    #[test]
    fn test_single_sphere() {
        let builder = BrushOctreeBuilder::new(64.0, 5);
        let mut session = BrushSession::new();

        let voxel = Voxel::new(255, 0, 0, 1);
        session.sphere(Vec3::ZERO, 2.0, voxel, 5);

        let octree = builder.build(&session);

        assert!(!octree.is_empty());
        assert!(octree.node_count() > 1);
        assert!(octree.brick_count() > 0);
    }

    #[test]
    fn test_blend_modes() {
        let builder = BrushOctreeBuilder::new(64.0, 5);
        let mut session = BrushSession::new();

        let red = Voxel::new(255, 0, 0, 1);
        let green = Voxel::new(0, 255, 0, 1);

        // Replace: green overwrites red
        session
            .sphere(Vec3::ZERO, 2.0, red, 5)
            .set_blend(BlendMode::Replace)
            .sphere(Vec3::ZERO, 1.5, green, 5);

        let octree = builder.build(&session);
        assert!(octree.brick_count() > 0);
    }

    #[test]
    fn test_child_offset() {
        // Child 0: (-1, -1, -1)
        assert_eq!(child_offset(0), Vec3::new(-1.0, -1.0, -1.0));
        // Child 7: (+1, +1, +1)
        assert_eq!(child_offset(7), Vec3::new(1.0, 1.0, 1.0));
        // Child 1: (+1, -1, -1)
        assert_eq!(child_offset(1), Vec3::new(1.0, -1.0, -1.0));
    }

    #[test]
    fn test_brick_voxel_offset() {
        // Voxel 0: (-0.5, -0.5, -0.5)
        assert_eq!(brick_voxel_offset(0), Vec3::new(-0.5, -0.5, -0.5));
        // Voxel 7: (+0.5, +0.5, +0.5)
        assert_eq!(brick_voxel_offset(7), Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_subtract_blend() {
        let builder = BrushOctreeBuilder::new(64.0, 5);
        let mut session = BrushSession::new();

        let red = Voxel::new(255, 0, 0, 1);

        // Large sphere, then carve out center
        session
            .sphere(Vec3::ZERO, 3.0, red, 5)
            .set_blend(BlendMode::Subtract)
            .sphere(Vec3::ZERO, 1.0, red, 5);

        let octree = builder.build(&session);
        assert!(octree.brick_count() > 0);
    }

    #[test]
    fn test_multiple_shapes() {
        let builder = BrushOctreeBuilder::new(64.0, 6);
        let mut session = BrushSession::new();

        let red = Voxel::new(255, 0, 0, 1);
        let green = Voxel::new(0, 255, 0, 1);
        let blue = Voxel::new(0, 0, 255, 1);

        session
            .sphere(Vec3::new(-5.0, 0.0, 0.0), 2.0, red, 6)
            .sphere(Vec3::new(0.0, 0.0, 0.0), 2.0, green, 6)
            .sphere(Vec3::new(5.0, 0.0, 0.0), 2.0, blue, 6);

        let octree = builder.build(&session);
        assert!(octree.node_count() > 1);
        assert!(octree.brick_count() >= 3);
    }

    #[test]
    fn test_multi_level_early_termination() {
        // Test that coarse strokes (low target_level) terminate early
        let builder = BrushOctreeBuilder::new(64.0, 8);
        let mut session = BrushSession::new();

        let voxel = Voxel::new(255, 0, 0, 1);

        // Large sphere at coarse level (should terminate early, fewer nodes)
        session.sphere(Vec3::ZERO, 20.0, voxel, 3);
        let octree_coarse = builder.build(&session);

        // Same sphere at fine level (should subdivide more, more nodes)
        let mut session_fine = BrushSession::new();
        session_fine.sphere(Vec3::ZERO, 20.0, voxel, 8);
        let octree_fine = builder.build(&session_fine);

        // Coarse should have fewer nodes due to early termination
        assert!(octree_coarse.node_count() <= octree_fine.node_count(),
            "Coarse ({}) should have <= nodes than fine ({})",
            octree_coarse.node_count(), octree_fine.node_count());
    }

    #[test]
    fn test_mixed_level_strokes() {
        // Coarse trunk + fine detail leaves
        let builder = BrushOctreeBuilder::new(64.0, 7);
        let mut session = BrushSession::new();

        let bark = Voxel::new(139, 90, 43, 2);
        let leaf = Voxel::new(34, 139, 34, 3);

        // Coarse trunk at level 3
        session.capsule(
            Vec3::new(0.0, -10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            3.0,
            bark,
            3
        );

        // Fine leaves at level 6
        session.sphere(Vec3::new(0.0, 12.0, 0.0), 8.0, leaf, 6);

        let octree = builder.build(&session);
        assert!(octree.node_count() > 1);
        assert!(octree.brick_count() > 0);
    }
}
