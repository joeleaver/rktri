//! Octree construction from dense voxel arrays

use super::{Octree, OctreeNode};
use crate::voxel::voxel::Voxel;
use crate::voxel::brick::VoxelBrick;
use crate::voxel::voxel::rgb_to_565;

/// Builder for constructing octrees from dense voxel data
pub struct OctreeBuilder {
    /// Cubic octree size (power of 2, >= max data dimension)
    size: u32,
    /// Actual data dimensions (may be non-cubic)
    data_width: u32,
    data_height: u32,
    data_depth: u32,
    /// Maximum octree depth
    max_depth: u8,
}

impl OctreeBuilder {
    /// Create a new builder for a cubic grid of given size
    /// Size must be power of 2
    pub fn new(size: u32) -> Self {
        assert!(size.is_power_of_two(), "Size must be power of 2");
        let max_depth = (size as f32).log2() as u8;
        Self { size, data_width: size, data_height: size, data_depth: size, max_depth }
    }

    /// Create a builder for non-cubic data.
    /// The octree size is the next power-of-two >= max(width, height, depth).
    /// Out-of-bounds voxels are treated as empty.
    pub fn new_rectangular(width: u32, height: u32, depth: u32) -> Self {
        let max_dim = width.max(height).max(depth);
        let size = max_dim.next_power_of_two();
        let max_depth = (size as f32).log2() as u8;
        Self { size, data_width: width, data_height: height, data_depth: depth, max_depth }
    }

    /// Build octree from dense voxel array
    /// Array should be data_width * data_height * data_depth elements in Z-Y-X order (x varies fastest)
    pub fn build(&self, voxels: &[Voxel], root_size: f32) -> Octree {
        assert_eq!(voxels.len(), (self.data_width as usize * self.data_height as usize * self.data_depth as usize));

        let mut octree = Octree::with_capacity(root_size, self.max_depth, 1024, 512);

        // Build recursively starting from root
        self.build_node(&mut octree, voxels, 0, 0, 0, self.size, 0);

        octree
    }

    /// Recursively build octree node
    /// Returns (lod_color, lod_material, is_empty)
    fn build_node(
        &self,
        octree: &mut Octree,
        voxels: &[Voxel],
        x: u32, y: u32, z: u32,
        size: u32,
        node_index: u32,
    ) -> (u16, u8, bool) {
        // Base case: 2x2x2 brick level
        if size == 2 {
            return self.build_leaf(octree, voxels, x, y, z, node_index);
        }

        let half = size / 2;
        let mut child_valid_mask: u8 = 0;
        let mut child_leaf_mask: u8 = 0;
        let mut child_indices: [u32; 8] = [0; 8];
        let mut child_colors: [(u16, u8); 8] = [(0, 0); 8];
        let mut all_empty = true;

        // Check each octant
        for child_idx in 0..8u8 {
            let cx = x + if child_idx & 1 != 0 { half } else { 0 };
            let cy = y + if child_idx & 2 != 0 { half } else { 0 };
            let cz = z + if child_idx & 4 != 0 { half } else { 0 };

            // Check if this subtree is empty
            if self.is_region_empty(voxels, cx, cy, cz, half) {
                continue;
            }

            all_empty = false;
            child_valid_mask |= 1 << child_idx;

            if half == 2 {
                // Next level is leaf level
                child_leaf_mask |= 1 << child_idx;
                // Will create brick later
                child_indices[child_idx as usize] = 0; // Placeholder
            } else {
                // Create child node
                let child_node_idx = octree.add_node(OctreeNode::empty());
                child_indices[child_idx as usize] = child_node_idx;
            }
        }

        if all_empty {
            return (0, 0, true);
        }

        // Now process non-empty children
        // Track both node offset and brick offset separately
        let (first_child_offset, first_brick_offset) = if child_leaf_mask == child_valid_mask {
            // All children are leaves - only need brick offset
            let first_brick = octree.brick_count() as u32;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) != 0 {
                    let cx = x + if child_idx & 1 != 0 { half } else { 0 };
                    let cy = y + if child_idx & 2 != 0 { half } else { 0 };
                    let cz = z + if child_idx & 4 != 0 { half } else { 0 };

                    let (color, mat, _) = self.build_leaf(octree, voxels, cx, cy, cz, u32::MAX);
                    child_colors[child_idx as usize] = (color, mat);
                }
            }

            (0, first_brick) // No internal children, only bricks
        } else {
            // Mixed: some internal nodes, some leaves
            let mut first_child = u32::MAX;
            // Record brick offset BEFORE creating any bricks for this node
            let first_brick = octree.brick_count() as u32;

            for child_idx in 0..8u8 {
                if child_valid_mask & (1 << child_idx) == 0 {
                    continue;
                }

                let cx = x + if child_idx & 1 != 0 { half } else { 0 };
                let cy = y + if child_idx & 2 != 0 { half } else { 0 };
                let cz = z + if child_idx & 4 != 0 { half } else { 0 };

                if child_leaf_mask & (1 << child_idx) != 0 {
                    // Leaf child - create brick
                    let (color, mat, _) = self.build_leaf(octree, voxels, cx, cy, cz, u32::MAX);
                    child_colors[child_idx as usize] = (color, mat);
                } else {
                    // Internal child - recurse
                    let child_node_idx = child_indices[child_idx as usize];
                    if first_child == u32::MAX {
                        first_child = child_node_idx;
                    }
                    let (color, mat, _) = self.build_node(octree, voxels, cx, cy, cz, half, child_node_idx);
                    child_colors[child_idx as usize] = (color, mat);
                }
            }

            (first_child, first_brick)
        };

        // Calculate LOD color as average of children
        let (lod_color, lod_material) = self.average_child_colors(&child_colors, child_valid_mask);

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
    fn build_leaf(
        &self,
        octree: &mut Octree,
        voxels: &[Voxel],
        x: u32, y: u32, z: u32,
        _node_index: u32,
    ) -> (u16, u8, bool) {
        let mut brick = VoxelBrick::EMPTY;
        let mut all_empty = true;

        for dz in 0..2u32 {
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let voxel = self.get_voxel(voxels, x + dx, y + dy, z + dz);
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

    /// Check if a region is completely empty
    fn is_region_empty(&self, voxels: &[Voxel], x: u32, y: u32, z: u32, size: u32) -> bool {
        // If entire region is out of bounds, it's empty
        if x >= self.data_width || y >= self.data_height || z >= self.data_depth {
            return true;
        }

        // Clamp iteration to actual data bounds
        let ex = size.min(self.data_width.saturating_sub(x));
        let ey = size.min(self.data_height.saturating_sub(y));
        let ez = size.min(self.data_depth.saturating_sub(z));

        for dz in 0..ez {
            for dy in 0..ey {
                for dx in 0..ex {
                    if !self.get_voxel(voxels, x + dx, y + dy, z + dz).is_empty() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Get voxel from dense array (Z-Y-X order), returns EMPTY for out-of-bounds
    fn get_voxel(&self, voxels: &[Voxel], x: u32, y: u32, z: u32) -> Voxel {
        if x >= self.data_width || y >= self.data_height || z >= self.data_depth {
            return Voxel::EMPTY;
        }
        let idx = z as usize * self.data_height as usize * self.data_width as usize
            + y as usize * self.data_width as usize
            + x as usize;
        voxels[idx]
    }

    /// Average child colors for LOD
    fn average_child_colors(&self, colors: &[(u16, u8); 8], valid_mask: u8) -> (u16, u8) {
        let mut r_sum: u32 = 0;
        let mut g_sum: u32 = 0;
        let mut b_sum: u32 = 0;
        let mut count: u32 = 0;
        let mut mat_counts = [0u8; 256];

        for i in 0..8 {
            if valid_mask & (1 << i) != 0 && colors[i].0 != 0 {
                let (r, g, b) = crate::voxel::voxel::rgb565_to_rgb(colors[i].0);
                r_sum += r as u32;
                g_sum += g as u32;
                b_sum += b as u32;
                mat_counts[colors[i].1 as usize] += 1;
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

/// Create a simple test sphere of voxels
pub fn create_test_sphere(size: u32, radius: f32) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];
    let center = size as f32 / 2.0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center + 0.5;
                let dy = y as f32 - center + 0.5;
                let dz = z as f32 - center + 0.5;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist <= radius {
                    let idx = (z * size * size + y * size + x) as usize;
                    // Color based on position for visual variety
                    let r = ((x as f32 / size as f32) * 255.0) as u8;
                    let g = ((y as f32 / size as f32) * 255.0) as u8;
                    let b = ((z as f32 / size as f32) * 255.0) as u8;
                    voxels[idx] = Voxel::new(r, g, b, 1);
                }
            }
        }
    }

    voxels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty() {
        let voxels = vec![Voxel::EMPTY; 8 * 8 * 8];
        let builder = OctreeBuilder::new(8);
        let octree = builder.build(&voxels, 8.0);

        assert!(octree.root().is_empty());
        assert_eq!(octree.brick_count(), 0);
    }

    #[test]
    fn test_build_sphere() {
        let size = 16u32;
        let voxels = create_test_sphere(size, 6.0);
        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        assert!(!octree.root().is_empty());
        assert!(octree.brick_count() > 0);
        println!("Built octree: {} nodes, {} bricks", octree.node_count(), octree.brick_count());
    }

    #[test]
    fn test_compression() {
        // A mostly empty volume should compress well
        let size = 64u32;
        let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];

        // Add a small cube
        for z in 10..20 {
            for y in 10..20 {
                for x in 10..20 {
                    let idx = (z * size * size + y * size + x) as usize;
                    voxels[idx] = Voxel::new(255, 0, 0, 1);
                }
            }
        }

        let builder = OctreeBuilder::new(size);
        let octree = builder.build(&voxels, size as f32);

        // Should be much smaller than dense array
        let dense_size = size * size * size * 4; // 4 bytes per voxel
        assert!(octree.memory_usage() < dense_size as usize / 10);
        println!("Compression: {} -> {} bytes", dense_size, octree.memory_usage());
    }
}
