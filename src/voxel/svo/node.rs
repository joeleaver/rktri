//! Sparse Voxel Octree node

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};

/// Octree node - exactly 32 bytes, 32-byte aligned for cache efficiency
///
/// Layout:
/// - flags (4 bytes): bits 0-7 child_valid, 8-15 child_leaf, 16-31 reserved
/// - child_offset (4 bytes): offset to first internal child in node array
/// - brick_offset (4 bytes): offset to first leaf child in brick array
/// - bounds_min (6 bytes): quantized min corner
/// - lod_color (2 bytes): RGB565 LOD color
/// - bounds_max (6 bytes): quantized max corner
/// - lod_material (2 bytes): LOD material ID
/// - padding (4 bytes): pad to 32 bytes
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Archive, Deserialize, Serialize)]
pub struct OctreeNode {
    /// Bits 0-7: child valid mask (which of 8 children exist)
    /// Bits 8-15: child leaf mask (which children are leaves vs internal)
    /// Bits 16-31: reserved (LOD level, flags)
    pub flags: u32,
    /// For internal children: offset to first child in node array
    pub child_offset: u32,
    /// For leaf children: offset to first brick in brick array
    pub brick_offset: u32,
    /// Bounding box min (quantized to u16)
    pub bounds_min: [u16; 3],
    /// LOD representative color (RGB565)
    pub lod_color: u16,
    /// Bounding box max (quantized to u16)
    pub bounds_max: [u16; 3],
    /// LOD representative material ID
    pub lod_material: u16,
    /// Padding to reach 32 bytes
    _padding: [u8; 4],
}

impl OctreeNode {
    /// Create an empty node
    pub const fn empty() -> Self {
        Self {
            flags: 0,
            child_offset: 0,
            brick_offset: 0,
            bounds_min: [0; 3],
            lod_color: 0,
            bounds_max: [0; 3],
            lod_material: 0,
            _padding: [0; 4],
        }
    }

    /// Get child valid mask (bits 0-7)
    pub fn child_valid_mask(&self) -> u8 {
        (self.flags & 0xFF) as u8
    }

    /// Get child leaf mask (bits 8-15)
    pub fn child_leaf_mask(&self) -> u8 {
        ((self.flags >> 8) & 0xFF) as u8
    }

    /// Set child valid mask
    pub fn set_child_valid_mask(&mut self, mask: u8) {
        self.flags = (self.flags & !0xFF) | (mask as u32);
    }

    /// Set child leaf mask
    pub fn set_child_leaf_mask(&mut self, mask: u8) {
        self.flags = (self.flags & !0xFF00) | ((mask as u32) << 8);
    }

    /// Check if child at index is valid (exists)
    pub fn is_child_valid(&self, index: u8) -> bool {
        debug_assert!(index < 8);
        (self.child_valid_mask() >> index) & 1 != 0
    }

    /// Check if child at index is a leaf
    pub fn is_child_leaf(&self, index: u8) -> bool {
        debug_assert!(index < 8);
        (self.child_leaf_mask() >> index) & 1 != 0
    }

    /// Set child valid bit
    pub fn set_child_valid(&mut self, index: u8, valid: bool) {
        debug_assert!(index < 8);
        let mask = self.child_valid_mask();
        let new_mask = if valid {
            mask | (1 << index)
        } else {
            mask & !(1 << index)
        };
        self.set_child_valid_mask(new_mask);
    }

    /// Set child leaf bit
    pub fn set_child_leaf(&mut self, index: u8, is_leaf: bool) {
        debug_assert!(index < 8);
        let mask = self.child_leaf_mask();
        let new_mask = if is_leaf {
            mask | (1 << index)
        } else {
            mask & !(1 << index)
        };
        self.set_child_leaf_mask(new_mask);
    }

    /// Count number of valid children
    pub fn child_count(&self) -> u8 {
        self.child_valid_mask().count_ones() as u8
    }

    /// Get LOD level (bits 16-23)
    pub fn lod_level(&self) -> u8 {
        ((self.flags >> 16) & 0xFF) as u8
    }

    /// Set LOD level
    pub fn set_lod_level(&mut self, level: u8) {
        self.flags = (self.flags & !0xFF0000) | ((level as u32) << 16);
    }

    /// Check if this is a fully empty node (no children and no terminal brick)
    pub fn is_empty(&self) -> bool {
        self.child_valid_mask() == 0 && self.brick_offset == 0
    }

    /// Check if this is a terminal leaf (single brick, no children)
    pub fn is_terminal_leaf(&self) -> bool {
        self.child_valid_mask() == 0 && self.brick_offset != 0
    }
}

impl Default for OctreeNode {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_and_alignment() {
        assert_eq!(std::mem::size_of::<OctreeNode>(), 32);
        assert_eq!(std::mem::align_of::<OctreeNode>(), 32);
    }

    #[test]
    fn test_child_masks() {
        let mut node = OctreeNode::empty();

        node.set_child_valid(0, true);
        node.set_child_valid(3, true);
        node.set_child_valid(7, true);

        assert!(node.is_child_valid(0));
        assert!(!node.is_child_valid(1));
        assert!(node.is_child_valid(3));
        assert!(node.is_child_valid(7));
        assert_eq!(node.child_count(), 3);
    }

    #[test]
    fn test_leaf_masks() {
        let mut node = OctreeNode::empty();

        node.set_child_valid(0, true);
        node.set_child_leaf(0, true);

        assert!(node.is_child_valid(0));
        assert!(node.is_child_leaf(0));
        assert!(!node.is_child_leaf(1));
    }

    #[test]
    fn test_lod_level() {
        let mut node = OctreeNode::empty();
        node.set_lod_level(5);
        assert_eq!(node.lod_level(), 5);
    }
}
