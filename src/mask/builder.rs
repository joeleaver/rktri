//! Mask octree builder: constructs MaskOctree<T> from a generator.
//!
//! Follows the same recursive subdivision pattern as AdaptiveOctreeBuilder,
//! using classify_region() for early-out and evaluate() at leaf depth.

use glam::Vec3;
use crate::math::Aabb;
use super::{MaskValue, MaskOctree};
use super::octree::MaskNode;

/// Hint from a generator about a region's content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskHint<T: MaskValue> {
    /// Entire region has this uniform value — stop subdividing.
    Uniform(T),
    /// Region contains mixed values — must subdivide.
    Mixed,
}

/// Trait for procedurally generating mask values.
///
/// Implementations classify regions for early-out and evaluate individual
/// positions at leaf depth.
pub trait MaskGenerator<T: MaskValue>: Send + Sync {
    /// Classify a 3D region. Return `Uniform(v)` to stop subdivision,
    /// or `Mixed` to recurse deeper.
    fn classify_region(&self, aabb: &Aabb) -> MaskHint<T>;

    /// Evaluate the mask value at a single world position.
    /// Called at leaf depth when classify_region returns Mixed.
    fn evaluate(&self, pos: Vec3) -> T;
}

/// Builds a `MaskOctree<T>` by recursively subdividing and querying a generator.
pub struct MaskBuilder {
    max_depth: u8,
}

impl MaskBuilder {
    /// Create a builder with the given max depth.
    ///
    /// Depth 3 = 8 cells/side, depth 5 = 32 cells/side.
    pub fn new(max_depth: u8) -> Self {
        Self { max_depth }
    }

    /// Build a mask octree from a generator.
    ///
    /// `origin` is the world-space origin (min corner) of the region.
    /// `size` is the region extent (typically CHUNK_SIZE = 4.0).
    pub fn build<T: MaskValue>(
        &self,
        generator: &dyn MaskGenerator<T>,
        origin: Vec3,
        size: f32,
    ) -> MaskOctree<T> {
        let cells_per_side = 1u32 << self.max_depth;
        let estimated_nodes = (cells_per_side as usize).min(512);
        let mut tree = MaskOctree::with_capacity(size, self.max_depth, estimated_nodes, estimated_nodes);

        self.build_node(&mut tree, generator, origin, size, 0, 0);
        tree
    }

    /// Recursively build a node.
    ///
    /// Returns the value if the subtree is uniform, or None if mixed.
    fn build_node<T: MaskValue>(
        &self,
        tree: &mut MaskOctree<T>,
        generator: &dyn MaskGenerator<T>,
        origin: Vec3,
        size: f32,
        node_idx: u32,
        depth: u8,
    ) -> Option<T> {
        let aabb = Aabb::new(origin, origin + Vec3::splat(size));

        // Ask generator for classification
        match generator.classify_region(&aabb) {
            MaskHint::Uniform(val) => {
                // Store uniform value as LOD on this node
                let val_idx = tree.add_value(val);
                tree.node_mut(node_idx).lod_value_idx = val_idx;
                return Some(val);
            }
            MaskHint::Mixed => {
                // At max depth, evaluate center position
                if depth >= self.max_depth {
                    let center = origin + Vec3::splat(size * 0.5);
                    let val = generator.evaluate(center);
                    let val_idx = tree.add_value(val);
                    tree.node_mut(node_idx).lod_value_idx = val_idx;
                    return Some(val);
                }
            }
        }

        // Subdivide into 8 children
        let half = size * 0.5;
        let mut child_mask: u8 = 0;
        let mut leaf_mask: u8 = 0;
        let mut child_values: [Option<T>; 8] = [None; 8];
        let mut child_node_indices: [u32; 8] = [0; 8];

        // First pass: classify all children, allocate internal nodes
        for ci in 0..8u8 {
            let child_origin = Vec3::new(
                if ci & 1 != 0 { origin.x + half } else { origin.x },
                if ci & 2 != 0 { origin.y + half } else { origin.y },
                if ci & 4 != 0 { origin.z + half } else { origin.z },
            );
            let child_aabb = Aabb::new(child_origin, child_origin + Vec3::splat(half));

            match generator.classify_region(&child_aabb) {
                MaskHint::Uniform(val) => {
                    if !val.is_default() {
                        child_mask |= 1 << ci;
                        leaf_mask |= 1 << ci;
                        child_values[ci as usize] = Some(val);
                    }
                    // Default values don't need storage
                }
                MaskHint::Mixed => {
                    if depth + 1 >= self.max_depth {
                        // Will be a leaf at next depth
                        let center = child_origin + Vec3::splat(half * 0.5);
                        let val = generator.evaluate(center);
                        if !val.is_default() {
                            child_mask |= 1 << ci;
                            leaf_mask |= 1 << ci;
                            child_values[ci as usize] = Some(val);
                        }
                    } else {
                        // Needs internal child node
                        child_mask |= 1 << ci;
                        let child_node = tree.add_node(MaskNode::empty());
                        child_node_indices[ci as usize] = child_node;
                    }
                }
            }
        }

        if child_mask == 0 {
            // All children are default — this node is effectively default
            let val_idx = tree.add_value(T::default());
            tree.node_mut(node_idx).lod_value_idx = val_idx;
            return Some(T::default());
        }

        // Second pass: store leaf values and recurse into internal children
        let first_value = tree.value_count() as u32;
        for ci in 0..8u8 {
            if child_mask & (1 << ci) != 0 && leaf_mask & (1 << ci) != 0 {
                if let Some(val) = child_values[ci as usize] {
                    tree.add_value(val);
                }
            }
        }

        let mut first_child_offset = u32::MAX;
        for ci in 0..8u8 {
            if child_mask & (1 << ci) != 0 && leaf_mask & (1 << ci) == 0 {
                let child_node_idx = child_node_indices[ci as usize];
                if first_child_offset == u32::MAX {
                    first_child_offset = child_node_idx;
                }

                let child_origin = Vec3::new(
                    if ci & 1 != 0 { origin.x + half } else { origin.x },
                    if ci & 2 != 0 { origin.y + half } else { origin.y },
                    if ci & 4 != 0 { origin.z + half } else { origin.z },
                );

                let child_result = self.build_node(
                    tree,
                    generator,
                    child_origin,
                    half,
                    child_node_idx,
                    depth + 1,
                );

                // If child turned out uniform, convert to leaf
                if let Some(val) = child_result {
                    if tree.node(child_node_idx).is_empty() {
                        // The child node is effectively a leaf — but since we already
                        // allocated it as internal, we leave the structure as-is.
                        // The LOD value is already stored on the child node.
                        child_values[ci as usize] = Some(val);
                    }
                }
            }
        }

        // Store LOD value for this node (most common child value)
        let lod_val = self.compute_lod(&child_values, child_mask);
        let lod_idx = tree.add_value(lod_val);

        // Update node
        let node = tree.node_mut(node_idx);
        node.child_mask = child_mask;
        node.leaf_mask = leaf_mask;
        node.value_offset = first_value;
        node.child_offset = if first_child_offset == u32::MAX { 0 } else { first_child_offset };
        node.lod_value_idx = lod_idx;

        // Check if all children have the same value (prune opportunity)
        if self.all_same(&child_values, child_mask) {
            if let Some(val) = child_values.iter().flatten().next() {
                // Could prune, but we leave nodes allocated — the tree is
                // still correct and sample() works via LOD fallback.
                return Some(*val);
            }
        }

        None
    }

    /// Compute LOD value as most common child value.
    fn compute_lod<T: MaskValue>(&self, child_values: &[Option<T>; 8], child_mask: u8) -> T {
        // Return first non-default value found (simple heuristic)
        for ci in 0..8u8 {
            if child_mask & (1 << ci) != 0 {
                if let Some(val) = child_values[ci as usize] {
                    if !val.is_default() {
                        return val;
                    }
                }
            }
        }
        T::default()
    }

    /// Check if all active children have the same value.
    fn all_same<T: MaskValue>(&self, child_values: &[Option<T>; 8], child_mask: u8) -> bool {
        let mut first: Option<T> = None;
        for ci in 0..8u8 {
            if child_mask & (1 << ci) != 0 {
                if let Some(val) = child_values[ci as usize] {
                    match first {
                        None => first = Some(val),
                        Some(f) if f != val => return false,
                        _ => {}
                    }
                }
            }
        }
        // Also need all missing children to be default
        if let Some(f) = first {
            if child_mask.count_ones() < 8 && !f.is_default() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::BiomeId;

    /// Generator that returns uniform value everywhere.
    struct UniformGenerator<T: MaskValue> {
        value: T,
    }

    impl<T: MaskValue> MaskGenerator<T> for UniformGenerator<T> {
        fn classify_region(&self, _aabb: &Aabb) -> MaskHint<T> {
            MaskHint::Uniform(self.value)
        }
        fn evaluate(&self, _pos: Vec3) -> T {
            self.value
        }
    }

    /// Generator that returns different biomes in different halves (x < mid vs x >= mid).
    struct SplitGenerator {
        mid_x: f32,
        left: BiomeId,
        right: BiomeId,
    }

    impl MaskGenerator<BiomeId> for SplitGenerator {
        fn classify_region(&self, aabb: &Aabb) -> MaskHint<BiomeId> {
            if aabb.max.x <= self.mid_x {
                MaskHint::Uniform(self.left)
            } else if aabb.min.x >= self.mid_x {
                MaskHint::Uniform(self.right)
            } else {
                MaskHint::Mixed
            }
        }
        fn evaluate(&self, pos: Vec3) -> BiomeId {
            if pos.x < self.mid_x { self.left } else { self.right }
        }
    }

    /// Gradient generator: f32 value = x / size.
    struct GradientGenerator {
        size: f32,
    }

    impl MaskGenerator<f32> for GradientGenerator {
        fn classify_region(&self, _aabb: &Aabb) -> MaskHint<f32> {
            MaskHint::Mixed
        }
        fn evaluate(&self, pos: Vec3) -> f32 {
            pos.x / self.size
        }
    }

    #[test]
    fn test_build_uniform() {
        let generator = UniformGenerator { value: BiomeId::FOREST };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        // Uniform tree should collapse
        let val = tree.sample(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(val, BiomeId::FOREST);

        // classify_region should return uniform
        let aabb = Aabb::new(Vec3::splat(0.5), Vec3::splat(3.5));
        assert_eq!(tree.classify_region(Vec3::ZERO, &aabb), Some(BiomeId::FOREST));
    }

    #[test]
    fn test_build_uniform_default() {
        let generator = UniformGenerator { value: BiomeId::default() };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        let val = tree.sample(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(val, BiomeId::default());
    }

    #[test]
    fn test_build_split() {
        let generator = SplitGenerator {
            mid_x: 2.0,
            left: BiomeId::FOREST,
            right: BiomeId::DESERT,
        };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        // Left half should be FOREST
        let left = tree.sample(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(left, BiomeId::FOREST);

        // Right half should be DESERT
        let right = tree.sample(Vec3::ZERO, Vec3::new(3.0, 1.0, 1.0));
        assert_eq!(right, BiomeId::DESERT);
    }

    #[test]
    fn test_build_gradient() {
        let generator = GradientGenerator { size: 4.0 };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        // Sample at various x positions
        let v0 = tree.sample(Vec3::ZERO, Vec3::new(0.25, 2.0, 2.0));
        let v1 = tree.sample(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let v2 = tree.sample(Vec3::ZERO, Vec3::new(3.75, 2.0, 2.0));

        // Values should increase with x
        assert!(v0 < v1, "v0={} should be < v1={}", v0, v1);
        assert!(v1 < v2, "v1={} should be < v2={}", v1, v2);
    }

    #[test]
    fn test_build_prunes_uniform_subtrees() {
        let generator = UniformGenerator { value: BiomeId::GRASSLAND };
        let builder = MaskBuilder::new(5);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        // Uniform generator should produce a very compact tree
        // (just the root with its LOD value)
        assert!(tree.node_count() <= 2, "Expected compact tree, got {} nodes", tree.node_count());
    }

    #[test]
    fn test_classify_region_uniform() {
        let generator = UniformGenerator { value: BiomeId::TAIGA };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(4.0));
        let result = tree.classify_region(Vec3::ZERO, &aabb);
        assert_eq!(result, Some(BiomeId::TAIGA));
    }

    #[test]
    fn test_classify_region_mixed() {
        let generator = SplitGenerator {
            mid_x: 2.0,
            left: BiomeId::FOREST,
            right: BiomeId::DESERT,
        };
        let builder = MaskBuilder::new(3);
        let tree = builder.build(&generator, Vec3::ZERO, 4.0);

        // Region spanning split should be mixed
        let aabb = Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 4.0));
        let result = tree.classify_region(Vec3::ZERO, &aabb);
        assert_eq!(result, None);

        // Region within a single octant that is uniform should return Some
        // (must fit within one octant: 0..2 in all axes)
        let single_octant = Aabb::new(Vec3::ZERO, Vec3::new(1.5, 1.5, 1.5));
        let octant_result = tree.classify_region(Vec3::ZERO, &single_octant);
        assert_eq!(octant_result, Some(BiomeId::FOREST));
    }
}
