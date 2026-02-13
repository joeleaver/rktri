//! Tree merging algorithm for placing trees into world octrees
//!
//! Trees are integrated into octrees during construction using the `MultiTreeClassifier`.
//! This classifier wraps a base terrain classifier and overlays multiple tree octrees,
//! allowing the AdaptiveOctreeBuilder to construct a single unified octree.

use glam::Vec3;

use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::svo::Octree;
use crate::voxel::voxel::Voxel;
use crate::math::Aabb;

/// A tree placement: an octree and its world position
#[derive(Clone)]
pub struct TreeInstance {
    /// The tree octree (centered at origin)
    pub octree: Octree,
    /// World position of tree base (trunk bottom)
    pub position: Vec3,
}

impl TreeInstance {
    pub fn new(octree: Octree, position: Vec3) -> Self {
        Self { octree, position }
    }

    /// Sample voxel at world position, returning None if outside tree bounds or empty
    pub fn sample_at_world(&self, world_pos: Vec3) -> Option<Voxel> {
        let tree_half = self.octree.root_size() / 2.0;

        // Transform world position to tree-local coordinates
        // tree_position is where the tree BASE should be in world space
        // The octree has the tree centered (Y from -half to +half)
        let tree_local = Vec3::new(
            world_pos.x - self.position.x,
            world_pos.y - self.position.y - tree_half,
            world_pos.z - self.position.z,
        );

        // Check if position is within tree bounds (centered at origin)
        if tree_local.x >= -tree_half
            && tree_local.x < tree_half
            && tree_local.y >= -tree_half
            && tree_local.y < tree_half
            && tree_local.z >= -tree_half
            && tree_local.z < tree_half
        {
            let voxel = self.octree.sample_voxel(tree_local);
            if !voxel.is_empty() {
                return Some(voxel);
            }
        }
        None
    }

    /// Get world-space bounding box (min corner)
    pub fn world_bounds_min(&self) -> Vec3 {
        let half = self.octree.root_size() / 2.0;
        Vec3::new(
            self.position.x - half,
            self.position.y,
            self.position.z - half,
        )
    }

    /// Get world-space bounding box (max corner)
    pub fn world_bounds_max(&self) -> Vec3 {
        let half = self.octree.root_size() / 2.0;
        Vec3::new(
            self.position.x + half,
            self.position.y + self.octree.root_size(),
            self.position.z + half,
        )
    }

    /// Get world-space AABB
    pub fn world_aabb(&self) -> Aabb {
        Aabb::new(self.world_bounds_min(), self.world_bounds_max())
    }
}

/// Classifier that overlays multiple trees onto a base terrain classifier.
///
/// Use this during octree construction to integrate trees directly into the octree
/// structure. This is the correct way to merge trees - not by appending to an
/// existing octree, but by including them during construction.
///
/// # Example
/// ```ignore
/// let terrain_classifier = MyTerrainClassifier::new();
/// let mut classifier = MultiTreeClassifier::new(&terrain_classifier);
/// classifier.add_tree(TreeInstance::new(tree_octree, Vec3::new(10.0, 0.5, 10.0)));
///
/// let builder = AdaptiveOctreeBuilder::new(512);
/// let octree = builder.build(&classifier, chunk_origin, chunk_size);
/// ```
pub struct MultiTreeClassifier<'a, C: RegionClassifier> {
    base: &'a C,
    trees: Vec<TreeInstance>,
}

impl<'a, C: RegionClassifier> MultiTreeClassifier<'a, C> {
    /// Create a new multi-tree classifier with a base terrain classifier
    pub fn new(base: &'a C) -> Self {
        Self {
            base,
            trees: Vec::new(),
        }
    }

    /// Add a tree to be overlaid on the terrain
    pub fn add_tree(&mut self, tree: TreeInstance) {
        self.trees.push(tree);
    }

    /// Add multiple trees
    pub fn add_trees(&mut self, trees: impl IntoIterator<Item = TreeInstance>) {
        self.trees.extend(trees);
    }

    /// Get the number of trees
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }
}

impl<'a, C: RegionClassifier> RegionClassifier for MultiTreeClassifier<'a, C> {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        // Check if any trees intersect this region
        let mut has_tree_intersection = false;
        for tree in &self.trees {
            if aabb.intersects(&tree.world_aabb()) {
                has_tree_intersection = true;
                break;
            }
        }

        // Get base classification
        let base_hint = self.base.classify_region(aabb);

        // If a tree intersects, we need to subdivide (Mixed)
        // regardless of what the base says
        if has_tree_intersection {
            return RegionHint::Mixed;
        }

        // No tree intersection - use base classification
        base_hint
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        // Check all trees first (trees overlay terrain)
        for tree in &self.trees {
            if let Some(voxel) = tree.sample_at_world(pos) {
                return voxel;
            }
        }

        // Fall back to base terrain
        self.base.evaluate(pos)
    }
}

/// Calculate which chunk coordinates a tree at a given position would touch
pub fn tree_affected_chunks(
    tree_bounds_min: Vec3,
    tree_bounds_max: Vec3,
    position: Vec3,
    chunk_size: f32,
) -> Vec<[i32; 3]> {
    let world_min = position + tree_bounds_min;
    let world_max = position + tree_bounds_max;

    let chunk_min_x = (world_min.x / chunk_size).floor() as i32;
    let chunk_min_y = (world_min.y / chunk_size).floor() as i32;
    let chunk_min_z = (world_min.z / chunk_size).floor() as i32;

    let chunk_max_x = (world_max.x / chunk_size).floor() as i32;
    let chunk_max_y = (world_max.y / chunk_size).floor() as i32;
    let chunk_max_z = (world_max.z / chunk_size).floor() as i32;

    let mut chunks = Vec::new();
    for x in chunk_min_x..=chunk_max_x {
        for y in chunk_min_y..=chunk_max_y {
            for z in chunk_min_z..=chunk_max_z {
                chunks.push([x, y, z]);
            }
        }
    }
    chunks
}

// Type alias for backwards compatibility
pub type MultiTreeEvaluator<'a, C> = MultiTreeClassifier<'a, C>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::procgen::{TreeGenerator, TreeStyle};
    use crate::voxel::svo::adaptive::AdaptiveOctreeBuilder;

    /// Simple terrain classifier for testing
    struct FlatTerrain {
        height: f32,
    }

    impl RegionClassifier for FlatTerrain {
        fn classify_region(&self, aabb: &Aabb) -> RegionHint {
            if aabb.min.y > self.height {
                RegionHint::Empty
            } else if aabb.max.y < self.height - 3.0 {
                RegionHint::Solid { material: 1, color: 0 }
            } else {
                RegionHint::Mixed
            }
        }

        fn evaluate(&self, pos: Vec3) -> Voxel {
            if pos.y <= self.height {
                Voxel::new(100, 80, 60, 1) // Brown dirt
            } else {
                Voxel::EMPTY
            }
        }
    }

    #[test]
    fn test_tree_instance_sampling() {
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let tree_octree = generator.generate(8.0, 6);

        let instance = TreeInstance::new(tree_octree, Vec3::new(10.0, 0.0, 10.0));

        // Sample at tree center (should find trunk)
        let center = Vec3::new(10.0, 4.0, 10.0); // Middle of tree height
        let _voxel = instance.sample_at_world(center);
        // May or may not hit a voxel depending on tree structure

        // Sample outside tree bounds (should be None)
        let outside = Vec3::new(100.0, 0.0, 100.0);
        assert!(instance.sample_at_world(outside).is_none());
    }

    #[test]
    fn test_multi_tree_classifier() {
        let terrain = FlatTerrain { height: 0.5 };
        let mut classifier = MultiTreeClassifier::new(&terrain);

        // Add a tree
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let tree = generator.generate(8.0, 6);
        classifier.add_tree(TreeInstance::new(tree, Vec3::new(4.0, 0.5, 4.0)));

        assert_eq!(classifier.tree_count(), 1);

        // Terrain below tree should still be terrain
        let ground = classifier.evaluate(Vec3::new(4.0, 0.3, 4.0));
        assert!(!ground.is_empty());

        // High above should be empty (above tree)
        let sky = classifier.evaluate(Vec3::new(4.0, 100.0, 4.0));
        assert!(sky.is_empty());
    }

    #[test]
    fn test_multi_tree_with_adaptive_builder() {
        let terrain = FlatTerrain { height: 0.5 };
        let mut classifier = MultiTreeClassifier::new(&terrain);

        // Add a tree at origin
        let mut generator = TreeGenerator::from_style(123, TreeStyle::Elm);
        let tree = generator.generate(4.0, 5);
        classifier.add_tree(TreeInstance::new(tree, Vec3::new(2.0, 0.5, 2.0)));

        // Build octree with trees integrated
        let builder = AdaptiveOctreeBuilder::new(64);
        let octree = builder.build(&classifier, Vec3::ZERO, 8.0);

        // Should have more than just terrain
        assert!(octree.node_count() > 1);
        assert!(octree.brick_count() > 0);

        println!(
            "Built octree with tree: {} nodes, {} bricks",
            octree.node_count(),
            octree.brick_count()
        );
    }

    #[test]
    fn test_affected_chunks() {
        let bounds_min = Vec3::new(-4.0, 0.0, -4.0);
        let bounds_max = Vec3::new(4.0, 8.0, 4.0);
        let position = Vec3::new(0.0, 0.0, 0.0);
        let chunk_size = 16.0;

        let chunks = tree_affected_chunks(bounds_min, bounds_max, position, chunk_size);

        assert!(!chunks.is_empty());
        assert!(chunks.contains(&[0, 0, 0]));
    }

    #[test]
    fn test_tree_aabb_intersection() {
        let terrain = FlatTerrain { height: 0.5 };
        let mut classifier = MultiTreeClassifier::new(&terrain);

        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let tree = generator.generate(8.0, 6);
        let tree_pos = Vec3::new(4.0, 0.5, 4.0);
        classifier.add_tree(TreeInstance::new(tree, tree_pos));

        // Region that intersects tree should be Mixed
        let tree_aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 10.0, 8.0));
        assert_eq!(classifier.classify_region(&tree_aabb), RegionHint::Mixed);

        // Region far from tree should use base classification
        let far_aabb = Aabb::new(Vec3::new(100.0, 100.0, 100.0), Vec3::new(110.0, 110.0, 110.0));
        assert_eq!(classifier.classify_region(&far_aabb), RegionHint::Empty);
    }
}
