//! Volumetric object trait for spatial composition of voxel data

use glam::Vec3;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::Octree;
use crate::math::aabb::Aabb;
use crate::voxel::voxel::Voxel;

/// Global counter for auto-generating unique instance IDs
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Trait for objects that can be sampled as volumetric voxel data in world space.
///
/// This enables spatial composition where multiple octrees or other volumetric
/// sources can be positioned, transformed, and sampled together to build complex scenes.
pub trait VolumetricObject: Send + Sync {
    /// Sample a voxel at the given world-space position.
    ///
    /// Returns `None` if the position is outside this object's bounds or in empty space.
    /// Returns `Some(voxel)` if a voxel exists at this position.
    fn sample_at(&self, world_pos: Vec3) -> Option<Voxel>;

    /// Get the axis-aligned bounding box of this object in world space.
    ///
    /// Used for spatial culling and determining which objects to sample.
    fn world_bounds(&self) -> Aabb;

    /// Get a unique identifier for this object instance.
    ///
    /// Used for removal, deduplication, and tracking in spatial databases.
    fn id(&self) -> u64;
}

/// An instance of an octree positioned in world space.
///
/// The octree data structure is centered at origin in local space.
/// The `position` field defines where the BASE (bottom-center) of the
/// octree should be placed in world coordinates.
///
/// # Coordinate Transform
///
/// Local octree coordinates range from `-root_size/2` to `+root_size/2`.
/// When positioning in world space:
/// - `position.xz` = horizontal position of base center
/// - `position.y` = Y coordinate of the bottom of the octree
///
/// Transform: `local = world - position - Vec3::Y * (root_size/2)`
#[derive(Debug, Clone)]
pub struct OctreeInstance {
    /// The octree data
    octree: Octree,
    /// World-space position of the octree base (bottom-center)
    position: Vec3,
    /// Unique instance identifier
    id: u64,
    /// Optional tighter content bounds in octree-local space.
    /// If set, world_bounds() uses these instead of the full octree volume.
    /// This avoids forcing unnecessary subdivision for empty octree regions.
    content_bounds_local: Option<Aabb>,
}

impl OctreeInstance {
    /// Create a new octree instance at the given world position.
    ///
    /// # Arguments
    /// * `octree` - The octree data to instance
    /// * `position` - World position for the base (bottom-center) of the octree
    ///
    /// # Example
    /// ```ignore
    /// let octree = Octree::new(64.0, 13);
    /// // Place octree with its base at ground level (y=0)
    /// let instance = OctreeInstance::new(octree, Vec3::new(100.0, 0.0, 50.0));
    /// ```
    pub fn new(octree: Octree, position: Vec3) -> Self {
        Self {
            octree,
            position,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            content_bounds_local: None,
        }
    }

    /// Create a new octree instance with an explicit ID.
    ///
    /// Useful for deserialization or when you need deterministic IDs.
    pub fn with_id(octree: Octree, position: Vec3, id: u64) -> Self {
        Self {
            octree,
            position,
            id,
            content_bounds_local: None,
        }
    }

    /// Set tighter content bounds in octree-local space.
    ///
    /// When set, `world_bounds()` returns these tighter bounds instead of the
    /// full octree volume. This is important for trees where content only
    /// occupies the upper half of the octree (local Y >= 0).
    pub fn with_content_bounds(mut self, local_min: Vec3, local_max: Vec3) -> Self {
        self.content_bounds_local = Some(Aabb::new(local_min, local_max));
        self
    }

    /// Get reference to the underlying octree
    pub fn octree(&self) -> &Octree {
        &self.octree
    }

    /// Get mutable reference to the underlying octree
    pub fn octree_mut(&mut self) -> &mut Octree {
        &mut self.octree
    }

    /// Get the world position (base point)
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Set the world position
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Transform world position to octree-local coordinates.
    ///
    /// The octree is centered at origin, but `position` represents the base.
    /// So we need to offset by half the root size in Y.
    fn world_to_local(&self, world_pos: Vec3) -> Vec3 {
        let tree_half = self.octree.root_size() / 2.0;
        Vec3::new(
            world_pos.x - self.position.x,
            world_pos.y - self.position.y - tree_half,
            world_pos.z - self.position.z,
        )
    }

    /// Get the center point of the octree in world space
    pub fn world_center(&self) -> Vec3 {
        let tree_half = self.octree.root_size() / 2.0;
        Vec3::new(
            self.position.x,
            self.position.y + tree_half,
            self.position.z,
        )
    }
}

impl VolumetricObject for OctreeInstance {
    fn sample_at(&self, world_pos: Vec3) -> Option<Voxel> {
        let local = self.world_to_local(world_pos);
        let voxel = self.octree.sample_voxel(local);

        // sample_voxel returns Voxel::EMPTY for out-of-bounds or empty regions
        if voxel.is_empty() {
            None
        } else {
            Some(voxel)
        }
    }

    fn world_bounds(&self) -> Aabb {
        if let Some(content) = &self.content_bounds_local {
            // Transform local content bounds to world space
            // world = local + position + Vec3::Y * tree_half
            let tree_half = self.octree.root_size() / 2.0;
            Aabb::new(
                Vec3::new(
                    self.position.x + content.min.x,
                    self.position.y + tree_half + content.min.y,
                    self.position.z + content.min.z,
                ),
                Vec3::new(
                    self.position.x + content.max.x,
                    self.position.y + tree_half + content.max.y,
                    self.position.z + content.max.z,
                ),
            )
        } else {
            let size = self.octree.root_size();
            let half = size / 2.0;

            // Base is at position, top is at position + size
            Aabb::new(
                Vec3::new(
                    self.position.x - half,
                    self.position.y,
                    self.position.z - half,
                ),
                Vec3::new(
                    self.position.x + half,
                    self.position.y + size,
                    self.position.z + half,
                ),
            )
        }
    }

    fn id(&self) -> u64 {
        self.id
    }
}

/// A spatial hash grid for efficient spatial queries of volumetric objects.
///
/// Uses a uniform grid to partition space into cells. Each cell contains references
/// to all volumetric objects whose bounding boxes overlap that cell.
///
/// Provides O(1) average-case lookups for point and AABB queries.
pub struct VolumetricGrid {
    /// Size of each grid cell in world units
    cell_size: f32,
    /// Mapping from cell coordinates to objects in that cell
    cells: HashMap<[i32; 3], Vec<Arc<dyn VolumetricObject>>>,
    /// All objects in the grid (for iteration and removal)
    all_objects: Vec<Arc<dyn VolumetricObject>>,
    /// Combined bounding box of all objects
    global_bounds: Option<Aabb>,
}

impl VolumetricGrid {
    /// Create a new empty volumetric grid.
    ///
    /// # Arguments
    /// * `cell_size` - Size of each grid cell in world units. Default 16.0 works well for tree-sized objects.
    ///
    /// # Example
    /// ```ignore
    /// let grid = VolumetricGrid::new(16.0);
    /// ```
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
            all_objects: Vec::new(),
            global_bounds: None,
        }
    }

    /// Convert a world position to grid cell coordinates.
    fn world_to_cell(&self, pos: Vec3) -> [i32; 3] {
        [
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        ]
    }

    /// Get all cells that overlap the given AABB.
    fn cells_for_aabb(&self, bounds: &Aabb) -> Vec<[i32; 3]> {
        let min_cell = self.world_to_cell(bounds.min);
        let max_cell = self.world_to_cell(bounds.max);

        let mut cells = Vec::new();
        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    cells.push([x, y, z]);
                }
            }
        }
        cells
    }

    /// Insert a volumetric object into the grid.
    ///
    /// The object is added to all grid cells that its bounding box overlaps.
    /// Updates the global bounds to include this object.
    pub fn insert(&mut self, obj: Arc<dyn VolumetricObject>) {
        let bounds = obj.world_bounds();
        let cells = self.cells_for_aabb(&bounds);

        // Insert into all overlapping cells
        for cell in cells {
            self.cells
                .entry(cell)
                .or_insert_with(Vec::new)
                .push(Arc::clone(&obj));
        }

        // Update global bounds
        self.global_bounds = match self.global_bounds {
            Some(existing) => Some(existing.merged(&bounds)),
            None => Some(bounds),
        };

        self.all_objects.push(obj);
    }

    /// Remove an object from the grid by its ID.
    ///
    /// Returns `true` if the object was found and removed, `false` otherwise.
    pub fn remove(&mut self, id: u64) -> bool {
        // Find and remove from all_objects
        let removed = if let Some(idx) = self.all_objects.iter().position(|o| o.id() == id) {
            self.all_objects.swap_remove(idx);
            true
        } else {
            return false;
        };

        // Remove from all cells
        for objects in self.cells.values_mut() {
            objects.retain(|o| o.id() != id);
        }

        // Remove empty cells
        self.cells.retain(|_, objects| !objects.is_empty());

        // Recalculate global bounds
        self.global_bounds = self.all_objects.iter().fold(None, |acc, obj| {
            let bounds = obj.world_bounds();
            match acc {
                Some(existing) => Some(existing.merged(&bounds)),
                None => Some(bounds),
            }
        });

        removed
    }

    /// Query all objects whose cells contain the given point.
    ///
    /// Returns references to objects in the cell containing this point.
    /// May return objects that don't actually contain the point (false positives),
    /// but will never miss objects that do contain it (no false negatives).
    pub fn query_point(&self, pos: Vec3) -> Vec<&Arc<dyn VolumetricObject>> {
        let cell = self.world_to_cell(pos);
        self.cells
            .get(&cell)
            .map(|objects| objects.iter().collect())
            .unwrap_or_default()
    }

    /// Query all objects whose bounding boxes overlap the given AABB.
    ///
    /// Returns deduplicated references to objects in cells overlapping the query bounds.
    /// May return objects that don't actually overlap (false positives),
    /// but will never miss objects that do overlap (no false negatives).
    pub fn query_aabb(&self, bounds: &Aabb) -> Vec<&Arc<dyn VolumetricObject>> {
        let cells = self.cells_for_aabb(bounds);
        let mut seen_ids = std::collections::HashSet::new();
        let mut results = Vec::new();

        for cell in cells {
            if let Some(objects) = self.cells.get(&cell) {
                for obj in objects {
                    let id = obj.id();
                    if seen_ids.insert(id) {
                        results.push(obj);
                    }
                }
            }
        }

        results
    }

    /// Get the combined bounding box of all objects in the grid.
    pub fn global_bounds(&self) -> Option<&Aabb> {
        self.global_bounds.as_ref()
    }

    /// Get the number of objects in the grid.
    pub fn len(&self) -> usize {
        self.all_objects.len()
    }

    /// Check if the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.all_objects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_ids() {
        let octree1 = Octree::new(64.0, 10);
        let octree2 = Octree::new(64.0, 10);

        let inst1 = OctreeInstance::new(octree1, Vec3::ZERO);
        let inst2 = OctreeInstance::new(octree2, Vec3::ZERO);

        assert_ne!(inst1.id(), inst2.id());
    }

    #[test]
    fn test_with_id() {
        let octree = Octree::new(64.0, 10);
        let inst = OctreeInstance::with_id(octree, Vec3::ZERO, 42);
        assert_eq!(inst.id(), 42);
    }

    #[test]
    fn test_world_bounds() {
        let octree = Octree::new(64.0, 10);
        let position = Vec3::new(100.0, 0.0, 50.0);
        let inst = OctreeInstance::new(octree, position);

        let bounds = inst.world_bounds();

        // Base at y=0, extends 64 units up
        assert_eq!(bounds.min.y, 0.0);
        assert_eq!(bounds.max.y, 64.0);

        // Centered at x=100, extends ±32
        assert_eq!(bounds.min.x, 68.0);
        assert_eq!(bounds.max.x, 132.0);

        // Centered at z=50, extends ±32
        assert_eq!(bounds.min.z, 18.0);
        assert_eq!(bounds.max.z, 82.0);
    }

    #[test]
    fn test_world_center() {
        let octree = Octree::new(64.0, 10);
        let position = Vec3::new(100.0, 0.0, 50.0);
        let inst = OctreeInstance::new(octree, position);

        let center = inst.world_center();

        // Center should be at position + half root size in Y
        assert_eq!(center, Vec3::new(100.0, 32.0, 50.0));
    }

    #[test]
    fn test_world_to_local() {
        let octree = Octree::new(64.0, 10);
        let position = Vec3::new(100.0, 10.0, 50.0);
        let inst = OctreeInstance::new(octree, position);

        // World position at the center of the octree
        let world_center = Vec3::new(100.0, 42.0, 50.0); // 10 + 32
        let local = inst.world_to_local(world_center);

        // Should map to local origin
        assert!((local.x).abs() < 0.001);
        assert!((local.y).abs() < 0.001);
        assert!((local.z).abs() < 0.001);
    }

    #[test]
    fn test_sample_empty_octree() {
        let octree = Octree::new(64.0, 10);
        let inst = OctreeInstance::new(octree, Vec3::ZERO);

        // Empty octree should return None everywhere
        assert_eq!(inst.sample_at(Vec3::ZERO), None);
        assert_eq!(inst.sample_at(Vec3::new(10.0, 10.0, 10.0)), None);
    }

    #[test]
    fn test_sample_out_of_bounds() {
        let octree = Octree::new(64.0, 10);
        let inst = OctreeInstance::new(octree, Vec3::ZERO);

        // Far outside bounds
        assert_eq!(inst.sample_at(Vec3::new(1000.0, 1000.0, 1000.0)), None);

        // Just outside bounds (octree extends from y=0 to y=64)
        assert_eq!(inst.sample_at(Vec3::new(0.0, -1.0, 0.0)), None);
        assert_eq!(inst.sample_at(Vec3::new(0.0, 65.0, 0.0)), None);
    }

    #[test]
    fn test_volumetric_trait_object() {
        let octree = Octree::new(64.0, 10);
        let inst = OctreeInstance::new(octree, Vec3::ZERO);

        // Should work through trait object
        let vol: &dyn VolumetricObject = &inst;
        assert_eq!(vol.sample_at(Vec3::ZERO), None);
        assert!(vol.id() > 0);

        let bounds = vol.world_bounds();
        assert_eq!(bounds.min.y, 0.0);
    }

    #[test]
    fn test_position_mutation() {
        let octree = Octree::new(64.0, 10);
        let mut inst = OctreeInstance::new(octree, Vec3::ZERO);

        assert_eq!(inst.position(), Vec3::ZERO);

        let new_pos = Vec3::new(100.0, 50.0, 200.0);
        inst.set_position(new_pos);

        assert_eq!(inst.position(), new_pos);

        // Bounds should update
        let bounds = inst.world_bounds();
        assert_eq!(bounds.min.y, 50.0);
        assert_eq!(bounds.max.y, 114.0);
    }
}
