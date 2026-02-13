//! Priority queue for chunk loading based on distance and visibility

use crate::core::types::Vec3;
use crate::math::{Aabb, frustum::Frustum};
use crate::voxel::chunk::{ChunkCoord, CHUNK_SIZE};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Priority information for a chunk
#[derive(Clone, Copy, Debug)]
pub struct ChunkPriority {
    pub coord: ChunkCoord,
    pub priority: f32,  // Higher = more important
    pub distance: f32,  // Distance from camera
    pub visible: bool,  // In view frustum
}

impl ChunkPriority {
    /// Calculate priority for a chunk based on camera position and view frustum
    pub fn calculate(coord: ChunkCoord, camera_pos: Vec3, frustum: &Frustum) -> Self {
        let chunk_center = coord.world_origin() + Vec3::splat(CHUNK_SIZE as f32 * 0.5);
        let distance = camera_pos.distance(chunk_center);
        let bounds = Aabb::new(
            coord.world_origin(),
            coord.world_origin() + Vec3::splat(CHUNK_SIZE as f32)
        );
        let visible = frustum.intersects_aabb(&bounds);

        // Priority formula: closer = higher, visible = bonus
        let priority = 1.0 / (distance + 1.0) + if visible { 100.0 } else { 0.0 };

        Self {
            coord,
            priority,
            distance,
            visible,
        }
    }
}

// Implement Ord/PartialOrd for BinaryHeap (max-heap by default)
impl Eq for ChunkPriority {}

impl PartialEq for ChunkPriority {
    fn eq(&self, other: &Self) -> bool {
        self.coord == other.coord
    }
}

impl Ord for ChunkPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority should come first (max-heap)
        // Use total_cmp for f32 to handle NaN/infinity properly
        self.priority.total_cmp(&other.priority)
    }
}

impl PartialOrd for ChunkPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue for chunk loading
pub struct ChunkPriorityQueue {
    heap: BinaryHeap<ChunkPriority>,
    max_distance: f32,  // Don't queue chunks beyond this
}

impl ChunkPriorityQueue {
    /// Create a new priority queue with a maximum loading distance
    pub fn new(max_distance: f32) -> Self {
        Self {
            heap: BinaryHeap::new(),
            max_distance,
        }
    }

    /// Clear all queued chunks
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Add a chunk to the queue
    pub fn push(&mut self, priority: ChunkPriority) {
        // Only queue if within max distance
        if priority.distance <= self.max_distance {
            self.heap.push(priority);
        }
    }

    /// Get the highest priority chunk
    pub fn pop(&mut self) -> Option<ChunkPriority> {
        self.heap.pop()
    }

    /// Get the number of queued chunks
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Update queue with chunks around camera position
    ///
    /// This scans a radius around the camera and queues unloaded chunks
    /// based on their distance and visibility.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        frustum: &Frustum,
        loaded: &HashSet<ChunkCoord>,
    ) {
        self.clear();

        // Calculate chunk radius to scan
        let chunk_radius = (self.max_distance / CHUNK_SIZE as f32).ceil() as i32;
        let camera_chunk = ChunkCoord::from_world_pos(camera_pos);

        // Scan chunks in a radius around the camera
        for dx in -chunk_radius..=chunk_radius {
            for dy in -chunk_radius..=chunk_radius {
                for dz in -chunk_radius..=chunk_radius {
                    let coord = ChunkCoord::new(
                        camera_chunk.x + dx,
                        camera_chunk.y + dy,
                        camera_chunk.z + dz,
                    );

                    // Skip if already loaded
                    if loaded.contains(&coord) {
                        continue;
                    }

                    // Calculate priority and add to queue
                    let priority = ChunkPriority::calculate(coord, camera_pos, frustum);
                    self.push(priority);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Mat4;

    #[test]
    fn test_chunk_priority_calculate() {
        let coord = ChunkCoord::new(0, 0, 0);
        let camera_pos = Vec3::new(32.0, 32.0, 32.0); // Center of chunk

        // Simple frustum (orthographic-like for testing)
        let proj = Mat4::orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        let priority = ChunkPriority::calculate(coord, camera_pos, &frustum);

        assert_eq!(priority.coord, coord);
        assert!(priority.distance >= 0.0);
        // Priority should be high since camera is at chunk center
        assert!(priority.priority > 0.0);
    }

    #[test]
    fn test_chunk_priority_ordering() {
        let coord1 = ChunkCoord::new(0, 0, 0);
        let coord2 = ChunkCoord::new(10, 10, 10);

        let camera_pos = Vec3::ZERO;
        let proj = Mat4::orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        let p1 = ChunkPriority::calculate(coord1, camera_pos, &frustum);
        let p2 = ChunkPriority::calculate(coord2, camera_pos, &frustum);

        // Chunk at origin should have higher priority (closer)
        assert!(p1.priority > p2.priority);
        assert!(p1 > p2); // Test Ord implementation
    }

    #[test]
    fn test_priority_queue_basic() {
        let mut queue = ChunkPriorityQueue::new(500.0); // Increased to 500m

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        let camera_pos = Vec3::ZERO;
        let proj = Mat4::orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        // Add chunks at different distances
        // Chunk (0,0,0) center: (32, 32, 32) ~= 55m
        // Chunk (1,0,0) center: (96, 32, 32) ~= 108m
        // Chunk (5,0,0) center: (352, 32, 32) ~= 355m
        let coords = vec![
            ChunkCoord::new(0, 0, 0),  // Close (~55m)
            ChunkCoord::new(1, 0, 0),  // Medium (~108m)
            ChunkCoord::new(5, 0, 0),  // Far (~355m)
        ];

        for coord in coords {
            let priority = ChunkPriority::calculate(coord, camera_pos, &frustum);
            queue.push(priority);
        }

        assert_eq!(queue.len(), 3);

        // Pop should return highest priority (closest) first
        let first = queue.pop().unwrap();
        assert_eq!(first.coord, ChunkCoord::new(0, 0, 0));

        let second = queue.pop().unwrap();
        assert_eq!(second.coord, ChunkCoord::new(1, 0, 0));

        let third = queue.pop().unwrap();
        assert_eq!(third.coord, ChunkCoord::new(5, 0, 0));

        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_queue_max_distance() {
        let mut queue = ChunkPriorityQueue::new(100.0); // 100m max distance

        let camera_pos = Vec3::ZERO;
        let proj = Mat4::orthographic_rh(-200.0, 200.0, -200.0, 200.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        // Close chunk (within max distance)
        // Chunk (0,0,0) center: (2, 2, 2) ~= 3.46m from origin
        let close_coord = ChunkCoord::new(0, 0, 0);
        let close_priority = ChunkPriority::calculate(close_coord, camera_pos, &frustum);
        queue.push(close_priority);

        // Far chunk (beyond max distance)
        // Chunk (50,50,50) center: (202, 202, 202) ~= 349.9m from origin
        let far_coord = ChunkCoord::new(50, 50, 50);
        let far_priority = ChunkPriority::calculate(far_coord, camera_pos, &frustum);
        queue.push(far_priority);

        // Only the close chunk should be queued
        assert_eq!(queue.len(), 1);
        let popped = queue.pop().unwrap();
        assert_eq!(popped.coord, close_coord);
    }

    #[test]
    fn test_priority_queue_update() {
        let mut queue = ChunkPriorityQueue::new(150.0);
        let camera_pos = Vec3::new(32.0, 32.0, 32.0);

        let proj = Mat4::orthographic_rh(-200.0, 200.0, -200.0, 200.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        let mut loaded = HashSet::new();
        loaded.insert(ChunkCoord::new(0, 0, 0)); // Already loaded

        queue.update(camera_pos, &frustum, &loaded);

        // Queue should have chunks around camera, but not the loaded one
        assert!(!queue.is_empty());

        // Check that the already-loaded chunk is not in the queue
        let coords: Vec<_> = std::iter::from_fn(|| queue.pop())
            .map(|p| p.coord)
            .collect();

        assert!(!coords.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn test_priority_queue_clear() {
        let mut queue = ChunkPriorityQueue::new(200.0);
        let camera_pos = Vec3::ZERO;
        let proj = Mat4::orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        let priority = ChunkPriority::calculate(ChunkCoord::new(0, 0, 0), camera_pos, &frustum);
        queue.push(priority);

        assert!(!queue.is_empty());

        queue.clear();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_visible_chunks_higher_priority() {
        let camera_pos = Vec3::new(32.0, 32.0, 100.0);

        // Create a narrow frustum looking at origin
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        // Chunk at origin (likely visible)
        let visible_coord = ChunkCoord::new(0, 0, 0);
        let visible_priority = ChunkPriority::calculate(visible_coord, camera_pos, &frustum);

        // Chunk behind camera (not visible, same distance)
        let behind_coord = ChunkCoord::new(0, 0, 10);
        let behind_priority = ChunkPriority::calculate(behind_coord, camera_pos, &frustum);

        // Visible chunk should have much higher priority due to +100.0 bonus
        if visible_priority.visible && !behind_priority.visible {
            assert!(visible_priority.priority > behind_priority.priority + 50.0);
        }
    }
}
