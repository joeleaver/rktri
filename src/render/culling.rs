//! CPU-side frustum culling and front-to-back chunk sorting

use glam::{Vec3, Vec4, Mat4};
use crate::render::buffer::octree_buffer::{GpuChunkInfo, MAX_CHUNKS};

/// A frustum plane in Hessian normal form (normal.xyz, distance)
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    /// Signed distance from point to plane (positive = in front)
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// 6-plane frustum extracted from a view-projection matrix
pub struct Frustum {
    pub planes: [Plane; 6], // left, right, bottom, top, near, far
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix.
    /// Uses the Gribb/Hartmann method.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        // Extract rows from the VP matrix (column-major storage)
        let rows = [
            Vec4::new(vp.col(0).x, vp.col(1).x, vp.col(2).x, vp.col(3).x),
            Vec4::new(vp.col(0).y, vp.col(1).y, vp.col(2).y, vp.col(3).y),
            Vec4::new(vp.col(0).z, vp.col(1).z, vp.col(2).z, vp.col(3).z),
            Vec4::new(vp.col(0).w, vp.col(1).w, vp.col(2).w, vp.col(3).w),
        ];

        let mut planes = [Plane { normal: Vec3::ZERO, d: 0.0 }; 6];

        // Left:   row3 + row0
        // Right:  row3 - row0
        // Bottom: row3 + row1
        // Top:    row3 - row1
        // Near:   row3 + row2
        // Far:    row3 - row2
        let raw = [
            rows[3] + rows[0], // left
            rows[3] - rows[0], // right
            rows[3] + rows[1], // bottom
            rows[3] - rows[1], // top
            rows[3] + rows[2], // near
            rows[3] - rows[2], // far
        ];

        for (i, r) in raw.iter().enumerate() {
            let len = Vec3::new(r.x, r.y, r.z).length();
            if len > 0.0 {
                planes[i] = Plane {
                    normal: Vec3::new(r.x, r.y, r.z) / len,
                    d: r.w / len,
                };
            }
        }

        Self { planes }
    }

    /// Test if an AABB intersects the frustum.
    /// Returns true if the AABB is at least partially inside.
    pub fn test_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Find the positive vertex (most in the direction of the plane normal)
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { max.x } else { min.x },
                if plane.normal.y >= 0.0 { max.y } else { min.y },
                if plane.normal.z >= 0.0 { max.z } else { min.z },
            );

            // If the positive vertex is behind the plane, AABB is fully outside
            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Stores the full chunk list and provides per-frame culling + front-to-back sorting.
///
/// Re-uses allocations across frames to avoid per-frame heap churn.
pub struct ChunkCuller {
    /// All chunk infos (mutable via add_chunk/replace_all)
    all_chunks: Vec<GpuChunkInfo>,
    /// Visible chunks after culling and sorting (reused each frame)
    visible_chunks: Vec<GpuChunkInfo>,
    /// Sort keys: (distance_squared, original_index)
    sort_keys: Vec<(f32, usize)>,
}

impl ChunkCuller {
    pub fn new(chunks: Vec<GpuChunkInfo>) -> Self {
        let cap = chunks.len();
        Self {
            all_chunks: chunks,
            visible_chunks: Vec::with_capacity(cap),
            sort_keys: Vec::with_capacity(cap),
        }
    }

    /// Cull chunks against the frustum and sort visible ones front-to-back.
    ///
    /// Returns a slice of visible GpuChunkInfo entries (valid until next call)
    /// and the count of visible chunks.
    pub fn cull_and_sort(&mut self, frustum: &Frustum, camera_pos: Vec3) -> (&[GpuChunkInfo], u32) {
        self.sort_keys.clear();

        for (idx, chunk) in self.all_chunks.iter().enumerate() {
            let chunk_min = Vec3::from_array(chunk.world_min);
            let chunk_max = chunk_min + Vec3::splat(chunk.root_size);

            if frustum.test_aabb(chunk_min, chunk_max) {
                let center = (chunk_min + chunk_max) * 0.5;
                let dist_sq = camera_pos.distance_squared(center);
                self.sort_keys.push((dist_sq, idx));
            }
        }

        // Sort front-to-back by squared distance
        self.sort_keys
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        self.visible_chunks.clear();
        for &(_, idx) in &self.sort_keys {
            if self.visible_chunks.len() >= MAX_CHUNKS as usize {
                break; // GPU buffer limit
            }
            self.visible_chunks.push(self.all_chunks[idx]);
        }

        let count = self.visible_chunks.len() as u32;
        (&self.visible_chunks, count)
    }

    /// Total number of chunks (before culling)
    pub fn total_chunks(&self) -> u32 {
        self.all_chunks.len() as u32
    }

    /// Append a single chunk (for incremental loading).
    pub fn add_chunk(&mut self, info: GpuChunkInfo) {
        self.all_chunks.push(info);
    }

    /// Replace the entire chunk list (for full scene graph rebuilds).
    pub fn replace_all(&mut self, infos: Vec<GpuChunkInfo>) {
        self.all_chunks = infos;
        // Resize scratch buffers to match new capacity
        self.visible_chunks = Vec::with_capacity(self.all_chunks.len());
        self.sort_keys = Vec::with_capacity(self.all_chunks.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustum_extraction_identity() {
        // A simple perspective projection looking down -Z
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        // All 6 planes should be non-zero
        for plane in &frustum.planes {
            assert!(plane.normal.length() > 0.9, "Plane normal should be normalized");
        }
    }

    #[test]
    fn test_aabb_inside_frustum() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        // A box directly in front of the camera should be visible
        let visible = frustum.test_aabb(
            Vec3::new(-1.0, -1.0, -10.0),
            Vec3::new(1.0, 1.0, -5.0),
        );
        assert!(visible, "Box in front of camera should be visible");
    }

    #[test]
    fn test_aabb_behind_frustum() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        // A box behind the camera should be culled
        let visible = frustum.test_aabb(
            Vec3::new(-1.0, -1.0, 5.0),
            Vec3::new(1.0, 1.0, 10.0),
        );
        assert!(!visible, "Box behind camera should be culled");
    }

    #[test]
    fn test_aabb_far_outside() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        // A box way to the left should be culled
        let visible = frustum.test_aabb(
            Vec3::new(-1000.0, -1.0, -10.0),
            Vec3::new(-999.0, 1.0, -5.0),
        );
        assert!(!visible, "Box far to the left should be culled");
    }

    #[test]
    fn test_aabb_beyond_far_plane() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        // A box beyond the far plane should be culled
        let visible = frustum.test_aabb(
            Vec3::new(-1.0, -1.0, -200.0),
            Vec3::new(1.0, 1.0, -150.0),
        );
        assert!(!visible, "Box beyond far plane should be culled");
    }

    #[test]
    fn test_chunk_culler_add_chunk() {
        let chunks = vec![
            GpuChunkInfo {
                world_min: [0.0, 0.0, -10.0],
                root_size: 8.0,
                root_node: 0,
                max_depth: 5,
                layer_id: 0,
                flags: 0,
            },
        ];
        let mut culler = ChunkCuller::new(chunks);
        assert_eq!(culler.total_chunks(), 1);

        culler.add_chunk(GpuChunkInfo {
            world_min: [16.0, 0.0, -10.0],
            root_size: 8.0,
            root_node: 100,
            max_depth: 5,
            layer_id: 1,
            flags: 0,
        });
        assert_eq!(culler.total_chunks(), 2);
    }

    #[test]
    fn test_chunk_culler_replace_all() {
        let chunks = vec![
            GpuChunkInfo {
                world_min: [0.0, 0.0, -10.0],
                root_size: 8.0,
                root_node: 0,
                max_depth: 5,
                layer_id: 0,
                flags: 0,
            },
        ];
        let mut culler = ChunkCuller::new(chunks);
        assert_eq!(culler.total_chunks(), 1);

        let new_chunks = vec![
            GpuChunkInfo {
                world_min: [0.0, 0.0, -20.0],
                root_size: 8.0,
                root_node: 200,
                max_depth: 5,
                layer_id: 0,
                flags: 0,
            },
            GpuChunkInfo {
                world_min: [16.0, 0.0, -20.0],
                root_size: 8.0,
                root_node: 300,
                max_depth: 5,
                layer_id: 1,
                flags: 0,
            },
        ];
        culler.replace_all(new_chunks);
        assert_eq!(culler.total_chunks(), 2);
    }

    #[test]
    fn test_chunk_culler_sort_order() {
        use crate::render::buffer::octree_buffer::GpuChunkInfo;

        // Both chunks in front of camera, one closer than the other
        let chunks = vec![
            GpuChunkInfo {
                world_min: [0.0, 0.0, -50.0],
                root_size: 8.0,
                root_node: 0,
                max_depth: 5,
                layer_id: 0,
                flags: 0,
            },
            GpuChunkInfo {
                world_min: [0.0, 0.0, -20.0],
                root_size: 8.0,
                root_node: 100,
                max_depth: 5,
                layer_id: 0,
                flags: 0,
            },
        ];

        let mut culler = ChunkCuller::new(chunks);

        // Camera at origin looking down -Z
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        let camera_pos = Vec3::ZERO;
        let (visible, count) = culler.cull_and_sort(&frustum, camera_pos);

        assert_eq!(count, 2);
        // The chunk at z=-20 is closer, should come first
        assert_eq!(visible[0].root_node, 100, "Closer chunk should be first");
        assert_eq!(visible[1].root_node, 0, "Farther chunk should be second");
    }
}
