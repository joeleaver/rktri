//! View frustum for culling

use crate::core::types::{Vec3, Vec4, Mat4};
use super::aabb::Aabb;

/// A plane defined by normal and distance from origin
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    pub fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Signed distance from point to plane (positive = in front)
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
}

/// View frustum with 6 planes (Near, Far, Left, Right, Top, Bottom)
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extract frustum planes from view-projection matrix
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = vp.to_cols_array_2d();

        // Extract and normalize planes
        // Left: row3 + row0
        let left = Self::normalize_plane(Vec4::new(
            m[0][3] + m[0][0],
            m[1][3] + m[1][0],
            m[2][3] + m[2][0],
            m[3][3] + m[3][0],
        ));

        // Right: row3 - row0
        let right = Self::normalize_plane(Vec4::new(
            m[0][3] - m[0][0],
            m[1][3] - m[1][0],
            m[2][3] - m[2][0],
            m[3][3] - m[3][0],
        ));

        // Bottom: row3 + row1
        let bottom = Self::normalize_plane(Vec4::new(
            m[0][3] + m[0][1],
            m[1][3] + m[1][1],
            m[2][3] + m[2][1],
            m[3][3] + m[3][1],
        ));

        // Top: row3 - row1
        let top = Self::normalize_plane(Vec4::new(
            m[0][3] - m[0][1],
            m[1][3] - m[1][1],
            m[2][3] - m[2][1],
            m[3][3] - m[3][1],
        ));

        // Near: row3 + row2
        let near = Self::normalize_plane(Vec4::new(
            m[0][3] + m[0][2],
            m[1][3] + m[1][2],
            m[2][3] + m[2][2],
            m[3][3] + m[3][2],
        ));

        // Far: row3 - row2
        let far = Self::normalize_plane(Vec4::new(
            m[0][3] - m[0][2],
            m[1][3] - m[1][2],
            m[2][3] - m[2][2],
            m[3][3] - m[3][2],
        ));

        Self {
            planes: [near, far, left, right, top, bottom],
        }
    }

    fn normalize_plane(plane: Vec4) -> Plane {
        let normal = Vec3::new(plane.x, plane.y, plane.z);
        let len = normal.length();
        Plane {
            normal: normal / len,
            distance: plane.w / len,
        }
    }

    /// Check if point is inside frustum
    pub fn contains_point(&self, point: Vec3) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(point) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Check if AABB intersects frustum (conservative test)
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            // Find the corner most aligned with plane normal (p-vertex)
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );

            // If p-vertex is outside, AABB is completely outside
            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_distance() {
        let plane = Plane::new(Vec3::Y, 0.0); // XZ plane
        assert_eq!(plane.distance_to_point(Vec3::new(0.0, 5.0, 0.0)), 5.0);
        assert_eq!(plane.distance_to_point(Vec3::new(0.0, -3.0, 0.0)), -3.0);
    }

    #[test]
    fn test_frustum_contains_point() {
        // Simple orthographic-like projection for testing
        let proj = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&(proj * view));

        // Point at origin should be visible
        assert!(frustum.contains_point(Vec3::ZERO));
    }
}
