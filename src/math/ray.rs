//! Ray type and operations

use crate::core::types::{Vec3, Mat4};
use super::aabb::Aabb;

/// A ray defined by origin and direction
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    /// Precomputed 1/direction for fast AABB intersection
    pub inv_direction: Vec3,
}

impl Ray {
    /// Create a new ray (direction should be normalized)
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction,
            inv_direction: Vec3::new(
                1.0 / direction.x,
                1.0 / direction.y,
                1.0 / direction.z,
            ),
        }
    }

    /// Get point along ray at parameter t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Ray-AABB intersection using slab method
    /// Returns Some((t_near, t_far)) if intersection, None otherwise
    pub fn intersects_aabb(&self, aabb: &Aabb) -> Option<(f32, f32)> {
        let t1 = (aabb.min - self.origin) * self.inv_direction;
        let t2 = (aabb.max - self.origin) * self.inv_direction;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t_near = t_min.x.max(t_min.y).max(t_min.z);
        let t_far = t_max.x.min(t_max.y).min(t_max.z);

        if t_near <= t_far && t_far >= 0.0 {
            Some((t_near.max(0.0), t_far))
        } else {
            None
        }
    }

    /// Transform ray by matrix
    pub fn transform(&self, matrix: &Mat4) -> Ray {
        let new_origin = matrix.transform_point3(self.origin);
        let new_direction = matrix.transform_vector3(self.direction).normalize();
        Ray::new(new_origin, new_direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at() {
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert_eq!(ray.at(5.0), Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn test_intersects_aabb_hit() {
        let ray = Ray::new(Vec3::new(-2.0, 0.5, 0.5), Vec3::X);
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let hit = ray.intersects_aabb(&aabb);
        assert!(hit.is_some());
        let (t_near, t_far) = hit.unwrap();
        assert!((t_near - 2.0).abs() < 0.001);
        assert!((t_far - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_intersects_aabb_miss() {
        let ray = Ray::new(Vec3::new(-2.0, 5.0, 0.5), Vec3::X);
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(ray.intersects_aabb(&aabb).is_none());
    }

    #[test]
    fn test_intersects_aabb_inside() {
        let ray = Ray::new(Vec3::splat(0.5), Vec3::X);
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let hit = ray.intersects_aabb(&aabb);
        assert!(hit.is_some());
        let (t_near, _) = hit.unwrap();
        assert_eq!(t_near, 0.0); // Inside, so t_near clamped to 0
    }
}
