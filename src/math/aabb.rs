//! Axis-aligned bounding box

use crate::core::types::Vec3;

/// Axis-aligned bounding box defined by min and max corners
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create AABB from min and max corners
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create AABB from center and half-extents
    pub fn from_center_half_extent(center: Vec3, half_extent: Vec3) -> Self {
        Self {
            min: center - half_extent,
            max: center + half_extent,
        }
    }

    /// Get center point
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get size (max - min)
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Get half-extents
    pub fn half_extent(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Check if point is inside AABB
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }

    /// Check if two AABBs intersect
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Expand AABB to include point
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Return merged AABB containing both
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Get child octant AABB for octree subdivision
    /// index: 0-7 representing xyz octant (bit 0=x, bit 1=y, bit 2=z)
    pub fn child_octant(&self, index: u8) -> Aabb {
        let center = self.center();
        let half = self.half_extent() * 0.5;

        let offset = Vec3::new(
            if index & 1 != 0 { half.x } else { -half.x },
            if index & 2 != 0 { half.y } else { -half.y },
            if index & 4 != 0 { half.z } else { -half.z },
        );

        Aabb::from_center_half_extent(center + offset, half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_accessors() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(aabb.center(), Vec3::splat(0.5));
        assert_eq!(aabb.size(), Vec3::ONE);
    }

    #[test]
    fn test_contains_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains_point(Vec3::splat(0.5)));
        assert!(!aabb.contains_point(Vec3::splat(2.0)));
    }

    #[test]
    fn test_intersects() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
        let c = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_child_octant() {
        let parent = Aabb::new(Vec3::ZERO, Vec3::splat(2.0));
        let child0 = parent.child_octant(0); // -x, -y, -z
        assert_eq!(child0.min, Vec3::ZERO);
        assert_eq!(child0.max, Vec3::ONE);
    }
}
