use glam::{Vec3, Mat4};
use crate::math::Aabb;

/// Axis for oriented primitives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Brush primitive shapes with SDF evaluation
#[derive(Debug, Clone, Copy)]
pub enum BrushPrimitive {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, half_height: f32, axis: Axis },
    Cylinder { radius: f32, half_height: f32, axis: Axis },
    Cloud { radius: f32, density: f32, seed: u32 },
}

impl Axis {
    /// Get unit vector for this axis
    pub fn to_vec3(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
}

/// Deterministic 3D hash for cloud primitive
fn hash_3d(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= x as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h ^= y as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h ^= z as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h
}

impl BrushPrimitive {
    /// Signed distance from point to surface (negative = inside)
    /// Point is in local space (primitive centered at origin)
    pub fn sdf(&self, local_point: Vec3) -> f32 {
        match self {
            BrushPrimitive::Sphere { radius } => {
                local_point.length() - radius
            }
            BrushPrimitive::Box { half_extents } => {
                let q = local_point.abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }
            BrushPrimitive::Capsule { radius, half_height, axis } => {
                // Project point onto axis, clamp to segment
                let axis_vec = axis.to_vec3();
                let t = local_point.dot(axis_vec).clamp(-*half_height, *half_height);
                let closest = axis_vec * t;
                (local_point - closest).length() - radius
            }
            BrushPrimitive::Cylinder { radius, half_height, axis } => {
                // Infinite cylinder distance in 2D, then combine with height
                let axis_vec = axis.to_vec3();
                let h = local_point.dot(axis_vec);
                let radial = local_point - axis_vec * h;
                let d_radial = radial.length() - radius;
                let d_height = h.abs() - half_height;
                // Outside: max of both distances
                // Inside: max of negative distances
                let outside = d_radial.max(d_height).max(0.0);
                let inside = d_radial.max(d_height).min(0.0);
                outside + inside
            }
            BrushPrimitive::Cloud { radius, .. } => {
                local_point.length() - radius
            }
        }
    }

    /// Check if point is inside primitive (SDF <= 0)
    pub fn contains_point(&self, local_point: Vec3) -> bool {
        match self {
            BrushPrimitive::Cloud { radius, density, seed } => {
                let dist = local_point.length();
                if dist > *radius {
                    return false;
                }

                // Distance-based density falloff (denser in center, sparse at edges)
                let normalized_dist = dist / radius;
                let falloff = 1.0 - normalized_dist * normalized_dist; // Quadratic falloff
                let local_density = density * falloff;

                // Hash the position to get deterministic per-voxel randomness
                // Quantize to a fine grid to ensure voxel-level variation
                let grid_size = *radius * 0.08; // ~12 grid cells across diameter
                let gx = (local_point.x / grid_size).floor() as i32;
                let gy = (local_point.y / grid_size).floor() as i32;
                let gz = (local_point.z / grid_size).floor() as i32;

                let hash = hash_3d(gx, gy, gz, *seed);
                let hash_float = (hash & 0xFFFF) as f32 / 65535.0;

                hash_float < local_density
            }
            _ => self.sdf(local_point) <= 0.0,
        }
    }

    /// Get local-space AABB (primitive centered at origin)
    pub fn local_bounds(&self) -> Aabb {
        match self {
            BrushPrimitive::Sphere { radius } => {
                Aabb::from_center_half_extent(Vec3::ZERO, Vec3::splat(*radius))
            }
            BrushPrimitive::Box { half_extents } => {
                Aabb::from_center_half_extent(Vec3::ZERO, *half_extents)
            }
            BrushPrimitive::Capsule { radius, half_height, axis } => {
                let axis_extent = axis.to_vec3() * (*half_height + *radius);
                let radial_extent = Vec3::splat(*radius);
                // Combine axis extent with radial extent
                let half = axis_extent.abs() + radial_extent;
                Aabb::from_center_half_extent(Vec3::ZERO, half)
            }
            BrushPrimitive::Cylinder { radius, half_height, axis } => {
                let axis_extent = axis.to_vec3() * *half_height;
                let radial_extent = Vec3::splat(*radius);
                let half = axis_extent.abs() + radial_extent;
                Aabb::from_center_half_extent(Vec3::ZERO, half)
            }
            BrushPrimitive::Cloud { radius, .. } => {
                Aabb::from_center_half_extent(Vec3::ZERO, Vec3::splat(*radius))
            }
        }
    }

    /// Get world-space AABB given a transform
    /// Note: This is conservative (may be larger than tight bound for rotated shapes)
    pub fn world_bounds(&self, transform: &Mat4) -> Aabb {
        let local = self.local_bounds();
        // Transform all 8 corners and compute bounding box
        let corners = [
            Vec3::new(local.min.x, local.min.y, local.min.z),
            Vec3::new(local.max.x, local.min.y, local.min.z),
            Vec3::new(local.min.x, local.max.y, local.min.z),
            Vec3::new(local.max.x, local.max.y, local.min.z),
            Vec3::new(local.min.x, local.min.y, local.max.z),
            Vec3::new(local.max.x, local.min.y, local.max.z),
            Vec3::new(local.min.x, local.max.y, local.max.z),
            Vec3::new(local.max.x, local.max.y, local.max.z),
        ];

        let first = transform.transform_point3(corners[0]);
        let mut result = Aabb::new(first, first);
        for corner in &corners[1..] {
            let transformed = transform.transform_point3(*corner);
            result.expand(transformed);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sdf() {
        let sphere = BrushPrimitive::Sphere { radius: 1.0 };
        assert!(sphere.sdf(Vec3::ZERO) < 0.0); // Inside
        assert!((sphere.sdf(Vec3::X) - 0.0).abs() < 0.001); // On surface
        assert!(sphere.sdf(Vec3::X * 2.0) > 0.0); // Outside
    }

    #[test]
    fn test_box_sdf() {
        let box_prim = BrushPrimitive::Box { half_extents: Vec3::ONE };
        assert!(box_prim.sdf(Vec3::ZERO) < 0.0); // Inside
        assert!(box_prim.sdf(Vec3::splat(2.0)) > 0.0); // Outside
    }

    #[test]
    fn test_capsule_sdf() {
        let capsule = BrushPrimitive::Capsule { radius: 0.5, half_height: 1.0, axis: Axis::Y };
        assert!(capsule.sdf(Vec3::ZERO) < 0.0); // Inside
        assert!(capsule.sdf(Vec3::Y * 1.3) < 0.0); // Inside cap
        assert!(capsule.sdf(Vec3::Y * 2.0) > 0.0); // Outside
    }

    #[test]
    fn test_cylinder_sdf() {
        let cylinder = BrushPrimitive::Cylinder { radius: 1.0, half_height: 1.0, axis: Axis::Y };
        assert!(cylinder.sdf(Vec3::ZERO) < 0.0); // Inside
        assert!(cylinder.sdf(Vec3::X * 2.0) > 0.0); // Outside radially
        assert!(cylinder.sdf(Vec3::Y * 2.0) > 0.0); // Outside axially
    }

    #[test]
    fn test_sphere_bounds() {
        let sphere = BrushPrimitive::Sphere { radius: 2.0 };
        let bounds = sphere.local_bounds();
        assert_eq!(bounds.min, Vec3::splat(-2.0));
        assert_eq!(bounds.max, Vec3::splat(2.0));
    }

    #[test]
    fn test_contains_point() {
        let sphere = BrushPrimitive::Sphere { radius: 1.0 };
        assert!(sphere.contains_point(Vec3::ZERO));
        assert!(sphere.contains_point(Vec3::X * 0.9));
        assert!(!sphere.contains_point(Vec3::X * 1.1));
    }
}
