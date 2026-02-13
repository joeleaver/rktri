//! Brush stroke representation

use glam::{Mat4, Vec3};
use crate::math::Aabb;
use crate::voxel::voxel::Voxel;
use super::primitive::{BrushPrimitive, Axis};

/// How brush strokes combine with existing voxels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Replace,  // Overwrite existing voxels
    Add,      // Only fill empty space
    Subtract, // Remove voxels (carving)
}

/// A single brush stroke in world space
#[derive(Debug, Clone)]
pub struct BrushStroke {
    /// The primitive shape
    pub primitive: BrushPrimitive,
    /// World-space transform (position, rotation, scale)
    pub transform: Mat4,
    /// Cached inverse transform for point queries
    pub inverse_transform: Mat4,
    /// Voxel to paint (color + material)
    pub voxel: Voxel,
    /// Target octree level (0 = root/coarsest, higher = finer)
    pub target_level: u8,
    /// How this stroke combines with existing voxels
    pub blend_mode: BlendMode,
    /// Cached world-space AABB
    pub world_bounds: Aabb,
}

impl BrushStroke {
    /// Create a new brush stroke with identity transform
    pub fn new(primitive: BrushPrimitive, voxel: Voxel, target_level: u8) -> Self {
        let transform = Mat4::IDENTITY;
        let inverse_transform = Mat4::IDENTITY;
        let world_bounds = primitive.local_bounds();

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Create a sphere stroke
    pub fn sphere(center: Vec3, radius: f32, voxel: Voxel, target_level: u8) -> Self {
        let primitive = BrushPrimitive::Sphere { radius };
        let transform = Mat4::from_translation(center);
        let inverse_transform = Mat4::from_translation(-center);
        let world_bounds = primitive.world_bounds(&transform);

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Create a box stroke
    pub fn box_stroke(center: Vec3, half_extents: Vec3, voxel: Voxel, target_level: u8) -> Self {
        let primitive = BrushPrimitive::Box { half_extents };
        let transform = Mat4::from_translation(center);
        let inverse_transform = Mat4::from_translation(-center);
        let world_bounds = primitive.world_bounds(&transform);

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Create a capsule stroke from start to end points
    pub fn capsule(start: Vec3, end: Vec3, radius: f32, voxel: Voxel, target_level: u8) -> Self {
        let center = (start + end) * 0.5;
        let diff = end - start;
        let length = diff.length();
        let half_height = length * 0.5;

        // Determine primary axis and rotation
        let (axis, transform) = if length < 0.0001 {
            // Degenerate: just a sphere
            (Axis::Y, Mat4::from_translation(center))
        } else {
            let dir = diff / length;
            // Default capsule is Y-axis aligned, compute rotation to target direction
            let rotation = if dir.y.abs() > 0.999 {
                // Nearly vertical, use Y axis
                if dir.y > 0.0 {
                    Mat4::IDENTITY
                } else {
                    Mat4::from_rotation_x(std::f32::consts::PI)
                }
            } else {
                // Compute rotation from Y to dir
                let axis = Vec3::Y.cross(dir).normalize();
                let angle = Vec3::Y.dot(dir).acos();
                Mat4::from_axis_angle(axis, angle)
            };
            (Axis::Y, Mat4::from_translation(center) * rotation)
        };

        let primitive = BrushPrimitive::Capsule { radius, half_height, axis };
        let inverse_transform = transform.inverse();
        let world_bounds = primitive.world_bounds(&transform);

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Create a cylinder stroke
    pub fn cylinder(center: Vec3, axis: Axis, half_height: f32, radius: f32, voxel: Voxel, target_level: u8) -> Self {
        let primitive = BrushPrimitive::Cylinder { radius, half_height, axis };
        let transform = Mat4::from_translation(center);
        let inverse_transform = Mat4::from_translation(-center);
        let world_bounds = primitive.world_bounds(&transform);

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Create a cloud stroke (stochastic fill within sphere)
    pub fn cloud(center: Vec3, radius: f32, density: f32, seed: u32, voxel: Voxel, target_level: u8) -> Self {
        let primitive = BrushPrimitive::Cloud { radius, density, seed };
        let transform = Mat4::from_translation(center);
        let inverse_transform = Mat4::from_translation(-center);
        let world_bounds = primitive.world_bounds(&transform);

        Self {
            primitive,
            transform,
            inverse_transform,
            voxel,
            target_level,
            blend_mode: BlendMode::Replace,
            world_bounds,
        }
    }

    /// Set blend mode (builder pattern)
    pub fn with_blend(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Check if world-space point is inside this stroke
    pub fn contains_point(&self, world_point: Vec3) -> bool {
        let local_point = self.inverse_transform.transform_point3(world_point);
        self.primitive.contains_point(local_point)
    }

    /// Get signed distance from world-space point to stroke surface
    pub fn sdf(&self, world_point: Vec3) -> f32 {
        let local_point = self.inverse_transform.transform_point3(world_point);
        self.primitive.sdf(local_point)
    }

    /// Check if stroke's AABB intersects the given AABB
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        self.world_bounds.intersects(aabb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_stroke() {
        let voxel = Voxel::new(255, 0, 0, 1);
        let stroke = BrushStroke::sphere(Vec3::new(5.0, 5.0, 5.0), 2.0, voxel, 5);

        assert!(stroke.contains_point(Vec3::new(5.0, 5.0, 5.0))); // Center
        assert!(stroke.contains_point(Vec3::new(6.0, 5.0, 5.0))); // Inside
        assert!(!stroke.contains_point(Vec3::new(8.0, 5.0, 5.0))); // Outside
    }

    #[test]
    fn test_capsule_stroke() {
        let voxel = Voxel::new(139, 90, 43, 2); // Brown bark
        let stroke = BrushStroke::capsule(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            0.5,
            voxel,
            4
        );

        assert!(stroke.contains_point(Vec3::new(0.0, 1.0, 0.0))); // Center
        assert!(stroke.contains_point(Vec3::new(0.3, 1.0, 0.0))); // Inside radius
        assert!(!stroke.contains_point(Vec3::new(1.0, 1.0, 0.0))); // Outside
    }

    #[test]
    fn test_blend_mode() {
        let voxel = Voxel::new(0, 255, 0, 1);
        let stroke = BrushStroke::sphere(Vec3::ZERO, 1.0, voxel, 5)
            .with_blend(BlendMode::Add);

        assert_eq!(stroke.blend_mode, BlendMode::Add);
    }

    #[test]
    fn test_aabb_intersection() {
        let voxel = Voxel::new(255, 255, 255, 1);
        let stroke = BrushStroke::sphere(Vec3::new(5.0, 5.0, 5.0), 1.0, voxel, 5);

        let intersecting = Aabb::new(Vec3::new(4.0, 4.0, 4.0), Vec3::new(6.0, 6.0, 6.0));
        let not_intersecting = Aabb::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(12.0, 12.0, 12.0));

        assert!(stroke.intersects_aabb(&intersecting));
        assert!(!stroke.intersects_aabb(&not_intersecting));
    }
}
