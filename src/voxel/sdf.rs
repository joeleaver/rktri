//! Shared SDF (Signed Distance Field) utilities for procedural voxel geometry.
//!
//! This module provides:
//! - Trait-based SDF primitives (sphere, capsule, plane, etc.)
//! - Distance encoding/decoding for voxel storage
//! - Gradient normal encoding (for smooth shading)
//!
//! The workflow is:
//! 1. Generate geometry using SDF primitives
//! 2. Compute gradient at each voxel surface
//! 3. Store encoded gradient in voxel color field
//! 4. Shader decodes gradient to get smooth normal

use glam::Vec3;

/// Maximum distance that can be stored (in meters)
pub const MAX_SDF_DISTANCE: f32 = 32.0;

/// Encode a distance into u16 for voxel storage.
/// Distances are clamped to [0, MAX_SDF_DISTANCE] and mapped to [0, 65535].
#[inline]
pub fn encode_distance(d: f32) -> u16 {
    let d = d.clamp(0.0, MAX_SDF_DISTANCE);
    (d / MAX_SDF_DISTANCE * 65535.0) as u16
}

/// Decode a distance from u16 voxel storage.
#[inline]
pub fn decode_distance(encoded: u16) -> f32 {
    (encoded as f32 / 65535.0) * MAX_SDF_DISTANCE
}

// =============================================================================
// Gradient Normal Encoding (for heightfield terrain)
// =============================================================================

/// Encode a gradient (dh/dx, dh/dz) into u16 for voxel storage.
/// Each component is mapped from [-GRADIENT_RANGE, +GRADIENT_RANGE] → [0, 255].
pub const GRADIENT_RANGE: f32 = 4.0; // max slope = 4

/// Encode gradient into color field: dx maps to low byte, dz to high byte.
#[inline]
pub fn encode_gradient(dh_dx: f32, dh_dz: f32) -> u16 {
    let dx_enc = (((dh_dx + GRADIENT_RANGE) / (2.0 * GRADIENT_RANGE)).clamp(0.0, 1.0) * 255.0) as u8;
    let dz_enc = (((dh_dz + GRADIENT_RANGE) / (2.0 * GRADIENT_RANGE)).clamp(0.0, 1.0) * 255.0) as u8;
    (dz_enc as u16) << 8 | dx_enc as u16
}

/// Decode gradient from voxel color field.
#[inline]
pub fn decode_gradient(encoded: u16) -> (f32, f32) {
    let dx_enc = (encoded & 0xFF) as f32;
    let dz_enc = ((encoded >> 8) & 0xFF) as f32;
    let dh_dx = dx_enc / 255.0 * 2.0 * GRADIENT_RANGE - GRADIENT_RANGE;
    let dh_dz = dz_enc / 255.0 * 2.0 * GRADIENT_RANGE - GRADIENT_RANGE;
    (dh_dx, dh_dz)
}

/// Decode gradient and compute surface normal.
/// For a heightfield surface y = h(x,z), the normal is:
/// n = normalize(-dh/dx, 1, -dh/dz)
#[inline]
pub fn decode_gradient_normal(encoded: u16) -> Vec3 {
    let (dh_dx, dh_dz) = decode_gradient(encoded);
    Vec3::new(-dh_dx, 1.0, -dh_dz).normalize()
}

// =============================================================================
// SDF Primitives
// =============================================================================

/// Distance to a sphere center
#[inline]
pub fn sdf_sphere(p: Vec3, center: Vec3, radius: f32) -> f32 {
    (p - center).length() - radius
}

/// Encode a unit normal vector into RGB565 format (16 bits).
/// Each component [-1,1] is mapped to its channel: R(5 bits), G(6 bits), B(5 bits).
/// Shader decodes with decode_bark_normal().
#[inline]
pub fn encode_normal_rgb565(n: Vec3) -> u16 {
    let r5 = ((n.x * 0.5 + 0.5).clamp(0.0, 1.0) * 31.0) as u16;
    let g6 = ((n.y * 0.5 + 0.5).clamp(0.0, 1.0) * 63.0) as u16;
    let b5 = ((n.z * 0.5 + 0.5).clamp(0.0, 1.0) * 31.0) as u16;
    (r5 << 11) | (g6 << 5) | b5
}

/// Distance to a capsule (line segment with radius)
/// Capsule from point `a` to `b` with radius `r`
#[inline]
pub fn sdf_capsule(p: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = (ap.dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    let closest = a + ab * t;
    (p - closest).length() - radius
}

/// Distance to an infinite plane at height y = 0
#[inline]
pub fn sdf_plane(p: Vec3) -> f32 {
    p.y
}

/// Distance to a plane at arbitrary position and orientation
/// Normal should be normalized
#[inline]
pub fn sdf_plane_point_normal(p: Vec3, plane_point: Vec3, normal: Vec3) -> f32 {
    (p - plane_point).dot(normal)
}

// =============================================================================
// SDF Combinations
// =============================================================================

/// Smooth minimum (polynomial smooth blend)
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (b - a).abs()).max(0.0) / k;
    b.min(a) - h * h * k * 0.25
}

/// Union (min of two distances)
#[inline]
pub fn sdf_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Subtraction: a - b (inside a, outside b)
#[inline]
pub fn sdf_subtraction(a: f32, b: f32) -> f32 {
    (-b).max(a)
}

/// Intersection (max of two distances)
#[inline]
pub fn sdf_intersection(a: f32, b: f32) -> f32 {
    a.max(b)
}

// =============================================================================
// Voxel Builder Helper
// =============================================================================

use crate::voxel::voxel::Voxel;

/// Create a voxel with gradient-encoded normal for heightfield terrain.
/// This is for surfaces defined by y = h(x,z).
pub fn voxel_with_gradient(dh_dx: f32, dh_dz: f32, height_fraction: f32, material_id: u8) -> Voxel {
    let color = encode_gradient(dh_dx, dh_dz);
    let flags = ((height_fraction.clamp(0.0, 1.0) * 254.0) as u8).max(1);
    Voxel {
        color,
        material_id,
        flags,
    }
}

/// Create a voxel with distance-encoded SDF value.
/// For non-heightfield geometry (spheres, capsules, etc.)
pub fn voxel_with_distance(distance: f32, distance_fraction: f32, material_id: u8) -> Voxel {
    let color = encode_distance(distance.abs());
    let flags = ((distance_fraction.clamp(0.0, 1.0) * 254.0) as u8).max(1);
    Voxel {
        color,
        material_id,
        flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_encoding() {
        let encoded = encode_distance(5.0);
        let decoded = decode_distance(encoded);
        assert!((decoded - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_encoding() {
        let encoded = encode_gradient(2.0, -1.0);
        let (dx, dz) = decode_gradient(encoded);
        assert!((dx - 2.0).abs() < 0.1);
        assert!((dz - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn test_sdf_sphere() {
        let d = sdf_sphere(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 1.0);
        assert!((d - 1.0).abs() < 0.001); // |(0,0,0) - (2,0,0)| - 1 = 2 - 1 = 1
    }

    #[test]
    fn test_sdf_capsule() {
        let d = sdf_capsule(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        assert!((d - 0.5).abs() < 0.001); // Point at origin, distance to line = 0.5
    }

    #[test]
    fn test_sdf_plane() {
        let d = sdf_plane(Vec3::new(5.0, 2.0, 3.0));
        assert!((d - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_smin() {
        let result = smin(0.5, 0.3, 0.5);
        assert!(result < 0.3); // Smooth min should be less than or equal to min
    }
}
