//! Level of Detail (LOD) system for distance-based detail reduction
//!
//! This module implements a 6-level LOD system that reduces voxel detail
//! based on distance from the camera/viewer. Each level doubles the voxel
//! size, allowing efficient rendering of distant geometry.

/// LOD level definitions
/// - LOD 0: Full detail (1cm voxels) - 0-64m
/// - LOD 1: 2cm voxels - 64-128m
/// - LOD 2: 4cm voxels - 128-256m
/// - LOD 3: 8cm voxels - 256-512m
/// - LOD 4: 16cm voxels - 512-1024m
/// - LOD 5: 32cm voxels - 1024m+
pub const LOD_DISTANCES: [f32; 6] = [64.0, 128.0, 256.0, 512.0, 1024.0, f32::MAX];

/// Maximum LOD level
pub const MAX_LOD: u32 = 5;

/// Configuration for LOD behavior
#[derive(Clone, Debug)]
pub struct LodConfig {
    /// Distance thresholds for each LOD level
    pub distances: [f32; 6],
    /// Whether to enable smooth LOD transitions
    pub blend_enabled: bool,
    /// Portion of each LOD band to use for blending (0.0-1.0)
    pub blend_width: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            distances: LOD_DISTANCES,
            blend_enabled: true,
            blend_width: 0.2,
        }
    }
}

/// Calculate LOD level from distance
///
/// # Arguments
/// * `distance` - Distance from viewer to point
///
/// # Returns
/// LOD level (0-5)
///
/// # Examples
/// ```
/// use rktri::streaming::lod::lod_from_distance;
///
/// assert_eq!(lod_from_distance(32.0), 0);  // Full detail
/// assert_eq!(lod_from_distance(96.0), 1);  // 2cm voxels
/// assert_eq!(lod_from_distance(200.0), 2); // 4cm voxels
/// assert_eq!(lod_from_distance(2000.0), 5); // Maximum LOD
/// ```
pub fn lod_from_distance(distance: f32) -> u32 {
    for (level, &max_dist) in LOD_DISTANCES.iter().enumerate() {
        if distance < max_dist {
            return level as u32;
        }
    }
    MAX_LOD
}

/// Get octree traversal depth for a given LOD level
///
/// At LOD 0, traverse full depth. Each LOD level reduces depth by 1,
/// as higher LOD levels use larger voxels and require less detail.
///
/// # Arguments
/// * `base_depth` - Maximum octree depth
/// * `lod` - LOD level
///
/// # Returns
/// Traversal depth for this LOD level
///
/// # Examples
/// ```
/// use rktri::streaming::lod::traversal_depth_for_lod;
///
/// assert_eq!(traversal_depth_for_lod(10, 0), 10); // Full depth
/// assert_eq!(traversal_depth_for_lod(10, 1), 9);  // One level less
/// assert_eq!(traversal_depth_for_lod(10, 5), 5);  // Five levels less
/// assert_eq!(traversal_depth_for_lod(3, 5), 0);   // Saturating sub
/// ```
pub fn traversal_depth_for_lod(base_depth: u32, lod: u32) -> u32 {
    base_depth.saturating_sub(lod)
}

/// Calculate voxel size at a given LOD level
///
/// Each LOD level doubles the voxel size from the base.
///
/// # Arguments
/// * `base_voxel_size` - Voxel size at LOD 0
/// * `lod` - LOD level
///
/// # Returns
/// Voxel size at the given LOD level
///
/// # Examples
/// ```
/// use rktri::streaming::lod::voxel_size_at_lod;
///
/// assert_eq!(voxel_size_at_lod(0.01, 0), 0.01);  // 1cm
/// assert_eq!(voxel_size_at_lod(0.01, 1), 0.02);  // 2cm
/// assert_eq!(voxel_size_at_lod(0.01, 2), 0.04);  // 4cm
/// assert_eq!(voxel_size_at_lod(0.01, 5), 0.32);  // 32cm
/// ```
pub fn voxel_size_at_lod(base_voxel_size: f32, lod: u32) -> f32 {
    base_voxel_size * (1 << lod) as f32
}

/// Calculate blend factor for LOD transitions (0.0 = current LOD, 1.0 = next LOD)
///
/// Provides smooth transitions between LOD levels by blending in the last
/// portion of each LOD distance band.
///
/// # Arguments
/// * `distance` - Distance from viewer
/// * `lod` - Current LOD level
///
/// # Returns
/// Blend factor (0.0-1.0), where 0.0 means use current LOD only,
/// and 1.0 means transition to next LOD
///
/// # Examples
/// ```
/// use rktri::streaming::lod::lod_blend_factor;
///
/// // No blending at start of LOD band
/// assert_eq!(lod_blend_factor(0.0, 0), 0.0);
/// assert_eq!(lod_blend_factor(32.0, 0), 0.0);
///
/// // Full blend at end of LOD band
/// assert_eq!(lod_blend_factor(64.0, 0), 1.0);
///
/// // Partial blend in transition zone (last 20%)
/// let blend = lod_blend_factor(60.0, 0); // 60m is in the 51.2-64m blend zone
/// assert!(blend > 0.0 && blend < 1.0);
/// ```
pub fn lod_blend_factor(distance: f32, lod: u32) -> f32 {
    if lod >= MAX_LOD {
        return 0.0;
    }

    let near = if lod == 0 { 0.0 } else { LOD_DISTANCES[lod as usize - 1] };
    let far = LOD_DISTANCES[lod as usize];

    let t = (distance - near) / (far - near);

    // Smooth transition in the last 20% of each LOD band
    let blend_start = 0.8;
    ((t - blend_start) / (1.0 - blend_start)).clamp(0.0, 1.0)
}

/// Calculate blend factor with custom blend width
///
/// # Arguments
/// * `distance` - Distance from viewer
/// * `lod` - Current LOD level
/// * `blend_width` - Portion of LOD band to use for blending (0.0-1.0)
///
/// # Returns
/// Blend factor (0.0-1.0)
pub fn lod_blend_factor_custom(distance: f32, lod: u32, blend_width: f32) -> f32 {
    if lod >= MAX_LOD {
        return 0.0;
    }

    let near = if lod == 0 { 0.0 } else { LOD_DISTANCES[lod as usize - 1] };
    let far = LOD_DISTANCES[lod as usize];

    let t = (distance - near) / (far - near);

    let blend_start = 1.0 - blend_width.clamp(0.0, 1.0);
    ((t - blend_start) / (1.0 - blend_start)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_from_distance() {
        // LOD 0: 0-64m
        assert_eq!(lod_from_distance(0.0), 0);
        assert_eq!(lod_from_distance(32.0), 0);
        assert_eq!(lod_from_distance(63.9), 0);

        // LOD 1: 64-128m
        assert_eq!(lod_from_distance(64.0), 1);
        assert_eq!(lod_from_distance(96.0), 1);
        assert_eq!(lod_from_distance(127.9), 1);

        // LOD 2: 128-256m
        assert_eq!(lod_from_distance(128.0), 2);
        assert_eq!(lod_from_distance(200.0), 2);

        // LOD 3: 256-512m
        assert_eq!(lod_from_distance(256.0), 3);
        assert_eq!(lod_from_distance(400.0), 3);

        // LOD 4: 512-1024m
        assert_eq!(lod_from_distance(512.0), 4);
        assert_eq!(lod_from_distance(800.0), 4);

        // LOD 5: 1024m+
        assert_eq!(lod_from_distance(1024.0), 5);
        assert_eq!(lod_from_distance(2000.0), 5);
        assert_eq!(lod_from_distance(100000.0), 5);
    }

    #[test]
    fn test_traversal_depth_for_lod() {
        // Full depth at LOD 0
        assert_eq!(traversal_depth_for_lod(10, 0), 10);

        // Decreasing depth with LOD
        assert_eq!(traversal_depth_for_lod(10, 1), 9);
        assert_eq!(traversal_depth_for_lod(10, 2), 8);
        assert_eq!(traversal_depth_for_lod(10, 5), 5);

        // Saturating subtraction
        assert_eq!(traversal_depth_for_lod(3, 5), 0);
        assert_eq!(traversal_depth_for_lod(0, 1), 0);
    }

    #[test]
    fn test_voxel_size_at_lod() {
        let base_size = 0.01; // 1cm

        // Each level doubles the size
        assert_eq!(voxel_size_at_lod(base_size, 0), 0.01);  // 1cm
        assert_eq!(voxel_size_at_lod(base_size, 1), 0.02);  // 2cm
        assert_eq!(voxel_size_at_lod(base_size, 2), 0.04);  // 4cm
        assert_eq!(voxel_size_at_lod(base_size, 3), 0.08);  // 8cm
        assert_eq!(voxel_size_at_lod(base_size, 4), 0.16);  // 16cm
        assert_eq!(voxel_size_at_lod(base_size, 5), 0.32);  // 32cm
    }

    #[test]
    fn test_lod_blend_factor() {
        // No blending at start of LOD 0 band
        assert_eq!(lod_blend_factor(0.0, 0), 0.0);
        assert_eq!(lod_blend_factor(32.0, 0), 0.0);

        // No blending before blend zone (80% mark = 51.2m for LOD 0)
        assert_eq!(lod_blend_factor(50.0, 0), 0.0);

        // Partial blending in transition zone
        let blend_mid = lod_blend_factor(57.6, 0); // Midpoint of blend zone
        assert!(blend_mid > 0.0 && blend_mid < 1.0);
        assert!((blend_mid - 0.5).abs() < 0.01); // Should be ~0.5

        // Full blending at end of LOD band
        assert_eq!(lod_blend_factor(64.0, 0), 1.0);

        // Test LOD 1 band (64-128m, blend starts at 115.2m)
        assert_eq!(lod_blend_factor(64.0, 1), 0.0);
        assert_eq!(lod_blend_factor(100.0, 1), 0.0);
        assert!(lod_blend_factor(120.0, 1) > 0.0);
        assert_eq!(lod_blend_factor(128.0, 1), 1.0);

        // No blending at max LOD
        assert_eq!(lod_blend_factor(2000.0, MAX_LOD), 0.0);
    }

    #[test]
    fn test_lod_blend_factor_custom() {
        // 50% blend width
        let blend_50 = lod_blend_factor_custom(48.0, 0, 0.5);
        assert!(blend_50 > 0.0);

        // 10% blend width - narrower transition
        let blend_10 = lod_blend_factor_custom(60.0, 0, 0.1);
        assert!(blend_10 > 0.0);

        // 100% blend width - blend across entire band
        let blend_100_start = lod_blend_factor_custom(0.0, 0, 1.0);
        assert_eq!(blend_100_start, 0.0);
        let blend_100_end = lod_blend_factor_custom(64.0, 0, 1.0);
        assert_eq!(blend_100_end, 1.0);

        // 0% blend width - no blending (division by zero protection results in 0.0)
        let blend_0 = lod_blend_factor_custom(63.0, 0, 0.0);
        assert_eq!(blend_0, 0.0); // No blend zone means always 0.0 until boundary
    }

    #[test]
    fn test_lod_config_default() {
        let config = LodConfig::default();
        assert_eq!(config.distances, LOD_DISTANCES);
        assert!(config.blend_enabled);
        assert_eq!(config.blend_width, 0.2);
    }

    #[test]
    fn test_lod_constants() {
        assert_eq!(LOD_DISTANCES.len(), 6);
        assert_eq!(MAX_LOD, 5);
        assert_eq!(LOD_DISTANCES[5], f32::MAX);
    }

    #[test]
    fn test_lod_boundary_conditions() {
        // Test exact boundary values
        assert_eq!(lod_from_distance(64.0), 1);
        assert_eq!(lod_from_distance(63.99999), 0);

        // Test blend at exact boundaries
        assert_eq!(lod_blend_factor(64.0, 0), 1.0);
        assert_eq!(lod_blend_factor(64.0, 1), 0.0);
    }

    #[test]
    fn test_lod_progression() {
        // Verify each LOD covers expected distance range
        let test_distances = [
            (0.0, 0),
            (63.9, 0),
            (64.0, 1),
            (127.9, 1),
            (128.0, 2),
            (255.9, 2),
            (256.0, 3),
            (511.9, 3),
            (512.0, 4),
            (1023.9, 4),
            (1024.0, 5),
        ];

        for (distance, expected_lod) in test_distances {
            assert_eq!(
                lod_from_distance(distance),
                expected_lod,
                "Distance {} should be LOD {}",
                distance,
                expected_lod
            );
        }
    }

    #[test]
    fn test_voxel_size_powers_of_two() {
        // Verify voxel sizes are exact powers of 2
        for lod in 0..=5 {
            let size = voxel_size_at_lod(1.0, lod);
            let expected = 2.0_f32.powi(lod as i32);
            assert_eq!(size, expected);
        }
    }

    #[test]
    fn test_blend_monotonicity() {
        // Blend factor should increase monotonically with distance
        for lod in 0..MAX_LOD {
            let near = if lod == 0 { 0.0 } else { LOD_DISTANCES[lod as usize - 1] };
            let far = LOD_DISTANCES[lod as usize];

            let mut prev_blend = 0.0;
            for i in 0..100 {
                let t = i as f32 / 99.0;
                let distance = near + t * (far - near);
                let blend = lod_blend_factor(distance, lod);

                assert!(
                    blend >= prev_blend,
                    "Blend factor should be monotonically increasing"
                );
                prev_blend = blend;
            }
        }
    }
}
