//! Water volume representation and properties.

use glam::Vec3;
use crate::math::Aabb;

/// Type of water body affects rendering and simulation.
#[derive(Clone, Debug)]
pub enum WaterBodyType {
    /// Large static body (ocean, lake) - flat surface with waves
    Ocean { sea_level: f32 },
    /// Flowing water (river, stream) - has flow direction
    River {
        flow_direction: Vec3,
        flow_speed: f32,
    },
    /// Vertical flow (waterfall)
    Waterfall { flow_speed: f32 },
    /// Dynamic water (rain puddles, flooding)
    Procedural,
}

/// Surface representation for water bodies.
#[derive(Clone, Debug)]
pub enum WaterSurface {
    /// Flat plane at constant Y (oceans, lakes)
    Flat { y: f32 },
    /// Heightfield for rivers following terrain
    Heightfield {
        /// Heights in row-major order
        heights: Vec<f32>,
        /// Grid resolution
        resolution: u32,
        /// World-space bounds of the heightfield
        bounds: Aabb,
    },
}

impl WaterSurface {
    /// Create a flat water surface.
    pub fn flat(y: f32) -> Self {
        Self::Flat { y }
    }

    /// Get water surface height at world position.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        match self {
            Self::Flat { y } => *y,
            Self::Heightfield { heights, resolution, bounds } => {
                // Normalize position to [0, 1] within bounds
                let nx = (x - bounds.min.x) / (bounds.max.x - bounds.min.x);
                let nz = (z - bounds.min.z) / (bounds.max.z - bounds.min.z);

                if nx < 0.0 || nx > 1.0 || nz < 0.0 || nz > 1.0 {
                    return 0.0; // Outside bounds
                }

                // Sample heightfield with bilinear interpolation
                let fx = nx * (*resolution - 1) as f32;
                let fz = nz * (*resolution - 1) as f32;
                let ix = fx.floor() as usize;
                let iz = fz.floor() as usize;
                let tx = fx.fract();
                let tz = fz.fract();

                let res = *resolution as usize;
                let i00 = iz * res + ix;
                let i10 = iz * res + (ix + 1).min(res - 1);
                let i01 = (iz + 1).min(res - 1) * res + ix;
                let i11 = (iz + 1).min(res - 1) * res + (ix + 1).min(res - 1);

                let h00 = heights.get(i00).copied().unwrap_or(0.0);
                let h10 = heights.get(i10).copied().unwrap_or(0.0);
                let h01 = heights.get(i01).copied().unwrap_or(0.0);
                let h11 = heights.get(i11).copied().unwrap_or(0.0);

                let h0 = h00 * (1.0 - tx) + h10 * tx;
                let h1 = h01 * (1.0 - tx) + h11 * tx;
                h0 * (1.0 - tz) + h1 * tz
            }
        }
    }
}

/// Water material properties for rendering.
#[derive(Clone, Copy, Debug)]
pub struct WaterProperties {
    /// Base color (tint)
    pub color: Vec3,
    /// Absorption coefficient (how quickly light fades underwater)
    /// Higher values = faster absorption. RGB for wavelength-dependent absorption.
    pub absorption: Vec3,
    /// Index of refraction (1.333 for water)
    pub ior: f32,
    /// Surface roughness (affects reflection sharpness)
    pub roughness: f32,
    /// Foam threshold (wave height that generates foam)
    pub foam_threshold: f32,
    /// Caustics intensity (0.0 = none, 1.0 = full)
    pub caustics_intensity: f32,
}

impl Default for WaterProperties {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.1, 0.4, 0.6),  // Blue-green
            absorption: Vec3::new(0.4, 0.2, 0.1), // Red absorbed fastest
            ior: 1.333,
            roughness: 0.05,
            foam_threshold: 0.5,
            caustics_intensity: 0.3,
        }
    }
}

impl WaterProperties {
    /// Create ocean-style water properties.
    pub fn ocean() -> Self {
        Self {
            color: Vec3::new(0.05, 0.2, 0.4),
            absorption: Vec3::new(0.3, 0.15, 0.08),
            ior: 1.333,
            roughness: 0.1,
            foam_threshold: 0.3,
            caustics_intensity: 0.2,
        }
    }

    /// Create river-style water properties.
    pub fn river() -> Self {
        Self {
            color: Vec3::new(0.15, 0.35, 0.45),
            absorption: Vec3::new(0.5, 0.3, 0.2),
            ior: 1.333,
            roughness: 0.15,
            foam_threshold: 0.2,
            caustics_intensity: 0.4,
        }
    }

    /// Create clear pool water properties.
    pub fn pool() -> Self {
        Self {
            color: Vec3::new(0.2, 0.5, 0.7),
            absorption: Vec3::new(0.1, 0.05, 0.02),
            ior: 1.333,
            roughness: 0.02,
            foam_threshold: 1.0, // No foam
            caustics_intensity: 0.5,
        }
    }
}

/// A water body in the world.
#[derive(Clone, Debug)]
pub struct WaterBody {
    /// Unique identifier
    pub id: u64,
    /// Type of water body
    pub body_type: WaterBodyType,
    /// World-space bounds of the water volume
    pub bounds: Aabb,
    /// Water surface representation
    pub surface: WaterSurface,
    /// Material properties for rendering
    pub properties: WaterProperties,
}

impl WaterBody {
    /// Create a new ocean water body.
    pub fn ocean(id: u64, bounds: Aabb, sea_level: f32) -> Self {
        Self {
            id,
            body_type: WaterBodyType::Ocean { sea_level },
            bounds,
            surface: WaterSurface::flat(sea_level),
            properties: WaterProperties::ocean(),
        }
    }

    /// Create a new river water body.
    pub fn river(id: u64, bounds: Aabb, surface: WaterSurface, flow_direction: Vec3, flow_speed: f32) -> Self {
        Self {
            id,
            body_type: WaterBodyType::River { flow_direction: flow_direction.normalize(), flow_speed },
            bounds,
            surface,
            properties: WaterProperties::river(),
        }
    }

    /// Check if a point is inside this water body.
    pub fn contains_point(&self, pos: Vec3) -> bool {
        if !self.bounds.contains_point(pos) {
            return false;
        }
        pos.y < self.surface.height_at(pos.x, pos.z)
    }

    /// Get water depth at a point (0 if not underwater).
    pub fn depth_at(&self, pos: Vec3) -> f32 {
        if !self.bounds.contains_point(pos) {
            return 0.0;
        }
        let surface_y = self.surface.height_at(pos.x, pos.z);
        (surface_y - pos.y).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_surface() {
        let surface = WaterSurface::flat(10.0);
        assert_eq!(surface.height_at(0.0, 0.0), 10.0);
        assert_eq!(surface.height_at(100.0, -50.0), 10.0);
    }

    #[test]
    fn test_water_properties_presets() {
        let ocean = WaterProperties::ocean();
        let river = WaterProperties::river();
        let pool = WaterProperties::pool();

        // Ocean is darker than pool
        assert!(ocean.color.x < pool.color.x);
        // Pool has less absorption than river
        assert!(pool.absorption.x < river.absorption.x);
    }

    #[test]
    fn test_water_body_contains() {
        let bounds = Aabb::new(Vec3::ZERO, Vec3::new(100.0, 20.0, 100.0));
        let body = WaterBody::ocean(1, bounds, 10.0);

        // Point underwater
        assert!(body.contains_point(Vec3::new(50.0, 5.0, 50.0)));
        // Point above water
        assert!(!body.contains_point(Vec3::new(50.0, 15.0, 50.0)));
        // Point outside bounds
        assert!(!body.contains_point(Vec3::new(150.0, 5.0, 50.0)));
    }

    #[test]
    fn test_water_depth() {
        let bounds = Aabb::new(Vec3::ZERO, Vec3::new(100.0, 20.0, 100.0));
        let body = WaterBody::ocean(1, bounds, 10.0);

        // 5m underwater
        assert_eq!(body.depth_at(Vec3::new(50.0, 5.0, 50.0)), 5.0);
        // At surface
        assert_eq!(body.depth_at(Vec3::new(50.0, 10.0, 50.0)), 0.0);
        // Above water
        assert_eq!(body.depth_at(Vec3::new(50.0, 15.0, 50.0)), 0.0);
    }
}
