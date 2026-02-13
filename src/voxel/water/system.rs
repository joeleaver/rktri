//! Water system manager for all water bodies.

use glam::Vec3;
use std::collections::HashMap;

use super::{WaterBody, WaterBodyType, WaterProperties};
use crate::math::Aabb;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::voxel::Voxel;

/// Material ID for water voxels
pub const MATERIAL_WATER: u8 = 200;

/// Central manager for all water bodies in the world.
pub struct WaterSystem {
    /// All water bodies by ID
    bodies: HashMap<u64, WaterBody>,
    /// Next available ID
    next_id: u64,
    /// Global sea level (for ocean)
    pub sea_level: f32,
    /// Current animation time
    time: f32,
    /// Whether the system has an ocean
    has_ocean: bool,
}

impl WaterSystem {
    /// Create a new empty water system.
    pub fn new() -> Self {
        Self {
            bodies: HashMap::new(),
            next_id: 1,
            sea_level: 0.0,
            time: 0.0,
            has_ocean: false,
        }
    }

    /// Create with a global ocean at the given sea level.
    pub fn with_ocean(sea_level: f32) -> Self {
        let mut system = Self::new();
        system.sea_level = sea_level;
        system.has_ocean = true;
        system
    }

    /// Add a water body and return its ID.
    pub fn add_body(&mut self, mut body: WaterBody) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;

        // Track if this is an ocean
        if matches!(body.body_type, WaterBodyType::Ocean { .. }) {
            self.has_ocean = true;
            if let WaterBodyType::Ocean { sea_level } = body.body_type {
                self.sea_level = sea_level;
            }
        }

        self.bodies.insert(id, body);
        id
    }

    /// Remove a water body.
    pub fn remove_body(&mut self, id: u64) -> Option<WaterBody> {
        self.bodies.remove(&id)
    }

    /// Get a water body by ID.
    pub fn get_body(&self, id: u64) -> Option<&WaterBody> {
        self.bodies.get(&id)
    }

    /// Iterate all water bodies.
    pub fn bodies(&self) -> impl Iterator<Item = &WaterBody> {
        self.bodies.values()
    }

    /// Update animation time.
    pub fn update(&mut self, delta_time: f32) {
        self.time += delta_time;
    }

    /// Get current animation time.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Check if a point is underwater.
    pub fn is_underwater(&self, pos: Vec3) -> bool {
        // Check global ocean first (most common case)
        if self.has_ocean && pos.y < self.sea_level {
            return true;
        }

        // Check individual water bodies
        for body in self.bodies.values() {
            if body.contains_point(pos) {
                return true;
            }
        }
        false
    }

    /// Get water depth at a point (0 if not underwater).
    pub fn water_depth(&self, pos: Vec3) -> f32 {
        // Check global ocean
        if self.has_ocean && pos.y < self.sea_level {
            return self.sea_level - pos.y;
        }

        // Check individual water bodies
        for body in self.bodies.values() {
            let depth = body.depth_at(pos);
            if depth > 0.0 {
                return depth;
            }
        }
        0.0
    }

    /// Get water surface height at XZ position (includes wave animation).
    pub fn surface_height_at(&self, x: f32, z: f32) -> f32 {
        let mut height = if self.has_ocean {
            self.sea_level + self.wave_height(x, z)
        } else {
            f32::MIN
        };

        // Check individual water bodies for higher surfaces
        for body in self.bodies.values() {
            if body.bounds.min.x <= x && x <= body.bounds.max.x
                && body.bounds.min.z <= z && z <= body.bounds.max.z
            {
                let body_height = body.surface.height_at(x, z);
                height = height.max(body_height);
            }
        }

        height
    }

    /// Calculate wave height using Gerstner waves.
    pub fn wave_height(&self, x: f32, z: f32) -> f32 {
        let t = self.time;

        // Sum of multiple wave frequencies for realistic ocean
        let mut h = 0.0;

        // Wave 1: Large slow swells
        h += 0.5 * (x * 0.02 + t * 0.5).sin() * (z * 0.015 + t * 0.3).cos();

        // Wave 2: Medium waves
        h += 0.2 * (x * 0.05 + t * 1.2).sin() * (z * 0.04 - t * 0.8).cos();

        // Wave 3: Small choppy waves
        h += 0.05 * (x * 0.15 + t * 2.5).sin() * (z * 0.12 + t * 1.8).cos();

        h
    }

    /// Calculate wave normal at position.
    pub fn wave_normal(&self, x: f32, z: f32) -> Vec3 {
        let eps = 0.1;
        let h = self.wave_height(x, z);
        let hx = self.wave_height(x + eps, z);
        let hz = self.wave_height(x, z + eps);

        let dx = (hx - h) / eps;
        let dz = (hz - h) / eps;

        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Get water properties at a point.
    pub fn properties_at(&self, pos: Vec3) -> WaterProperties {
        // Check individual water bodies first
        for body in self.bodies.values() {
            if body.contains_point(pos) {
                return body.properties;
            }
        }

        // Fall back to ocean properties
        if self.has_ocean && pos.y < self.sea_level {
            return WaterProperties::ocean();
        }

        WaterProperties::default()
    }

    /// Query water bodies intersecting an AABB.
    pub fn query_aabb(&self, aabb: &Aabb) -> Vec<&WaterBody> {
        self.bodies
            .values()
            .filter(|body| body.bounds.intersects(aabb))
            .collect()
    }
}

impl Default for WaterSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionClassifier for WaterSystem {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        // Region entirely above all water
        if !self.has_ocean || aabb.min.y > self.sea_level + 2.0 {
            let bodies = self.query_aabb(aabb);
            if bodies.is_empty() {
                return RegionHint::Empty;
            }
        }

        // Region entirely below water surface (deep underwater)
        if self.has_ocean && aabb.max.y < self.sea_level - 1.0 {
            return RegionHint::Solid {
                material: MATERIAL_WATER,
                color: 0x1E5B, // Blue-green encoded
            };
        }

        // Region intersects water surface or contains water bodies
        RegionHint::Mixed
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        if self.is_underwater(pos) {
            // Create blue-green water voxel
            // 0x1E5B in RGB565: R=3, G=114, B=27 -> roughly (24, 228, 216) in RGB888
            Voxel::from_rgb565(0x1E5B, MATERIAL_WATER)
        } else {
            Voxel::EMPTY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_system() {
        let system = WaterSystem::new();
        assert!(!system.is_underwater(Vec3::ZERO));
        assert_eq!(system.water_depth(Vec3::ZERO), 0.0);
    }

    #[test]
    fn test_ocean() {
        let system = WaterSystem::with_ocean(10.0);

        // Underwater
        assert!(system.is_underwater(Vec3::new(0.0, 5.0, 0.0)));
        assert_eq!(system.water_depth(Vec3::new(0.0, 5.0, 0.0)), 5.0);

        // Above water
        assert!(!system.is_underwater(Vec3::new(0.0, 15.0, 0.0)));
    }

    #[test]
    fn test_wave_animation() {
        let mut system = WaterSystem::with_ocean(10.0);

        let h1 = system.wave_height(0.0, 0.0);
        system.update(1.0);
        let h2 = system.wave_height(0.0, 0.0);

        // Waves should animate
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_region_classifier() {
        let system = WaterSystem::with_ocean(10.0);

        // High above water - empty
        let high = Aabb::new(Vec3::new(0.0, 50.0, 0.0), Vec3::new(1.0, 51.0, 1.0));
        assert_eq!(system.classify_region(&high), RegionHint::Empty);

        // Deep underwater - solid water
        let deep = Aabb::new(Vec3::new(0.0, -50.0, 0.0), Vec3::new(1.0, -49.0, 1.0));
        match system.classify_region(&deep) {
            RegionHint::Solid { material, .. } => assert_eq!(material, MATERIAL_WATER),
            _ => panic!("Expected Solid"),
        }

        // At surface - mixed
        let surface = Aabb::new(Vec3::new(0.0, 9.0, 0.0), Vec3::new(1.0, 11.0, 1.0));
        assert_eq!(system.classify_region(&surface), RegionHint::Mixed);
    }
}
