//! Scene configuration for test scenes

use glam::Vec3;
use crate::terrain::generator::TerrainParams;

/// Debug visualization modes
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugMode {
    #[default]
    None,
    Albedo,
    Normal,
    Depth,
    Material,
    Biome,
}

/// Configuration for a test scene
#[derive(Clone, Debug)]
pub struct SceneConfig {
    /// Random seed for terrain and biome generation
    pub seed: u32,
    /// Terrain generation parameters
    pub terrain_params: TerrainParams,
    /// Initial camera position
    pub initial_camera_pos: Vec3,
    /// View distance in meters (chunks loaded within this radius)
    pub view_distance: f32,
    /// Time of day (0.0-24.0 hours)
    pub time_of_day: f32,
    /// Sun intensity multiplier
    pub sun_intensity: f32,
    /// Debug visualization mode
    pub debug_mode: DebugMode,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            seed: 12345,
            terrain_params: TerrainParams {
                scale: 150.0,        // Larger for biome variety
                height_scale: 80.0,  // Taller terrain
                octaves: 5,
                sea_level: 20.0,     // Lower for more land
                ..Default::default()
            },
            initial_camera_pos: Vec3::new(2.0, 55.0, 2.0),  // Above terrain surface (height_scale=80)
            view_distance: 32.0,  // 8-chunk radius, fast with adaptive generation
            time_of_day: 10.0,     // Morning
            sun_intensity: 1.5,
            debug_mode: DebugMode::None,
        }
    }
}

impl SceneConfig {
    /// Calculate sun direction from time of day
    pub fn sun_direction(&self) -> Vec3 {
        // Sun rises at 6:00, peaks at 12:00, sets at 18:00
        let hour_angle = (self.time_of_day - 12.0) * 15.0_f32.to_radians();
        let altitude = (90.0 - (self.time_of_day - 12.0).abs() * 7.5).to_radians();

        Vec3::new(
            hour_angle.sin() * altitude.cos(),
            altitude.sin().max(0.1), // Keep sun above horizon
            hour_angle.cos() * altitude.cos(),
        ).normalize()
    }
}
