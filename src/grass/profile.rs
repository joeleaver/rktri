//! Grass profile types: per-biome grass appearance definitions.
//!
//! `GrassProfile` is a thin index type implementing `MaskValue` for storage
//! in `MaskOctree<GrassCell>`. `GpuGrassProfile` is the GPU-side struct
//! with all per-profile rendering parameters. `GrassProfileTable` manages
//! the collection of profile definitions.

use bytemuck::{Pod, Zeroable};
use crate::mask::MaskValue;

/// Index into the grass profile table. 0 = no grass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GrassProfile(pub u8);

impl GrassProfile {
    pub const NONE: Self = Self(0);
}

impl MaskValue for GrassProfile {}

/// Grass cell stored in mask octrees: profile index + density packed into u16.
///
/// - Low 8 bits: profile index (0 = no grass, 1-255 = profile ID)
/// - High 8 bits: density (0-255, maps to 0.0-1.0 in shader)
///
/// PartialEq compares the full value so the octree subdivides at density transitions.
/// Density is quantized to ~16 levels in the generator so uniform regions stay pruned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GrassCell(pub u16);

impl GrassCell {
    pub const NONE: Self = Self(0);

    /// Create a grass cell with profile and density (0.0-1.0, quantized to 16 levels).
    pub fn new(profile: GrassProfile, density: f32) -> Self {
        if profile == GrassProfile::NONE {
            return Self::NONE;
        }
        // Quantize to 16 levels (0, 17, 34, ..., 255) for better octree pruning
        let d_u8 = ((density.clamp(0.0, 1.0) * 15.0).round() as u8) * 17;
        Self((d_u8 as u16) << 8 | profile.0 as u16)
    }

    /// Extract the profile index.
    pub fn profile(self) -> GrassProfile {
        GrassProfile((self.0 & 0xFF) as u8)
    }

    /// Extract the density as 0.0-1.0.
    pub fn density(self) -> f32 {
        ((self.0 >> 8) as u8) as f32 / 255.0
    }

    /// Returns true if this cell has no grass.
    pub fn is_none(self) -> bool {
        (self.0 & 0xFF) == 0
    }
}

impl MaskValue for GrassCell {}

/// GPU-side grass profile data (64 bytes, 16-byte aligned).
/// Must match `GrassProfileGpu` in svo_trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGrassProfile {
    pub height_min: f32,
    pub height_max: f32,
    pub width: f32,
    pub density: f32,
    // -- 16 bytes --
    pub color_base: [f32; 3],
    pub sway_amount: f32,
    // -- 16 bytes --
    pub color_variation: f32,
    pub sway_frequency: f32,
    pub blade_spacing: f32,
    pub slope_threshold: f32,
    // -- 16 bytes --
    pub coverage_scale: f32,
    pub coverage_amount: f32,
    pub _pad: [f32; 2],
    // -- 16 bytes --
    // Total: 64 bytes
}

/// CPU-side profile definition with name and GPU data.
#[derive(Clone, Debug)]
pub struct GrassProfileDef {
    pub name: String,
    pub gpu: GpuGrassProfile,
}

/// Table of grass profile definitions.
pub struct GrassProfileTable {
    profiles: Vec<GrassProfileDef>,
}

impl GrassProfileTable {
    /// Number of profiles (including the NONE sentinel at index 0).
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Get a profile definition by index.
    pub fn get(&self, index: u8) -> Option<&GrassProfileDef> {
        self.profiles.get(index as usize)
    }

    /// Get mutable reference for runtime editing.
    pub fn get_mut(&mut self, index: u8) -> Option<&mut GrassProfileDef> {
        self.profiles.get_mut(index as usize)
    }

    /// Produce a flat Vec of GPU profile data for upload.
    pub fn gpu_data(&self) -> Vec<GpuGrassProfile> {
        self.profiles.iter().map(|p| p.gpu).collect()
    }

    /// Map a biome ID to a default grass profile index.
    ///
    /// Returns the primary profile for the given biome. Some biomes
    /// use noise-based mixing between two profiles (handled in the generator).
    pub fn biome_default_profile(biome_id: u8) -> GrassProfile {
        match biome_id {
            0 => GrassProfile::NONE, // Ocean
            1 => GrassProfile::NONE, // Beach
            2 => GrassProfile::NONE, // Desert
            3 => GrassProfile(1),    // Grassland -> Tall Grass
            4 => GrassProfile(2),    // Forest -> Meadow
            5 => GrassProfile(4),    // Taiga -> Tundra Scrub
            6 => GrassProfile(4),    // Tundra -> Tundra Scrub
            7 => GrassProfile(5),    // Mountains -> Alpine
            8 => GrassProfile::NONE, // Snow
            _ => GrassProfile::NONE,
        }
    }

    /// Secondary profile for biome mixing (if any).
    /// Returns None for biomes without mixing.
    pub fn biome_secondary_profile(biome_id: u8) -> Option<(GrassProfile, f32)> {
        match biome_id {
            3 => Some((GrassProfile(2), 0.30)), // Grassland: 30% Meadow
            4 => Some((GrassProfile::NONE, 0.20)), // Forest: 20% clearings
            _ => None,
        }
    }
}

impl Default for GrassProfileTable {
    fn default() -> Self {
        Self {
            profiles: vec![
                // 0: None (sentinel)
                GrassProfileDef {
                    name: "None".into(),
                    gpu: GpuGrassProfile::zeroed(),
                },
                // 1: Tall Grass — thin blades, dense field
                GrassProfileDef {
                    name: "Tall Grass".into(),
                    gpu: GpuGrassProfile {
                        height_min: 0.30,
                        height_max: 0.55,
                        width: 0.025,
                        density: 0.85,
                        color_base: [0.20, 0.48, 0.10],
                        sway_amount: 0.5,
                        color_variation: 0.55,
                        sway_frequency: 1.5,
                        blade_spacing: 0.057,
                        slope_threshold: 0.8,
                        coverage_scale: 5.0,
                        coverage_amount: 0.6,
                        _pad: [0.0; 2],
                    },
                },
                // 2: Meadow — slightly shorter, lighter green
                GrassProfileDef {
                    name: "Meadow".into(),
                    gpu: GpuGrassProfile {
                        height_min: 0.20,
                        height_max: 0.40,
                        width: 0.020,
                        density: 0.80,
                        color_base: [0.28, 0.52, 0.13],
                        sway_amount: 0.6,
                        color_variation: 0.55,
                        sway_frequency: 1.8,
                        blade_spacing: 0.057,
                        slope_threshold: 1.0,
                        coverage_scale: 4.0,
                        coverage_amount: 0.5,
                        _pad: [0.0; 2],
                    },
                },
                // 3: Forest Floor — short undergrowth
                GrassProfileDef {
                    name: "Forest Floor".into(),
                    gpu: GpuGrassProfile {
                        height_min: 0.12,
                        height_max: 0.25,
                        width: 0.015,
                        density: 0.70,
                        color_base: [0.15, 0.35, 0.10],
                        sway_amount: 0.3,
                        color_variation: 0.35,
                        sway_frequency: 1.2,
                        blade_spacing: 0.050,
                        slope_threshold: 0.7,
                        coverage_scale: 3.0,
                        coverage_amount: 0.7,
                        _pad: [0.0; 2],
                    },
                },
                // 4: Tundra Scrub — sparse, yellowish, short
                GrassProfileDef {
                    name: "Tundra Scrub".into(),
                    gpu: GpuGrassProfile {
                        height_min: 0.08,
                        height_max: 0.18,
                        width: 0.010,
                        density: 0.50,
                        color_base: [0.35, 0.40, 0.20],
                        sway_amount: 0.2,
                        color_variation: 0.3,
                        sway_frequency: 1.0,
                        blade_spacing: 0.057,
                        slope_threshold: 0.5,
                        coverage_scale: 8.0,
                        coverage_amount: 0.6,
                        _pad: [0.0; 2],
                    },
                },
                // 5: Alpine — sparse mountain grass
                GrassProfileDef {
                    name: "Alpine".into(),
                    gpu: GpuGrassProfile {
                        height_min: 0.10,
                        height_max: 0.22,
                        width: 0.012,
                        density: 0.45,
                        color_base: [0.25, 0.45, 0.18],
                        sway_amount: 0.25,
                        color_variation: 0.35,
                        sway_frequency: 1.0,
                        blade_spacing: 0.057,
                        slope_threshold: 0.4,
                        coverage_scale: 6.0,
                        coverage_amount: 0.5,
                        _pad: [0.0; 2],
                    },
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_grass_profile_size() {
        assert_eq!(std::mem::size_of::<GpuGrassProfile>(), 64);
    }

    #[test]
    fn test_gpu_grass_profile_alignment() {
        assert_eq!(std::mem::size_of::<GpuGrassProfile>() % 16, 0);
    }

    #[test]
    fn test_grass_profile_is_mask_value() {
        assert!(GrassProfile::NONE.is_default());
        assert!(!GrassProfile(1).is_default());
    }

    #[test]
    fn test_default_profile_table() {
        let table = GrassProfileTable::default();
        assert_eq!(table.len(), 6);
        assert_eq!(table.get(0).unwrap().name, "None");
        assert_eq!(table.get(1).unwrap().name, "Tall Grass");
        assert_eq!(table.get(5).unwrap().name, "Alpine");
    }

    #[test]
    fn test_gpu_data() {
        let table = GrassProfileTable::default();
        let data = table.gpu_data();
        assert_eq!(data.len(), 6);
        // None profile should have zero density
        assert_eq!(data[0].density, 0.0);
        // Tall Grass should have high density
        assert!(data[1].density > 0.8);
    }

    #[test]
    fn test_biome_mapping() {
        // Grassland -> Tall Grass
        assert_eq!(GrassProfileTable::biome_default_profile(3), GrassProfile(1));
        // Forest -> Meadow
        assert_eq!(GrassProfileTable::biome_default_profile(4), GrassProfile(2));
        // Ocean -> None
        assert_eq!(GrassProfileTable::biome_default_profile(0), GrassProfile::NONE);
        // Desert -> None
        assert_eq!(GrassProfileTable::biome_default_profile(2), GrassProfile::NONE);
    }

    #[test]
    fn test_bytemuck_cast() {
        let p = GpuGrassProfile::zeroed();
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 64);
    }
}
