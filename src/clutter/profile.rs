//! Clutter profile types: per-biome clutter object definitions.
//!
//! `ClutterProfile` is a thin index type implementing `MaskValue` for storage
//! in `MaskOctree<ClutterCell>`. `ClutterObject` defines individual clutter
//! objects with placement rules. `ClutterProfileTable` maps biomes to clutter
//! configurations.

use bytemuck::{Pod, Zeroable};
use crate::mask::MaskValue;

/// Index into the clutter profile table. 0 = no clutter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ClutterProfile(pub u8);

impl ClutterProfile {
    pub const NONE: Self = Self(0);
}

impl MaskValue for ClutterProfile {}

/// Clutter cell stored in mask octrees: profile index + variant packed into u16.
///
/// - Low 8 bits: profile index (0 = no clutter, 1-255 = profile ID)
/// - High 8 bits: variant (0-255, for random variation in placement)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ClutterCell(pub u16);

impl ClutterCell {
    pub const NONE: Self = Self(0);

    /// Create a clutter cell with profile and variant.
    pub fn new(profile: ClutterProfile, variant: u8) -> Self {
        if profile == ClutterProfile::NONE {
            return Self::NONE;
        }
        Self((variant as u16) << 8 | profile.0 as u16)
    }

    /// Extract the profile index.
    pub fn profile(self) -> ClutterProfile {
        ClutterProfile((self.0 & 0xFF) as u8)
    }

    /// Extract the variant.
    pub fn variant(self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Returns true if this cell has no clutter.
    pub fn is_none(self) -> bool {
        (self.0 & 0xFF) == 0
    }
}

impl MaskValue for ClutterCell {}

/// Placement rules for a single clutter object type.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ClutterObject {
    /// Reference to voxel object ID in ClutterLibrary
    pub voxel_object_id: u16,
    /// Random rotation variations (number of discrete rotations)
    pub rotation_variations: u8,
    /// Padding for alignment
    pub _pad1: [u8; 1],
    /// Chance to place (0.0-1.0)
    pub probability: f32,
    /// Minimum terrain slope (0.0 = flat, 1.0 = vertical)
    pub min_slope: f32,
    /// Maximum terrain slope
    pub max_slope: f32,
    /// Minimum world height for placement (meters)
    pub min_height: f32,
    /// Maximum world height for placement
    pub max_height: f32,
    /// Scale variation range (min, max)
    pub scale_range: [f32; 2],
    /// Padding for alignment (to 48 bytes)
    pub _pad2: [f32; 8],
}

impl Default for ClutterObject {
    fn default() -> Self {
        Self {
            voxel_object_id: 0,
            probability: 0.5,
            min_slope: 0.0,
            max_slope: 0.5,
            min_height: 0.0,
            max_height: 1000.0,
            rotation_variations: 4,
            scale_range: [0.8, 1.2],
            _pad1: [0],
            _pad2: [0.0; 8],
        }
    }
}

/// A clutter profile defines a collection of clutter objects for a biome.
#[derive(Clone, Debug)]
pub struct ClutterProfileDef {
    pub name: String,
    pub objects: Vec<ClutterObject>,
}

/// Table of clutter profile definitions per biome.
#[derive(Clone, Debug)]
pub struct ClutterProfileTable {
    /// Profiles indexed by biome ID (0-8)
    profiles: Vec<ClutterProfileDef>,
}

impl ClutterProfileTable {
    /// Get a profile by biome ID.
    pub fn get(&self, biome_id: u8) -> &ClutterProfileDef {
        self.profiles.get(biome_id as usize).unwrap_or(&self.profiles[0])
    }

    /// Get a mutable reference.
    pub fn get_mut(&mut self, biome_id: u8) -> &mut ClutterProfileDef {
        if biome_id as usize >= self.profiles.len() {
            return &mut self.profiles[0];
        }
        &mut self.profiles[biome_id as usize]
    }

    /// Get the default profile for a biome (returns profile index 1 + biome_id).
    pub fn biome_default_profile(biome_id: u8) -> ClutterProfile {
        // Profile indices: 1-9 for biomes 0-8
        ClutterProfile(biome_id.saturating_add(1).min(9))
    }

    /// Number of profiles (including NONE sentinel at index 0).
    pub fn len(&self) -> usize {
        self.profiles.len()
    }
}

impl Default for ClutterProfileTable {
    fn default() -> Self {
        Self {
            profiles: vec![
                // 0: None (Ocean)
                ClutterProfileDef {
                    name: "None".into(),
                    objects: vec![],
                },
                // 1: Beach (biome 0) - shells, driftwood
                ClutterProfileDef {
                    name: "Beach".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 1,  // shell
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 0.0,
                            max_height: 25.0,
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 2,  // driftwood
                            probability: 0.08,
                            min_slope: 0.0,
                            max_slope: 0.2,
                            min_height: 0.0,
                            max_height: 22.0,
                            ..Default::default()
                        },
                    ],
                },
                // 2: Desert (biome 1) - rocks, tumbleweeds
                ClutterProfileDef {
                    name: "Desert".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.20,
                            min_slope: 0.0,
                            max_slope: 0.6,
                            min_height: 20.0,
                            max_height: 85.0,
                            scale_range: [0.5, 1.2],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 15,  // flat rock
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 20.0,
                            max_height: 70.0,
                            scale_range: [0.8, 1.5],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 4,  // tumbleweed
                            probability: 0.10,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 20.0,
                            max_height: 60.0,
                            ..Default::default()
                        },
                    ],
                },
                // 3: Grassland (biome 2) - rocks, wildflowers, flat rocks
                ClutterProfileDef {
                    name: "Grassland".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.12,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 20.0,
                            max_height: 60.0,
                            scale_range: [0.6, 1.0],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 15,  // flat rock
                            probability: 0.18,
                            min_slope: 0.0,
                            max_slope: 0.35,
                            min_height: 18.0,
                            max_height: 55.0,
                            scale_range: [0.8, 1.6],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 16,  // medium rock
                            probability: 0.08,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 22.0,
                            max_height: 50.0,
                            scale_range: [0.7, 1.2],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 5,  // wildflowers
                            probability: 0.20,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 20.0,
                            max_height: 50.0,
                            ..Default::default()
                        },
                    ],
                },
                // 4: Forest (biome 3) - sticks, logs, mushrooms, rocks, flat rocks
                ClutterProfileDef {
                    name: "Forest".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 6,  // stick
                            probability: 0.28,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 20.0,
                            max_height: 60.0,
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 7,  // fallen log
                            probability: 0.10,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 20.0,
                            max_height: 50.0,
                            scale_range: [1.0, 2.0],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 8,  // mushroom
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.35,
                            min_height: 25.0,
                            max_height: 55.0,
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.10,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 20.0,
                            max_height: 55.0,
                            scale_range: [0.6, 1.1],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 15,  // flat rock
                            probability: 0.12,
                            min_slope: 0.0,
                            max_slope: 0.35,
                            min_height: 18.0,
                            max_height: 50.0,
                            scale_range: [0.7, 1.3],
                            ..Default::default()
                        },
                    ],
                },
                // 5: Taiga (biome 4) - pine cones, fallen branches, rocks
                ClutterProfileDef {
                    name: "Taiga".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 9,  // pine cone
                            probability: 0.22,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 30.0,
                            max_height: 70.0,
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 6,  // stick
                            probability: 0.20,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 30.0,
                            max_height: 65.0,
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 30.0,
                            max_height: 75.0,
                            scale_range: [0.5, 1.0],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 16,  // medium rock
                            probability: 0.08,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 35.0,
                            max_height: 65.0,
                            scale_range: [0.6, 1.1],
                            ..Default::default()
                        },
                    ],
                },
                // 6: Tundra (biome 5) - rocks, moss, flat rocks
                ClutterProfileDef {
                    name: "Tundra".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.18,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 40.0,
                            max_height: 80.0,
                            scale_range: [0.5, 1.1],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 15,  // flat rock
                            probability: 0.12,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 38.0,
                            max_height: 75.0,
                            scale_range: [0.8, 1.4],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 16,  // medium rock
                            probability: 0.08,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 45.0,
                            max_height: 70.0,
                            scale_range: [0.6, 1.0],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 10,  // moss patch
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.3,
                            min_height: 45.0,
                            max_height: 75.0,
                            ..Default::default()
                        },
                    ],
                },
                // 7: Mountains (biome 6) - variety of rocks: small, medium, large, snow-covered
                ClutterProfileDef {
                    name: "Mountains".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.25,
                            min_slope: 0.05,
                            max_slope: 0.7,
                            min_height: 55.0,
                            max_height: 125.0,
                            scale_range: [0.4, 0.9],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 16,  // medium rock
                            probability: 0.20,
                            min_slope: 0.1,
                            max_slope: 0.75,
                            min_height: 58.0,
                            max_height: 115.0,
                            scale_range: [0.6, 1.2],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 11,  // large rock
                            probability: 0.18,
                            min_slope: 0.15,
                            max_slope: 0.85,
                            min_height: 65.0,
                            max_height: 120.0,
                            scale_range: [0.5, 1.6],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 17,  // huge rock
                            probability: 0.10,
                            min_slope: 0.2,
                            max_slope: 0.9,
                            min_height: 70.0,
                            max_height: 115.0,
                            scale_range: [0.7, 1.8],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 12,  // snow boulder
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.6,
                            min_height: 68.0,
                            max_height: 110.0,
                            scale_range: [0.5, 1.3],
                            ..Default::default()
                        },
                    ],
                },
                // 8: Snow (biome 7) - ice chunks, snow rocks, frozen rocks
                ClutterProfileDef {
                    name: "Snow".into(),
                    objects: vec![
                        ClutterObject {
                            voxel_object_id: 13,  // ice chunk
                            probability: 0.12,
                            min_slope: 0.0,
                            max_slope: 0.4,
                            min_height: 70.0,
                            max_height: 120.0,
                            scale_range: [0.6, 1.2],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 14,  // snow rock
                            probability: 0.15,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 72.0,
                            max_height: 115.0,
                            scale_range: [0.5, 1.2],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 3,  // small rock
                            probability: 0.10,
                            min_slope: 0.0,
                            max_slope: 0.5,
                            min_height: 68.0,
                            max_height: 110.0,
                            scale_range: [0.4, 0.8],
                            ..Default::default()
                        },
                        ClutterObject {
                            voxel_object_id: 16,  // medium rock
                            probability: 0.08,
                            min_slope: 0.0,
                            max_slope: 0.45,
                            min_height: 75.0,
                            max_height: 105.0,
                            scale_range: [0.5, 1.0],
                            ..Default::default()
                        },
                    ],
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clutter_profile_is_mask_value() {
        assert!(ClutterProfile::NONE.is_default());
        assert!(!ClutterProfile(1).is_default());
    }

    #[test]
    fn test_clutter_cell_new() {
        let cell = ClutterCell::new(ClutterProfile(3), 42);
        assert_eq!(cell.profile(), ClutterProfile(3));
        assert_eq!(cell.variant(), 42);
    }

    #[test]
    fn test_clutter_cell_none() {
        let cell = ClutterCell::new(ClutterProfile::NONE, 0);
        assert!(cell.is_none());
    }

    #[test]
    fn test_default_profile_table() {
        let table = ClutterProfileTable::default();
        assert_eq!(table.len(), 9); // None + 8 biomes
        
        // Check biome mappings
        assert_eq!(table.get(0).name, "None");
        assert_eq!(table.get(1).name, "Beach");
        assert_eq!(table.get(4).name, "Forest");
        assert_eq!(table.get(8).name, "Snow");
    }

    #[test]
    fn test_biome_default_profile() {
        assert_eq!(ClutterProfileTable::biome_default_profile(0), ClutterProfile(1)); // Ocean -> Beach
        assert_eq!(ClutterProfileTable::biome_default_profile(4), ClutterProfile(5)); // Forest -> profile 5
        assert_eq!(ClutterProfileTable::biome_default_profile(8), ClutterProfile(9)); // Snow -> profile 9
    }

    #[test]
    fn test_clutter_object_size() {
        assert_eq!(std::mem::size_of::<ClutterObject>(), 64);
    }

    #[test]
    fn test_forest_profile_has_objects() {
        let table = ClutterProfileTable::default();
        let forest = table.get(4);
        assert!(!forest.objects.is_empty());
        assert!(forest.objects.len() >= 4);
    }
}
