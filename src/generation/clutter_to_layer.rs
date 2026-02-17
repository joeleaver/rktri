//! Clutter-to-layer conversion — converts clutter mask to rock/vegetation octrees.
//!
//! This module bridges the generation pipeline by:
//! 1. Sampling the clutter mask at configured intervals
//! 2. Using ClutterProfileTable to look up voxel_object_id from profile + variant
//! 3. Using RockGenerator to create rock geometry
//! 4. Merging rocks into rock octrees, trees into vegetation octrees

use glam::Vec3;
use rayon::prelude::*;

use crate::clutter::profile::{ClutterCell, ClutterProfile, ClutterProfileTable};
use crate::mask::MaskOctree;
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::rock_library::{RockGenerator, RockParams};
use crate::voxel::svo::Octree;

/// Configuration for clutter-to-layer conversion.
#[derive(Clone, Debug)]
pub struct ClutterLayerConfig {
    /// Sample spacing for clutter placement (in meters)
    pub sample_spacing: f32,
    /// Voxel object ID range for rocks (inclusive)
    pub rock_id_range: (u16, u16),
    /// Voxel object ID range for vegetation (inclusive)
    pub vegetation_id_range: (u16, u16),
    /// Maximum rocks per chunk (for performance)
    pub max_rocks_per_chunk: usize,
    /// Maximum vegetation per chunk (for performance)
    pub max_vegetation_per_chunk: usize,
}

impl Default for ClutterLayerConfig {
    fn default() -> Self {
        Self {
            sample_spacing: 1.0,
            rock_id_range: (3, 17),  // Rocks: 3, 11-17
            vegetation_id_range: (5, 10), // Vegetation: 5-10 (trees, flowers, etc.)
            max_rocks_per_chunk: 100,
            max_vegetation_per_chunk: 50,
        }
    }
}

/// Result of converting clutter to layer octrees.
#[derive(Clone, Debug)]
pub struct LayerOctrees {
    /// Rock octree for this chunk (layer 2)
    pub rocks_octree: Octree,
    /// Vegetation octree for this chunk (layer 3)
    pub vegetation_octree: Octree,
    /// Number of rocks placed
    pub rock_count: usize,
    /// Number of vegetation placed
    pub vegetation_count: usize,
}

impl Default for LayerOctrees {
    fn default() -> Self {
        Self {
            rocks_octree: Octree::new(CHUNK_SIZE as f32, 7),
            vegetation_octree: Octree::new(CHUNK_SIZE as f32, 7),
            rock_count: 0,
            vegetation_count: 0,
        }
    }
}

/// Converts clutter mask cells to rock and vegetation octrees.
///
/// This converter:
/// 1. Iterates through clutter mask positions at sample spacing
/// 2. For each occupied cell, looks up the voxel_object_id from ClutterProfileTable
/// 3. Generates rock geometry for rock IDs, stores vegetation IDs for later processing
pub struct ClutterToLayerConverter {
    config: ClutterLayerConfig,
    profile_table: ClutterProfileTable,
    seed: u32,
}

impl ClutterToLayerConverter {
    /// Create a new converter with configuration.
    pub fn new(config: ClutterLayerConfig, seed: u32) -> Self {
        Self {
            config,
            profile_table: ClutterProfileTable::default(),
            seed,
        }
    }

    /// Convert clutter mask to rock and vegetation octrees.
    ///
    /// # Arguments
    /// * `clutter_mask` - MaskOctree containing ClutterCell values
    /// * `chunk_origin` - World position of chunk origin
    /// * `terrain_octree` - Terrain octree for surface height reference
    pub fn convert(
        &self,
        clutter_mask: &MaskOctree<ClutterCell>,
        chunk_origin: Vec3,
        terrain_octree: &Octree,
    ) -> LayerOctrees {
        let mut result = LayerOctrees::default();
        let chunk_size = CHUNK_SIZE as f32;

        // Collect placement candidates
        let mut rock_candidates: Vec<(Vec3, u16)> = Vec::new();
        let mut vegetation_candidates: Vec<(Vec3, u16)> = Vec::new();

        // Sample clutter mask at regular intervals
        let spacing = self.config.sample_spacing;
        let steps = (chunk_size / spacing).ceil() as i32;

        for ix in 0..=steps {
            for iz in 0..=steps {
                let wx = chunk_origin.x + ix as f32 * spacing;
                let wz = chunk_origin.z + iz as f32 * spacing;

                // Sample clutter at this position (using terrain height as Y)
                let terrain_height = self.get_surface_height(terrain_octree, wx, wz, chunk_origin.y);
                if terrain_height < chunk_origin.y {
                    continue;
                }

                let sample_pos = Vec3::new(wx, terrain_height, wz);
                let local_pos = sample_pos - chunk_origin;
                let cell = clutter_mask.sample(chunk_origin, local_pos);

                if cell.is_none() {
                    continue;
                }

                // Look up the clutter object
                let voxel_object_id = self.lookup_voxel_object_id(cell);
                if voxel_object_id == 0 {
                    continue;
                }

                // Check if it's a rock or vegetation
                let (min_id, max_id) = self.config.rock_id_range;
                if voxel_object_id >= min_id && voxel_object_id <= max_id {
                    if rock_candidates.len() < self.config.max_rocks_per_chunk {
                        rock_candidates.push((sample_pos, voxel_object_id));
                    }
                } else {
                    let (vmin, vmax) = self.config.vegetation_id_range;
                    if voxel_object_id >= vmin && voxel_object_id <= vmax {
                        if vegetation_candidates.len() < self.config.max_vegetation_per_chunk {
                            vegetation_candidates.push((sample_pos, voxel_object_id));
                        }
                    }
                }
            }
        }

        // Generate rock octree
        if !rock_candidates.is_empty() {
            result.rocks_octree = self.generate_rock_octree(&rock_candidates, chunk_origin);
            result.rock_count = rock_candidates.len();
        }

        // For vegetation, we just track the positions for now
        // Full tree generation would require a tree generator
        if !vegetation_candidates.is_empty() {
            result.vegetation_count = vegetation_candidates.len();
            // TODO: Generate vegetation octree when tree generator is available
            // For now, create a placeholder with just positions
            result.vegetation_octree = self.generate_vegetation_placeholder(
                &vegetation_candidates,
                chunk_origin,
            );
        }

        result
    }

    /// Look up voxel_object_id from a ClutterCell using the profile table.
    fn lookup_voxel_object_id(&self, cell: ClutterCell) -> u16 {
        if cell.is_none() {
            return 0;
        }

        let profile = cell.profile();
        if profile == ClutterProfile::NONE {
            return 0;
        }

        let profile_def = self.profile_table.get(profile.0 as u8);

        // Use variant to select which object from the profile
        let variant = cell.variant() as usize;
        let object_idx = variant % profile_def.objects.len().max(1);

        profile_def
            .objects
            .get(object_idx)
            .map(|obj| obj.voxel_object_id)
            .unwrap_or(0)
    }

    /// Get surface height from terrain octree at world position.
    fn get_surface_height(&self, octree: &Octree, x: f32, z: f32, chunk_base_y: f32) -> f32 {
        // Sample at multiple heights to find the surface
        let chunk_size = CHUNK_SIZE as f32;
        let mut best_y = chunk_base_y;

        // Convert world position to local octree position
        // Octree is centered at chunk origin
        let half = chunk_size / 2.0;

        // Simple approach: sample at the top of the chunk and work down
        for y_try in (chunk_base_y as i32..=(chunk_base_y + chunk_size) as i32).rev() {
            let local_x = x - (chunk_base_y + half); // Approximate - needs proper origin
            let local_y = y_try as f32 - chunk_base_y - half;
            let local_z = z - (chunk_base_y + half);

            let local_pos = Vec3::new(local_x, local_y, local_z);
            let voxel = octree.sample_voxel(local_pos);
            if !voxel.is_empty() {
                best_y = y_try as f32;
                break;
            }
        }

        best_y
    }

    /// Generate rock octree from placement candidates.
    fn generate_rock_octree(
        &self,
        candidates: &[(Vec3, u16)],
        chunk_origin: Vec3,
    ) -> Octree {
        log::debug!("generate_rock_octree: {} candidates at chunk_origin={:?}", candidates.len(), chunk_origin);
        if candidates.is_empty() {
            return Octree::new(CHUNK_SIZE as f32, 7);
        }

        // Convert world positions to local chunk coordinates
        let local_candidates: Vec<(Vec3, u16)> = candidates
            .iter()
            .map(|&(world_pos, voxel_id)| {
                let local_pos = world_pos - chunk_origin;
                (local_pos, voxel_id)
            })
            .collect();

        // Generate individual rock octrees at LOCAL positions and merge
        let mut merged = Octree::new(CHUNK_SIZE as f32, 7);

        for (i, &(local_pos, voxel_id)) in local_candidates.iter().enumerate() {
            let rock_seed = self.seed.wrapping_add((i as u32).wrapping_mul(12345)) as u64;
            let rock = self.generate_rock_at(local_pos, voxel_id, rock_seed);
            if rock.brick_count() > 0 {
                merged = merged.merge_union(&rock);
            }
        }

        merged
    }

    /// Generate a single rock at the given position.
    fn generate_rock_at(&self, _pos: Vec3, voxel_id: u16, seed: u64) -> Octree {
        // Select rock params based on voxel_id
        let params = match voxel_id {
            3 => RockParams::small_boulder(),
            11 => RockParams::large_boulder(),
            12 => RockParams::snowy_rock(),
            13 => RockParams::medium_rock(),  // ice chunk
            14 => RockParams::snowy_rock(),  // snow rock
            15 => RockParams::flat_rock(),
            16 => RockParams::medium_rock(),
            17 => RockParams::huge_boulder(),
            _ => RockParams::random(seed),
        };

        let height = params.height;
        let mut generator = RockGenerator::with_params(seed, params);
        generator.generate(height)
    }

    /// Generate placeholder vegetation octree.
    /// TODO: Replace with proper tree generation
    fn generate_vegetation_placeholder(
        &self,
        candidates: &[(Vec3, u16)],
        _chunk_origin: Vec3,
    ) -> Octree {
        // For now, return empty octree
        // Full implementation would use tree generator
        if candidates.is_empty() {
            return Octree::new(CHUNK_SIZE as f32, 7);
        }

        // Placeholder: create tiny voxels at vegetation positions
        // This is a placeholder until tree generation is implemented
        Octree::new(CHUNK_SIZE as f32, 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::MaskBuilder;
    use crate::voxel::svo::adaptive::AdaptiveOctreeBuilder;

    #[test]
    fn test_converter_create() {
        let config = ClutterLayerConfig::default();
        let converter = ClutterToLayerConverter::new(config, 12345);
        assert_eq!(converter.config.sample_spacing, 1.0);
    }

    #[test]
    fn test_voxel_id_lookup() {
        let config = ClutterLayerConfig::default();
        let converter = ClutterToLayerConverter::new(config, 12345);

        // Create a cell with profile 3 (mountains) and variant 128
        let cell = ClutterCell::new(ClutterProfile(3), 128);
        let voxel_id = converter.lookup_voxel_object_id(cell);

        // Should get a valid rock ID from mountains profile
        assert!(voxel_id > 0);
    }

    #[test]
    fn test_clutter_cell_packing() {
        let cell = ClutterCell::new(ClutterProfile(5), 42);
        assert_eq!(cell.profile(), ClutterProfile(5));
        assert_eq!(cell.variant(), 42);
    }
}
