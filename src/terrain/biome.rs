//! Biome system based on temperature and moisture

use noise::{NoiseFn, Perlin};

/// Biome types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Biome {
    Ocean,
    Beach,
    Desert,
    Grassland,
    Forest,
    Taiga,
    Tundra,
    Mountains,
    Snow,
}

impl Biome {
    /// Convert biome to a numeric BiomeId for mask storage.
    pub fn to_id(&self) -> crate::mask::BiomeId {
        use crate::mask::BiomeId;
        match self {
            Biome::Ocean => BiomeId::OCEAN,
            Biome::Beach => BiomeId::BEACH,
            Biome::Desert => BiomeId::DESERT,
            Biome::Grassland => BiomeId::GRASSLAND,
            Biome::Forest => BiomeId::FOREST,
            Biome::Taiga => BiomeId::TAIGA,
            Biome::Tundra => BiomeId::TUNDRA,
            Biome::Mountains => BiomeId::MOUNTAINS,
            Biome::Snow => BiomeId::SNOW,
        }
    }

    /// Convert a numeric BiomeId back to a Biome enum.
    pub fn from_id(id: crate::mask::BiomeId) -> Self {
        use crate::mask::BiomeId;
        match id {
            BiomeId::BEACH => Biome::Beach,
            BiomeId::DESERT => Biome::Desert,
            BiomeId::GRASSLAND => Biome::Grassland,
            BiomeId::FOREST => Biome::Forest,
            BiomeId::TAIGA => Biome::Taiga,
            BiomeId::TUNDRA => Biome::Tundra,
            BiomeId::MOUNTAINS => Biome::Mountains,
            BiomeId::SNOW => Biome::Snow,
            _ => Biome::Ocean, // Default/unknown → Ocean
        }
    }

    /// Get surface voxel color for this biome.
    /// Material IDs 7-15 are biome-specific terrain surfaces.
    /// The actual color is stored in the shader LUT (terrain_base_color),
    /// since the voxel color field encodes terrain gradient for smooth normals.
    pub fn surface_color(&self) -> crate::voxel::voxel::Voxel {
        use crate::voxel::voxel::Voxel;

        match self {
            Biome::Ocean => Voxel::new(30, 80, 150, 15),      // Ocean floor
            Biome::Beach => Voxel::new(238, 214, 175, 7),     // Beach sand
            Biome::Desert => Voxel::new(237, 201, 175, 8),    // Desert sand
            Biome::Grassland => Voxel::new(100, 180, 80, 9),  // Grassland
            Biome::Forest => Voxel::new(50, 120, 40, 10),     // Forest
            Biome::Taiga => Voxel::new(80, 100, 60, 11),      // Taiga
            Biome::Tundra => Voxel::new(160, 180, 170, 12),   // Tundra
            Biome::Mountains => Voxel::new(120, 120, 120, 13), // Mountains
            Biome::Snow => Voxel::new(240, 248, 255, 14),     // Snow
        }
    }

    /// Get underground color for this biome
    pub fn underground_color(&self) -> crate::voxel::voxel::Voxel {
        use crate::voxel::voxel::Voxel;

        match self {
            Biome::Ocean => Voxel::new(100, 100, 120, 4),    // Dark stone
            Biome::Beach => Voxel::new(180, 160, 140, 1),    // Packed sand
            Biome::Desert => Voxel::new(200, 180, 150, 4),   // Sandstone
            Biome::Grassland => Voxel::new(130, 100, 70, 6), // Brown dirt
            Biome::Forest => Voxel::new(110, 85, 60, 6),     // Rich soil
            Biome::Taiga => Voxel::new(140, 110, 80, 6),     // Rocky soil
            Biome::Tundra => Voxel::new(120, 120, 130, 4),   // Frozen earth
            Biome::Mountains => Voxel::new(90, 90, 90, 4),   // Deep stone
            Biome::Snow => Voxel::new(100, 100, 110, 4),     // Ice/stone
        }
    }

    /// Whether this biome can have vegetation
    pub fn has_vegetation(&self) -> bool {
        matches!(
            self,
            Biome::Grassland | Biome::Forest | Biome::Taiga
        )
    }

    /// Vegetation density (0.0-1.0)
    pub fn vegetation_density(&self) -> f32 {
        match self {
            Biome::Forest => 0.8,
            Biome::Taiga => 0.5,
            Biome::Grassland => 0.3,
            _ => 0.0,
        }
    }
}

/// Biome map generator
pub struct BiomeMap {
    temperature_noise: Perlin,
    moisture_noise: Perlin,
    temp_scale: f32,
    moisture_scale: f32,
}

impl BiomeMap {
    /// Create new biome map with given seed
    pub fn new(seed: u32) -> Self {
        Self {
            temperature_noise: Perlin::new(seed),
            moisture_noise: Perlin::new(seed.wrapping_add(1000)),
            temp_scale: 0.0008,    // Large-scale temperature zones
            moisture_scale: 0.0012, // Medium-scale moisture patterns
        }
    }

    /// Get temperature at world position (-1 to 1, cold to hot)
    pub fn temperature_at(&self, x: f32, z: f32) -> f32 {
        // Base temperature from noise
        let temp = self.temperature_noise.get([
            (x * self.temp_scale) as f64,
            (z * self.temp_scale) as f64,
        ]) as f32;

        // Add latitude gradient (gets colder towards edges)
        let latitude_factor = (z * 0.0002).abs().min(1.0);
        temp - latitude_factor * 0.5
    }

    /// Get moisture at world position (0 to 1, dry to wet)
    pub fn moisture_at(&self, x: f32, z: f32) -> f32 {
        let moisture = self.moisture_noise.get([
            (x * self.moisture_scale) as f64,
            (z * self.moisture_scale) as f64,
        ]) as f32;

        // Normalize from [-1, 1] to [0, 1]
        (moisture + 1.0) * 0.5
    }

    /// Get biome at world position based on temp, moisture, and height
    pub fn biome_at(&self, x: f32, z: f32, height: f32, sea_level: f32) -> Biome {
        let temp = self.temperature_at(x, z);
        let moisture = self.moisture_at(x, z);

        // Water biomes
        if height < sea_level - 2.0 {
            return Biome::Ocean;
        }
        if height < sea_level + 1.0 && height >= sea_level - 2.0 {
            return Biome::Beach;
        }

        // High altitude biomes
        let altitude_factor = (height - sea_level) / 100.0;
        if altitude_factor > 0.8 {
            return Biome::Snow;
        }
        if altitude_factor > 0.6 {
            return Biome::Mountains;
        }

        // Temperature-moisture based biomes
        match (temp, moisture) {
            // Cold biomes (temp < -0.3)
            (t, m) if t < -0.3 => {
                if m > 0.5 {
                    Biome::Taiga // Cold + wet
                } else {
                    Biome::Tundra // Cold + dry
                }
            }
            // Hot biomes (temp > 0.3)
            (t, m) if t > 0.3 => {
                if m > 0.6 {
                    Biome::Forest // Hot + very wet (tropical)
                } else if m > 0.3 {
                    Biome::Grassland // Hot + moderate moisture
                } else {
                    Biome::Desert // Hot + dry
                }
            }
            // Temperate biomes (-0.3 to 0.3)
            (_, m) => {
                if m > 0.6 {
                    Biome::Forest // Temperate forest
                } else if m > 0.3 {
                    Biome::Grassland // Temperate grassland
                } else {
                    Biome::Desert // Temperate desert
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biome_colors() {
        // Ensure all biomes return valid voxels
        for biome in [
            Biome::Ocean,
            Biome::Beach,
            Biome::Desert,
            Biome::Grassland,
            Biome::Forest,
            Biome::Taiga,
            Biome::Tundra,
            Biome::Mountains,
            Biome::Snow,
        ] {
            let surface = biome.surface_color();
            let underground = biome.underground_color();

            // Colors should not be empty
            assert!(!surface.is_empty());
            assert!(!underground.is_empty());
        }
    }

    #[test]
    fn test_vegetation() {
        assert!(Biome::Forest.has_vegetation());
        assert!(Biome::Grassland.has_vegetation());
        assert!(Biome::Taiga.has_vegetation());
        assert!(!Biome::Desert.has_vegetation());
        assert!(!Biome::Ocean.has_vegetation());

        // Check density ranges
        assert_eq!(Biome::Forest.vegetation_density(), 0.8);
        assert_eq!(Biome::Desert.vegetation_density(), 0.0);
    }

    #[test]
    fn test_temperature_range() {
        let biome_map = BiomeMap::new(12345);

        // Test multiple positions
        for x in [-1000.0, 0.0, 1000.0] {
            for z in [-1000.0, 0.0, 1000.0] {
                let temp = biome_map.temperature_at(x, z);
                // Temperature should be reasonable range
                assert!(temp >= -2.0 && temp <= 2.0, "temp {} out of range", temp);
            }
        }
    }

    #[test]
    fn test_moisture_range() {
        let biome_map = BiomeMap::new(12345);

        // Test multiple positions
        for x in [-1000.0, 0.0, 1000.0] {
            for z in [-1000.0, 0.0, 1000.0] {
                let moisture = biome_map.moisture_at(x, z);
                // Moisture should be in [0, 1]
                assert!(moisture >= 0.0 && moisture <= 1.0, "moisture {} out of range", moisture);
            }
        }
    }

    #[test]
    fn test_ocean_biome() {
        let biome_map = BiomeMap::new(12345);
        let sea_level = 64.0;

        // Below sea level should be ocean
        let biome = biome_map.biome_at(0.0, 0.0, sea_level - 10.0, sea_level);
        assert_eq!(biome, Biome::Ocean);
    }

    #[test]
    fn test_beach_biome() {
        let biome_map = BiomeMap::new(12345);
        let sea_level = 64.0;

        // At sea level should be beach
        let biome = biome_map.biome_at(0.0, 0.0, sea_level, sea_level);
        assert_eq!(biome, Biome::Beach);
    }

    #[test]
    fn test_high_altitude_biomes() {
        let biome_map = BiomeMap::new(12345);
        let sea_level = 64.0;

        // Very high altitude should be snow
        let biome = biome_map.biome_at(0.0, 0.0, sea_level + 100.0, sea_level);
        assert_eq!(biome, Biome::Snow);

        // High altitude should be mountains
        let biome = biome_map.biome_at(0.0, 0.0, sea_level + 70.0, sea_level);
        assert_eq!(biome, Biome::Mountains);
    }

    #[test]
    fn test_temp_moisture_biomes() {
        let biome_map = BiomeMap::new(12345);
        let sea_level = 64.0;
        let height = sea_level + 10.0; // Above water, below mountains

        // Find positions with different temp/moisture combinations
        let mut found_biomes = std::collections::HashSet::new();

        // Sample various positions to find different biomes
        for x in (0..20).map(|i| i as f32 * 500.0) {
            for z in (0..20).map(|i| i as f32 * 500.0) {
                let biome = biome_map.biome_at(x, z, height, sea_level);
                found_biomes.insert(biome);
            }
        }

        // Should find at least a few different land biomes
        assert!(found_biomes.len() >= 2, "Expected variety in biomes, found {:?}", found_biomes);
    }

    #[test]
    fn test_biome_determinism() {
        let biome_map = BiomeMap::new(12345);
        let sea_level = 64.0;

        // Same position should always return same biome
        let biome1 = biome_map.biome_at(100.0, 200.0, sea_level + 10.0, sea_level);
        let biome2 = biome_map.biome_at(100.0, 200.0, sea_level + 10.0, sea_level);
        assert_eq!(biome1, biome2);
    }
}
