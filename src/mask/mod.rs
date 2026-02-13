//! Mask layers: sparse 3D octrees for spatial classification data.
//!
//! Masks are the spatial primitive for biome classification, grass density,
//! decoration scattering, and future editor paint operations. Each mask is
//! a chunk-aligned octree with configurable resolution per mask type.

pub mod octree;
pub mod builder;

pub use octree::{MaskOctree, MaskNode};
pub use builder::{MaskGenerator, MaskHint, MaskBuilder};

/// Trait for values stored in mask octrees.
///
/// Must be cheap to copy, have a meaningful default (typically "no data"),
/// and support equality for tree pruning.
pub trait MaskValue: Copy + Clone + Default + PartialEq + Send + Sync + 'static {
    /// Returns true if this value is the default (empty/unset).
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

// Built-in impls
impl MaskValue for f32 {}
impl MaskValue for u8 {}

/// Biome identifier — mirrors `terrain::biome::Biome` variants as a compact u8.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BiomeId(pub u8);

impl BiomeId {
    pub const OCEAN: Self = Self(0);
    pub const BEACH: Self = Self(1);
    pub const DESERT: Self = Self(2);
    pub const GRASSLAND: Self = Self(3);
    pub const FOREST: Self = Self(4);
    pub const TAIGA: Self = Self(5);
    pub const TUNDRA: Self = Self(6);
    pub const MOUNTAINS: Self = Self(7);
    pub const SNOW: Self = Self(8);
}

impl MaskValue for BiomeId {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_value_default() {
        assert!(f32::default().is_default());
        assert!(u8::default().is_default());
        assert!(BiomeId::default().is_default());
    }

    #[test]
    fn test_mask_value_non_default() {
        assert!(!1.0_f32.is_default());
        assert!(!1_u8.is_default());
        assert!(!BiomeId::FOREST.is_default());
    }

    #[test]
    fn test_biome_id_constants() {
        assert_eq!(BiomeId::OCEAN.0, 0);
        assert_eq!(BiomeId::SNOW.0, 8);
        assert_ne!(BiomeId::FOREST, BiomeId::GRASSLAND);
    }
}
