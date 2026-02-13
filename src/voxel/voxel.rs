//! Voxel data type

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};

/// Convert RGB888 to RGB565
pub fn rgb_to_565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 >> 3) & 0x1F;
    let g6 = (g as u16 >> 2) & 0x3F;
    let b5 = (b as u16 >> 3) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

/// Convert RGB565 to RGB888
pub fn rgb565_to_rgb(color: u16) -> (u8, u8, u8) {
    let r5 = (color >> 11) & 0x1F;
    let g6 = (color >> 5) & 0x3F;
    let b5 = color & 0x1F;
    (
        ((r5 << 3) | (r5 >> 2)) as u8,
        ((g6 << 2) | (g6 >> 4)) as u8,
        ((b5 << 3) | (b5 >> 2)) as u8,
    )
}

/// Voxel flags
pub mod flags {
    pub const TRANSPARENT: u8 = 1 << 0;
    pub const EMISSIVE: u8 = 1 << 1;
}

/// Single voxel - exactly 4 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable, Archive, Deserialize, Serialize)]
pub struct Voxel {
    /// RGB565 encoded color
    pub color: u16,
    /// Material ID (index into PBR material LUT)
    pub material_id: u8,
    /// Flags (transparent, emissive, etc.)
    pub flags: u8,
}

impl Voxel {
    /// Empty/air voxel
    pub const EMPTY: Voxel = Voxel {
        color: 0,
        material_id: 0,
        flags: 0,
    };

    /// Create voxel from RGB888 values
    pub fn new(r: u8, g: u8, b: u8, material_id: u8) -> Self {
        Self {
            color: rgb_to_565(r, g, b),
            material_id,
            flags: 0,
        }
    }

    /// Create voxel from RGB565 color
    pub fn from_rgb565(color: u16, material_id: u8) -> Self {
        Self {
            color,
            material_id,
            flags: 0,
        }
    }

    /// Create a copy of this voxel with the given flags value
    pub fn with_flags_value(self, flags: u8) -> Self {
        Self { flags, ..self }
    }

    /// Create a copy with both color and flags overridden
    pub fn with_color_and_flags(self, color: u16, flags: u8) -> Self {
        Self { color, flags, ..self }
    }

    /// Get RGB888 color
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        rgb565_to_rgb(self.color)
    }

    /// Check if voxel is empty (air)
    pub fn is_empty(&self) -> bool {
        self.color == 0 && self.material_id == 0 && self.flags == 0
    }

    /// Check if voxel is transparent
    pub fn is_transparent(&self) -> bool {
        self.flags & flags::TRANSPARENT != 0
    }

    /// Set transparency flag
    pub fn set_transparent(&mut self, transparent: bool) {
        if transparent {
            self.flags |= flags::TRANSPARENT;
        } else {
            self.flags &= !flags::TRANSPARENT;
        }
    }

    /// Check if voxel is emissive
    pub fn is_emissive(&self) -> bool {
        self.flags & flags::EMISSIVE != 0
    }

    /// Set emissive flag
    pub fn set_emissive(&mut self, emissive: bool) {
        if emissive {
            self.flags |= flags::EMISSIVE;
        } else {
            self.flags &= !flags::EMISSIVE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<Voxel>(), 4);
    }

    #[test]
    fn test_rgb565_roundtrip() {
        // Test common colors
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)] {
            let color = rgb_to_565(r, g, b);
            let (r2, g2, b2) = rgb565_to_rgb(color);
            // Allow small error due to bit depth reduction
            assert!((r as i32 - r2 as i32).abs() <= 8);
            assert!((g as i32 - g2 as i32).abs() <= 4);
            assert!((b as i32 - b2 as i32).abs() <= 8);
        }
    }

    #[test]
    fn test_empty() {
        assert!(Voxel::EMPTY.is_empty());
        assert!(!Voxel::new(255, 0, 0, 1).is_empty());
    }

    #[test]
    fn test_flags() {
        let mut voxel = Voxel::new(255, 255, 255, 0);
        assert!(!voxel.is_transparent());
        voxel.set_transparent(true);
        assert!(voxel.is_transparent());
    }
}
