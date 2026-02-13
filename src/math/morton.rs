//! Morton encoding (Z-order curve) for spatial indexing

/// Spread bits of a 21-bit integer into every third bit of a 64-bit integer
fn spread_bits(x: u32) -> u64 {
    let mut x = x as u64 & 0x1fffff; // 21 bits max
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

/// Compact every third bit of a 64-bit integer into a 21-bit integer
fn compact_bits(x: u64) -> u32 {
    let mut x = x & 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x | (x >> 4)) & 0x100f00f00f00f00f;
    x = (x | (x >> 8)) & 0x1f0000ff0000ff;
    x = (x | (x >> 16)) & 0x1f00000000ffff;
    x = (x | (x >> 32)) & 0x1fffff;
    x as u32
}

/// Encode 3D coordinates into Morton code (Z-order curve)
/// Each coordinate can be up to 21 bits (0..2097151)
pub fn encode_morton_3d(x: u32, y: u32, z: u32) -> u64 {
    spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
}

/// Decode Morton code back to 3D coordinates
pub fn decode_morton_3d(code: u64) -> (u32, u32, u32) {
    (
        compact_bits(code),
        compact_bits(code >> 1),
        compact_bits(code >> 2),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        for x in [0, 1, 10, 100, 500, 1000, 1023] {
            for y in [0, 1, 10, 100, 500, 1000, 1023] {
                for z in [0, 1, 10, 100, 500, 1000, 1023] {
                    let code = encode_morton_3d(x, y, z);
                    let (dx, dy, dz) = decode_morton_3d(code);
                    assert_eq!((x, y, z), (dx, dy, dz), "Failed for ({}, {}, {})", x, y, z);
                }
            }
        }
    }

    #[test]
    fn test_ordering() {
        // Morton codes should interleave bits
        assert_eq!(encode_morton_3d(0, 0, 0), 0);
        assert_eq!(encode_morton_3d(1, 0, 0), 1);
        assert_eq!(encode_morton_3d(0, 1, 0), 2);
        assert_eq!(encode_morton_3d(0, 0, 1), 4);
        assert_eq!(encode_morton_3d(1, 1, 1), 7);
    }
}
