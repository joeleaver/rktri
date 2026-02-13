//! GPU-ready grass uniform (48 bytes, 16-byte aligned).
//!
//! Contains only global settings. Per-profile parameters live in the
//! `GrassProfileTable` storage buffer.

use bytemuck::{Pod, Zeroable};

/// GPU uniform for the grass system. Must match `GrassParams` in svo_trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GrassParams {
    pub enabled: u32,
    pub max_distance: f32,
    pub fade_start: f32,
    pub time: f32,
    // -- 16 bytes --
    pub wind_direction: [f32; 3],
    pub wind_speed: f32,
    // -- 16 bytes --
    pub profile_count: u32,
    pub _pad: [f32; 3],
    // -- 16 bytes --
    // Total: 48 bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grass_params_size() {
        assert_eq!(std::mem::size_of::<GrassParams>(), 48);
    }

    #[test]
    fn test_grass_params_alignment() {
        assert_eq!(std::mem::size_of::<GrassParams>() % 16, 0);
    }

    #[test]
    fn test_bytemuck_cast() {
        let p = GrassParams::zeroed();
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 48);
    }
}
