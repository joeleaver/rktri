//! 2x2x2 voxel brick for octree leaf nodes

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};
use super::voxel::{Voxel, rgb_to_565};

/// Index into brick using Morton order for cache-friendly access
fn brick_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(x < 2 && y < 2 && z < 2);
    ((z as usize) << 2) | ((y as usize) << 1) | (x as usize)
}

/// 2x2x2 brick of voxels (8 voxels total) - 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Archive, Deserialize, Serialize)]
pub struct VoxelBrick {
    pub voxels: [Voxel; 8],
}

impl VoxelBrick {
    /// Empty brick (all air)
    pub const EMPTY: VoxelBrick = VoxelBrick {
        voxels: [Voxel::EMPTY; 8],
    };

    /// Create brick from voxel array
    pub fn new(voxels: [Voxel; 8]) -> Self {
        Self { voxels }
    }

    /// Get voxel at local coordinates (0-1 each axis)
    pub fn get(&self, x: u8, y: u8, z: u8) -> &Voxel {
        &self.voxels[brick_index(x, y, z)]
    }

    /// Get mutable voxel at local coordinates
    pub fn get_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Voxel {
        &mut self.voxels[brick_index(x, y, z)]
    }

    /// Set voxel at local coordinates
    pub fn set(&mut self, x: u8, y: u8, z: u8, voxel: Voxel) {
        self.voxels[brick_index(x, y, z)] = voxel;
    }

    /// Check if all voxels are empty
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|v| v.is_empty())
    }

    /// Check if all voxels are the same (uniform)
    /// Returns Some(voxel) if uniform, None otherwise
    pub fn is_uniform(&self) -> Option<Voxel> {
        let first = self.voxels[0];
        if self.voxels.iter().all(|v| *v == first) {
            Some(first)
        } else {
            None
        }
    }

    /// Compute average color for LOD representation
    pub fn average_color(&self) -> u16 {
        let mut r_sum: u32 = 0;
        let mut g_sum: u32 = 0;
        let mut b_sum: u32 = 0;
        let mut count: u32 = 0;

        for voxel in &self.voxels {
            if !voxel.is_empty() {
                let (r, g, b) = voxel.to_rgb();
                r_sum += r as u32;
                g_sum += g as u32;
                b_sum += b as u32;
                count += 1;
            }
        }

        if count == 0 {
            return 0;
        }

        rgb_to_565(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        )
    }

    /// Get most common material ID in brick
    pub fn average_material(&self) -> u8 {
        let mut counts = [0u8; 256];
        for voxel in &self.voxels {
            if !voxel.is_empty() {
                counts[voxel.material_id as usize] += 1;
            }
        }
        counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(id, _)| id as u8)
            .unwrap_or(0)
    }
}

impl Default for VoxelBrick {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<VoxelBrick>(), 32);
    }

    #[test]
    fn test_indexing() {
        let mut brick = VoxelBrick::EMPTY;
        let voxel = Voxel::new(255, 0, 0, 1);
        brick.set(1, 1, 1, voxel);
        assert_eq!(*brick.get(1, 1, 1), voxel);
        assert!(brick.get(0, 0, 0).is_empty());
    }

    #[test]
    fn test_is_empty() {
        assert!(VoxelBrick::EMPTY.is_empty());
        let mut brick = VoxelBrick::EMPTY;
        brick.set(0, 0, 0, Voxel::new(255, 0, 0, 1));
        assert!(!brick.is_empty());
    }

    #[test]
    fn test_is_uniform() {
        let uniform = VoxelBrick::new([Voxel::new(255, 0, 0, 1); 8]);
        assert!(uniform.is_uniform().is_some());

        let mut non_uniform = uniform;
        non_uniform.set(0, 0, 0, Voxel::new(0, 255, 0, 2));
        assert!(non_uniform.is_uniform().is_none());
    }
}
