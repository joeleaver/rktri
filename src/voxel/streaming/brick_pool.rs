//! Brick Pool - Fixed-size GPU buffer for active bricks
//!
//! Based on GigaVoxels brick pool concept:
//! - Fixed capacity to stay within GPU memory limits
//! - LRU eviction when pool is full
//! - Slots can be reused for different bricks

use crate::voxel::brick::VoxelBrick;

/// Maximum bricks in pool (256MB / 32 bytes per brick = ~8M bricks)
/// But we limit to stay well under GPU buffer limits
pub const DEFAULT_BRICK_POOL_SIZE: u32 = 6_000_000; // ~192MB

/// Sentinel value indicating brick is not loaded
pub const BRICK_NOT_LOADED: u32 = u32::MAX;

/// A slot in the brick pool
#[derive(Clone, Copy, Debug)]
pub struct BrickSlot {
    /// Which brick ID is in this slot (or BRICK_NOT_LOADED if empty)
    pub brick_id: u32,
    /// Frame number when last accessed (for LRU)
    pub last_access_frame: u32,
    /// Whether this slot is currently in use
    pub in_use: bool,
}

impl Default for BrickSlot {
    fn default() -> Self {
        Self {
            brick_id: BRICK_NOT_LOADED,
            last_access_frame: 0,
            in_use: false,
        }
    }
}

/// Brick pool managing GPU brick storage with LRU eviction
pub struct BrickPool {
    /// GPU buffer for brick data
    buffer: wgpu::Buffer,
    /// Capacity in number of bricks
    capacity: u32,
    /// Slot metadata (CPU-side tracking)
    slots: Vec<BrickSlot>,
    /// Map from brick_id -> slot_index
    brick_to_slot: std::collections::HashMap<u32, u32>,
    /// Current frame number for LRU tracking
    current_frame: u32,
    /// Number of bricks currently loaded
    loaded_count: u32,
    /// Queue of free slot indices
    free_slots: Vec<u32>,
}

impl BrickPool {
    /// Create a new brick pool with given capacity
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let buffer_size = (capacity as usize * std::mem::size_of::<VoxelBrick>()) as u64;

        // Check against typical GPU limits
        let max_buffer_size = 256 * 1024 * 1024; // 256MB typical limit
        let actual_capacity = if buffer_size > max_buffer_size {
            log::warn!(
                "Brick pool size {} exceeds GPU limit, reducing to {}MB",
                buffer_size / 1024 / 1024,
                max_buffer_size / 1024 / 1024
            );
            (max_buffer_size / std::mem::size_of::<VoxelBrick>() as u64) as u32
        } else {
            capacity
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: (actual_capacity as usize * std::mem::size_of::<VoxelBrick>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let slots = vec![BrickSlot::default(); actual_capacity as usize];
        let free_slots: Vec<u32> = (0..actual_capacity).collect();

        log::info!(
            "Created brick pool: {} slots, {}MB",
            actual_capacity,
            actual_capacity as usize * std::mem::size_of::<VoxelBrick>() / 1024 / 1024
        );

        Self {
            buffer,
            capacity: actual_capacity,
            slots,
            brick_to_slot: std::collections::HashMap::new(),
            current_frame: 0,
            loaded_count: 0,
            free_slots,
        }
    }

    /// Start a new frame (for LRU tracking)
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Check if a brick is currently loaded
    pub fn is_loaded(&self, brick_id: u32) -> bool {
        self.brick_to_slot.contains_key(&brick_id)
    }

    /// Get slot index for a brick (if loaded)
    pub fn get_slot(&mut self, brick_id: u32) -> Option<u32> {
        if let Some(&slot_idx) = self.brick_to_slot.get(&brick_id) {
            // Update LRU timestamp
            self.slots[slot_idx as usize].last_access_frame = self.current_frame;
            Some(slot_idx)
        } else {
            None
        }
    }

    /// Allocate a slot for a new brick, evicting if necessary
    /// Returns the slot index
    pub fn allocate_slot(&mut self, brick_id: u32) -> Option<u32> {
        // Check if already loaded
        if let Some(&existing) = self.brick_to_slot.get(&brick_id) {
            self.slots[existing as usize].last_access_frame = self.current_frame;
            return Some(existing);
        }

        // Try to get a free slot
        let slot_idx = if let Some(free_idx) = self.free_slots.pop() {
            free_idx
        } else {
            // Need to evict - find oldest slot
            self.evict_oldest()?
        };

        // Set up the slot
        self.slots[slot_idx as usize] = BrickSlot {
            brick_id,
            last_access_frame: self.current_frame,
            in_use: true,
        };
        self.brick_to_slot.insert(brick_id, slot_idx);
        self.loaded_count += 1;

        Some(slot_idx)
    }

    /// Upload brick data to a slot
    pub fn upload_brick(&self, queue: &wgpu::Queue, slot_idx: u32, brick: &VoxelBrick) {
        let offset = (slot_idx as usize * std::mem::size_of::<VoxelBrick>()) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(brick));
    }

    /// Evict the oldest (least recently used) brick
    fn evict_oldest(&mut self) -> Option<u32> {
        // Find slot with oldest access time
        let mut oldest_idx = None;
        let mut oldest_frame = u32::MAX;

        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.in_use && slot.last_access_frame < oldest_frame {
                oldest_frame = slot.last_access_frame;
                oldest_idx = Some(idx as u32);
            }
        }

        if let Some(idx) = oldest_idx {
            let old_brick_id = self.slots[idx as usize].brick_id;
            self.brick_to_slot.remove(&old_brick_id);
            self.slots[idx as usize].in_use = false;
            self.loaded_count -= 1;
            log::trace!("Evicted brick {} from slot {}", old_brick_id, idx);
            Some(idx)
        } else {
            None
        }
    }

    /// Get the GPU buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get capacity
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get number of loaded bricks
    pub fn loaded_count(&self) -> u32 {
        self.loaded_count
    }

    /// Get pool utilization percentage
    pub fn utilization(&self) -> f32 {
        self.loaded_count as f32 / self.capacity as f32 * 100.0
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests require a GPU device, so they're integration tests
    // Unit tests for the logic can be done with mock structures
}
