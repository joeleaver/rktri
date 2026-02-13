//! BrickHandle - individual brick streaming handle.

use crate::voxel::chunk::ChunkCoord;

/// Unique identifier for a brick across all chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickId {
    /// Chunk containing this brick
    pub chunk: ChunkCoord,
    /// Index within chunk's brick array
    pub local_index: u32,
}

impl BrickId {
    pub fn new(chunk: ChunkCoord, local_index: u32) -> Self {
        Self { chunk, local_index }
    }
}

/// Handle to a brick with streaming state.
#[derive(Clone, Debug)]
pub struct BrickHandle {
    /// Unique brick identifier
    pub id: BrickId,
    /// GPU brick pool slot (None if not resident)
    pub pool_slot: Option<u32>,
    /// Last frame this brick was accessed
    pub last_access_frame: u32,
    /// Streaming priority (higher = more important)
    pub priority: f32,
}

impl BrickHandle {
    /// Create a new brick handle (not yet in GPU pool).
    pub fn new(id: BrickId) -> Self {
        Self {
            id,
            pool_slot: None,
            last_access_frame: 0,
            priority: 0.0,
        }
    }

    /// Check if brick is resident in GPU memory.
    pub fn is_gpu_resident(&self) -> bool {
        self.pool_slot.is_some()
    }

    /// Mark as accessed this frame.
    pub fn touch(&mut self, frame: u32) {
        self.last_access_frame = frame;
    }

    /// Set GPU pool slot.
    pub fn set_pool_slot(&mut self, slot: u32) {
        self.pool_slot = Some(slot);
    }

    /// Clear GPU pool slot (evicted).
    pub fn clear_pool_slot(&mut self) {
        self.pool_slot = None;
    }

    /// Frames since last access.
    pub fn frames_since_access(&self, current_frame: u32) -> u32 {
        current_frame.saturating_sub(self.last_access_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_id() {
        let id = BrickId::new(ChunkCoord::new(1, 2, 3), 42);
        assert_eq!(id.chunk, ChunkCoord::new(1, 2, 3));
        assert_eq!(id.local_index, 42);
    }

    #[test]
    fn test_brick_handle() {
        let mut handle = BrickHandle::new(BrickId::new(ChunkCoord::new(0, 0, 0), 0));
        assert!(!handle.is_gpu_resident());

        handle.set_pool_slot(100);
        assert!(handle.is_gpu_resident());
        assert_eq!(handle.pool_slot, Some(100));

        handle.clear_pool_slot();
        assert!(!handle.is_gpu_resident());
    }

    #[test]
    fn test_access_tracking() {
        let mut handle = BrickHandle::new(BrickId::new(ChunkCoord::new(0, 0, 0), 0));
        handle.touch(10);
        assert_eq!(handle.last_access_frame, 10);
        assert_eq!(handle.frames_since_access(15), 5);
    }
}
