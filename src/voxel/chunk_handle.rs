//! ChunkHandle - lazy loading wrapper for chunks.
//!
//! Provides thread-safe access to chunk data with multiple loading states.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use crate::voxel::chunk::{Chunk, ChunkCoord};

/// State of a chunk in the streaming system.
pub enum ChunkState {
    /// Chunk data is not loaded (on disk or not generated)
    Unloaded,
    /// Chunk is currently being loaded or generated
    Loading {
        /// Progress (0.0 - 1.0) if available
        progress: f32,
    },
    /// Chunk is loaded in CPU memory
    Resident(Box<Chunk>),
    /// Chunk is loaded and uploaded to GPU
    GpuResident {
        /// CPU-side data
        chunk: Box<Chunk>,
        /// Handle to GPU resources
        gpu_handle: GpuChunkHandle,
    },
}

impl ChunkState {
    /// Check if chunk data is available for reading.
    pub fn is_ready(&self) -> bool {
        matches!(self, ChunkState::Resident(_) | ChunkState::GpuResident { .. })
    }

    /// Check if chunk is on GPU.
    pub fn is_gpu_ready(&self) -> bool {
        matches!(self, ChunkState::GpuResident { .. })
    }

    /// Get chunk data if available.
    pub fn get_chunk(&self) -> Option<&Chunk> {
        match self {
            ChunkState::Resident(chunk) => Some(chunk),
            ChunkState::GpuResident { chunk, .. } => Some(chunk),
            _ => None,
        }
    }

    /// Get mutable chunk data if available.
    pub fn get_chunk_mut(&mut self) -> Option<&mut Chunk> {
        match self {
            ChunkState::Resident(chunk) => Some(chunk),
            ChunkState::GpuResident { chunk, .. } => Some(chunk),
            _ => None,
        }
    }
}

/// Handle to GPU-uploaded chunk data.
#[derive(Clone, Debug)]
pub struct GpuChunkHandle {
    /// Index in the GPU node buffer
    pub node_buffer_offset: u32,
    /// Index in the GPU brick pool
    pub brick_pool_offset: u32,
    /// Number of nodes
    pub node_count: u32,
    /// Number of bricks
    pub brick_count: u32,
}

/// Thread-safe handle to a chunk with lazy loading support.
pub struct ChunkHandle {
    /// Chunk coordinate
    pub coord: ChunkCoord,
    /// Current state (thread-safe)
    state: RwLock<ChunkState>,
    /// Generation counter for invalidation
    generation: AtomicU32,
    /// Priority for streaming (higher = more important)
    priority: AtomicU32,
}

impl ChunkHandle {
    /// Create a new unloaded chunk handle.
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            state: RwLock::new(ChunkState::Unloaded),
            generation: AtomicU32::new(0),
            priority: AtomicU32::new(0),
        }
    }

    /// Create a handle with already-loaded data.
    pub fn with_chunk(coord: ChunkCoord, chunk: Chunk) -> Self {
        Self {
            coord,
            state: RwLock::new(ChunkState::Resident(Box::new(chunk))),
            generation: AtomicU32::new(0),
            priority: AtomicU32::new(0),
        }
    }

    /// Get current generation (increments on each edit).
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }

    /// Increment generation (call after editing).
    pub fn increment_generation(&self) -> u32 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Get current priority.
    pub fn priority(&self) -> u32 {
        self.priority.load(Ordering::Relaxed)
    }

    /// Set priority.
    pub fn set_priority(&self, priority: u32) {
        self.priority.store(priority, Ordering::Relaxed);
    }

    /// Check if chunk is ready (loaded).
    pub fn is_ready(&self) -> bool {
        self.state.read().unwrap().is_ready()
    }

    /// Check if chunk is on GPU.
    pub fn is_gpu_ready(&self) -> bool {
        self.state.read().unwrap().is_gpu_ready()
    }

    /// Get read access to chunk state.
    pub fn read_state(&self) -> std::sync::RwLockReadGuard<'_, ChunkState> {
        self.state.read().unwrap()
    }

    /// Get write access to chunk state.
    pub fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, ChunkState> {
        self.state.write().unwrap()
    }

    /// Transition to loading state.
    pub fn start_loading(&self) {
        let mut state = self.state.write().unwrap();
        *state = ChunkState::Loading { progress: 0.0 };
    }

    /// Update loading progress.
    pub fn update_loading_progress(&self, progress: f32) {
        let mut state = self.state.write().unwrap();
        if let ChunkState::Loading { progress: p } = &mut *state {
            *p = progress;
        }
    }

    /// Transition to resident state.
    pub fn set_resident(&self, chunk: Chunk) {
        let mut state = self.state.write().unwrap();
        *state = ChunkState::Resident(Box::new(chunk));
    }

    /// Transition to GPU resident state.
    pub fn set_gpu_resident(&self, gpu_handle: GpuChunkHandle) {
        let mut state = self.state.write().unwrap();
        if let ChunkState::Resident(chunk) = std::mem::replace(&mut *state, ChunkState::Unloaded) {
            *state = ChunkState::GpuResident { chunk, gpu_handle };
        }
    }

    /// Unload from GPU (keep CPU resident).
    pub fn unload_gpu(&self) {
        let mut state = self.state.write().unwrap();
        if let ChunkState::GpuResident { chunk, .. } = std::mem::replace(&mut *state, ChunkState::Unloaded) {
            *state = ChunkState::Resident(chunk);
        }
    }

    /// Fully unload chunk.
    pub fn unload(&self) {
        let mut state = self.state.write().unwrap();
        *state = ChunkState::Unloaded;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_handle_states() {
        let handle = ChunkHandle::new(ChunkCoord::new(0, 0, 0));
        assert!(!handle.is_ready());

        handle.start_loading();
        assert!(!handle.is_ready());

        // Can't easily test Resident without a real Chunk, but structure is verified
    }

    #[test]
    fn test_generation_counter() {
        let handle = ChunkHandle::new(ChunkCoord::new(0, 0, 0));
        assert_eq!(handle.generation(), 0);

        assert_eq!(handle.increment_generation(), 1);
        assert_eq!(handle.generation(), 1);
    }

    #[test]
    fn test_priority() {
        let handle = ChunkHandle::new(ChunkCoord::new(0, 0, 0));
        handle.set_priority(100);
        assert_eq!(handle.priority(), 100);
    }
}
