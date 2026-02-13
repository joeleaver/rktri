//! Async chunk loader with thread pool for disk I/O and generation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use crate::voxel::chunk::{Chunk, ChunkCoord};

/// Status of a chunk load operation.
#[derive(Clone, Debug)]
pub enum LoadStatus {
    /// Queued for loading
    Queued,
    /// Currently loading from disk
    LoadingFromDisk,
    /// Currently generating procedurally
    Generating,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
    /// Cancelled
    Cancelled,
}

/// A completed chunk load result.
pub struct LoadedChunk {
    /// The chunk coordinate
    pub coord: ChunkCoord,
    /// The loaded chunk data
    pub chunk: Chunk,
    /// Load time in milliseconds
    pub load_time_ms: f32,
    /// Whether this was loaded from disk (true) or generated (false)
    pub from_disk: bool,
}

/// Configuration for the async loader.
#[derive(Clone, Debug)]
pub struct AsyncLoaderConfig {
    /// Number of I/O threads
    pub io_threads: usize,
    /// Number of generation threads
    pub gen_threads: usize,
    /// Maximum pending requests
    pub max_pending: usize,
    /// Base directory for chunk files
    pub chunk_dir: PathBuf,
}

impl Default for AsyncLoaderConfig {
    fn default() -> Self {
        Self {
            io_threads: 2,
            gen_threads: 2,
            max_pending: 64,
            chunk_dir: PathBuf::from("assets/worlds"),
        }
    }
}

/// Async chunk loader using crossbeam channels for thread-safe communication.
///
/// In a full implementation, this would use a thread pool and async I/O.
/// For now, this provides the interface and synchronous fallback.
pub struct AsyncChunkLoader {
    /// Configuration
    config: AsyncLoaderConfig,
    /// Status of pending loads
    pending: Arc<Mutex<HashMap<ChunkCoord, LoadStatus>>>,
    /// Completed chunks ready for pickup
    completed: Arc<Mutex<Vec<LoadedChunk>>>,
    /// Whether the loader is running
    running: bool,
}

impl AsyncChunkLoader {
    /// Create a new async loader.
    pub fn new(config: AsyncLoaderConfig) -> Self {
        Self {
            config,
            pending: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(Vec::new())),
            running: true,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AsyncLoaderConfig::default())
    }

    /// Request a chunk to be loaded.
    pub fn request_load(&self, coord: ChunkCoord, _priority: f32) -> bool {
        let mut pending = self.pending.lock().unwrap();

        if pending.len() >= self.config.max_pending {
            return false; // Queue full
        }

        if pending.contains_key(&coord) {
            return false; // Already pending
        }

        pending.insert(coord, LoadStatus::Queued);
        true
    }

    /// Cancel a pending load.
    pub fn cancel(&self, coord: ChunkCoord) {
        let mut pending = self.pending.lock().unwrap();
        if let Some(status) = pending.get_mut(&coord) {
            *status = LoadStatus::Cancelled;
        }
    }

    /// Cancel all pending loads.
    pub fn cancel_all(&self) {
        let mut pending = self.pending.lock().unwrap();
        for status in pending.values_mut() {
            *status = LoadStatus::Cancelled;
        }
    }

    /// Poll for completed chunks.
    pub fn poll_completed(&self) -> Vec<LoadedChunk> {
        let mut completed = self.completed.lock().unwrap();
        std::mem::take(&mut *completed)
    }

    /// Check if a chunk is pending.
    pub fn is_pending(&self, coord: ChunkCoord) -> bool {
        self.pending.lock().unwrap().contains_key(&coord)
    }

    /// Get the status of a pending load.
    pub fn status(&self, coord: ChunkCoord) -> Option<LoadStatus> {
        self.pending.lock().unwrap().get(&coord).cloned()
    }

    /// Number of pending loads.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Number of completed loads ready for pickup.
    pub fn completed_count(&self) -> usize {
        self.completed.lock().unwrap().len()
    }

    /// Submit a completed chunk (for synchronous loading path).
    pub fn submit_completed(&self, loaded: LoadedChunk) {
        // Remove from pending
        {
            let mut pending = self.pending.lock().unwrap();
            pending.remove(&loaded.coord);
        }
        // Add to completed
        {
            let mut completed = self.completed.lock().unwrap();
            completed.push(loaded);
        }
    }

    /// Shutdown the loader.
    pub fn shutdown(&mut self) {
        self.running = false;
        self.cancel_all();
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running
    }
}

impl Default for AsyncChunkLoader {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Drop for AsyncChunkLoader {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_load() {
        let loader = AsyncChunkLoader::with_defaults();

        assert!(loader.request_load(ChunkCoord::new(0, 0, 0), 1.0));
        assert!(!loader.request_load(ChunkCoord::new(0, 0, 0), 1.0)); // Duplicate
        assert_eq!(loader.pending_count(), 1);
    }

    #[test]
    fn test_cancel() {
        let loader = AsyncChunkLoader::with_defaults();

        loader.request_load(ChunkCoord::new(0, 0, 0), 1.0);
        loader.cancel(ChunkCoord::new(0, 0, 0));

        let status = loader.status(ChunkCoord::new(0, 0, 0));
        assert!(matches!(status, Some(LoadStatus::Cancelled)));
    }

    #[test]
    fn test_max_pending() {
        let config = AsyncLoaderConfig {
            max_pending: 2,
            ..Default::default()
        };
        let loader = AsyncChunkLoader::new(config);

        assert!(loader.request_load(ChunkCoord::new(0, 0, 0), 1.0));
        assert!(loader.request_load(ChunkCoord::new(1, 0, 0), 1.0));
        assert!(!loader.request_load(ChunkCoord::new(2, 0, 0), 1.0)); // Queue full
    }
}
