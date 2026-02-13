//! Async chunk loading system with priority-based concurrent loading

use crate::streaming::disk_io::{Chunk, ChunkCoord, load_chunk};
use std::collections::HashSet;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio::runtime::Runtime;

/// Request to load a chunk with priority
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub coord: ChunkCoord,
    pub priority: f32,
}

/// Result of a chunk load operation
#[derive(Debug)]
pub enum LoadResult {
    /// Successfully loaded from disk
    Loaded(Chunk),
    /// Freshly generated, not from disk
    Generated(Chunk),
    /// Chunk file not found on disk
    NotFound(ChunkCoord),
    /// Error during loading
    Error(ChunkCoord, String),
}

/// Concurrent chunk loader with async I/O
pub struct ChunkLoader {
    /// Channel for sending load requests to worker tasks
    request_tx: mpsc::UnboundedSender<LoadRequest>,
    /// Channel for receiving load results
    result_rx: mpsc::UnboundedReceiver<LoadResult>,
    /// Set of chunks currently being loaded
    pending: HashSet<ChunkCoord>,
    /// Base directory for chunk storage
    base_dir: PathBuf,
    /// Tokio runtime handle (optional - if None, uses current runtime)
    #[allow(dead_code)]
    runtime: Option<Runtime>,
}

impl ChunkLoader {
    /// Create a new chunk loader
    ///
    /// # Arguments
    /// * `base_dir` - Directory where chunks are stored
    /// * `max_concurrent` - Maximum number of concurrent load operations
    pub fn new(base_dir: PathBuf, max_concurrent: usize) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<LoadRequest>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<LoadResult>();

        // Create a dedicated runtime for async operations
        let runtime = Runtime::new().expect("Failed to create tokio runtime");

        let base_dir_clone = base_dir.clone();

        // Spawn the worker task on the runtime
        runtime.spawn(async move {
            Self::worker_loop(base_dir_clone, max_concurrent, &mut request_rx, result_tx).await;
        });

        Self {
            request_tx,
            result_rx,
            pending: HashSet::new(),
            base_dir,
            runtime: Some(runtime),
        }
    }

    /// Create a chunk loader using the current tokio runtime
    ///
    /// This is useful when the caller already has a tokio runtime active.
    /// Panics if called outside a tokio runtime context.
    pub fn new_with_current_runtime(base_dir: PathBuf, max_concurrent: usize) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<LoadRequest>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<LoadResult>();

        let base_dir_clone = base_dir.clone();

        // Spawn on the current runtime
        tokio::spawn(async move {
            Self::worker_loop(base_dir_clone, max_concurrent, &mut request_rx, result_tx).await;
        });

        Self {
            request_tx,
            result_rx,
            pending: HashSet::new(),
            base_dir,
            runtime: None,
        }
    }

    /// Worker loop that processes load requests with concurrency control
    async fn worker_loop(
        base_dir: PathBuf,
        max_concurrent: usize,
        request_rx: &mut mpsc::UnboundedReceiver<LoadRequest>,
        result_tx: mpsc::UnboundedSender<LoadResult>,
    ) {
        use tokio::task::JoinSet;

        let mut active_tasks = JoinSet::new();
        let mut pending_requests: Vec<LoadRequest> = Vec::new();

        loop {
            tokio::select! {
                // Process incoming requests
                Some(request) = request_rx.recv() => {
                    // Add to pending queue
                    pending_requests.push(request);
                }

                // Wait for a task to complete
                Some(result) = active_tasks.join_next(), if !active_tasks.is_empty() => {
                    match result {
                        Ok(load_result) => {
                            // Send result back to main thread
                            let _ = result_tx.send(load_result);
                        }
                        Err(e) => {
                            eprintln!("Chunk loader task panicked: {}", e);
                        }
                    }
                }

                // Exit when channel is closed and no more work
                else => {
                    if pending_requests.is_empty() && active_tasks.is_empty() {
                        break;
                    }
                }
            }

            // Start new tasks if we have capacity and pending requests
            while active_tasks.len() < max_concurrent && !pending_requests.is_empty() {
                // Sort by priority and take highest priority request
                pending_requests.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
                let request = pending_requests.remove(0);

                let base_dir_clone = base_dir.clone();
                active_tasks.spawn(async move {
                    Self::load_chunk_task(base_dir_clone, request.coord).await
                });
            }
        }
    }

    /// Task that loads a single chunk
    async fn load_chunk_task(base_dir: PathBuf, coord: ChunkCoord) -> LoadResult {
        match load_chunk(&base_dir, coord).await {
            Ok(Some(chunk)) => LoadResult::Loaded(chunk),
            Ok(None) => LoadResult::NotFound(coord),
            Err(e) => LoadResult::Error(coord, e.to_string()),
        }
    }

    /// Request a chunk to be loaded
    ///
    /// Returns `false` if the chunk is already pending, `true` if the request was queued.
    pub fn request(&mut self, coord: ChunkCoord, priority: f32) -> bool {
        if self.pending.contains(&coord) {
            return false;
        }

        self.pending.insert(coord);

        let request = LoadRequest { coord, priority };
        self.request_tx.send(request).expect("Worker thread died");

        true
    }

    /// Poll for completed load results (non-blocking)
    ///
    /// Returns all currently available results.
    pub fn poll_results(&mut self) -> Vec<LoadResult> {
        let mut results = Vec::new();

        // Drain all available results
        while let Ok(result) = self.result_rx.try_recv() {
            // Remove from pending set
            let coord = match &result {
                LoadResult::Loaded(chunk) => chunk.coord,
                LoadResult::Generated(chunk) => chunk.coord,
                LoadResult::NotFound(coord) => *coord,
                LoadResult::Error(coord, _) => *coord,
            };
            self.pending.remove(&coord);

            results.push(result);
        }

        results
    }

    /// Get the number of pending load requests
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if a specific chunk is currently pending
    pub fn is_pending(&self, coord: ChunkCoord) -> bool {
        self.pending.contains(&coord)
    }

    /// Cancel a pending load request (best effort)
    ///
    /// Note: If the chunk is already being loaded, it cannot be cancelled.
    /// The coord will be removed from the pending set, but the result may
    /// still arrive and should be ignored by the caller.
    pub fn cancel(&mut self, coord: ChunkCoord) {
        self.pending.remove(&coord);
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }
}

impl Drop for ChunkLoader {
    fn drop(&mut self) {
        // Close the request channel to signal worker to shut down
        // The receiver in the worker will get None and exit gracefully
        drop(self.request_tx.clone());

        // Runtime will be dropped and shut down automatically
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_request_creation() {
        let coord = ChunkCoord::new(1, 2, 3);
        let request = LoadRequest {
            coord,
            priority: 1.0,
        };

        assert_eq!(request.coord, coord);
        assert_eq!(request.priority, 1.0);
    }

    #[test]
    fn test_chunk_loader_creation() {
        let temp_dir = std::env::temp_dir().join("rktri_loader_test");
        let loader = ChunkLoader::new(temp_dir.clone(), 4);

        assert_eq!(loader.pending_count(), 0);
        assert_eq!(loader.base_dir(), &temp_dir);
    }

    #[test]
    fn test_pending_tracking() {
        let temp_dir = std::env::temp_dir().join("rktri_loader_test");
        let mut loader = ChunkLoader::new(temp_dir, 4);

        let coord = ChunkCoord::new(5, 10, 15);

        // First request should succeed
        assert!(loader.request(coord, 1.0));
        assert_eq!(loader.pending_count(), 1);
        assert!(loader.is_pending(coord));

        // Second request for same chunk should fail
        assert!(!loader.request(coord, 2.0));
        assert_eq!(loader.pending_count(), 1);
    }

    #[test]
    fn test_cancel() {
        let temp_dir = std::env::temp_dir().join("rktri_loader_test");
        let mut loader = ChunkLoader::new(temp_dir, 4);

        let coord = ChunkCoord::new(1, 2, 3);
        loader.request(coord, 1.0);

        assert!(loader.is_pending(coord));
        loader.cancel(coord);
        assert!(!loader.is_pending(coord));
    }

    // Async tests require tokio test support
    // #[tokio::test]
    // async fn test_load_nonexistent_chunk() {
    //     let temp_dir = std::env::temp_dir().join("rktri_loader_async_test");
    //     let mut loader = ChunkLoader::new(temp_dir, 4);
    //
    //     let coord = ChunkCoord::new(999, 999, 999);
    //     loader.request(coord, 1.0);
    //
    //     // Wait a bit for processing
    //     tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    //
    //     let results = loader.poll_results();
    //     assert_eq!(results.len(), 1);
    //
    //     match &results[0] {
    //         LoadResult::NotFound(c) => assert_eq!(*c, coord),
    //         _ => panic!("Expected NotFound result"),
    //     }
    // }
}
