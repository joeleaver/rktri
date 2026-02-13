//! Brick streaming system based on GigaVoxels architecture
//!
//! Key concepts:
//! - Brick Pool: Fixed-size GPU buffer holding active bricks
//! - LRU Cache: Evict least-recently-used bricks when pool is full
//! - Ray-Guided Loading: Rays request missing bricks during traversal
//! - Async Streaming: CPU loads bricks on demand, uploads to GPU

pub mod brick_pool;
pub mod brick_cache;
pub mod request_queue;
pub mod manager;
pub mod orchestrator;
pub mod prefetch;
pub mod async_loader;
pub mod feedback;

pub use brick_pool::BrickPool;
pub use brick_cache::BrickCache;
pub use request_queue::BrickRequestQueue;
pub use manager::StreamingManager;
pub use orchestrator::{StreamingOrchestrator, MemoryBudget, StreamingStats, LoadPriority};
pub use prefetch::PrefetchPredictor;
pub use async_loader::{AsyncChunkLoader, AsyncLoaderConfig, LoadedChunk, LoadStatus};
pub use feedback::{FeedbackBuffer, BrickRequest, BRICK_NOT_RESIDENT};
