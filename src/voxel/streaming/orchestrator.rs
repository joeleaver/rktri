//! Streaming orchestrator for coordinating GPU feedback and chunk loading.

use std::collections::{HashSet, HashMap, VecDeque};
use glam::Vec3;

use crate::voxel::chunk::ChunkCoord;
use crate::voxel::hierarchy::world_to_chunk;

/// Memory budget configuration.
#[derive(Clone, Debug)]
pub struct MemoryBudget {
    /// Maximum GPU memory for bricks (bytes)
    pub gpu_brick_budget: usize,
    /// Maximum GPU memory for nodes (bytes)
    pub gpu_node_budget: usize,
    /// Maximum CPU memory for cached chunks (bytes)
    pub cpu_cache_budget: usize,
    /// Target GPU utilization (0.0 - 1.0)
    pub target_utilization: f32,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            gpu_brick_budget: 2 * 1024 * 1024 * 1024, // 2 GB
            gpu_node_budget: 512 * 1024 * 1024,       // 512 MB
            cpu_cache_budget: 1024 * 1024 * 1024,     // 1 GB
            target_utilization: 0.9,
        }
    }
}

/// Statistics from streaming operations.
#[derive(Clone, Debug, Default)]
pub struct StreamingStats {
    /// Bricks requested this frame
    pub bricks_requested: u32,
    /// Bricks uploaded this frame
    pub bricks_uploaded: u32,
    /// Bricks evicted this frame
    pub bricks_evicted: u32,
    /// Chunks loaded this frame
    pub chunks_loaded: u32,
    /// Current GPU brick memory usage (bytes)
    pub gpu_brick_usage: usize,
    /// Current GPU node memory usage (bytes)
    pub gpu_node_usage: usize,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f32,
    /// Average load latency (ms)
    pub avg_load_latency_ms: f32,
}

/// Priority for loading requests.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LoadPriority(pub f32);

impl LoadPriority {
    /// Highest priority (on-screen, close to camera)
    pub const CRITICAL: LoadPriority = LoadPriority(1000.0);
    /// High priority (on-screen, medium distance)
    pub const HIGH: LoadPriority = LoadPriority(100.0);
    /// Medium priority (edge of screen or prefetch)
    pub const MEDIUM: LoadPriority = LoadPriority(10.0);
    /// Low priority (background prefetch)
    pub const LOW: LoadPriority = LoadPriority(1.0);

    /// Calculate priority from distance to camera.
    pub fn from_distance(distance: f32) -> Self {
        if distance < 10.0 {
            Self::CRITICAL
        } else if distance < 50.0 {
            Self::HIGH
        } else if distance < 200.0 {
            Self::MEDIUM
        } else {
            Self::LOW
        }
    }
}

/// A request to load a chunk or brick.
#[derive(Clone, Debug)]
pub struct LoadRequest {
    /// Chunk to load
    pub chunk: ChunkCoord,
    /// Priority for ordering
    pub priority: LoadPriority,
    /// Frame when requested
    pub request_frame: u32,
}

/// Coordinates streaming of chunks and bricks.
pub struct StreamingOrchestrator {
    /// Memory budget configuration
    budget: MemoryBudget,
    /// Pending load requests (priority queue)
    load_queue: VecDeque<LoadRequest>,
    /// Chunks currently being loaded
    loading: HashSet<ChunkCoord>,
    /// Chunks that are resident
    resident: HashSet<ChunkCoord>,
    /// Current frame number
    frame: u32,
    /// Statistics
    stats: StreamingStats,
    /// Camera position for distance calculations
    camera_pos: Vec3,
    /// Maximum concurrent loads
    max_concurrent_loads: usize,
    /// Last access frame for each chunk (for LRU eviction)
    last_access: HashMap<ChunkCoord, u32>,
}

impl StreamingOrchestrator {
    /// Create a new streaming orchestrator.
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            budget,
            load_queue: VecDeque::new(),
            loading: HashSet::new(),
            resident: HashSet::new(),
            frame: 0,
            stats: StreamingStats::default(),
            camera_pos: Vec3::ZERO,
            max_concurrent_loads: 4,
            last_access: HashMap::new(),
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self, camera_pos: Vec3) {
        self.frame += 1;
        self.camera_pos = camera_pos;
        self.stats = StreamingStats::default();
    }

    /// Request a chunk to be loaded.
    pub fn request_chunk(&mut self, chunk: ChunkCoord, priority: LoadPriority) {
        // Skip if already resident or loading
        if self.resident.contains(&chunk) || self.loading.contains(&chunk) {
            return;
        }

        // Add to queue (insert by priority)
        let request = LoadRequest {
            chunk,
            priority,
            request_frame: self.frame,
        };

        // Simple insertion - could use a proper priority queue for better performance
        let pos = self.load_queue
            .iter()
            .position(|r| r.priority < priority)
            .unwrap_or(self.load_queue.len());
        self.load_queue.insert(pos, request);
    }

    /// Request chunks around a position.
    pub fn request_chunks_around(&mut self, pos: Vec3, radius: f32) {
        let (cx, cy, cz) = world_to_chunk(pos);
        let chunk_radius = (radius / 4.0).ceil() as i32; // 4m per chunk

        for dx in -chunk_radius..=chunk_radius {
            for dy in -chunk_radius..=chunk_radius {
                for dz in -chunk_radius..=chunk_radius {
                    let coord = ChunkCoord::new(cx + dx, cy + dy, cz + dz);
                    let chunk_center = Vec3::new(
                        (cx + dx) as f32 * 4.0 + 2.0,
                        (cy + dy) as f32 * 4.0 + 2.0,
                        (cz + dz) as f32 * 4.0 + 2.0,
                    );
                    let distance = (chunk_center - pos).length();
                    let priority = LoadPriority::from_distance(distance);
                    self.request_chunk(coord, priority);
                }
            }
        }
    }

    /// Get next chunks to load (respects max concurrent).
    pub fn get_pending_loads(&mut self) -> Vec<ChunkCoord> {
        let available = self.max_concurrent_loads.saturating_sub(self.loading.len());
        let mut result = Vec::with_capacity(available);

        while result.len() < available {
            if let Some(request) = self.load_queue.pop_front() {
                if !self.resident.contains(&request.chunk) {
                    self.loading.insert(request.chunk);
                    result.push(request.chunk);
                }
            } else {
                break;
            }
        }

        result
    }

    /// Mark a chunk as loaded.
    pub fn mark_loaded(&mut self, chunk: ChunkCoord) {
        self.loading.remove(&chunk);
        self.resident.insert(chunk);
        self.last_access.insert(chunk, self.frame);
        self.stats.chunks_loaded += 1;
    }

    /// Touch a chunk to update its last access time (for LRU tracking).
    pub fn touch_chunk(&mut self, coord: ChunkCoord) {
        if self.resident.contains(&coord) {
            self.last_access.insert(coord, self.frame);
        }
    }

    /// Mark a chunk as unloaded/evicted.
    pub fn mark_unloaded(&mut self, chunk: ChunkCoord) {
        self.resident.remove(&chunk);
    }

    /// Check if a chunk is resident.
    pub fn is_resident(&self, chunk: ChunkCoord) -> bool {
        self.resident.contains(&chunk)
    }

    /// Check if a chunk is loading.
    pub fn is_loading(&self, chunk: ChunkCoord) -> bool {
        self.loading.contains(&chunk)
    }

    /// Get current statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Get memory budget.
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Number of resident chunks.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Number of chunks in load queue.
    pub fn queue_length(&self) -> usize {
        self.load_queue.len()
    }

    /// Current frame number.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// Enforce memory budget by evicting chunks.
    pub fn enforce_budget(&mut self, current_usage: usize) -> Vec<ChunkCoord> {
        let mut evicted = Vec::new();
        let target = (self.budget.gpu_brick_budget as f32 * self.budget.target_utilization) as usize;

        if current_usage > target {
            // Estimate per-chunk memory usage
            const BYTES_PER_CHUNK: usize = 256 * 1024; // 256 KB per chunk estimate
            let chunks_to_evict = ((current_usage - target) / BYTES_PER_CHUNK) + 1;

            // Build eviction candidates with scores
            let mut candidates: Vec<(ChunkCoord, f32)> = self.resident
                .iter()
                .map(|&coord| {
                    // Calculate chunk center position
                    let chunk_center = Vec3::new(
                        coord.x as f32 * 4.0 + 2.0,
                        coord.y as f32 * 4.0 + 2.0,
                        coord.z as f32 * 4.0 + 2.0,
                    );

                    // Calculate distance from camera
                    let distance = (chunk_center - self.camera_pos).length();

                    // Calculate staleness (frames since last access)
                    let last_frame = self.last_access.get(&coord).copied().unwrap_or(0);
                    let staleness = self.frame.saturating_sub(last_frame);

                    // Eviction score: higher = evict first
                    // Prioritize distant chunks and stale chunks
                    let eviction_score = distance * 0.01 + staleness as f32 * 0.1;

                    (coord, eviction_score)
                })
                .collect();

            // Sort by eviction score (highest first)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Evict chunks until under budget
            for (coord, _score) in candidates.iter().take(chunks_to_evict) {
                self.resident.remove(coord);
                self.last_access.remove(coord);
                evicted.push(*coord);
                self.stats.bricks_evicted += 1;
            }

            if !evicted.is_empty() {
                log::debug!("Evicted {} chunks to enforce budget (usage: {} -> target: {})",
                    evicted.len(), current_usage, target);
            }
        }

        evicted
    }
}

impl Default for StreamingOrchestrator {
    fn default() -> Self {
        Self::new(MemoryBudget::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_chunk() {
        let mut orchestrator = StreamingOrchestrator::default();
        orchestrator.begin_frame(Vec3::ZERO);

        let coord = ChunkCoord::new(0, 0, 0);
        orchestrator.request_chunk(coord, LoadPriority::HIGH);

        assert_eq!(orchestrator.queue_length(), 1);
    }

    #[test]
    fn test_get_pending_loads() {
        let mut orchestrator = StreamingOrchestrator::default();
        orchestrator.begin_frame(Vec3::ZERO);

        for i in 0..10 {
            orchestrator.request_chunk(ChunkCoord::new(i, 0, 0), LoadPriority::MEDIUM);
        }

        let pending = orchestrator.get_pending_loads();
        assert_eq!(pending.len(), 4); // max_concurrent_loads
    }

    #[test]
    fn test_mark_loaded() {
        let mut orchestrator = StreamingOrchestrator::default();
        let coord = ChunkCoord::new(0, 0, 0);

        orchestrator.loading.insert(coord);
        orchestrator.mark_loaded(coord);

        assert!(orchestrator.is_resident(coord));
        assert!(!orchestrator.is_loading(coord));
    }

    #[test]
    fn test_priority_ordering() {
        let mut orchestrator = StreamingOrchestrator::default();
        orchestrator.begin_frame(Vec3::ZERO);

        orchestrator.request_chunk(ChunkCoord::new(0, 0, 0), LoadPriority::LOW);
        orchestrator.request_chunk(ChunkCoord::new(1, 0, 0), LoadPriority::CRITICAL);
        orchestrator.request_chunk(ChunkCoord::new(2, 0, 0), LoadPriority::MEDIUM);

        let pending = orchestrator.get_pending_loads();
        // Critical should be first
        assert_eq!(pending[0], ChunkCoord::new(1, 0, 0));
    }
}
