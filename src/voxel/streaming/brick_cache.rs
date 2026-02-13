//! Brick Cache - Manages brick loading state and indirection
//!
//! The cache provides:
//! - Indirection table: brick_id -> pool_slot (or NOT_LOADED sentinel)
//! - Loading state tracking: which bricks are pending, loading, or loaded
//! - Priority queue for loading based on distance/importance

use std::collections::{HashMap, VecDeque};
use super::brick_pool::BRICK_NOT_LOADED;

/// State of a brick in the cache
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrickState {
    /// Not loaded, not requested
    NotLoaded,
    /// Requested but not yet loading
    Pending,
    /// Currently being loaded
    Loading,
    /// Loaded and available in pool
    Loaded { slot: u32 },
}

/// Priority for loading a brick
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadPriority {
    /// Distance from camera (lower = higher priority)
    pub distance: f32,
    /// LOD level (lower = higher priority)
    pub lod_level: u8,
    /// Frame when requested
    pub request_frame: u32,
}

impl LoadPriority {
    /// Calculate priority score (lower = higher priority)
    pub fn score(&self) -> f32 {
        // Prioritize by LOD first, then distance
        self.lod_level as f32 * 1000.0 + self.distance
    }
}

/// Brick cache managing loading state and indirection
pub struct BrickCache {
    /// Brick state map
    states: HashMap<u32, BrickState>,
    /// Indirection table: brick_id -> pool_slot
    /// This is uploaded to GPU for shader access
    indirection: Vec<u32>,
    /// Maximum brick ID (determines indirection table size)
    max_brick_id: u32,
    /// Pending load requests with priorities
    pending_loads: VecDeque<(u32, LoadPriority)>,
    /// Current frame
    current_frame: u32,
    /// Stats: total requests this frame
    frame_requests: u32,
    /// Stats: cache hits this frame
    frame_hits: u32,
}

impl BrickCache {
    /// Create a new brick cache
    pub fn new(max_brick_id: u32) -> Self {
        let indirection = vec![BRICK_NOT_LOADED; max_brick_id as usize];

        Self {
            states: HashMap::new(),
            indirection,
            max_brick_id,
            pending_loads: VecDeque::new(),
            current_frame: 0,
            frame_requests: 0,
            frame_hits: 0,
        }
    }

    /// Start a new frame
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.frame_requests = 0;
        self.frame_hits = 0;
    }

    /// Request a brick, returns current state
    pub fn request(&mut self, brick_id: u32, priority: LoadPriority) -> BrickState {
        if brick_id >= self.max_brick_id {
            return BrickState::NotLoaded;
        }

        self.frame_requests += 1;

        let state = self.states.get(&brick_id).copied().unwrap_or(BrickState::NotLoaded);

        match state {
            BrickState::Loaded { .. } => {
                self.frame_hits += 1;
                state
            }
            BrickState::NotLoaded => {
                // Add to pending queue
                self.states.insert(brick_id, BrickState::Pending);
                self.pending_loads.push_back((brick_id, priority));
                BrickState::Pending
            }
            _ => state,
        }
    }

    /// Get next brick to load (highest priority)
    pub fn pop_pending(&mut self) -> Option<u32> {
        // Simple FIFO for now - could be priority queue
        while let Some((brick_id, _priority)) = self.pending_loads.pop_front() {
            if let Some(BrickState::Pending) = self.states.get(&brick_id) {
                self.states.insert(brick_id, BrickState::Loading);
                return Some(brick_id);
            }
        }
        None
    }

    /// Mark a brick as loaded in a specific slot
    pub fn mark_loaded(&mut self, brick_id: u32, slot: u32) {
        if brick_id < self.max_brick_id {
            self.states.insert(brick_id, BrickState::Loaded { slot });
            self.indirection[brick_id as usize] = slot;
        }
    }

    /// Mark a brick as evicted (no longer in pool)
    pub fn mark_evicted(&mut self, brick_id: u32) {
        if brick_id < self.max_brick_id {
            self.states.remove(&brick_id);
            self.indirection[brick_id as usize] = BRICK_NOT_LOADED;
        }
    }

    /// Get slot for a brick (if loaded)
    pub fn get_slot(&self, brick_id: u32) -> Option<u32> {
        if brick_id >= self.max_brick_id {
            return None;
        }

        match self.states.get(&brick_id) {
            Some(BrickState::Loaded { slot }) => Some(*slot),
            _ => None,
        }
    }

    /// Get the indirection table for GPU upload
    pub fn indirection_table(&self) -> &[u32] {
        &self.indirection
    }

    /// Resize indirection table if needed
    pub fn resize(&mut self, new_max: u32) {
        if new_max > self.max_brick_id {
            self.indirection.resize(new_max as usize, BRICK_NOT_LOADED);
            self.max_brick_id = new_max;
        }
    }

    /// Get number of pending loads
    pub fn pending_count(&self) -> usize {
        self.pending_loads.len()
    }

    /// Get cache hit rate for current frame
    pub fn hit_rate(&self) -> f32 {
        if self.frame_requests == 0 {
            1.0
        } else {
            self.frame_hits as f32 / self.frame_requests as f32
        }
    }

    /// Get total loaded brick count
    pub fn loaded_count(&self) -> usize {
        self.states
            .values()
            .filter(|s| matches!(s, BrickState::Loaded { .. }))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_and_load() {
        let mut cache = BrickCache::new(100);

        let priority = LoadPriority {
            distance: 10.0,
            lod_level: 0,
            request_frame: 0,
        };

        // First request should be NotLoaded -> Pending
        assert_eq!(cache.request(5, priority), BrickState::Pending);

        // Pop should give us the brick
        assert_eq!(cache.pop_pending(), Some(5));

        // Mark as loaded
        cache.mark_loaded(5, 42);

        // Now request should hit
        assert!(matches!(cache.request(5, priority), BrickState::Loaded { slot: 42 }));
    }

    #[test]
    fn test_eviction() {
        let mut cache = BrickCache::new(100);

        cache.mark_loaded(10, 5);
        assert_eq!(cache.get_slot(10), Some(5));

        cache.mark_evicted(10);
        assert_eq!(cache.get_slot(10), None);
    }
}
