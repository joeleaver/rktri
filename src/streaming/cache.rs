//! LRU cache for chunks
//!
//! Provides an LRU (Least Recently Used) cache for managing chunks in memory.
//! When the cache is full, the oldest unused chunk is evicted automatically.

use crate::streaming::disk_io::{Chunk, ChunkCoord};
use std::collections::HashMap;

/// LRU cache for chunks
///
/// Maintains chunks in memory with automatic eviction of least recently used chunks.
/// Access order is tracked to determine which chunks to evict when the cache is full.
pub struct ChunkCache {
    /// Map of chunk coordinates to chunks
    chunks: HashMap<ChunkCoord, Chunk>,
    /// Access order: oldest first, newest last
    /// When a chunk is accessed, it's moved to the end
    access_order: Vec<ChunkCoord>,
    /// Maximum number of chunks to keep in cache
    max_chunks: usize,
}

impl ChunkCache {
    /// Create a new chunk cache with the given capacity
    ///
    /// # Arguments
    /// * `max_chunks` - Maximum number of chunks to keep in memory
    pub fn new(max_chunks: usize) -> Self {
        Self {
            chunks: HashMap::with_capacity(max_chunks),
            access_order: Vec::with_capacity(max_chunks),
            max_chunks,
        }
    }

    /// Get a chunk by coordinate (immutable)
    ///
    /// Updates the access order to mark this chunk as recently used.
    ///
    /// # Arguments
    /// * `coord` - Chunk coordinate
    ///
    /// # Returns
    /// Reference to the chunk if it exists in cache
    pub fn get(&mut self, coord: ChunkCoord) -> Option<&Chunk> {
        if self.chunks.contains_key(&coord) {
            self.update_access_order(coord);
            self.chunks.get(&coord)
        } else {
            None
        }
    }

    /// Get a mutable chunk by coordinate
    ///
    /// Updates the access order to mark this chunk as recently used.
    ///
    /// # Arguments
    /// * `coord` - Chunk coordinate
    ///
    /// # Returns
    /// Mutable reference to the chunk if it exists in cache
    pub fn get_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        if self.chunks.contains_key(&coord) {
            self.update_access_order(coord);
            self.chunks.get_mut(&coord)
        } else {
            None
        }
    }

    /// Insert a chunk into the cache
    ///
    /// If the cache is at capacity, the least recently used chunk is evicted first.
    /// If a chunk with the same coordinate already exists, it is replaced.
    ///
    /// # Arguments
    /// * `chunk` - Chunk to insert
    ///
    /// # Returns
    /// The evicted chunk if one was removed to make space, or the replaced chunk if it existed
    pub fn insert(&mut self, chunk: Chunk) -> Option<Chunk> {
        let coord = chunk.coord;

        // If chunk already exists, remove it from access order
        if self.chunks.contains_key(&coord) {
            self.remove_from_access_order(coord);
        }

        // Evict oldest chunk if at capacity
        let evicted = if self.chunks.len() >= self.max_chunks && !self.chunks.contains_key(&coord) {
            self.evict_oldest()
        } else {
            None
        };

        // Insert new chunk
        let replaced = self.chunks.insert(coord, chunk);
        self.access_order.push(coord);

        // Return either evicted or replaced chunk
        evicted.or(replaced)
    }

    /// Remove a chunk from the cache
    ///
    /// # Arguments
    /// * `coord` - Chunk coordinate
    ///
    /// # Returns
    /// The removed chunk if it existed
    pub fn remove(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.remove_from_access_order(coord);
        self.chunks.remove(&coord)
    }

    /// Check if the cache contains a chunk
    ///
    /// # Arguments
    /// * `coord` - Chunk coordinate
    ///
    /// # Returns
    /// True if the chunk is in the cache
    pub fn contains(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains_key(&coord)
    }

    /// Get the number of chunks in the cache
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Evict the oldest (least recently used) chunk
    ///
    /// # Returns
    /// The evicted chunk if any existed
    pub fn evict_oldest(&mut self) -> Option<Chunk> {
        if let Some(coord) = self.access_order.first().copied() {
            self.remove(coord)
        } else {
            None
        }
    }

    /// Get an iterator over all chunk coordinates
    pub fn coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.chunks.keys()
    }

    /// Update access order by moving a coordinate to the end (most recent)
    fn update_access_order(&mut self, coord: ChunkCoord) {
        self.remove_from_access_order(coord);
        self.access_order.push(coord);
    }

    /// Remove a coordinate from the access order
    fn remove_from_access_order(&mut self, coord: ChunkCoord) {
        if let Some(pos) = self.access_order.iter().position(|&c| c == coord) {
            self.access_order.remove(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(x: i32, y: i32, z: i32) -> Chunk {
        Chunk::new(ChunkCoord::new(x, y, z))
    }

    #[test]
    fn test_cache_new() {
        let cache = ChunkCache::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = ChunkCache::new(10);
        let chunk = make_chunk(1, 2, 3);
        let coord = chunk.coord;

        cache.insert(chunk);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(coord));

        let retrieved = cache.get(coord);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().coord, coord);
    }

    #[test]
    fn test_cache_insert_replace() {
        let mut cache = ChunkCache::new(10);
        let chunk1 = make_chunk(1, 2, 3);
        let chunk2 = make_chunk(1, 2, 3);
        let coord = chunk1.coord;

        let replaced = cache.insert(chunk1);
        assert!(replaced.is_none());
        assert_eq!(cache.len(), 1);

        let replaced = cache.insert(chunk2);
        assert!(replaced.is_some());
        assert_eq!(replaced.unwrap().coord, coord);
        assert_eq!(cache.len(), 1); // Still only 1 chunk
    }

    #[test]
    fn test_cache_get_mut() {
        let mut cache = ChunkCache::new(10);
        let chunk = make_chunk(5, 6, 7);
        let coord = chunk.coord;

        cache.insert(chunk);

        let chunk_mut = cache.get_mut(coord);
        assert!(chunk_mut.is_some());
        assert_eq!(chunk_mut.unwrap().coord, coord);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = ChunkCache::new(10);
        let chunk = make_chunk(1, 2, 3);
        let coord = chunk.coord;

        cache.insert(chunk);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(coord);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().coord, coord);
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(coord));
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = ChunkCache::new(3);

        // Insert 3 chunks (at capacity)
        cache.insert(make_chunk(1, 0, 0));
        cache.insert(make_chunk(2, 0, 0));
        cache.insert(make_chunk(3, 0, 0));
        assert_eq!(cache.len(), 3);

        // Insert 4th chunk - should evict oldest (1, 0, 0)
        let evicted = cache.insert(make_chunk(4, 0, 0));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().coord, ChunkCoord::new(1, 0, 0));
        assert_eq!(cache.len(), 3);

        // Verify (1, 0, 0) is gone
        assert!(!cache.contains(ChunkCoord::new(1, 0, 0)));
        assert!(cache.contains(ChunkCoord::new(2, 0, 0)));
        assert!(cache.contains(ChunkCoord::new(3, 0, 0)));
        assert!(cache.contains(ChunkCoord::new(4, 0, 0)));
    }

    #[test]
    fn test_cache_lru_access_order() {
        let mut cache = ChunkCache::new(3);

        // Insert 3 chunks
        cache.insert(make_chunk(1, 0, 0));
        cache.insert(make_chunk(2, 0, 0));
        cache.insert(make_chunk(3, 0, 0));

        // Access chunk (1, 0, 0) to move it to end
        cache.get(ChunkCoord::new(1, 0, 0));

        // Insert 4th chunk - should evict (2, 0, 0) now, not (1, 0, 0)
        let evicted = cache.insert(make_chunk(4, 0, 0));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().coord, ChunkCoord::new(2, 0, 0));

        // Verify (1, 0, 0) is still present
        assert!(cache.contains(ChunkCoord::new(1, 0, 0)));
        assert!(!cache.contains(ChunkCoord::new(2, 0, 0)));
        assert!(cache.contains(ChunkCoord::new(3, 0, 0)));
        assert!(cache.contains(ChunkCoord::new(4, 0, 0)));
    }

    #[test]
    fn test_cache_lru_get_mut_updates_order() {
        let mut cache = ChunkCache::new(3);

        cache.insert(make_chunk(1, 0, 0));
        cache.insert(make_chunk(2, 0, 0));
        cache.insert(make_chunk(3, 0, 0));

        // Access chunk (1, 0, 0) with get_mut to move it to end
        cache.get_mut(ChunkCoord::new(1, 0, 0));

        // Insert 4th chunk - should evict (2, 0, 0)
        let evicted = cache.insert(make_chunk(4, 0, 0));
        assert_eq!(evicted.unwrap().coord, ChunkCoord::new(2, 0, 0));
        assert!(cache.contains(ChunkCoord::new(1, 0, 0)));
    }

    #[test]
    fn test_cache_evict_oldest() {
        let mut cache = ChunkCache::new(10);

        cache.insert(make_chunk(1, 0, 0));
        cache.insert(make_chunk(2, 0, 0));
        cache.insert(make_chunk(3, 0, 0));

        let evicted = cache.evict_oldest();
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().coord, ChunkCoord::new(1, 0, 0));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_coords_iterator() {
        let mut cache = ChunkCache::new(10);

        cache.insert(make_chunk(1, 2, 3));
        cache.insert(make_chunk(4, 5, 6));
        cache.insert(make_chunk(7, 8, 9));

        let coords: Vec<_> = cache.coords().copied().collect();
        assert_eq!(coords.len(), 3);
        assert!(coords.contains(&ChunkCoord::new(1, 2, 3)));
        assert!(coords.contains(&ChunkCoord::new(4, 5, 6)));
        assert!(coords.contains(&ChunkCoord::new(7, 8, 9)));
    }

    #[test]
    fn test_cache_empty_evict() {
        let mut cache = ChunkCache::new(10);
        let evicted = cache.evict_oldest();
        assert!(evicted.is_none());
    }
}
