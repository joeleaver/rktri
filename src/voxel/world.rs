//! World container managing multiple chunks

use super::chunk::{Chunk, ChunkCoord};
use std::collections::HashMap;

/// Container for managing a world composed of multiple chunks
pub struct World {
    /// Map from chunk coordinates to loaded chunks
    chunks: HashMap<ChunkCoord, Chunk>,
    /// List of chunk coordinates that have been modified and need saving
    modified_chunks: Vec<ChunkCoord>,
}

impl World {
    /// Create a new empty world
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            modified_chunks: Vec::new(),
        }
    }

    /// Get immutable reference to a chunk by coordinate
    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    /// Get mutable reference to a chunk by coordinate
    pub fn get_chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord)
    }

    /// Insert a chunk into the world
    /// If a chunk already exists at this coordinate, it will be replaced
    pub fn insert_chunk(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.coord, chunk);
    }

    /// Remove a chunk from the world and return it
    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        // Also remove from modified list if present
        self.modified_chunks.retain(|&c| c != coord);
        self.chunks.remove(&coord)
    }

    /// Get the number of loaded chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get an iterator over all loaded chunk coordinates
    pub fn loaded_coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.chunks.keys()
    }

    /// Mark a chunk as modified (needs to be saved)
    pub fn mark_modified(&mut self, coord: ChunkCoord) {
        // Only add if not already in the list
        if !self.modified_chunks.contains(&coord) {
            self.modified_chunks.push(coord);
        }

        // Also mark the chunk itself as modified if it exists
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            chunk.modified = true;
        }
    }

    /// Take the list of modified chunks and clear the internal list
    /// Returns the coordinates of chunks that need to be saved
    pub fn take_modified(&mut self) -> Vec<ChunkCoord> {
        std::mem::take(&mut self.modified_chunks)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_world() {
        let world = World::new();
        assert_eq!(world.chunk_count(), 0);
    }

    #[test]
    fn test_insert_and_get_chunk() {
        let mut world = World::new();
        let coord = ChunkCoord::new(1, 2, 3);
        let chunk = Chunk::new(coord);

        world.insert_chunk(chunk);
        assert_eq!(world.chunk_count(), 1);

        let retrieved = world.get_chunk(coord);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().coord, coord);
    }

    #[test]
    fn test_get_chunk_mut() {
        let mut world = World::new();
        let coord = ChunkCoord::new(0, 0, 0);
        world.insert_chunk(Chunk::new(coord));

        {
            let chunk = world.get_chunk_mut(coord).unwrap();
            chunk.modified = true;
        }

        assert!(world.get_chunk(coord).unwrap().modified);
    }

    #[test]
    fn test_remove_chunk() {
        let mut world = World::new();
        let coord = ChunkCoord::new(5, 6, 7);
        world.insert_chunk(Chunk::new(coord));

        assert_eq!(world.chunk_count(), 1);

        let removed = world.remove_chunk(coord);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().coord, coord);
        assert_eq!(world.chunk_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_chunk() {
        let mut world = World::new();
        let removed = world.remove_chunk(ChunkCoord::new(0, 0, 0));
        assert!(removed.is_none());
    }

    #[test]
    fn test_loaded_coords() {
        let mut world = World::new();
        world.insert_chunk(Chunk::new(ChunkCoord::new(1, 0, 0)));
        world.insert_chunk(Chunk::new(ChunkCoord::new(0, 1, 0)));
        world.insert_chunk(Chunk::new(ChunkCoord::new(0, 0, 1)));

        let coords: Vec<_> = world.loaded_coords().copied().collect();
        assert_eq!(coords.len(), 3);
        assert!(coords.contains(&ChunkCoord::new(1, 0, 0)));
        assert!(coords.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(coords.contains(&ChunkCoord::new(0, 0, 1)));
    }

    #[test]
    fn test_mark_modified() {
        let mut world = World::new();
        let coord = ChunkCoord::new(1, 2, 3);
        world.insert_chunk(Chunk::new(coord));

        world.mark_modified(coord);

        // Check that chunk is marked modified
        assert!(world.get_chunk(coord).unwrap().modified);

        // Check that coord is in modified list
        let modified = world.take_modified();
        assert_eq!(modified.len(), 1);
        assert_eq!(modified[0], coord);
    }

    #[test]
    fn test_mark_modified_multiple() {
        let mut world = World::new();
        let coord1 = ChunkCoord::new(1, 0, 0);
        let coord2 = ChunkCoord::new(0, 1, 0);

        world.insert_chunk(Chunk::new(coord1));
        world.insert_chunk(Chunk::new(coord2));

        world.mark_modified(coord1);
        world.mark_modified(coord2);

        let modified = world.take_modified();
        assert_eq!(modified.len(), 2);
        assert!(modified.contains(&coord1));
        assert!(modified.contains(&coord2));
    }

    #[test]
    fn test_mark_modified_duplicate() {
        let mut world = World::new();
        let coord = ChunkCoord::new(0, 0, 0);
        world.insert_chunk(Chunk::new(coord));

        world.mark_modified(coord);
        world.mark_modified(coord); // Mark same chunk twice

        let modified = world.take_modified();
        assert_eq!(modified.len(), 1); // Should only appear once
    }

    #[test]
    fn test_take_modified_clears_list() {
        let mut world = World::new();
        let coord = ChunkCoord::new(1, 1, 1);
        world.insert_chunk(Chunk::new(coord));

        world.mark_modified(coord);

        let modified1 = world.take_modified();
        assert_eq!(modified1.len(), 1);

        let modified2 = world.take_modified();
        assert_eq!(modified2.len(), 0); // List should be cleared
    }

    #[test]
    fn test_remove_chunk_clears_modified() {
        let mut world = World::new();
        let coord = ChunkCoord::new(2, 3, 4);
        world.insert_chunk(Chunk::new(coord));

        world.mark_modified(coord);
        world.remove_chunk(coord);

        let modified = world.take_modified();
        assert_eq!(modified.len(), 0); // Removed chunk shouldn't be in modified list
    }

    #[test]
    fn test_replace_chunk() {
        let mut world = World::new();
        let coord = ChunkCoord::new(0, 0, 0);

        world.insert_chunk(Chunk::new(coord));
        world.insert_chunk(Chunk::new(coord)); // Replace with new chunk

        assert_eq!(world.chunk_count(), 1); // Should still be 1 chunk
    }

    #[test]
    fn test_mark_modified_nonexistent_chunk() {
        let mut world = World::new();
        let coord = ChunkCoord::new(10, 10, 10);

        // Mark a chunk that doesn't exist yet
        world.mark_modified(coord);

        let modified = world.take_modified();
        assert_eq!(modified.len(), 1); // Coordinate is still tracked
    }
}
