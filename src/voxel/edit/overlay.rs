//! Edit overlay - spatial index of active edits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::types::Vec3;
use super::delta::{EditDelta, EditOp};
use crate::math::aabb::Aabb;
use crate::voxel::chunk::ChunkCoord;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::voxel::voxel::Voxel;

/// Manages active edits as a spatial overlay.
pub struct EditOverlay {
    /// All edits by ID
    edits: HashMap<u64, EditDelta>,
    /// Index: chunk -> edit IDs
    chunk_index: HashMap<ChunkCoord, Vec<u64>>,
    /// Next edit ID
    next_id: AtomicU64,
    /// Dirty chunks needing rebuild
    dirty_chunks: Vec<ChunkCoord>,
}

impl EditOverlay {
    /// Create a new empty overlay.
    pub fn new() -> Self {
        Self {
            edits: HashMap::new(),
            chunk_index: HashMap::new(),
            next_id: AtomicU64::new(1),
            dirty_chunks: Vec::new(),
        }
    }

    /// Add an edit and return its ID.
    pub fn add_edit(&mut self, op: EditOp, frame: u32) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let delta = EditDelta::new(id, frame, op);

        // Index by affected chunks
        for chunk in &delta.affected_chunks {
            self.chunk_index
                .entry(*chunk)
                .or_insert_with(Vec::new)
                .push(id);

            // Mark chunk as dirty
            if !self.dirty_chunks.contains(chunk) {
                self.dirty_chunks.push(*chunk);
            }
        }

        self.edits.insert(id, delta);
        id
    }

    /// Remove an edit by ID.
    pub fn remove_edit(&mut self, id: u64) -> Option<EditDelta> {
        if let Some(delta) = self.edits.remove(&id) {
            // Remove from chunk index
            for chunk in &delta.affected_chunks {
                if let Some(ids) = self.chunk_index.get_mut(chunk) {
                    ids.retain(|&i| i != id);
                }
                if !self.dirty_chunks.contains(chunk) {
                    self.dirty_chunks.push(*chunk);
                }
            }
            Some(delta)
        } else {
            None
        }
    }

    /// Get edits affecting a specific chunk.
    pub fn edits_for_chunk(&self, chunk: ChunkCoord) -> Vec<&EditDelta> {
        self.chunk_index
            .get(&chunk)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.edits.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get edits intersecting an AABB.
    pub fn edits_in_region(&self, aabb: &Aabb) -> Vec<&EditDelta> {
        self.edits
            .values()
            .filter(|delta| delta.affected_region().intersects(aabb))
            .collect()
    }

    /// Take dirty chunks (clears the dirty list).
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkCoord> {
        std::mem::take(&mut self.dirty_chunks)
    }

    /// Check if any chunks are dirty.
    pub fn has_dirty_chunks(&self) -> bool {
        !self.dirty_chunks.is_empty()
    }

    /// Total number of edits.
    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    /// Clear all edits.
    pub fn clear(&mut self) {
        let affected: Vec<_> = self.chunk_index.keys().copied().collect();
        self.edits.clear();
        self.chunk_index.clear();
        self.dirty_chunks = affected;
    }

    /// Evaluate edits at a position (latest edit wins).
    pub fn evaluate_at(&self, pos: Vec3) -> Option<Voxel> {
        // Find edits that might affect this position
        let mut latest: Option<(u64, Voxel)> = None;

        for delta in self.edits.values() {
            if let Some(voxel) = delta.evaluate_at(pos) {
                match latest {
                    Some((prev_id, _)) if delta.id > prev_id => {
                        latest = Some((delta.id, voxel));
                    }
                    None => {
                        latest = Some((delta.id, voxel));
                    }
                    _ => {}
                }
            }
        }

        latest.map(|(_, v)| v)
    }
}

impl Default for EditOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionClassifier for EditOverlay {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        let edits = self.edits_in_region(aabb);

        if edits.is_empty() {
            // No edits here - defer to base classifier
            return RegionHint::Unknown;
        }

        // If any edit affects this region, it's mixed
        RegionHint::Mixed
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        self.evaluate_at(pos).unwrap_or(Voxel::EMPTY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_edit() {
        let mut overlay = EditOverlay::new();

        let _id = overlay.add_edit(
            EditOp::SetVoxel {
                position: Vec3::new(2.0, 2.0, 2.0),
                voxel: Voxel::from_rgb565(0, 1),
            },
            0,
        );

        assert_eq!(overlay.edit_count(), 1);
        assert!(overlay.has_dirty_chunks());
    }

    #[test]
    fn test_edits_for_chunk() {
        let mut overlay = EditOverlay::new();

        overlay.add_edit(
            EditOp::SetVoxel {
                position: Vec3::new(2.0, 2.0, 2.0), // Chunk (0,0,0)
                voxel: Voxel::from_rgb565(0, 1),
            },
            0,
        );
        overlay.add_edit(
            EditOp::SetVoxel {
                position: Vec3::new(6.0, 2.0, 2.0), // Chunk (1,0,0)
                voxel: Voxel::from_rgb565(0, 2),
            },
            0,
        );

        let edits = overlay.edits_for_chunk(ChunkCoord::new(0, 0, 0));
        assert_eq!(edits.len(), 1);
    }

    #[test]
    fn test_evaluate_latest_wins() {
        let mut overlay = EditOverlay::new();
        let pos = Vec3::new(2.0, 2.0, 2.0);

        // First edit: set to material 1
        overlay.add_edit(
            EditOp::SetVoxel { position: pos, voxel: Voxel::from_rgb565(0, 1) },
            0,
        );
        // Second edit: set to material 5
        overlay.add_edit(
            EditOp::SetVoxel { position: pos, voxel: Voxel::from_rgb565(0, 5) },
            1,
        );

        let result = overlay.evaluate_at(pos);
        assert!(result.is_some());
        assert_eq!(result.unwrap().material_id, 5); // Latest wins
    }

    #[test]
    fn test_remove_edit() {
        let mut overlay = EditOverlay::new();

        let id = overlay.add_edit(
            EditOp::SetVoxel {
                position: Vec3::new(2.0, 2.0, 2.0),
                voxel: Voxel::from_rgb565(0, 1),
            },
            0,
        );

        overlay.take_dirty_chunks(); // Clear dirty
        overlay.remove_edit(id);

        assert_eq!(overlay.edit_count(), 0);
        assert!(overlay.has_dirty_chunks()); // Removal marks chunks dirty again
    }

    #[test]
    fn test_region_classifier() {
        let mut overlay = EditOverlay::new();

        overlay.add_edit(
            EditOp::FillRegion {
                region: Aabb::new(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0)),
                voxel: Voxel::from_rgb565(0, 1),
            },
            0,
        );

        // Region intersecting the fill
        let inside = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(overlay.classify_region(&inside), RegionHint::Mixed);

        // Region outside the fill
        let outside = Aabb::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(12.0, 12.0, 12.0));
        assert_eq!(overlay.classify_region(&outside), RegionHint::Unknown);
    }
}
