//! Brush session for collecting strokes

use glam::Vec3;
use crate::voxel::voxel::Voxel;
use super::primitive::Axis;
use super::stroke::{BrushStroke, BlendMode};
use crate::voxel::edit::{EditOp, EditOverlay};
use crate::voxel::edit::invalidator::ChunkInvalidator;

/// A session for collecting brush strokes before building an octree
#[derive(Debug, Clone, Default)]
pub struct BrushSession {
    /// All strokes in this session
    strokes: Vec<BrushStroke>,
    /// Current blend mode for new strokes
    current_blend: BlendMode,
}

impl BrushSession {
    /// Create a new empty brush session
    pub fn new() -> Self {
        Self {
            strokes: Vec::new(),
            current_blend: BlendMode::Replace,
        }
    }

    /// Create session with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strokes: Vec::with_capacity(capacity),
            current_blend: BlendMode::Replace,
        }
    }

    /// Set the blend mode for subsequent strokes
    pub fn set_blend(&mut self, mode: BlendMode) -> &mut Self {
        self.current_blend = mode;
        self
    }

    /// Add a sphere stroke
    pub fn sphere(&mut self, center: Vec3, radius: f32, voxel: Voxel, target_level: u8) -> &mut Self {
        let stroke = BrushStroke::sphere(center, radius, voxel, target_level)
            .with_blend(self.current_blend);
        self.strokes.push(stroke);
        self
    }

    /// Add a box stroke
    pub fn box_stroke(&mut self, center: Vec3, half_extents: Vec3, voxel: Voxel, target_level: u8) -> &mut Self {
        let stroke = BrushStroke::box_stroke(center, half_extents, voxel, target_level)
            .with_blend(self.current_blend);
        self.strokes.push(stroke);
        self
    }

    /// Add a capsule stroke from start to end points
    pub fn capsule(&mut self, start: Vec3, end: Vec3, radius: f32, voxel: Voxel, target_level: u8) -> &mut Self {
        let stroke = BrushStroke::capsule(start, end, radius, voxel, target_level)
            .with_blend(self.current_blend);
        self.strokes.push(stroke);
        self
    }

    /// Add a cylinder stroke
    pub fn cylinder(&mut self, center: Vec3, axis: Axis, half_height: f32, radius: f32, voxel: Voxel, target_level: u8) -> &mut Self {
        let stroke = BrushStroke::cylinder(center, axis, half_height, radius, voxel, target_level)
            .with_blend(self.current_blend);
        self.strokes.push(stroke);
        self
    }

    /// Add a cloud stroke (stochastic fill for foliage)
    pub fn cloud(&mut self, center: Vec3, radius: f32, density: f32, seed: u32, voxel: Voxel, target_level: u8) -> &mut Self {
        let stroke = BrushStroke::cloud(center, radius, density, seed, voxel, target_level)
            .with_blend(self.current_blend);
        self.strokes.push(stroke);
        self
    }

    /// Add a pre-built stroke directly
    pub fn add_stroke(&mut self, stroke: BrushStroke) -> &mut Self {
        self.strokes.push(stroke);
        self
    }

    /// Get all strokes
    pub fn strokes(&self) -> &[BrushStroke] {
        &self.strokes
    }

    /// Get mutable access to strokes
    pub fn strokes_mut(&mut self) -> &mut Vec<BrushStroke> {
        &mut self.strokes
    }

    /// Consume session and return strokes
    pub fn into_strokes(self) -> Vec<BrushStroke> {
        self.strokes
    }

    /// Get number of strokes
    pub fn len(&self) -> usize {
        self.strokes.len()
    }

    /// Check if session has no strokes
    pub fn is_empty(&self) -> bool {
        self.strokes.is_empty()
    }

    /// Clear all strokes
    pub fn clear(&mut self) {
        self.strokes.clear();
    }

    /// Convert all strokes in this session to EditOps for the edit system.
    /// Each stroke becomes a FillRegion (for Replace/Add) or ClearRegion (for Subtract).
    pub fn to_edit_ops(&self) -> Vec<EditOp> {
        self.strokes
            .iter()
            .map(|stroke| {
                match stroke.blend_mode {
                    BlendMode::Replace | BlendMode::Add => EditOp::FillRegion {
                        region: stroke.world_bounds,
                        voxel: stroke.voxel,
                    },
                    BlendMode::Subtract => EditOp::ClearRegion {
                        region: stroke.world_bounds,
                    },
                }
            })
            .collect()
    }

    /// Apply all strokes to an edit overlay, returning the IDs of created edits.
    pub fn apply_to_overlay(&self, overlay: &mut EditOverlay, frame: u32) -> Vec<u64> {
        self.to_edit_ops()
            .into_iter()
            .map(|op| overlay.add_edit(op, frame))
            .collect()
    }

    /// Mark all regions affected by this session's strokes as dirty.
    pub fn invalidate_chunks(&self, invalidator: &mut ChunkInvalidator) {
        for stroke in &self.strokes {
            invalidator.mark_dirty(&stroke.world_bounds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session() {
        let session = BrushSession::new();
        assert!(session.is_empty());
        assert_eq!(session.len(), 0);
    }

    #[test]
    fn test_add_strokes() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(255, 0, 0, 1);

        session
            .sphere(Vec3::ZERO, 1.0, voxel, 5)
            .sphere(Vec3::X, 0.5, voxel, 6);

        assert_eq!(session.len(), 2);
    }

    #[test]
    fn test_blend_mode() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(0, 255, 0, 1);

        session.set_blend(BlendMode::Add);
        session.sphere(Vec3::ZERO, 1.0, voxel, 5);

        assert_eq!(session.strokes()[0].blend_mode, BlendMode::Add);
    }

    #[test]
    fn test_fluent_api() {
        let mut session = BrushSession::new();
        let bark = Voxel::new(139, 90, 43, 2);
        let leaf = Voxel::new(34, 139, 34, 3);

        // Build a simple tree with fluent API
        session
            .capsule(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), 0.2, bark, 4)
            .sphere(Vec3::new(0.0, 2.5, 0.0), 1.0, leaf, 5);

        assert_eq!(session.len(), 2);
    }

    #[test]
    fn test_into_strokes() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(255, 255, 255, 1);
        session.sphere(Vec3::ZERO, 1.0, voxel, 5);

        let strokes = session.into_strokes();
        assert_eq!(strokes.len(), 1);
    }

    #[test]
    fn test_to_edit_ops_replace() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(255, 0, 0, 1);

        session.set_blend(BlendMode::Replace);
        session.sphere(Vec3::ZERO, 1.0, voxel, 5);

        let ops = session.to_edit_ops();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            EditOp::FillRegion { region: _, voxel: v } => {
                assert_eq!(v.material_id, 1);
            }
            _ => panic!("Expected FillRegion for Replace blend mode"),
        }
    }

    #[test]
    fn test_to_edit_ops_subtract() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(0, 0, 0, 0);

        session.set_blend(BlendMode::Subtract);
        session.sphere(Vec3::ZERO, 1.0, voxel, 5);

        let ops = session.to_edit_ops();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            EditOp::ClearRegion { .. } => {
                // Expected
            }
            _ => panic!("Expected ClearRegion for Subtract blend mode"),
        }
    }

    #[test]
    fn test_apply_to_overlay() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(0, 255, 0, 2);

        session.sphere(Vec3::ZERO, 1.0, voxel, 5);
        session.sphere(Vec3::new(5.0, 5.0, 5.0), 0.5, voxel, 6);

        let mut overlay = EditOverlay::new();
        let ids = session.apply_to_overlay(&mut overlay, 0);

        assert_eq!(ids.len(), 2);
        assert_eq!(overlay.edit_count(), 2);
    }

    #[test]
    fn test_invalidate_chunks() {
        let mut session = BrushSession::new();
        let voxel = Voxel::new(100, 100, 100, 3);

        // Add strokes in different regions
        session.sphere(Vec3::new(2.0, 2.0, 2.0), 0.5, voxel, 5);
        session.sphere(Vec3::new(10.0, 10.0, 10.0), 0.5, voxel, 5);

        let mut invalidator = ChunkInvalidator::new();
        session.invalidate_chunks(&mut invalidator);

        // Should have marked at least some chunks dirty
        assert!(invalidator.has_dirty());
        let dirty = invalidator.take_dirty_chunks();
        assert!(dirty.len() > 0);
    }
}
