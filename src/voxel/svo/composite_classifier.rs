//! Composite classifier that layers multiple classifiers.
//!
//! Provides priority-based composition: edits > volumes > terrain.
//! This replaces the old CompositeEvaluator with proper 3D support.

use glam::Vec3;
use crate::voxel::svo::classifier::{RegionClassifier, RegionHint};
use crate::math::Aabb;
use crate::voxel::voxel::Voxel;

/// Combines multiple RegionClassifiers with priority ordering.
///
/// Classifiers are evaluated in order until a definitive answer is found.
/// This allows edits to override volumes, which override terrain.
pub struct CompositeRegionClassifier {
    /// Classifiers in priority order (first = highest priority)
    classifiers: Vec<Box<dyn RegionClassifier>>,
}

impl CompositeRegionClassifier {
    /// Create an empty composite classifier.
    pub fn new() -> Self {
        Self {
            classifiers: Vec::new(),
        }
    }

    /// Create from a list of classifiers (first = highest priority).
    pub fn from_classifiers(classifiers: Vec<Box<dyn RegionClassifier>>) -> Self {
        Self { classifiers }
    }

    /// Add a classifier with highest priority.
    pub fn push_front(&mut self, classifier: Box<dyn RegionClassifier>) {
        self.classifiers.insert(0, classifier);
    }

    /// Add a classifier with lowest priority.
    pub fn push_back(&mut self, classifier: Box<dyn RegionClassifier>) {
        self.classifiers.push(classifier);
    }

    /// Number of classifiers.
    pub fn len(&self) -> usize {
        self.classifiers.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.classifiers.is_empty()
    }
}

impl Default for CompositeRegionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionClassifier for CompositeRegionClassifier {
    fn classify_region(&self, aabb: &Aabb) -> RegionHint {
        // Track if any classifier returned Mixed
        let mut has_mixed = false;

        for classifier in &self.classifiers {
            match classifier.classify_region(aabb) {
                // Terminal results from high-priority classifiers override lower
                RegionHint::Empty => {
                    // Empty from high priority means no content from this layer
                    // Continue checking lower priority
                }
                RegionHint::Solid { material, color } => {
                    // Solid from any layer means we have content
                    // But other layers might have different content - treat as mixed
                    // unless this is the only layer
                    if self.classifiers.len() == 1 {
                        return RegionHint::Solid { material, color };
                    }
                    has_mixed = true;
                }
                RegionHint::Mixed => {
                    has_mixed = true;
                }
                RegionHint::Unknown => {
                    // Unknown means we need to evaluate - treat as mixed
                    has_mixed = true;
                }
            }
        }

        if has_mixed {
            RegionHint::Mixed
        } else {
            // All classifiers returned Empty
            RegionHint::Empty
        }
    }

    fn evaluate(&self, pos: Vec3) -> Voxel {
        // Evaluate in priority order, return first non-empty voxel
        for classifier in &self.classifiers {
            let voxel = classifier.evaluate(pos);
            if !voxel.is_empty() {
                return voxel;
            }
        }
        Voxel::EMPTY
    }
}

// Allow using as trait object
unsafe impl Send for CompositeRegionClassifier {}
unsafe impl Sync for CompositeRegionClassifier {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test classifier that always returns a specific hint
    struct ConstantClassifier {
        hint: RegionHint,
        voxel: Voxel,
    }

    impl ConstantClassifier {
        fn empty() -> Self {
            Self { hint: RegionHint::Empty, voxel: Voxel::EMPTY }
        }

        fn solid(material: u8) -> Self {
            Self {
                hint: RegionHint::Solid { material, color: 0 },
                voxel: Voxel::new(255, 255, 255, material),
            }
        }

        fn mixed(voxel: Voxel) -> Self {
            Self { hint: RegionHint::Mixed, voxel }
        }
    }

    impl RegionClassifier for ConstantClassifier {
        fn classify_region(&self, _aabb: &Aabb) -> RegionHint {
            self.hint
        }

        fn evaluate(&self, _pos: Vec3) -> Voxel {
            self.voxel
        }
    }

    #[test]
    fn test_empty_composite() {
        let classifier = CompositeRegionClassifier::new();
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(classifier.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_single_empty_classifier() {
        let mut composite = CompositeRegionClassifier::new();
        composite.push_back(Box::new(ConstantClassifier::empty()));

        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(composite.classify_region(&aabb), RegionHint::Empty);
    }

    #[test]
    fn test_mixed_overrides_empty() {
        let mut composite = CompositeRegionClassifier::new();
        composite.push_back(Box::new(ConstantClassifier::mixed(Voxel::new(255, 255, 255, 1))));
        composite.push_back(Box::new(ConstantClassifier::empty()));

        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(composite.classify_region(&aabb), RegionHint::Mixed);
    }

    #[test]
    fn test_evaluate_returns_first_non_empty() {
        let mut composite = CompositeRegionClassifier::new();
        composite.push_back(Box::new(ConstantClassifier::empty()));
        composite.push_back(Box::new(ConstantClassifier::solid(5)));

        let voxel = composite.evaluate(Vec3::ZERO);
        assert!(!voxel.is_empty());
        assert_eq!(voxel.material_id, 5);
    }
}
