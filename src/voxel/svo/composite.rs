//! DEPRECATED: Use CompositeRegionClassifier instead.
//!
//! This module is kept for backwards compatibility but will be removed.
//! The new CompositeRegionClassifier in composite_classifier.rs provides
//! proper 3D region classification that works with caves, floating islands,
//! and arbitrary volumetric content.

// Re-export the new implementation for migration
pub use super::composite_classifier::CompositeRegionClassifier;
