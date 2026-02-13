//! Sparse Voxel Octree implementation

pub mod node;
pub mod octree;
pub mod builder;
pub mod svdag;
pub mod hashdag;
pub mod adaptive;
pub mod volumetric;
pub mod composite;
pub mod classifier;
pub mod composite_classifier;

pub use node::OctreeNode;
pub use octree::Octree;
pub use builder::{OctreeBuilder, create_test_sphere};
pub use svdag::SvdagBuilder;
pub use hashdag::HashDag;
pub use adaptive::AdaptiveOctreeBuilder;
pub use volumetric::{VolumetricObject, OctreeInstance, VolumetricGrid};
// CompositeEvaluator is deprecated - use CompositeRegionClassifier instead
#[deprecated(since = "0.1.0", note = "Use CompositeRegionClassifier instead")]
pub use composite::CompositeRegionClassifier as CompositeEvaluator;
pub use classifier::{RegionHint, RegionClassifier};
pub use composite_classifier::CompositeRegionClassifier;
