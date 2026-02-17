//! GPU buffer management

pub mod octree_buffer;
pub mod camera_buffer;

pub use octree_buffer::{OctreeBuffer, GpuGrassMaskInfo, GpuGrassMaskNode, pack_grass_masks, LayerDescriptor};
pub use camera_buffer::{CameraBuffer, CameraUniform};
