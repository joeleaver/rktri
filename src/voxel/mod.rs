//! Voxel data structures and operations

pub mod voxel;
pub mod brick;
pub mod chunk;
pub mod chunk_handle;
pub mod brick_handle;
pub mod world;
pub mod world_index;
// pub mod material;
// pub mod query;
pub mod svo;
pub mod instancing;
pub mod streaming;
pub mod brush;
pub mod procgen;
pub mod tree_data;
pub mod tree_library;
pub mod tree_merge;
pub mod layer;
pub mod volume;
pub mod terrain;
pub mod hierarchy;
pub mod super_chunk;
pub mod water;
pub mod edit;

pub use chunk::{Chunk, ChunkCoord, CHUNK_SIZE};
pub use chunk_handle::{ChunkHandle, ChunkState, GpuChunkHandle};
pub use brick_handle::{BrickId, BrickHandle};
pub use world_index::WorldIndex;
pub use tree_data::TreeData;
pub use procgen::{TreeGenerator, TreeParams, TreeStyle};
pub use world::World;
pub use instancing::{VoxelModel, ModelInstance, ModelLibrary};
pub use streaming::{BrickPool, BrickCache, BrickRequestQueue, StreamingManager, StreamingOrchestrator, MemoryBudget, StreamingStats, LoadPriority};
// TerrainClassifier moved to generation::terrain_gen::MaskDrivenTerrainClassifier
