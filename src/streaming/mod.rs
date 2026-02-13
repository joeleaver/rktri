//! Dynamic chunk loading and LOD management

pub mod disk_io;
pub mod priority;
pub mod chunk_loader;
pub mod cache;
pub mod budget;
pub mod lod;

pub use disk_io::{
    Chunk, ChunkCoord, ChunkData,
    compress_chunk, decompress_chunk,
    serialize_chunk, deserialize_chunk,
    save_chunk, load_chunk, delete_chunk, chunk_exists,
    chunk_path,
};
pub use priority::{ChunkPriority, ChunkPriorityQueue};
pub use chunk_loader::{ChunkLoader, LoadRequest, LoadResult};
pub use cache::ChunkCache;
pub use budget::MemoryBudget;
pub use lod::{
    LodConfig, lod_from_distance, traversal_depth_for_lod,
    voxel_size_at_lod, lod_blend_factor, LOD_DISTANCES, MAX_LOD,
};
