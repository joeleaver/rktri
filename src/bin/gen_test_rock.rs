//! Generate a single rock and save to .rkc file
//!
//! Usage: cargo run --release --bin gen_test_rock

use rktri::voxel::rock_library::{RockGenerator, RockParams};
use rktri::streaming::disk_io::{self, Chunk, ChunkCoord};

fn main() {
    let seed = 42u64;
    let size = 1.0f32;

    // Generate rock
    let mut generator = RockGenerator::with_params(seed, RockParams::medium_rock());
    let octree = generator.generate(size);

    println!("Generated rock: {} nodes, {} bricks", octree.node_count(), octree.brick_count());

    // Create chunk at origin
    let chunk = Chunk::from_octree(ChunkCoord::new(0, 0, 0), octree);

    // Compress and save
    let compressed = disk_io::compress_chunk(&chunk).expect("Failed to compress");
    let path = std::path::Path::new("assets/rocks/test_rock.rkc");
    std::fs::write(path, compressed).expect("Failed to write");
    println!("Saved to {}", path.display());
}
