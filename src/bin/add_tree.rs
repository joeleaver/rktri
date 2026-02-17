//! Add one or more trees to an existing world for visual testing.
//!
//! Usage:
//!   cargo run --release --bin add_tree -- --world grass_test
//!   cargo run --release --bin add_tree -- --world forest_world --count 50 --radius 50

use glam::Vec3;

use rktri::generation::{GenerationConfig, GenerationPipeline, BiomeNoiseGenerator, MaskDrivenTerrainClassifier};
use rktri::mask::MaskBuilder;
use rktri::terrain::generator::TerrainParams;
use rktri::voxel::chunk::{ChunkCoord, CHUNK_SIZE};
use rktri::voxel::procgen::{TreeGenerator, TreeStyle};
use rktri::voxel::svo::adaptive::AdaptiveOctreeBuilder;
use rktri::voxel::tree_merge::{TreeInstance, MultiTreeClassifier};
use rktri::streaming::disk_io;

use std::path::PathBuf;

/// Simple LCG random number generator for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn gen_f32(&mut self) -> f32 {
        (self.next() >> 32) as f32 / u32::MAX as f32
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        (self.gen_f32() * range.len() as f32) as usize + range.start
    }
}

const TREE_STYLES: &[TreeStyle] = &[
    TreeStyle::Oak,
    TreeStyle::Willow,
    TreeStyle::Elm,
    TreeStyle::WinterOak,
    TreeStyle::WinterWillow,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let world_name = args.iter()
        .position(|a| a == "--world")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: add_tree --world <name> [--count N] [--radius R]");

    let count = args.iter()
        .position(|a| a == "--count")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let radius = args.iter()
        .position(|a| a == "--radius")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    let world_dir = PathBuf::from(format!("assets/worlds/{}", world_name));
    let manifest_path = world_dir.join("manifest.json");

    // Read manifest
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("Failed to read manifest")
    ).expect("Failed to parse manifest");

    let tp = &manifest["terrain_params"];
    let seed = manifest["seed"].as_u64().unwrap_or(12345) as u32;

    let terrain_params = TerrainParams {
        seed,
        scale: tp["scale"].as_f64().unwrap_or(150.0) as f32,
        height_scale: tp["height_scale"].as_f64().unwrap_or(80.0) as f32,
        octaves: tp["octaves"].as_u64().unwrap_or(5) as u32,
        persistence: tp["persistence"].as_f64().unwrap_or(0.5) as f32,
        lacunarity: tp["lacunarity"].as_f64().unwrap_or(2.0) as f32,
        sea_level: tp["sea_level"].as_f64().unwrap_or(20.0) as f32,
    };

    let config = GenerationConfig {
        seed,
        terrain_params,
        biome_mask_depth: 3,
        grass_mask_depth: 5,
        clutter_mask_depth: 4,
    };
    let pipeline = GenerationPipeline::new(&config);

    let size = manifest["size"].as_f64().unwrap_or(200.0) as f32;
    let center = size / 2.0;

    // Generate random tree positions
    let mut rng = SimpleRng::new(12345);
    let mut tree_positions: Vec<(Vec3, TreeStyle)> = Vec::with_capacity(count);

    println!("=== Adding {} trees to world '{}' ===", count, world_name);

    for i in 0..count {
        // Random position within radius around center
        let angle = rng.gen_f32() * std::f32::consts::TAU;
        let dist = rng.gen_f32() * radius;
        let tree_x = center + angle.cos() * dist;
        let tree_z = center + angle.sin() * dist;

        let terrain_height = pipeline.height_at(tree_x, tree_z);
        let tree_pos = Vec3::new(tree_x, terrain_height, tree_z);

        // Pick random tree style
        let style = TREE_STYLES[rng.gen_range(0..TREE_STYLES.len())];

        if i < 5 || count <= 10 {
            println!("Tree {}: style={:?}, pos=({:.1}, {:.1}, {:.1})", i, style, tree_pos.x, tree_pos.y, tree_pos.z);
        } else if i == 5 {
            println!("  ... ({} more trees)", count - 5);
        }

        tree_positions.push((tree_pos, style));
    }

    // Collect all tree AABBs to determine affected chunks
    let mut tree_instances: Vec<TreeInstance> = Vec::with_capacity(count);
    let mut combined_min = Vec3::splat(f32::MAX);
    let mut combined_max = Vec3::splat(f32::MIN);

    for (tree_pos, style) in &tree_positions {
        let mut tree_gen = TreeGenerator::from_style(42, *style);
        let tree_octree = tree_gen.generate(16.0, 7);
        let tree_instance = TreeInstance::new(tree_octree, *tree_pos);
        let tree_aabb = tree_instance.world_aabb();

        combined_min = combined_min.min(tree_aabb.min);
        combined_max = combined_max.max(tree_aabb.max);

        tree_instances.push(tree_instance);
    }

    println!("Combined AABB: ({:.1},{:.1},{:.1}) -> ({:.1},{:.1},{:.1})",
        combined_min.x, combined_min.y, combined_min.z,
        combined_max.x, combined_max.y, combined_max.z);

    // Find affected chunks from combined AABB
    let chunk_f = CHUNK_SIZE as f32;
    let min_cx = (combined_min.x / chunk_f).floor() as i32;
    let max_cx = (combined_max.x / chunk_f).floor() as i32;
    let min_cy = (combined_min.y / chunk_f).floor() as i32;
    let max_cy = (combined_max.y / chunk_f).floor() as i32;
    let min_cz = (combined_min.z / chunk_f).floor() as i32;
    let max_cz = (combined_max.z / chunk_f).floor() as i32;

    println!("Affected chunks: x={}..{}, y={}..{}, z={}..{}",
        min_cx, max_cx, min_cy, max_cy, min_cz, max_cz);

    let terrain_dir = world_dir.join("terrain");
    let mut chunks_written = 0;
    let mut new_chunks = Vec::new();

    // Extract sea_level before the loop to ensure config doesn't need to live into the loop
    let sea_level = config.terrain_params.sea_level;

    for cx in min_cx..=max_cx {
        for cy in min_cy..=max_cy {
            for cz in min_cz..=max_cz {
                let coord = ChunkCoord::new(cx, cy, cz);
                let origin = Vec3::new(
                    cx as f32 * chunk_f,
                    cy as f32 * chunk_f,
                    cz as f32 * chunk_f,
                );

                // Build biome mask
                let biome_gen = BiomeNoiseGenerator::new(
                    pipeline.biome_map(), pipeline.terrain(), sea_level);
                let biome_mask = MaskBuilder::new(3)
                    .build(&biome_gen, origin, chunk_f);

                // Build terrain classifier
                let voxel_size = chunk_f / 128.0;
                let terrain_classifier = MaskDrivenTerrainClassifier::new(
                    pipeline.terrain(), &biome_mask, origin, voxel_size);

                // Wrap with all trees
                let mut tree_classifier = MultiTreeClassifier::new(&terrain_classifier);
                for tree_instance in &tree_instances {
                    // Check if tree AABB overlaps this chunk before adding
                    let tree_aabb = tree_instance.world_aabb();
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(chunk_f);

                    let overlaps = tree_aabb.min.x < chunk_max.x && tree_aabb.max.x > chunk_min.x
                        && tree_aabb.min.y < chunk_max.y && tree_aabb.max.y > chunk_min.y
                        && tree_aabb.min.z < chunk_max.z && tree_aabb.max.z > chunk_min.z;

                    if overlaps {
                        tree_classifier.add_tree(tree_instance.clone());
                    }
                }

                // Build chunk octree
                let builder = AdaptiveOctreeBuilder::new(128);
                let octree = builder.build(&tree_classifier, origin, chunk_f);

                let node_count = octree.node_count();
                if node_count == 0 {
                    continue;
                }

                // Serialize and write
                let disk_coord = disk_io::ChunkCoord::new(cx, cy, cz);
                let disk_chunk = disk_io::Chunk::from_octree(disk_coord, octree);
                let compressed = disk_io::compress_chunk(&disk_chunk)
                    .expect("Failed to compress chunk");

                let chunk_file = terrain_dir.join(format!("chunk_{}_{}_{}.rkc", cx, cy, cz));
                let is_new = !chunk_file.exists();
                std::fs::write(&chunk_file, &compressed)
                    .expect("Failed to write chunk file");

                if is_new {
                    new_chunks.push((cx, cy, cz));
                }

                println!("  {} chunk ({},{},{}) - {} nodes, {} bytes",
                    if is_new { "NEW" } else { "UPD" },
                    cx, cy, cz, node_count, compressed.len());
                chunks_written += 1;
            }
        }
    }

    // Update manifest with new chunks
    if !new_chunks.is_empty() {
        let manifest_str = std::fs::read_to_string(&manifest_path).unwrap();
        let mut manifest: serde_json::Value = serde_json::from_str(&manifest_str).unwrap();

        if let Some(layers) = manifest["layers"].as_array_mut() {
            if let Some(terrain_layer) = layers.get_mut(0) {
                if let Some(chunks) = terrain_layer["chunks"].as_array_mut() {
                    for (x, y, z) in &new_chunks {
                        chunks.push(serde_json::json!({"x": x, "y": y, "z": z}));
                    }
                }
                let old_count = terrain_layer["chunk_count"].as_u64().unwrap_or(0);
                terrain_layer["chunk_count"] = serde_json::json!(old_count + new_chunks.len() as u64);
            }
        }

        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap())
            .expect("Failed to write manifest");
        println!("Updated manifest with {} new chunks", new_chunks.len());
    }

    println!();
    println!("Done! {} chunks written.", chunks_written);
    println!("Run: cargo run --release --bin rktri -- --world {}", world_name);
}
