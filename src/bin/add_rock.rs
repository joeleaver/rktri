//! Add one or more rocks to an existing world by merging into terrain chunks.
//!
//! Usage:
//!   cargo run --release --bin add_rock -- --world my_world --count 30 --radius 40

use glam::Vec3;

use rktri::generation::{GenerationConfig, GenerationPipeline, BiomeNoiseGenerator, MaskDrivenTerrainClassifier};
use rktri::mask::MaskBuilder;
use rktri::terrain::generator::TerrainParams;
use rktri::voxel::chunk::{ChunkCoord, CHUNK_SIZE};
use rktri::voxel::rock_library::{RockGenerator, RockParams};
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

/// Available rock size presets
const ROCK_PRESETS: &[(&str, fn() -> RockParams, f32)] = &[
    ("small", RockParams::small_boulder, 0.25),
    ("medium", RockParams::medium_rock, 0.5),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let world_name = args.iter()
        .position(|a| a == "--world")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: add_rock --world <name> [--count N] [--radius R]");

    let count = args.iter()
        .position(|a| a == "--count")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let radius = args.iter()
        .position(|a| a == "--radius")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(40.0);

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

    // Generate random rock positions with sizes
    let mut rng = SimpleRng::new(54321);
    let mut rock_instances: Vec<TreeInstance> = Vec::with_capacity(count);
    let mut combined_min = Vec3::splat(f32::MAX);
    let mut combined_max = Vec3::splat(f32::MIN);

    let embed_ratio = 0.35; // 35% embed

    println!("=== Adding {} rocks to world '{}' ===", count, world_name);
    println!("Distribution radius: {}m", radius);

    for i in 0..count {
        // Random position within radius around center
        let angle = rng.gen_f32() * std::f32::consts::TAU;
        let dist = rng.gen_f32() * radius;
        let rock_x = center + angle.cos() * dist;
        let rock_z = center + angle.sin() * dist;

        let terrain_height = pipeline.height_at(rock_x, rock_z);

        // Pick random rock size preset
        let preset_idx = rng.gen_range(0..ROCK_PRESETS.len());
        let (name, params_fn, base_size) = ROCK_PRESETS[preset_idx];

        // Add some variation to the size (0.8x to 1.2x of base)
        let size_variation = 0.8 + rng.gen_f32() * 0.4;
        let rock_size = base_size * size_variation;

        // Position rock: embed bottom 35% into terrain
        let rock_pos = Vec3::new(rock_x, terrain_height - (rock_size * embed_ratio), rock_z);

        if i < 5 {
            println!("Rock {}: {} ({:.2}m), pos=({:.1}, {:.1}, {:.1})",
                i, name, rock_size, rock_pos.x, rock_pos.y, rock_pos.z);
        } else if i == 5 {
            println!("  ... ({} more rocks)", count - 5);
        }

        // Generate rock octree
        let params = params_fn();
        let mut rock_gen = RockGenerator::with_params(42 + i as u64, params);
        let rock_octree = rock_gen.generate(rock_size);

        let rock_instance = TreeInstance::new(rock_octree, rock_pos);
        let rock_aabb = rock_instance.world_aabb();

        combined_min = combined_min.min(rock_aabb.min);
        combined_max = combined_max.max(rock_aabb.max);

        rock_instances.push(rock_instance);
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

    let sea_level = config.terrain_params.sea_level;

    for cx in min_cx..=max_cx {
        for cy in min_cy..=max_cy {
            for cz in min_cz..=max_cz {
                let _coord = ChunkCoord::new(cx, cy, cz);
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

                // Wrap with all rocks
                let mut rock_classifier = MultiTreeClassifier::new(&terrain_classifier);
                for rock_instance in &rock_instances {
                    // Check if rock AABB overlaps this chunk before adding
                    let rock_aabb = rock_instance.world_aabb();
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(chunk_f);

                    let overlaps = rock_aabb.min.x < chunk_max.x && rock_aabb.max.x > chunk_min.x
                        && rock_aabb.min.y < chunk_max.y && rock_aabb.max.y > chunk_min.y
                        && rock_aabb.min.z < chunk_max.z && rock_aabb.max.z > chunk_min.z;

                    if overlaps {
                        rock_classifier.add_tree(rock_instance.clone());
                    }
                }

                // Build chunk octree
                let builder = AdaptiveOctreeBuilder::new(128);
                let octree = builder.build(&rock_classifier, origin, chunk_f);

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
    println!("=== Done ===");
    println!("Added {} rocks, wrote {} chunks", count, chunks_written);
    println!();
    println!("Run: cargo run --release --bin rktri -- --world {}", world_name);
}
