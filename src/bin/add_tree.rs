//! Add a single tree to an existing world for visual testing.
//!
//! Usage: cargo run --release --bin add_tree -- --world grass_test

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let world_name = args.iter()
        .position(|a| a == "--world")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: add_tree --world <name>");

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
    };
    let pipeline = GenerationPipeline::new(&config);

    // Place tree ~12m ahead of camera start position (camera starts at world center looking NE)
    let size = manifest["size"].as_f64().unwrap_or(200.0) as f32;
    let center = size / 2.0;
    let tree_x = center + 10.0;
    let tree_z = center + 10.0;
    let terrain_height = pipeline.height_at(tree_x, tree_z);
    let tree_pos = Vec3::new(tree_x, terrain_height, tree_z);

    println!("=== Adding Oak tree to world '{}' ===", world_name);
    println!("Tree position: ({:.1}, {:.1}, {:.1})", tree_pos.x, tree_pos.y, tree_pos.z);
    println!("Terrain height: {:.1}m", terrain_height);

    // Generate tree with updated generator (encodes SDF flags)
    let mut tree_gen = TreeGenerator::from_style(42, TreeStyle::Oak);
    let tree_octree = tree_gen.generate(16.0, 7);
    println!("Tree: {} nodes, {} bricks", tree_octree.node_count(), tree_octree.brick_count());

    let tree_instance = TreeInstance::new(tree_octree, tree_pos);
    let tree_aabb = tree_instance.world_aabb();
    println!("Tree AABB: ({:.1},{:.1},{:.1}) -> ({:.1},{:.1},{:.1})",
        tree_aabb.min.x, tree_aabb.min.y, tree_aabb.min.z,
        tree_aabb.max.x, tree_aabb.max.y, tree_aabb.max.z);

    // Find affected chunks
    let chunk_f = CHUNK_SIZE as f32;
    let min_cx = (tree_aabb.min.x / chunk_f).floor() as i32;
    let max_cx = (tree_aabb.max.x / chunk_f).floor() as i32;
    let min_cy = (tree_aabb.min.y / chunk_f).floor() as i32;
    let max_cy = (tree_aabb.max.y / chunk_f).floor() as i32;
    let min_cz = (tree_aabb.min.z / chunk_f).floor() as i32;
    let max_cz = (tree_aabb.max.z / chunk_f).floor() as i32;

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

                // Wrap with tree
                let mut tree_classifier = MultiTreeClassifier::new(&terrain_classifier);
                tree_classifier.add_tree(tree_instance.clone());

                // Build chunk octree
                let builder = AdaptiveOctreeBuilder::new(128);
                let octree = builder.build(&tree_classifier, origin, chunk_f);

                if octree.brick_count() == 0 {
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

                println!("  {} chunk ({},{},{}) - {} bytes",
                    if is_new { "NEW" } else { "UPD" },
                    cx, cy, cz, compressed.len());
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
