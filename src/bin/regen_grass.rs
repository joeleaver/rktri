//! Regenerate grass masks for an existing world.
//!
//! Usage: cargo run --release --bin regen_grass -- --world grass_test

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rktri::generation::{GenerationConfig, GenerationPipeline};
use rktri::terrain::generator::TerrainParams;
use rktri::voxel::chunk::ChunkCoord;
use rktri::streaming::disk_io;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format_timestamp_millis()
    .init();

    let args: Vec<String> = std::env::args().collect();
    let world_name = args.iter()
        .position(|a| a == "--world")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: regen_grass --world <name>");

    let world_dir = PathBuf::from(format!("assets/worlds/{}", world_name));
    let manifest_path = world_dir.join("manifest.json");

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("Failed to read manifest")
    ).expect("Failed to parse manifest");

    let tp = &manifest["terrain_params"];
    let seed = manifest["seed"].as_u64().unwrap_or(12345) as u32;
    let size = manifest["size"].as_f64().unwrap_or(200.0) as f32;

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

    // Collect chunk coords from terrain layer
    let terrain_chunks: Vec<ChunkCoord> = manifest["layers"]
        .as_array().unwrap()
        .iter()
        .find(|l| l["name"].as_str() == Some("terrain"))
        .and_then(|l| l["chunks"].as_array())
        .expect("No terrain chunks in manifest")
        .iter()
        .map(|c| ChunkCoord::new(
            c["x"].as_i64().unwrap() as i32,
            c["y"].as_i64().unwrap() as i32,
            c["z"].as_i64().unwrap() as i32,
        ))
        .collect();

    println!("=== Regenerating grass masks for '{}' ===", world_name);
    println!("Terrain chunks: {}", terrain_chunks.len());

    let grass_dir = world_dir.join("grass");
    std::fs::create_dir_all(&grass_dir).expect("Failed to create grass directory");

    let start = Instant::now();
    let generated = AtomicUsize::new(0);
    let grass_count = AtomicUsize::new(0);
    let total = terrain_chunks.len();

    terrain_chunks.par_iter().for_each(|&coord| {
        let result = pipeline.generate_chunk_with_grass(coord);
        let done = generated.fetch_add(1, Ordering::Relaxed) + 1;

        if done % 500 == 0 || done == total {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let remaining = (total - done) as f64 / rate;
            eprintln!("  [{}/{}] {:.0}/sec, ~{:.0}s remaining", done, total, rate, remaining);
        }

        if !result.grass_mask.is_empty() {
            let disk_coord = disk_io::ChunkCoord::new(coord.x, coord.y, coord.z);
            let compressed = disk_io::compress_grass_mask(disk_coord, &result.grass_mask)
                .expect("Failed to compress grass mask");
            let grass_file = grass_dir.join(format!("chunk_{}_{}_{}.rkm", coord.x, coord.y, coord.z));
            std::fs::write(&grass_file, &compressed)
                .expect("Failed to write grass mask");
            grass_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let elapsed = start.elapsed();
    let count = grass_count.load(Ordering::Relaxed);
    println!("\nDone! {} grass masks generated in {:.1}s", count, elapsed.as_secs_f64());
}
