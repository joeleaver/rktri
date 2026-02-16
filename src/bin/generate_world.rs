//! World generator binary — pre-generates terrain chunks to disk.
//!
//! Usage: cargo run --release --bin generate_world -- [OPTIONS]
//!
//! Options:
//!   --size <METERS>   World size in meters (default: 1000)
//!   --seed <SEED>     Random seed (default: 12345)
//!   --name <NAME>     World name / output directory (default: "terrain")
//!   --scale <SCALE>   Terrain noise scale (default: 150.0)
//!   --height <H>      Terrain height scale (default: 80.0)
//!   --jobs <N>         Max parallel chunk builds (default: 4)
//!
//! Output structure:
//!   assets/worlds/<name>/
//!     manifest.json           # World metadata + per-layer chunk lists
//!     terrain/                # Terrain layer chunks
//!       chunk_0_10_0.rkc
//!       ...
//!     grass/                  # Grass mask layer
//!       chunk_0_10_0.rkm
//!       ...

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;
use serde_json::json;

use rktri::generation::{GenerationConfig, GenerationPipeline};
use rktri::streaming::disk_io;
use rktri::terrain::generator::TerrainParams;
use rktri::voxel::chunk::{ChunkCoord, CHUNK_SIZE};
use rktri::voxel::svo::svdag::SvdagBuilder;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format_timestamp_millis()
    .init();

    let args: Vec<String> = std::env::args().collect();
    let size = parse_f32_arg(&args, "--size").unwrap_or(1000.0);
    let seed = parse_u32_arg(&args, "--seed").unwrap_or(12345);
    let name = parse_str_arg(&args, "--name").unwrap_or_else(|| "terrain".to_string());
    let scale = parse_f32_arg(&args, "--scale").unwrap_or(150.0);
    let height_scale = parse_f32_arg(&args, "--height").unwrap_or(80.0);
    let jobs = parse_usize_arg(&args, "--jobs").unwrap_or(4);

    // Limit rayon's thread pool to cap peak memory usage
    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .expect("Failed to configure thread pool");

    let output_dir = PathBuf::from(format!("assets/worlds/{}", name));

    println!("=== Rktri World Generator ===");
    println!("World: {}", name);
    println!("Size:  {}m x {}m", size, size);
    println!("Seed:  {}", seed);
    println!("Scale: {}, Height: {}", scale, height_scale);
    println!("Jobs:  {} parallel", jobs);
    println!("Output: {}", output_dir.display());
    println!();

    let terrain_params = TerrainParams {
        seed,
        scale,
        height_scale,
        octaves: 5,
        persistence: 0.5,
        lacunarity: 2.0,
        sea_level: 20.0,
    };

    let config = GenerationConfig {
        seed,
        terrain_params: terrain_params.clone(),
        biome_mask_depth: 3,
        grass_mask_depth: 5,
    };
    let pipeline = GenerationPipeline::new(&config);

    // Phase 1: Collect chunk coordinates
    let radius = size / 2.0;
    let chunk_f = CHUNK_SIZE as f32;
    let chunk_radius = (radius / chunk_f).floor() as i32;
    let center = Vec3::new(radius, 0.0, radius);
    let center_coord = ChunkCoord::from_world_pos(center);

    let mut coords: Vec<ChunkCoord> = Vec::new();
    for dx in -chunk_radius..=chunk_radius {
        for dz in -chunk_radius..=chunk_radius {
            let cx = (center_coord.x + dx) as f32 * chunk_f + chunk_f * 0.5;
            let cz = (center_coord.z + dz) as f32 * chunk_f + chunk_f * 0.5;
            let h = pipeline.height_at(cx, cz);

            let min_y = ((h - chunk_f) / chunk_f).floor().max(0.0) as i32;
            let max_y = ((h + chunk_f) / chunk_f).ceil() as i32;

            for dy in min_y..=max_y {
                coords.push(ChunkCoord::new(center_coord.x + dx, dy, center_coord.z + dz));
            }
        }
    }

    let total = coords.len();
    let side = chunk_radius * 2 + 1;
    println!("Chunks: {} candidates ({} x {} XZ grid, height-filtered)", total, side, side);
    println!();

    // Phase 2: Generate terrain + grass layers
    let terrain_dir = output_dir.join("terrain");
    let grass_dir = output_dir.join("grass");
    std::fs::create_dir_all(&terrain_dir).expect("Failed to create terrain directory");
    std::fs::create_dir_all(&grass_dir).expect("Failed to create grass directory");

    let start = Instant::now();
    let generated = AtomicUsize::new(0);
    let total_terrain_bytes = AtomicUsize::new(0);
    let total_grass_bytes = AtomicUsize::new(0);
    let grass_chunk_count = AtomicUsize::new(0);

    let terrain_chunks: Vec<(i32, i32, i32)> = coords
        .par_iter()
        .filter_map(|&coord| {
            let result = pipeline.generate_chunk_with_grass(coord);
            let done = generated.fetch_add(1, Ordering::Relaxed) + 1;

            if done % 1000 == 0 || done == total {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let remaining = (total - done) as f64 / rate;
                eprintln!("  [{}/{}] {:.0} chunks/sec, ~{:.0}s remaining",
                    done, total, rate, remaining);
            }

            if result.chunk.octree.brick_count() == 0 {
                return None;
            }

            // Write terrain chunk with SVDAG pre-compression (v3 format)
            let disk_coord = disk_io::ChunkCoord::new(coord.x, coord.y, coord.z);

            // Apply SVDAG compression ONCE during world generation
            let pruned = result.chunk.octree.prune();
            let svdag = SvdagBuilder::new().build(&pruned);

            let disk_chunk = disk_io::Chunk::from_octree(disk_coord, svdag);
            let compressed = disk_io::serialize_svdag_chunk(&disk_chunk)
                .expect("Failed to serialize SVDAG chunk");
            let compressed = lz4_flex::compress_prepend_size(&compressed);

            let chunk_file = terrain_dir.join(format!("chunk_{}_{}_{}.rkc", coord.x, coord.y, coord.z));
            total_terrain_bytes.fetch_add(compressed.len(), Ordering::Relaxed);
            std::fs::write(&chunk_file, &compressed)
                .expect("Failed to write chunk file");

            // Write grass mask (only if non-empty)
            if !result.grass_mask.is_empty() {
                let grass_compressed = disk_io::compress_grass_mask(disk_coord, &result.grass_mask)
                    .expect("Failed to compress grass mask");
                let grass_file = grass_dir.join(format!("chunk_{}_{}_{}.rkm", coord.x, coord.y, coord.z));
                total_grass_bytes.fetch_add(grass_compressed.len(), Ordering::Relaxed);
                std::fs::write(&grass_file, &grass_compressed)
                    .expect("Failed to write grass mask file");
                grass_chunk_count.fetch_add(1, Ordering::Relaxed);
            }

            Some((coord.x, coord.y, coord.z))
        })
        .collect();

    let elapsed = start.elapsed();
    let terrain_bytes = total_terrain_bytes.load(Ordering::Relaxed);
    let grass_bytes = total_grass_bytes.load(Ordering::Relaxed);
    let grass_count = grass_chunk_count.load(Ordering::Relaxed);

    println!();
    println!("Terrain: {} chunks in {:.1}s ({:.0} chunks/sec, {:.1} MB)",
        terrain_chunks.len(), elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64(),
        terrain_bytes as f64 / (1024.0 * 1024.0));
    println!("Grass:   {} chunks with masks ({:.1} KB)",
        grass_count, grass_bytes as f64 / 1024.0);

    // Phase 3: Write manifest with per-layer structure
    let mut y_groups: BTreeMap<i32, usize> = BTreeMap::new();
    for &(_, y, _) in &terrain_chunks {
        *y_groups.entry(y).or_insert(0) += 1;
    }

    let manifest = json!({
        "name": name,
        "version": 3,
        "seed": seed,
        "size": size,
        "chunk_size": CHUNK_SIZE,
        "terrain_params": {
            "scale": scale,
            "height_scale": height_scale,
            "octaves": 5,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "sea_level": 20.0,
        },
        "layers": [
            {
                "name": "terrain",
                "id": 0,
                "directory": "terrain",
                "chunk_count": terrain_chunks.len(),
                "total_bytes": terrain_bytes,
                "y_levels": y_groups,
                "chunks": terrain_chunks.iter().map(|(x, y, z)| {
                    json!({"x": x, "y": y, "z": z})
                }).collect::<Vec<_>>(),
            },
            {
                "name": "grass",
                "id": 1,
                "directory": "grass",
                "chunk_count": grass_count,
                "total_bytes": grass_bytes,
            }
        ],
    });

    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap())
        .expect("Failed to write manifest");

    println!();
    println!("=== Generation Complete ===");
    println!("Layers: 2 (terrain, grass)");
    println!("Chunks: {} with geometry (of {} candidates)", terrain_chunks.len(), total);
    println!("Grass:  {} chunks with masks", grass_count);
    println!("Size:   {:.1} MB terrain + {:.1} KB grass on disk",
        terrain_bytes as f64 / (1024.0 * 1024.0),
        grass_bytes as f64 / 1024.0);
    println!("Output: {}", output_dir.display());
    println!();
    println!("To load this world:");
    println!("  cargo run --release --bin rktri -- --world {}", name);
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_u32_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone())
}
