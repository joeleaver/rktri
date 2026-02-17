//! Rock generation utility.
//!
//! Generates procedural rocks using the rock_library and saves them to disk.

use std::path::PathBuf;
use std::time::Instant;

use rktri::voxel::rock_library::{RockGenerator, RockParams, MATERIAL_ROCK};

const DEFAULT_OUTPUT_DIR: &str = "assets/rocks";
const DEFAULT_COUNT: usize = 30;

#[derive(Debug)]
struct Args {
    output_dir: PathBuf,
    count: usize,
    seed: u64,
    preset: Option<String>,
    height: f32,
    mossy: bool,
    snowy: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    
    let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
    let mut count = DEFAULT_COUNT;
    let mut seed = 42u64;
    let mut preset: Option<String> = None;
    let mut height = 0.5f32;
    let mut mossy = true;
    let mut snowy = true;
    
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output-dir" => {
                if let Some(v) = args.next() {
                    output_dir = PathBuf::from(v);
                }
            }
            "-c" | "--count" => {
                if let Some(v) = args.next() {
                    count = v.parse().unwrap_or(DEFAULT_COUNT);
                }
            }
            "-s" | "--seed" => {
                if let Some(v) = args.next() {
                    seed = v.parse().unwrap_or(42);
                }
            }
            "-p" | "--preset" => {
                if let Some(v) = args.next() {
                    preset = Some(v);
                }
            }
            "-H" | "--height" => {
                if let Some(v) = args.next() {
                    height = v.parse().unwrap_or(0.5);
                }
            }
            "--mossy" => { mossy = true; }
            "--no-mossy" => { mossy = false; }
            "--snowy" => { snowy = true; }
            "--no-snowy" => { snowy = false; }
            "-h" | "--help" | "help" => {
                return Err("show_help".to_string());
            }
            _ => {}
        }
    }
    
    Ok(Args {
        output_dir,
        count,
        seed,
        preset,
        height,
        mossy,
        snowy,
    })
}

fn print_help() {
    println!("Rock Generation Utility");
    println!("======================");
    println!();
    println!("Usage: generate_rocks [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -o, --output-dir <DIR>   Output directory (default: assets/rocks)");
    println!("  -c, --count <N>         Number of rocks to generate (default: 30)");
    println!("  -s, --seed <N>          Random seed (default: 42)");
    println!("  -p, --preset <PRESET>   Rock preset: small, medium, large, huge, flat, mossy, snowy, mixed");
    println!("  -H, --height <N>        Rock height in meters (for custom)");
    println!("  --mossy                 Include mossy rocks (default: true)");
    println!("  --no-mossy              Exclude mossy rocks");
    println!("  --snowy                 Include snowy rocks (default: true)");
    println!("  --no-snowy              Exclude snowy rocks");
    println!();
    println!("Examples:");
    println!("  generate_rocks --count 20");
    println!("  generate_rocks --preset large --count 10");
    println!("  generate_rocks --height 1.5 --count 5");
}

fn generate_preset(
    preset: &str,
    count: usize,
    base_seed: u64,
    _output_dir: &PathBuf,
) -> std::io::Result<usize> {
    let mut generated = 0;
    let start = Instant::now();

    for i in 0..count {
        let seed = base_seed.wrapping_add(i as u64).wrapping_mul(0x517cc1b727220a95);
        
        let params = match preset {
            "small" => RockParams::small_boulder(),
            "medium" => RockParams::medium_rock(),
            "large" => RockParams::large_boulder(),
            "huge" => RockParams::huge_boulder(),
            "flat" => RockParams::flat_rock(),
            "mossy" => RockParams::mossy_rock(),
            "snowy" => RockParams::snowy_rock(),
            _ => RockParams::random(seed),
        };

        let mut generator = RockGenerator::with_params(seed, params.clone());
        let _octree = generator.generate(params.height);
        
        generated += 1;
        
        if (generated) % 5 == 0 || generated == count {
            print!("\r  {}/{} rocks generated", generated, count);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    
    let elapsed = start.elapsed();
    println!("\n  Completed in {:.2}s ({:.1} rocks/sec)",
             elapsed.as_secs_f64(),
             generated as f64 / elapsed.as_secs_f64());
    
    Ok(generated)
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            if e == "show_help" {
                print_help();
                return;
            }
            eprintln!("Error: {}", e);
            print_help();
            std::process::exit(1);
        }
    };

    println!("Rock Generation Utility");
    println!("=======================");
    println!("Output directory: {}", args.output_dir.display());
    println!("Number of rocks: {}", args.count);
    println!("Base seed: {}", args.seed);
    
    if let Some(preset) = &args.preset {
        println!("Preset: {}", preset);
    } else {
        println!("Height: {}m", args.height);
    }
    
    println!("Mossy rocks: {}", args.mossy);
    println!("Snowy rocks: {}", args.snowy);
    println!();

    // Create output directory (not strictly needed since we're not saving yet)
    std::fs::create_dir_all(&args.output_dir).ok();

    let mut total_generated = 0;
    let base_seed = args.seed;

    // Generate rocks based on preset or custom settings
    if let Some(ref preset) = args.preset {
        let count = args.count;
        println!("Generating {} {} rocks...", count, preset);
        total_generated += generate_preset(&preset, count, base_seed, &args.output_dir).unwrap();
    } else {
        // Custom height - generate rocks of specified size
        let count = args.count;
        let height = args.height;
        
        println!("Generating {} rocks with height {}m...", count, height);
        
        let start = Instant::now();
        for i in 0..count {
            let seed = base_seed.wrapping_add(i as u64).wrapping_mul(0x517cc1b727220a95);
            
            let mut params = RockParams::random(seed);
            params.height = height;
            
            let mut generator = RockGenerator::with_params(seed, params);
            let _octree = generator.generate(height);
            
            total_generated += 1;
            
            if (i + 1) % 5 == 0 || i + 1 == count {
                print!("\r  {}/{} rocks generated", i + 1, count);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
        
        let elapsed = start.elapsed();
        println!("\n  Completed in {:.2}s ({:.1} rocks/sec)",
                 elapsed.as_secs_f64(),
                 total_generated as f64 / elapsed.as_secs_f64());
    }

    // Generate mossy rocks if enabled
    if args.mossy && args.preset.is_none() {
        let mossy_count = args.count / 5;
        if mossy_count > 0 {
            println!("Generating {} mossy rocks...", mossy_count);
            total_generated += generate_preset("mossy", mossy_count, base_seed + 10000, &args.output_dir).unwrap();
        }
    }

    // Generate snowy rocks if enabled
    if args.snowy && args.preset.is_none() {
        let snowy_count = args.count / 5;
        if snowy_count > 0 {
            println!("Generating {} snowy rocks...", snowy_count);
            total_generated += generate_preset("snowy", snowy_count, base_seed + 20000, &args.output_dir).unwrap();
        }
    }

    println!();
    println!("Total rocks generated: {}", total_generated);
    println!("Output directory: {}", args.output_dir.display());
}
