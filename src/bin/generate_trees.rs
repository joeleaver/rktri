//! Batch tree generation utility
//!
//! Generates a library of pre-built trees for use in world generation.
//!
//! Usage:
//!     generate_trees [OPTIONS] <OUTPUT_DIR>
//!
//! Options:
//!     -s, --style <STYLE>     Tree style: oak, willow, elm, or all (default: all)
//!     -n, --count <N>         Number of trees per style (default: 100)
//!     --seed <SEED>           Base seed for RNG (default: 12345)
//!     --root-size <SIZE>      Octree root size in meters (default: 16.0)
//!     --max-depth <DEPTH>     Octree max depth (default: 7)
//!     -h, --help              Show this help message

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use rktri::voxel::procgen::{TreeGenerator, TreeStyle};
use rktri::voxel::tree_data::TreeData;
use rktri::voxel::tree_library::TreeLibrary;

fn print_help() {
    eprintln!("generate_trees - Batch tree generation utility");
    eprintln!();
    eprintln!("Usage: generate_trees [OPTIONS] <OUTPUT_DIR>");
    eprintln!();
    eprintln!("Options:");
    eprintln!("    -s, --style <STYLE>     Tree style: oak, willow, elm, or all (default: all)");
    eprintln!("    -n, --count <N>         Number of trees per style (default: 100)");
    eprintln!("    --seed <SEED>           Base seed for RNG (default: 12345)");
    eprintln!("    --root-size <SIZE>      Octree root size in meters (default: 16.0)");
    eprintln!("    --max-depth <DEPTH>     Octree max depth (default: 7)");
    eprintln!("    -h, --help              Show this help message");
    eprintln!();
    eprintln!("Example:");
    eprintln!("    generate_trees -s oak -n 50 ./assets/trees");
    eprintln!("    generate_trees --seed 42 -n 200 ./assets/trees");
}

#[derive(Debug)]
struct Args {
    output_dir: PathBuf,
    style: Option<TreeStyle>, // None means all styles
    count: u32,
    seed: u64,
    root_size: f32,
    max_depth: u8,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        return Err("Missing output directory".to_string());
    }

    let mut style: Option<TreeStyle> = None;
    let mut count: u32 = 100;
    let mut seed: u64 = 12345;
    let mut root_size: f32 = 16.0;
    let mut max_depth: u8 = 7;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-s" | "--style" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --style".to_string());
                }
                style = match args[i].to_lowercase().as_str() {
                    "oak" => Some(TreeStyle::Oak),
                    "willow" => Some(TreeStyle::Willow),
                    "elm" => Some(TreeStyle::Elm),
                    "all" => None,
                    other => return Err(format!("Unknown style: {}. Valid styles: oak, willow, elm, all", other)),
                };
            }
            "-n" | "--count" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --count".to_string());
                }
                count = args[i].parse().map_err(|_| format!("Invalid count: {}", args[i]))?;
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --seed".to_string());
                }
                seed = args[i].parse().map_err(|_| format!("Invalid seed: {}", args[i]))?;
            }
            "--root-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --root-size".to_string());
                }
                root_size = args[i].parse().map_err(|_| format!("Invalid root-size: {}", args[i]))?;
            }
            "--max-depth" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --max-depth".to_string());
                }
                max_depth = args[i].parse().map_err(|_| format!("Invalid max-depth: {}", args[i]))?;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            path => {
                if output_dir.is_some() {
                    return Err("Multiple output directories specified".to_string());
                }
                output_dir = Some(PathBuf::from(path));
            }
        }
        i += 1;
    }

    let output_dir = output_dir.ok_or("Missing output directory")?;

    Ok(Args {
        output_dir,
        style,
        count,
        seed,
        root_size,
        max_depth,
    })
}

fn generate_trees_for_style(
    library: &mut TreeLibrary,
    style: TreeStyle,
    count: u32,
    base_seed: u64,
    root_size: f32,
    max_depth: u8,
) -> Result<(), std::io::Error> {
    let style_name = match style {
        TreeStyle::Oak => "oak",
        TreeStyle::Willow => "willow",
        TreeStyle::Elm => "elm",
    };

    println!("Generating {} {} trees...", count, style_name);
    let start = Instant::now();

    for i in 0..count {
        let seed = base_seed.wrapping_add(i as u64).wrapping_mul(0x517cc1b727220a95);
        let mut generator = TreeGenerator::from_style(seed, style);
        let octree = generator.generate(root_size, max_depth);
        let tree_data = TreeData::from_octree(&octree, style, seed);

        library.add_tree_sync(&tree_data)?;

        // Progress indicator every 10 trees
        if (i + 1) % 10 == 0 || i + 1 == count {
            print!("\r  {}/{} trees generated", i + 1, count);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }

    let elapsed = start.elapsed();
    println!("\n  Completed in {:.2}s ({:.1} trees/sec)",
             elapsed.as_secs_f64(),
             count as f64 / elapsed.as_secs_f64());

    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            print_help();
            std::process::exit(1);
        }
    };

    println!("Tree Generation Utility");
    println!("=======================");
    println!("Output directory: {}", args.output_dir.display());
    println!("Trees per style: {}", args.count);
    println!("Base seed: {}", args.seed);
    println!("Root size: {}m", args.root_size);
    println!("Max depth: {}", args.max_depth);
    println!();

    // Create or open library
    let mut library = match TreeLibrary::open_sync(args.output_dir.clone()) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("Error opening library: {}", e);
            std::process::exit(1);
        }
    };

    let styles: Vec<TreeStyle> = match args.style {
        Some(style) => vec![style],
        None => vec![
            TreeStyle::Oak,
            TreeStyle::Willow,
            TreeStyle::Elm,
        ],
    };

    let total_start = Instant::now();

    for style in &styles {
        if let Err(e) = generate_trees_for_style(
            &mut library,
            *style,
            args.count,
            args.seed,
            args.root_size,
            args.max_depth,
        ) {
            eprintln!("Error generating trees: {}", e);
            std::process::exit(1);
        }
    }

    let total_elapsed = total_start.elapsed();
    let total_trees = styles.len() as u32 * args.count;

    println!();
    println!("Summary:");
    println!("  Total trees generated: {}", total_trees);
    println!("  Library now contains: {} trees", library.len());
    println!("  Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("  Output: {}", args.output_dir.display());
}
