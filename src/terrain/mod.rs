//! Procedural terrain generation

pub mod generator;
pub use generator::{TerrainGenerator, TerrainParams};

pub mod biome;
pub use biome::{Biome, BiomeMap};
