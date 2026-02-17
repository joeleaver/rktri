//! Procedural rock generation using SDF-based shape modeling.
//!
//! Rocks are generated using SDF primitives with smooth unions, domain warping,
//! ridge noise, and multi-scale detail layers for organic, natural-looking shapes.
//! Uses AdaptiveOctreeBuilder with direct voxel evaluation.

use glam::Vec3;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::voxel::sdf::{encode_normal_rgb565, smin};
use crate::voxel::svo::{AdaptiveOctreeBuilder, Octree};
use crate::voxel::voxel::Voxel;

/// Material ID for rock voxels
pub const MATERIAL_ROCK: u8 = 20;

/// Rock generation parameters
#[derive(Clone, Debug)]
pub struct RockParams {
    /// Height in meters (0.15 - 3.0)
    pub height: f32,
    /// Width/depth ratio (0.5 - 1.5)
    pub aspect_ratio: f32,
    /// Flatness factor (0.3 - 1.0, lower = flatter)
    pub flatness: f32,
    /// Bumpiness of surface (0.0 - 1.0)
    pub bumpiness: f32,
    /// Sharpness of features (0.0 - 1.0)
    pub sharpness: f32,
    /// Color variation (0.0 - 1.0)
    pub color_variation: f32,
    /// Base color (RGB)
    pub base_color: [u8; 3],
    /// Rock voxel with PBR material
    pub rock_voxel: Voxel,
}

impl Default for RockParams {
    fn default() -> Self {
        Self {
            height: 0.5,
            aspect_ratio: 0.8,
            flatness: 0.6,
            bumpiness: 0.4,
            sharpness: 0.3,
            color_variation: 0.3,
            base_color: [120, 120, 120],
            rock_voxel: Voxel::new(120, 120, 120, MATERIAL_ROCK),
        }
    }
}

impl RockParams {
    /// Create a random rock preset
    pub fn random(seed: u64) -> Self {
        let mut rng = SimpleHash::new(seed);
        
        // Determine size category
        let size_category = rng.float();
        
        let height = if size_category < 0.3 {
            // Small: 0.15 - 0.5m
            0.15 + rng.float() * 0.35
        } else if size_category < 0.7 {
            // Medium: 0.5 - 1.5m
            0.5 + rng.float() * 1.0
        } else {
            // Large: 1.5 - 3.0m
            1.5 + rng.float() * 1.5
        };

        Self {
            height,
            aspect_ratio: 0.6 + rng.float() * 0.8,
            flatness: 0.4 + rng.float() * 0.6,
            bumpiness: rng.float() * 0.8,
            sharpness: rng.float() * 0.7,
            color_variation: rng.float() * 0.5,
            base_color: [120, 120, 120],
            rock_voxel: Voxel::new(120, 120, 120, MATERIAL_ROCK),
        }
    }

    /// Small boulder (6-18 inches)
    pub fn small_boulder() -> Self {
        Self {
            height: 0.25,
            aspect_ratio: 0.9,
            flatness: 0.7,
            bumpiness: 0.5,
            sharpness: 0.4,
            color_variation: 0.3,
            base_color: [140, 140, 140],
            rock_voxel: Voxel::new(140, 140, 140, MATERIAL_ROCK),
        }
    }

    /// Medium rock (1-2 feet)
    pub fn medium_rock() -> Self {
        Self {
            height: 0.5,
            aspect_ratio: 0.8,
            flatness: 0.6,
            bumpiness: 0.4,
            sharpness: 0.3,
            color_variation: 0.3,
            base_color: [130, 130, 130],
            rock_voxel: Voxel::new(130, 130, 130, MATERIAL_ROCK),
        }
    }

    /// Large boulder (2-3 feet)
    pub fn large_boulder() -> Self {
        Self {
            height: 0.8,
            aspect_ratio: 0.75,
            flatness: 0.55,
            bumpiness: 0.35,
            sharpness: 0.25,
            color_variation: 0.25,
            base_color: [120, 120, 120],
            rock_voxel: Voxel::new(120, 120, 120, MATERIAL_ROCK),
        }
    }

    /// Huge rock (3-10 feet)
    pub fn huge_boulder() -> Self {
        Self {
            height: 2.0,
            aspect_ratio: 0.7,
            flatness: 0.5,
            bumpiness: 0.3,
            sharpness: 0.2,
            color_variation: 0.2,
            base_color: [110, 110, 110],
            rock_voxel: Voxel::new(110, 110, 110, MATERIAL_ROCK),
        }
    }

    /// Flat rock (pancake shape)
    pub fn flat_rock() -> Self {
        Self {
            height: 0.2,
            aspect_ratio: 1.2,
            flatness: 0.3,
            bumpiness: 0.2,
            sharpness: 0.1,
            color_variation: 0.2,
            base_color: [150, 145, 140],
            rock_voxel: Voxel::new(150, 145, 140, MATERIAL_ROCK),
        }
    }

    /// Mossy rock (forest rocks with green tint)
    pub fn mossy_rock() -> Self {
        Self {
            height: 0.4,
            aspect_ratio: 0.85,
            flatness: 0.65,
            bumpiness: 0.45,
            sharpness: 0.35,
            color_variation: 0.4,
            base_color: [100, 110, 95],
            rock_voxel: Voxel::new(100, 110, 95, MATERIAL_ROCK),
        }
    }

    /// Snowy rock (mountain/snow biome)
    pub fn snowy_rock() -> Self {
        Self {
            height: 0.6,
            aspect_ratio: 0.8,
            flatness: 0.6,
            bumpiness: 0.3,
            sharpness: 0.2,
            color_variation: 0.15,
            base_color: [220, 225, 230],
            rock_voxel: Voxel::new(220, 225, 230, MATERIAL_ROCK),
        }
    }

    /// Set base color
    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.base_color = [r, g, b];
        self.rock_voxel = Voxel::new(r, g, b, MATERIAL_ROCK);
        self
    }
}

/// Simple hash-based RNG for deterministic random values
#[derive(Clone, Debug)]
struct SimpleHash(u64);

impl SimpleHash {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn float(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) as f32 / 65536.0
    }
}

/// Evaluate the rock SDF at a given position (for gradient computation)
/// This is a helper that replicates the SDF logic from the evaluator closure
fn eval_sdf_at(
    pos: Vec3,
    params: &RockParams,
    warp_noise: &Fbm<Perlin>,
    shape_noise: &Fbm<Perlin>,
    ridge_noise: &Fbm<Perlin>,
    half_height: f32,
    rx: f32,
    ry: f32,
    rz: f32,
) -> f32 {
    let height = params.height;

    // Offset to center the rock at origin
    let p = pos - Vec3::new(0.0, half_height, 0.0);

    // Helper to clamp noise coordinates
    let clamp_coord = |c: f64| c.clamp(-1e6, 1e6);

    // Noise scales
    let safe_height = height.max(0.1);
    let warp_scale = (3.0 / safe_height as f64).min(20.0);
    let shape_scale = (5.0 / safe_height as f64).min(25.0);
    let ridge_scale = (7.0 / safe_height as f64).min(30.0);

    // Domain warping
    let wx = clamp_coord(p.x as f64 * warp_scale);
    let wy = clamp_coord(p.y as f64 * warp_scale);
    let wz = clamp_coord(p.z as f64 * warp_scale);
    let warp_x = warp_noise.get([wx, wy, wz]) as f32;
    let warp_y = warp_noise.get([wx + 100.0, wy + 100.0, wz + 100.0]) as f32;
    let warp_z = warp_noise.get([wx + 200.0, wy + 200.0, wz + 200.0]) as f32;

    let warped_p = p + Vec3::new(warp_x, warp_y, warp_z) * params.bumpiness * 0.1;

    // Main ellipsoid
    let unit_p = Vec3::new(
        warped_p.x / rx,
        warped_p.y / ry,
        warped_p.z / rz
    );
    let main_ellipsoid = (unit_p.length() - 1.0) * ry.min(rx).min(rz);

    // Secondary ellipsoids
    let offset1 = Vec3::new(rx * 0.3, -ry * 0.2, rz * 0.2);
    let unit_p1 = Vec3::new(
        (warped_p.x - offset1.x) / (rx * 0.5),
        (warped_p.y - offset1.y) / (ry * 0.6),
        (warped_p.z - offset1.z) / (rz * 0.5)
    );
    let ellipsoid1 = (unit_p1.length() - 1.0) * ry.min(rx).min(rz) * 0.6;

    let offset2 = Vec3::new(-rx * 0.25, -ry * 0.3, -rz * 0.15);
    let unit_p2 = Vec3::new(
        (warped_p.x - offset2.x) / (rx * 0.4),
        (warped_p.y - offset2.y) / (ry * 0.5),
        (warped_p.z - offset2.z) / (rz * 0.4)
    );
    let ellipsoid2 = (unit_p2.length() - 1.0) * ry.min(rx).min(rz) * 0.5;

    // Smooth union
    let smooth_k = ry.min(rx).min(rz) * 0.3;
    let mut sdf = smin(main_ellipsoid, ellipsoid1, smooth_k);
    sdf = smin(sdf, ellipsoid2, smooth_k * 0.8);

    // Shape displacement
    let shape_px = clamp_coord(warped_p.x as f64 * shape_scale);
    let shape_py = clamp_coord(warped_p.y as f64 * shape_scale);
    let shape_pz = clamp_coord(warped_p.z as f64 * shape_scale);
    let shape_disp = shape_noise.get([shape_px, shape_py, shape_pz]) as f32;

    // Ridge noise
    let ridge_px = clamp_coord(warped_p.x as f64 * ridge_scale);
    let ridge_py = clamp_coord(warped_p.y as f64 * ridge_scale);
    let ridge_pz = clamp_coord(warped_p.z as f64 * ridge_scale);
    let ridge_val = ridge_noise.get([ridge_px, ridge_py, ridge_pz]) as f32;
    let ridge_disp = ridge_val.abs() * params.sharpness;

    // Total displacement
    let total_displacement =
        shape_disp * 0.5 * params.bumpiness +
        ridge_disp * 0.4 * params.bumpiness;

    sdf + total_displacement * (height * 0.15)
}

/// Procedural rock generator
pub struct RockGenerator {
    params: RockParams,
    seed: u64,
}

impl RockGenerator {
    /// Create a new rock generator
    pub fn new(seed: u64) -> Self {
        Self {
            params: RockParams::default(),
            seed,
        }
    }

    /// Create generator with specific parameters
    pub fn with_params(seed: u64, params: RockParams) -> Self {
        Self {
            params,
            seed,
        }
    }

    /// Generate a rock octree
    pub fn generate(&mut self, height: f32) -> Octree {
        self.params.height = height;
        self.generate_octree()
    }

    /// Get the rock voxel with PBR material
    pub fn rock_voxel(&self) -> Voxel {
        self.params.rock_voxel
    }

    /// Get the parameters
    pub fn params(&self) -> &RockParams {
        &self.params
    }

    /// Generate rock octree using SDF-based noise evaluation
    /// This produces smooth, organic rock shapes using direct voxel evaluation
    /// with domain warping, ridge noise, and multi-scale detail layers
    fn generate_octree(&mut self) -> Octree {
        let height = self.params.height;
        let half_height = height / 2.0;
        let width = height * self.params.aspect_ratio * self.params.flatness;
        let depth = width;

        // Root size should exactly fit the rock (origin at bottom-center)
        let root_size = height.max(width).max(depth);

        // Use high resolution for smooth rocks
        let grid_size = if height < 0.3 {
            128 // Small rocks
        } else if height < 1.0 {
            256 // Medium
        } else {
            256 // Large (capped for performance)
        };

        // Seed offset for different noise types
        let seed = self.seed as u32;

        // Domain warping - distorts the space before evaluating SDF
        let warp_noise = Fbm::<Perlin>::new(seed)
            .set_octaves(3)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        // Shape noise - large-scale deformations
        let shape_noise = Fbm::<Perlin>::new(seed.wrapping_add(1000))
            .set_octaves(4)
            .set_persistence(0.6)
            .set_lacunarity(2.0);

        // Ridge noise - creates sharp ridges and crags (abs of noise)
        let ridge_noise = Fbm::<Perlin>::new(seed.wrapping_add(2000))
            .set_octaves(4)
            .set_persistence(0.5)
            .set_lacunarity(2.2);

        // Detail noise - fine surface detail
        let detail_noise = Fbm::<Perlin>::new(seed.wrapping_add(3000))
            .set_octaves(3)
            .set_persistence(0.4)
            .set_lacunarity(2.5);

        // Micro detail - very fine bumps
        let micro_noise = Fbm::<Perlin>::new(seed.wrapping_add(4000))
            .set_octaves(2)
            .set_persistence(0.3)
            .set_lacunarity(3.0);

        // Color variation noise
        let color_noise = Fbm::<Perlin>::new(seed.wrapping_add(5000))
            .set_octaves(4)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        // Store params for the closure
        let params = self.params.clone();

        // Noise scales - clamp to prevent extreme values
        let safe_height = height.max(0.1);
        let warp_scale = (3.0 / safe_height as f64).min(20.0);
        let shape_scale = (5.0 / safe_height as f64).min(25.0);
        let ridge_scale = (7.0 / safe_height as f64).min(30.0);
        let detail_scale = (15.0 / safe_height as f64).min(40.0);
        let micro_scale = (30.0 / safe_height as f64).min(60.0);

        // Helper to clamp noise coordinates to safe range
        let clamp_coord = |c: f64| c.clamp(-1e6, 1e6);

        // Create evaluator closure using SDF primitives
        let evaluator = move |pos: Vec3| -> Voxel {
            // Offset to center the rock at origin
            let p = pos - Vec3::new(0.0, half_height, 0.0);

            // === Domain Warping ===
            // Warp the domain using low-frequency noise for interesting base shapes
            let wx = clamp_coord(p.x as f64 * warp_scale);
            let wy = clamp_coord(p.y as f64 * warp_scale);
            let wz = clamp_coord(p.z as f64 * warp_scale);
            let warp_x = warp_noise.get([wx, wy, wz]) as f32;
            let warp_y = warp_noise.get([wx + 100.0, wy + 100.0, wz + 100.0]) as f32;
            let warp_z = warp_noise.get([wx + 200.0, wy + 200.0, wz + 200.0]) as f32;

            let warped_p = p + Vec3::new(warp_x, warp_y, warp_z) * params.bumpiness * 0.1;

            // === SDF Composition using multiple ellipsoids with smooth union ===
            // Main ellipsoid (primary rock body)
            let rx = width / 2.0;
            let ry = height / 2.0;
            let rz = depth / 2.0;

            // Scale to unit sphere space
            let unit_p = Vec3::new(
                warped_p.x / rx,
                warped_p.y / ry,
                warped_p.z / rz
            );
            let main_ellipsoid = (unit_p.length() - 1.0) * ry.min(rx).min(rz);

            // Secondary ellipsoids for more interesting shapes
            // These get blended with smooth union
            let offset1 = Vec3::new(rx * 0.3, -ry * 0.2, rz * 0.2);
            let unit_p1 = Vec3::new(
                (warped_p.x - offset1.x) / (rx * 0.5),
                (warped_p.y - offset1.y) / (ry * 0.6),
                (warped_p.z - offset1.z) / (rz * 0.5)
            );
            let ellipsoid1 = (unit_p1.length() - 1.0) * ry.min(rx).min(rz) * 0.6;

            let offset2 = Vec3::new(-rx * 0.25, -ry * 0.3, -rz * 0.15);
            let unit_p2 = Vec3::new(
                (warped_p.x - offset2.x) / (rx * 0.4),
                (warped_p.y - offset2.y) / (ry * 0.5),
                (warped_p.z - offset2.z) / (rz * 0.4)
            );
            let ellipsoid2 = (unit_p2.length() - 1.0) * ry.min(rx).min(rz) * 0.5;

            // Blend ellipsoids with smooth union for organic shapes
            let smooth_k = ry.min(rx).min(rz) * 0.3; // Blend radius
            let mut sdf = smin(main_ellipsoid, ellipsoid1, smooth_k);
            sdf = smin(sdf, ellipsoid2, smooth_k * 0.8);

            // === Noise Displacement Layers ===

            // Shape displacement - large scale bulges
            let shape_px = clamp_coord(warped_p.x as f64 * shape_scale);
            let shape_py = clamp_coord(warped_p.y as f64 * shape_scale);
            let shape_pz = clamp_coord(warped_p.z as f64 * shape_scale);
            let shape_disp = shape_noise.get([shape_px, shape_py, shape_pz]) as f32;

            // Ridge noise - creates sharp, craggy features
            let ridge_px = clamp_coord(warped_p.x as f64 * ridge_scale);
            let ridge_py = clamp_coord(warped_p.y as f64 * ridge_scale);
            let ridge_pz = clamp_coord(warped_p.z as f64 * ridge_scale);
            let ridge_val = ridge_noise.get([ridge_px, ridge_py, ridge_pz]) as f32;
            let ridge_disp = ridge_val.abs() * params.sharpness;

            // Detail noise - medium surface detail
            let detail_px = clamp_coord(warped_p.x as f64 * detail_scale);
            let detail_py = clamp_coord(warped_p.y as f64 * detail_scale);
            let detail_pz = clamp_coord(warped_p.z as f64 * detail_scale);
            let detail_disp = detail_noise.get([detail_px, detail_py, detail_pz]) as f32;

            // Micro detail - fine bumps
            let micro_px = clamp_coord(warped_p.x as f64 * micro_scale);
            let micro_py = clamp_coord(warped_p.y as f64 * micro_scale);
            let micro_pz = clamp_coord(warped_p.z as f64 * micro_scale);
            let micro_disp = micro_noise.get([micro_px, micro_py, micro_pz]) as f32;

            // Combine all displacement layers
            // Weight: shape > ridge > detail > micro
            let total_displacement =
                shape_disp * 0.5 * params.bumpiness +
                ridge_disp * 0.4 * params.bumpiness +
                detail_disp * 0.2 * params.bumpiness +
                micro_disp * 0.1 * params.bumpiness;

            let sdf_final = sdf + total_displacement * (height * 0.15);

            // If SDF < 0, we're inside the rock
            if sdf_final < 0.0 {
                // Calculate color variation based on position
                let color_px = clamp_coord(p.x as f64 * 3.0);
                let color_py = clamp_coord(p.y as f64 * 3.0);
                let color_pz = clamp_coord(p.z as f64 * 3.0);
                let color_val = color_noise.get([color_px, color_py, color_pz]) as f32;
                let color_factor = ((color_val + 1.0) / 2.0 * params.color_variation * 40.0) as i16;

                let base_r = params.base_color[0] as i16;
                let base_g = params.base_color[1] as i16;
                let base_b = params.base_color[2] as i16;

                // Add slight darkening on ridges for visual interest
                let ridge_darken = (ridge_disp * 10.0) as i16;

                let r = (base_r + color_factor - ridge_darken - params.color_variation as i16 * 20).clamp(0, 255) as u8;
                let g = (base_g + color_factor - ridge_darken - params.color_variation as i16 * 20).clamp(0, 255) as u8;
                let b = (base_b + color_factor - ridge_darken - params.color_variation as i16 * 20).clamp(0, 255) as u8;

                // Compute gradient normal for smooth shading using central differences
                let eps = 0.01;
                let sdf_xp = eval_sdf_at(pos + Vec3::new(eps, 0.0, 0.0), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);
                let sdf_xn = eval_sdf_at(pos - Vec3::new(eps, 0.0, 0.0), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);
                let sdf_yp = eval_sdf_at(pos + Vec3::new(0.0, eps, 0.0), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);
                let sdf_yn = eval_sdf_at(pos - Vec3::new(0.0, eps, 0.0), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);
                let sdf_zp = eval_sdf_at(pos + Vec3::new(0.0, 0.0, eps), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);
                let sdf_zn = eval_sdf_at(pos - Vec3::new(0.0, 0.0, eps), &params, &warp_noise, &shape_noise, &ridge_noise, half_height, rx, ry, rz);

                // Gradient is the direction of steepest ascent; surface normal is opposite
                let grad = Vec3::new(sdf_xp - sdf_xn, sdf_yp - sdf_yn, sdf_zp - sdf_zn) / (2.0 * eps);
                let normal = if grad.length_squared() > 0.001 {
                    -grad.normalize()
                } else {
                    Vec3::Y // Default up normal for flat areas
                };

                // Encode normal into color field using RGB565
                let normal_color = encode_normal_rgb565(normal);

                // Use distance fraction for flags (how close to surface)
                let dist_frac = ((-sdf_final / (height * 0.15) * 254.0).clamp(1.0, 255.0)) as u8;

                Voxel::new(r, g, b, MATERIAL_ROCK).with_color_and_flags(normal_color, dist_frac)
            } else {
                Voxel::EMPTY
            }
        };

        // Build octree using adaptive builder with direct evaluation
        let builder = AdaptiveOctreeBuilder::new(grid_size);
        builder.build_simple(&evaluator, Vec3::ZERO, root_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rock_generator_create() {
        let mut generator = RockGenerator::new(12345);
        let _octree = generator.generate(0.5);
        // Just verify it runs without panicking
    }

    #[test]
    fn test_rock_params_presets() {
        let small = RockParams::small_boulder();
        assert!(small.height < 0.5);
        
        let large = RockParams::large_boulder();
        assert!(large.height > 0.5);
        
        let flat = RockParams::flat_rock();
        assert!(flat.flatness < 0.5);
    }

    #[test]
    fn test_random_rock() {
        // Test with a few different seeds to verify randomness works
        let mut any_positive = false;
        for seed in 0..10 {
            let params = RockParams::random(seed);
            if params.height > 0.0 {
                any_positive = true;
            }
        }
        assert!(any_positive, "At least one random rock should have positive height");
    }

    #[test]
    fn test_rock_voxel_material() {
        let generator = RockGenerator::new(42);
        let voxel = generator.rock_voxel();
        assert_eq!(voxel.material_id, MATERIAL_ROCK);
    }
}
