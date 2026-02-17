//! Procedural tree generation using Space Colonization algorithm
//!
//! Generates realistic tree structures with:
//! - Space Colonization for natural branching patterns
//! - Pipe model for branch thickness (parent = sum of children cross-sections)
//! - Root flare at trunk base
//! - Dense foliage with color variation along outer branches

use std::collections::HashMap;

use glam::Vec3;

use crate::math::Aabb;
use crate::voxel::sdf::{sdf_capsule, sdf_sphere, encode_normal_rgb565 as encode_sdf_normal};
use crate::voxel::voxel::{Voxel, rgb565_to_rgb, rgb_to_565};
use crate::voxel::svo::Octree;
use crate::voxel::svo::node::OctreeNode;
use crate::voxel::brick::VoxelBrick;
use crate::voxel::brush::BrushSession;

/// Simple deterministic RNG using hash function
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    /// Advance state and return next u32
    fn next_u32(&mut self) -> u32 {
        // PCG-like state update
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Output function
        let mut h = (self.state >> 32) as u32;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        h
    }

    /// Generate f32 in range [0, 1)
    fn next_float(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    /// Generate f32 in range [min, max)
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_float() * (max - min)
    }
}

/// Tree visual style presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TreeStyle {
    #[default]
    Oak,
    Willow,
    Elm,
    /// Winter variant — bare branches with no leaves
    WinterOak,
    WinterWillow,
}

/// Crown volume shape for Space Colonization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrownShape {
    #[default]
    Sphere,
    Cone,
    Ellipsoid,
}

/// Parameters for tree generation (Space Colonization algorithm)
#[derive(Debug, Clone)]
pub struct TreeParams {
    /// Total tree height in world units
    pub height: f32,
    /// Trunk radius at base
    pub trunk_radius: f32,
    /// Trunk taper (ratio of top radius to base radius)
    pub trunk_taper: f32,
    /// Crown radius
    pub crown_radius: f32,
    /// Crown density (0.0 - 1.0), controls number of attraction points
    pub crown_density: f32,
    /// Bark voxel (color + material)
    pub bark_voxel: Voxel,
    /// Leaf voxel (color + material)
    pub leaf_voxel: Voxel,
    /// Octree level for trunk (coarse)
    pub trunk_level: u8,
    /// Octree level for branches
    pub branch_level: u8,
    /// Octree level for leaves (fine)
    pub leaf_level: u8,
    /// How far nodes can see attractors
    pub attraction_distance: f32,
    /// When attractors are removed (node reaches this close)
    pub kill_distance: f32,
    /// Growth step size per iteration
    pub segment_length: f32,
    /// Shape of the crown volume
    pub crown_shape: CrownShape,
    /// Direction bias (gravity, light, etc.)
    pub tropism: Vec3,
    /// Tropism strength (0.0-1.0)
    pub tropism_strength: f32,
    /// Maximum growth iterations
    pub max_iterations: u32,
    /// Minimum leaf cluster radius
    pub leaf_radius_min: f32,
    /// Maximum leaf cluster radius
    pub leaf_radius_max: f32,
    /// How many overlapping spheres per leaf cluster (1-4)
    pub leaf_cluster_count: u8,
    /// Fraction of outer branches that also get leaves (0.0-1.0)
    pub branch_leaf_density: f32,
    /// Color variation range for leaves (0-30)
    pub leaf_color_variation: u8,
    /// Root flare multiplier (1.0 = no flare, 2.0 = 2x wider at base)
    pub root_flare: f32,
    /// Foliage density (0.0-1.0) — fraction of leaf voxels that are solid.
    /// Lower values create more porous, natural-looking canopy with light gaps.
    pub foliage_density: f32,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self::oak()
    }
}

impl TreeParams {
    /// Create oak tree preset — wide spreading crown, dense rounded canopy
    pub fn oak() -> Self {
        Self {
            height: 6.0,
            trunk_radius: 0.20,
            trunk_taper: 0.4,
            crown_radius: 3.5,
            crown_density: 0.6,
            bark_voxel: Voxel::new(90, 60, 30, 2),   // Dark brown bark
            leaf_voxel: Voxel::new(40, 130, 30, 3),   // Deep forest green
            trunk_level: 8,
            branch_level: 9,
            leaf_level: 10,
            attraction_distance: 2.5,
            kill_distance: 0.4,
            segment_length: 0.2,
            crown_shape: CrownShape::Ellipsoid,
            tropism: Vec3::Y,
            tropism_strength: 0.08,
            max_iterations: 300,
            leaf_radius_min: 0.25,
            leaf_radius_max: 0.5,
            leaf_cluster_count: 2,
            branch_leaf_density: 0.25,
            leaf_color_variation: 20,
            root_flare: 1.4,
            foliage_density: 0.55,
        }
    }

    /// Create willow tree preset — drooping canopy, cascading branches
    pub fn willow() -> Self {
        Self {
            height: 5.5,
            trunk_radius: 0.18,
            trunk_taper: 0.45,
            crown_radius: 4.0,
            crown_density: 0.55,
            bark_voxel: Voxel::new(110, 85, 50, 2),   // Lighter brown bark
            leaf_voxel: Voxel::new(120, 200, 100, 3),  // Light yellow-green
            trunk_level: 8,
            branch_level: 9,
            leaf_level: 10,
            attraction_distance: 2.0,
            kill_distance: 0.35,
            segment_length: 0.25,
            crown_shape: CrownShape::Sphere,
            tropism: Vec3::NEG_Y,
            tropism_strength: 0.3,
            max_iterations: 300,
            leaf_radius_min: 0.18,
            leaf_radius_max: 0.35,
            leaf_cluster_count: 2,
            branch_leaf_density: 0.35,
            leaf_color_variation: 15,
            root_flare: 1.3,
            foliage_density: 0.50,
        }
    }

    /// Create elm tree preset — tall vase-shaped crown, upward reaching
    pub fn elm() -> Self {
        Self {
            height: 8.0,
            trunk_radius: 0.22,
            trunk_taper: 0.4,
            crown_radius: 3.0,
            crown_density: 0.55,
            bark_voxel: Voxel::new(75, 55, 35, 2),    // Gray-brown bark
            leaf_voxel: Voxel::new(50, 150, 40, 3),   // Medium green
            trunk_level: 8,
            branch_level: 9,
            leaf_level: 10,
            attraction_distance: 2.5,
            kill_distance: 0.45,
            segment_length: 0.22,
            crown_shape: CrownShape::Ellipsoid,
            tropism: Vec3::Y,
            tropism_strength: 0.15,
            max_iterations: 300,
            leaf_radius_min: 0.22,
            leaf_radius_max: 0.45,
            leaf_cluster_count: 2,
            branch_leaf_density: 0.25,
            leaf_color_variation: 18,
            root_flare: 1.3,
            foliage_density: 0.50,
        }
    }

    /// Create params from style preset
    pub fn from_style(style: TreeStyle) -> Self {
        match style {
            TreeStyle::Oak => Self::oak(),
            TreeStyle::Willow => Self::willow(),
            TreeStyle::Elm => Self::elm(),
            TreeStyle::WinterOak => Self::winter_oak(),
            TreeStyle::WinterWillow => Self::winter_willow(),
        }
    }

    /// Create winter oak — bare branches, no leaves
    pub fn winter_oak() -> Self {
        let mut params = Self::oak();
        params.leaf_voxel = Voxel::EMPTY; // No leaves
        params.branch_leaf_density = 0.0; // No leaves on branches
        params.foliage_density = 0.0; // No foliage
        params
    }

    /// Create winter willow — drooping bare branches, no leaves
    pub fn winter_willow() -> Self {
        let mut params = Self::willow();
        params.leaf_voxel = Voxel::EMPTY; // No leaves
        params.branch_leaf_density = 0.0; // No leaves on branches
        params.foliage_density = 0.0; // No foliage
        params
    }
}

/// A node in the growing tree skeleton (internal)
struct ColonizationNode {
    position: Vec3,
    parent: Option<usize>,
    /// Number of descendant leaf nodes (for pipe model)
    leaf_count: u32,
}

/// An attraction point in crown volume (internal)
struct Attractor {
    position: Vec3,
    active: bool,
}

/// A branch segment with tapered radius
#[derive(Clone)]
struct BranchSegment {
    start: Vec3,
    end: Vec3,
    start_radius: f32,
    end_radius: f32,
    voxel: Voxel,
}

impl BranchSegment {
    fn aabb(&self) -> Aabb {
        let max_radius = self.start_radius.max(self.end_radius);
        let min = self.start.min(self.end) - Vec3::splat(max_radius);
        let max = self.start.max(self.end) + Vec3::splat(max_radius);
        Aabb::new(min, max)
    }
}

/// A foliage cloud region
#[derive(Clone)]
struct FoliageCloud {
    center: Vec3,
    radius: f32,
    density: f32,
    seed: u32,
    voxel: Voxel,
}

/// Complete tree skeleton for direct voxelization
struct TreeSkeleton {
    branches: Vec<BranchSegment>,
    foliage: Vec<FoliageCloud>,
}

// --- Helper functions for direct octree construction ---

/// Deterministic 3D hash for organic noise
fn hash_3d(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= x as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h ^= y as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h ^= z as u32;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h
}

/// Get child octant offset (-1 or +1 for each axis)
fn child_offset(child_idx: u8) -> Vec3 {
    Vec3::new(
        if child_idx & 1 != 0 { 1.0 } else { -1.0 },
        if child_idx & 2 != 0 { 1.0 } else { -1.0 },
        if child_idx & 4 != 0 { 1.0 } else { -1.0 },
    )
}

/// Get brick voxel offset within octant (-0.5 or +0.5 for each axis)
fn brick_voxel_offset(voxel_idx: usize) -> Vec3 {
    Vec3::new(
        if voxel_idx & 1 != 0 { 0.5 } else { -0.5 },
        if voxel_idx & 2 != 0 { 0.5 } else { -0.5 },
        if voxel_idx & 4 != 0 { 0.5 } else { -0.5 },
    )
}

/// Average child colors for LOD propagation
fn average_child_colors(colors: &[(u16, u8); 8], valid_mask: u8) -> (u16, u8) {
    let mut r_sum: u32 = 0;
    let mut g_sum: u32 = 0;
    let mut b_sum: u32 = 0;
    let mut count: u32 = 0;
    let mut mat_counts = [0u8; 256];

    for i in 0..8 {
        if valid_mask & (1 << i) != 0 {
            let (color, mat) = colors[i];
            let (r, g, b) = rgb565_to_rgb(color);
            r_sum += r as u32;
            g_sum += g as u32;
            b_sum += b as u32;
            count += 1;
            mat_counts[mat as usize] = mat_counts[mat as usize].saturating_add(1);
        }
    }

    if count == 0 {
        return (0, 0);
    }

    let avg_r = (r_sum / count) as u8;
    let avg_g = (g_sum / count) as u8;
    let avg_b = (b_sum / count) as u8;
    let avg_color = rgb_to_565(avg_r, avg_g, avg_b);

    let avg_mat = mat_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map(|(i, _)| i as u8)
        .unwrap_or(0);

    (avg_color, avg_mat)
}

// --- TreeVoxelizer: builds octree directly from skeleton SDF ---

/// Builds an octree directly from a tree skeleton SDF
struct TreeVoxelizer {
    skeleton: TreeSkeleton,
    root_size: f32,
    max_depth: u8,
    /// Half-voxel padding added to branch radius during voxelization
    /// to ensure thin branches produce continuous voxel coverage
    voxel_padding: f32,
}

impl TreeVoxelizer {
    fn new(skeleton: TreeSkeleton, root_size: f32, max_depth: u8) -> Self {
        let voxel_size = root_size / (1u32 << max_depth) as f32;
        Self {
            skeleton,
            root_size,
            max_depth,
            // Full voxel padding ensures branches are at least 2 voxels wide,
            // eliminating gaps between segments and at branch joints
            voxel_padding: voxel_size * 1.0,
        }
    }

    fn build(&self) -> Octree {
        let mut octree = Octree::with_capacity(self.root_size, self.max_depth, 1024, 512);
        octree.set_dense_children(true);

        // Center octree so it spans y=[0, root_size] to fit upward-growing trees
        let center = Vec3::new(0.0, self.root_size / 2.0, 0.0);
        self.build_recursive(
            &mut octree,
            0,
            center,
            self.root_size,
            0,
            &self.skeleton.branches,
            &self.skeleton.foliage,
        );

        octree
    }

    fn build_recursive(
        &self,
        octree: &mut Octree,
        node_index: u32,
        center: Vec3,
        size: f32,
        depth: u8,
        branches: &[BranchSegment],
        clouds: &[FoliageCloud],
    ) -> (u16, u8, bool) {
        let half_size = size * 0.5;
        let node_aabb = Aabb::from_center_half_extent(center, Vec3::splat(half_size));

        // Filter to segments/clouds that intersect this node
        let relevant_branches: Vec<&BranchSegment> = branches
            .iter()
            .filter(|b| b.aabb().intersects(&node_aabb))
            .collect();
        let relevant_clouds: Vec<&FoliageCloud> = clouds
            .iter()
            .filter(|c| {
                let cloud_aabb =
                    Aabb::from_center_half_extent(c.center, Vec3::splat(c.radius));
                cloud_aabb.intersects(&node_aabb)
            })
            .collect();

        if relevant_branches.is_empty() && relevant_clouds.is_empty() {
            return (0, 0, true);
        }

        // At max depth, create leaf brick
        if depth >= self.max_depth {
            return self.create_leaf_brick(
                octree,
                node_index,
                center,
                size,
                &relevant_branches,
                &relevant_clouds,
            );
        }

        // Subdivide
        let child_size = size * 0.5;
        let quarter_size = size * 0.25;
        let mut child_valid_mask = 0u8;
        let first_child_index = octree.node_count() as u32;
        let mut child_colors: [(u16, u8); 8] = [(0, 0); 8];

        // Pre-allocate 8 child nodes
        for _ in 0..8 {
            octree.add_node(OctreeNode::empty());
        }

        for child_idx in 0..8u8 {
            let child_center = center + child_offset(child_idx) * quarter_size;
            let child_node_index = first_child_index + child_idx as u32;

            let (child_color, child_mat, child_empty) = self.build_recursive(
                octree,
                child_node_index,
                child_center,
                child_size,
                depth + 1,
                branches,
                clouds,
            );

            child_colors[child_idx as usize] = (child_color, child_mat);
            if !child_empty {
                child_valid_mask |= 1 << child_idx;
            }
        }

        let (lod_color, lod_material) = average_child_colors(&child_colors, child_valid_mask);

        let node = octree.node_mut(node_index);
        node.set_child_valid_mask(child_valid_mask);
        node.set_child_leaf_mask(0);
        node.child_offset = if child_valid_mask != 0 {
            first_child_index
        } else {
            0
        };
        node.lod_color = lod_color;
        node.lod_material = lod_material as u16;
        node.set_lod_level(depth);

        (lod_color, lod_material, child_valid_mask == 0)
    }

    fn create_leaf_brick(
        &self,
        octree: &mut Octree,
        node_index: u32,
        center: Vec3,
        size: f32,
        branches: &[&BranchSegment],
        clouds: &[&FoliageCloud],
    ) -> (u16, u8, bool) {
        let quarter_size = size * 0.25;
        let mut voxels = [Voxel::EMPTY; 8];

        for voxel_idx in 0..8 {
            let offset = brick_voxel_offset(voxel_idx);
            let voxel_center = center + offset * quarter_size;
            voxels[voxel_idx] = self.evaluate_at(voxel_center, branches, clouds);
        }

        let brick = VoxelBrick::new(voxels);
        if !brick.is_empty() {
            let brick_index = octree.add_brick(brick);
            let lod_color = brick.average_color();
            let lod_material = brick.average_material();

            let node = octree.node_mut(node_index);
            node.brick_offset = brick_index;
            node.set_child_valid_mask(0);
            node.set_child_leaf_mask(0);
            node.lod_color = lod_color;
            node.lod_material = lod_material as u16;

            (lod_color, lod_material, false)
        } else {
            (0, 0, true)
        }
    }

    /// Evaluate skeleton SDF at a point -- returns bark voxel, leaf voxel, or empty
    /// Uses proper capsule SDF to eliminate gaps at branch joints
    fn evaluate_at(
        &self,
        point: Vec3,
        branches: &[&BranchSegment],
        clouds: &[&FoliageCloud],
    ) -> Voxel {
        // Check branches first (bark takes priority)
        for seg in branches {
            // Use SDF capsule for proper interpolation between endpoints
            // This eliminates gaps at branch joints
            let dist = sdf_capsule(point, seg.start, seg.end, 0.0);

            if dist <= 0.0 {
                // Point is inside or on surface of the capsule
                // Compute tapered radius at closest point on line segment
                let dir = seg.end - seg.start;
                let len_sq = dir.length_squared();

                let (closest, t, dist_to_axis) = if len_sq < 0.0001 {
                    // Degenerate segment — treat as sphere
                    (seg.start, 0.0, (point - seg.start).length())
                } else {
                    let t = ((point - seg.start).dot(dir) / len_sq).clamp(0.0, 1.0);
                    let closest = seg.start + dir * t;
                    let dist_to_axis = (point - closest).length();
                    (closest, t, dist_to_axis)
                };

                // Tapered radius along the segment
                let radius_at_t = seg.start_radius + (seg.end_radius - seg.start_radius) * t;

                // Organic noise: slight radius perturbation
                let grid = 0.05;
                let gx = (point.x / grid).floor() as i32;
                let gy = (point.y / grid).floor() as i32;
                let gz = (point.z / grid).floor() as i32;
                let hash = hash_3d(gx, gy, gz, 0xDEADBEEF);
                let noise = ((hash & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.15 * radius_at_t;

                // Apply voxel padding to ensure continuous coverage
                let effective_radius = (radius_at_t + noise).max(self.voxel_padding);

                // Check if point is within the effective radius
                let dist_from_surface = effective_radius - dist_to_axis;
                if dist_from_surface >= -self.voxel_padding * 0.5 {
                    // Compute radial normal for shading
                    let radial = if dist_to_axis > 0.001 {
                        (point - closest).normalize()
                    } else {
                        // At axis center: use perpendicular to branch direction
                        let branch_dir = if len_sq < 0.0001 {
                            Vec3::Y
                        } else {
                            dir.normalize()
                        };
                        if branch_dir.y.abs() < 0.9 {
                            branch_dir.cross(Vec3::Y).normalize()
                        } else {
                            branch_dir.cross(Vec3::X).normalize()
                        }
                    };
                    let normal_color = encode_sdf_normal(radial);

                    // SDF flags based on distance from surface
                    let dist_frac = ((effective_radius - dist_to_axis) / effective_radius.max(0.001) * 254.0 + 1.0).clamp(1.0, 255.0) as u8;
                    return seg.voxel.with_color_and_flags(normal_color, dist_frac);
                }
            }
        }

        // Check foliage clouds
        for cloud in clouds {
            // Use SDF sphere for foliage
            let dist = sdf_sphere(point, cloud.center, cloud.radius);

            if dist <= 0.0 {
                let dist_from_center = (point - cloud.center).length();
                let normalized_dist = dist_from_center / cloud.radius;
                let falloff = 1.0 - normalized_dist * normalized_dist;
                let local_density = cloud.density * falloff;

                let grid_size = cloud.radius * 0.08;
                let gx = (point.x / grid_size).floor() as i32;
                let gy = (point.y / grid_size).floor() as i32;
                let gz = (point.z / grid_size).floor() as i32;
                let hash = hash_3d(gx, gy, gz, cloud.seed);
                let hash_float = (hash & 0xFFFF) as f32 / 65535.0;

                if hash_float < local_density {
                    let leaf_normal = if dist_from_center > 0.001 {
                        (point - cloud.center).normalize()
                    } else {
                        Vec3::Y
                    };
                    let normal_color = encode_sdf_normal(leaf_normal);
                    let dist_flag = ((1.0 - normalized_dist) * 255.0).clamp(0.0, 255.0) as u8;
                    return cloud.voxel.with_color_and_flags(normal_color, dist_flag);
                }
            }
        }

        Voxel::EMPTY
    }
}

/// Procedural tree generator using Space Colonization algorithm
pub struct TreeGenerator {
    rng: SimpleRng,
    params: TreeParams,
}

impl TreeGenerator {
    /// Create a new tree generator with seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            params: TreeParams::default(),
        }
    }

    /// Create generator with specific parameters
    pub fn with_params(seed: u64, params: TreeParams) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            params,
        }
    }

    /// Create generator from style preset
    pub fn from_style(seed: u64, style: TreeStyle) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            params: TreeParams::from_style(style),
        }
    }

    /// Generate a tree octree using direct skeleton voxelization
    pub fn generate(&mut self, root_size: f32, max_depth: u8) -> Octree {
        let skeleton = self.build_skeleton(Vec3::ZERO);
        let voxelizer = TreeVoxelizer::new(skeleton, root_size, max_depth);
        let dense_octree = voxelizer.build();
        dense_octree.compact_from_dense()
    }

    /// Generate tree strokes into an existing session (brush-based path, kept for compatibility)
    pub fn generate_into(&mut self, session: &mut BrushSession) {
        self.generate_at(session, Vec3::ZERO);
    }

    /// Generate tree strokes at a world position into an existing session (brush-based path)
    pub fn generate_at(&mut self, session: &mut BrushSession, world_pos: Vec3) {
        let trunk_height = self.params.height * 0.4;
        let trunk_top = world_pos + Vec3::Y * trunk_height;

        // --- Root flare ---
        if self.params.root_flare > 1.0 {
            let flare_radius = self.params.trunk_radius * self.params.root_flare;
            let flare_height = trunk_height * 0.15;
            let flare_top = world_pos + Vec3::Y * flare_height;
            session.capsule(
                world_pos,
                flare_top,
                flare_radius,
                self.params.bark_voxel,
                self.params.trunk_level,
            );
            session.capsule(
                flare_top,
                trunk_top,
                self.params.trunk_radius,
                self.params.bark_voxel,
                self.params.trunk_level,
            );
        } else {
            session.capsule(
                world_pos,
                trunk_top,
                self.params.trunk_radius,
                self.params.bark_voxel,
                self.params.trunk_level,
            );
        }

        let (nodes, children) = self.run_colonization(world_pos);
        let node_count = nodes.len();
        let trunk_top_radius = self.params.trunk_radius * self.params.trunk_taper;
        let max_leaf_count = nodes[0].leaf_count.max(1) as f32;
        let min_branch_radius = 0.02;

        // Emit branch capsules
        for i in 0..node_count {
            if let Some(parent_idx) = nodes[i].parent {
                let parent_thickness = trunk_top_radius
                    * (nodes[parent_idx].leaf_count as f32 / max_leaf_count).sqrt();
                let child_thickness = trunk_top_radius
                    * (nodes[i].leaf_count as f32 / max_leaf_count).sqrt();
                let radius = ((parent_thickness + child_thickness) * 0.5).max(min_branch_radius);
                let level = if radius > trunk_top_radius * 0.5 {
                    self.params.trunk_level
                } else {
                    self.params.branch_level
                };
                session.capsule(
                    nodes[parent_idx].position,
                    nodes[i].position,
                    radius,
                    self.params.bark_voxel,
                    level,
                );
            }
        }

        // Foliage
        let (depths, outer_threshold) = Self::compute_depths(&nodes);
        let leaf_radius_min = self.params.leaf_radius_min;
        let leaf_radius_max = self.params.leaf_radius_max;
        let branch_leaf_density = self.params.branch_leaf_density;
        let color_var = self.params.leaf_color_variation as i16;
        let foliage_density = self.params.foliage_density;

        for i in 0..node_count {
            if nodes[i].parent.is_none() {
                continue;
            }
            let is_terminal = children[i].is_empty();
            let is_outer = depths[i] >= outer_threshold;
            let should_leaf = is_terminal
                || (is_outer && self.rng.next_float() < branch_leaf_density);
            if !should_leaf {
                continue;
            }

            let radius = self.rng.range(leaf_radius_min, leaf_radius_max);
            let cloud_density = if is_terminal { foliage_density } else { foliage_density * 0.6 };
            let cloud_seed = self.rng.next_u32();
            let offset = Vec3::new(
                self.rng.range(-radius * 0.3, radius * 0.3),
                self.rng.range(-radius * 0.2, radius * 0.3),
                self.rng.range(-radius * 0.3, radius * 0.3),
            );
            let varied_voxel = self.vary_leaf_color(color_var);

            session.cloud(
                nodes[i].position + offset,
                radius,
                cloud_density,
                cloud_seed,
                varied_voxel,
                self.params.leaf_level,
            );
        }
    }

    /// Build a tree skeleton with direct voxelization data (no brush strokes)
    fn build_skeleton(&mut self, world_pos: Vec3) -> TreeSkeleton {
        let trunk_height = self.params.height * 0.4;
        let trunk_top = world_pos + Vec3::Y * trunk_height;
        let trunk_top_radius = self.params.trunk_radius * self.params.trunk_taper;
        let min_branch_radius = 0.02;

        let mut branches = Vec::new();
        let mut foliage = Vec::new();

        // --- Root flare + trunk as tapered segments ---
        if self.params.root_flare > 1.0 {
            let flare_radius = self.params.trunk_radius * self.params.root_flare;
            let flare_height = trunk_height * 0.15;
            let flare_top = world_pos + Vec3::Y * flare_height;
            // Root flare: wide base tapering to trunk radius
            branches.push(BranchSegment {
                start: world_pos,
                end: flare_top,
                start_radius: flare_radius,
                end_radius: self.params.trunk_radius,
                voxel: self.params.bark_voxel,
            });
            // Trunk above flare
            branches.push(BranchSegment {
                start: flare_top,
                end: trunk_top,
                start_radius: self.params.trunk_radius,
                end_radius: trunk_top_radius,
                voxel: self.params.bark_voxel,
            });
        } else {
            branches.push(BranchSegment {
                start: world_pos,
                end: trunk_top,
                start_radius: self.params.trunk_radius,
                end_radius: trunk_top_radius,
                voxel: self.params.bark_voxel,
            });
        }

        // --- Space Colonization + Pipe Model ---
        let (nodes, children) = self.run_colonization(world_pos);
        let node_count = nodes.len();
        let max_leaf_count = nodes[0].leaf_count.max(1) as f32;

        // Emit tapered branch segments
        for i in 0..node_count {
            if let Some(parent_idx) = nodes[i].parent {
                let parent_thickness = trunk_top_radius
                    * (nodes[parent_idx].leaf_count as f32 / max_leaf_count).sqrt();
                let child_thickness = trunk_top_radius
                    * (nodes[i].leaf_count as f32 / max_leaf_count).sqrt();

                branches.push(BranchSegment {
                    start: nodes[parent_idx].position,
                    end: nodes[i].position,
                    start_radius: parent_thickness.max(min_branch_radius),
                    end_radius: child_thickness.max(min_branch_radius),
                    voxel: self.params.bark_voxel,
                });
            }
        }

        // --- Foliage clouds ---
        let (depths, outer_threshold) = Self::compute_depths(&nodes);
        let leaf_radius_min = self.params.leaf_radius_min;
        let leaf_radius_max = self.params.leaf_radius_max;
        let branch_leaf_density = self.params.branch_leaf_density;
        let color_var = self.params.leaf_color_variation as i16;
        let foliage_density = self.params.foliage_density;

        for i in 0..node_count {
            if nodes[i].parent.is_none() {
                continue;
            }
            let is_terminal = children[i].is_empty();
            let is_outer = depths[i] >= outer_threshold;
            let should_leaf = is_terminal
                || (is_outer && self.rng.next_float() < branch_leaf_density);
            if !should_leaf {
                continue;
            }

            let radius = self.rng.range(leaf_radius_min, leaf_radius_max);
            let cloud_density = if is_terminal { foliage_density } else { foliage_density * 0.6 };
            let cloud_seed = self.rng.next_u32();
            let offset = Vec3::new(
                self.rng.range(-radius * 0.3, radius * 0.3),
                self.rng.range(-radius * 0.2, radius * 0.3),
                self.rng.range(-radius * 0.3, radius * 0.3),
            );
            let varied_voxel = self.vary_leaf_color(color_var);

            foliage.push(FoliageCloud {
                center: nodes[i].position + offset,
                radius,
                density: cloud_density,
                seed: cloud_seed,
                voxel: varied_voxel,
            });
        }

        TreeSkeleton { branches, foliage }
    }

    /// Run Space Colonization algorithm + pipe model, returning nodes and children map.
    /// Shared between build_skeleton() and generate_at().
    fn run_colonization(&mut self, world_pos: Vec3) -> (Vec<ColonizationNode>, Vec<Vec<usize>>) {
        let trunk_height = self.params.height * 0.4;
        let trunk_top = world_pos + Vec3::Y * trunk_height;

        let attractor_count = (self.params.crown_density * 1500.0) as usize;
        let crown_center = trunk_top + Vec3::Y * self.params.crown_radius * 0.6;

        let attraction_distance = self.params.attraction_distance;
        let kill_distance = self.params.kill_distance;
        let segment_length = self.params.segment_length;
        let tropism = self.params.tropism;
        let tropism_strength = self.params.tropism_strength;
        let max_iterations = self.params.max_iterations;

        let mut attractors = self.generate_attractors(attractor_count, crown_center);

        let mut nodes = vec![ColonizationNode {
            position: trunk_top,
            parent: None,
            leaf_count: 0,
        }];

        // Main growth loop
        for _ in 0..max_iterations {
            if !attractors.iter().any(|a| a.active) {
                break;
            }

            let mut node_influences: HashMap<usize, Vec<usize>> = HashMap::new();

            for (ai, attractor) in attractors.iter().enumerate() {
                if !attractor.active {
                    continue;
                }

                let mut nearest_node: Option<usize> = None;
                let mut nearest_dist = attraction_distance;

                for (ni, node) in nodes.iter().enumerate() {
                    let dist = (attractor.position - node.position).length();
                    if dist < nearest_dist {
                        nearest_dist = dist;
                        nearest_node = Some(ni);
                    }
                }

                if let Some(ni) = nearest_node {
                    node_influences.entry(ni).or_default().push(ai);
                }
            }

            let mut new_nodes = Vec::new();
            let mut sorted_node_indices: Vec<_> = node_influences.keys().copied().collect();
            sorted_node_indices.sort();
            for node_idx in sorted_node_indices {
                let attractor_indices = &node_influences[&node_idx];
                let node = &nodes[node_idx];

                let mut avg_dir = Vec3::ZERO;
                for &ai in attractor_indices {
                    avg_dir += (attractors[ai].position - node.position).normalize();
                }
                avg_dir = avg_dir.normalize();

                avg_dir = (avg_dir + tropism * tropism_strength).normalize();

                let new_pos = node.position + avg_dir * segment_length;
                new_nodes.push(ColonizationNode {
                    position: new_pos,
                    parent: Some(node_idx),
                    leaf_count: 0,
                });
            }

            if new_nodes.is_empty() {
                break;
            }

            for attractor in attractors.iter_mut() {
                if !attractor.active {
                    continue;
                }
                for node in nodes.iter().chain(new_nodes.iter()) {
                    if (attractor.position - node.position).length() < kill_distance {
                        attractor.active = false;
                        break;
                    }
                }
            }

            nodes.extend(new_nodes);
        }

        // --- Pipe model: compute thickness bottom-up ---
        let node_count = nodes.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        for (i, node) in nodes.iter().enumerate() {
            if let Some(parent_idx) = node.parent {
                children[parent_idx].push(i);
            }
        }

        for i in (0..node_count).rev() {
            if children[i].is_empty() {
                nodes[i].leaf_count = 1;
            } else {
                let sum: u32 = children[i].iter().map(|&c| nodes[c].leaf_count).sum();
                nodes[i].leaf_count = sum;
            }
        }

        (nodes, children)
    }

    /// Compute node depths and outer threshold for foliage placement
    fn compute_depths(nodes: &[ColonizationNode]) -> (Vec<u32>, u32) {
        let node_count = nodes.len();
        let mut depths = vec![0u32; node_count];
        for i in 0..node_count {
            if let Some(parent_idx) = nodes[i].parent {
                depths[i] = depths[parent_idx] + 1;
            }
        }
        let max_depth_from_root = *depths.iter().max().unwrap_or(&1);
        let outer_threshold = max_depth_from_root / 2;
        (depths, outer_threshold)
    }

    /// Create a color-varied leaf voxel
    fn vary_leaf_color(&mut self, variation: i16) -> Voxel {
        let base = self.params.leaf_voxel;
        let (br, bg, bb) = rgb565_to_rgb(base.color);

        let r = (br as i16 + self.rng.range(-variation as f32, variation as f32) as i16)
            .clamp(0, 255) as u8;
        let g = (bg as i16 + self.rng.range(-variation as f32, variation as f32) as i16)
            .clamp(0, 255) as u8;
        let b = (bb as i16 + self.rng.range(-variation as f32, variation as f32) as i16)
            .clamp(0, 255) as u8;

        Voxel::new(r, g, b, base.material_id)
    }

    /// Generate attractors in crown volume
    fn generate_attractors(&mut self, count: usize, center: Vec3) -> Vec<Attractor> {
        let crown_shape = self.params.crown_shape;
        let mut attractors = Vec::with_capacity(count);
        let radius = self.params.crown_radius;

        for _ in 0..count {
            // Rejection sampling
            loop {
                let x = self.rng.range(-radius, radius);
                let y = self.rng.range(-radius, radius);
                let z = self.rng.range(-radius, radius);
                let p = Vec3::new(x, y, z);

                let in_shape = match crown_shape {
                    CrownShape::Sphere => p.length() <= radius,
                    CrownShape::Cone => {
                        let height_ratio = (p.y + radius) / (2.0 * radius);
                        let max_r = radius * (1.0 - height_ratio * 0.8);
                        Vec3::new(p.x, 0.0, p.z).length() <= max_r
                            && p.y >= -radius * 0.2
                            && p.y <= radius
                    }
                    CrownShape::Ellipsoid => {
                        let ey = radius * 1.5;
                        (p.x / radius).powi(2) + (p.y / ey).powi(2) + (p.z / radius).powi(2) <= 1.0
                    }
                };

                if in_shape {
                    attractors.push(Attractor {
                        position: center + p,
                        active: true,
                    });
                    break;
                }
            }
        }

        attractors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_params_default() {
        let params = TreeParams::default();
        assert!(params.height > 0.0);
        assert!(params.trunk_radius > 0.0);
    }

    #[test]
    fn test_tree_params_presets() {
        let oak = TreeParams::oak();
        let elm = TreeParams::elm();
        let willow = TreeParams::willow();

        // Elm should be taller than oak
        assert!(elm.height > oak.height);
        // Willow should have downward tropism
        assert!(willow.tropism.y < 0.0);
        // Oak should have bigger leaf clusters
        assert!(oak.leaf_radius_max > willow.leaf_radius_max);
    }

    #[test]
    fn test_tree_generator_deterministic() {
        let mut generator1 = TreeGenerator::new(12345);
        let mut generator2 = TreeGenerator::new(12345);

        let tree1 = generator1.generate(16.0, 7);
        let tree2 = generator2.generate(16.0, 7);

        // Same seed should produce same tree
        assert_eq!(tree1.node_count(), tree2.node_count());
        assert_eq!(tree1.brick_count(), tree2.brick_count());
    }

    #[test]
    fn test_tree_generator_produces_octree() {
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let tree = generator.generate(8.0, 5);

        assert!(tree.node_count() > 1);
        assert!(tree.brick_count() > 0);
    }

    #[test]
    fn test_different_seeds_different_trees() {
        let mut generator1 = TreeGenerator::new(1);
        let mut generator2 = TreeGenerator::new(2);

        let tree1 = generator1.generate(16.0, 7);
        let tree2 = generator2.generate(16.0, 7);

        // Different seeds should produce different trees (very likely)
        assert!(tree1.node_count() != tree2.node_count() || tree1.brick_count() != tree2.brick_count());
    }

    #[test]
    fn test_colonization_deterministic() {
        let mut generator1 = TreeGenerator::from_style(42, TreeStyle::Oak);
        let mut generator2 = TreeGenerator::from_style(42, TreeStyle::Oak);

        let tree1 = generator1.generate(16.0, 7);
        let tree2 = generator2.generate(16.0, 7);

        assert_eq!(tree1.node_count(), tree2.node_count());
        assert_eq!(tree1.brick_count(), tree2.brick_count());
    }

    #[test]
    fn test_colonization_produces_branches() {
        let mut generator = TreeGenerator::from_style(123, TreeStyle::Oak);
        let tree = generator.generate(8.0, 5);

        assert!(tree.node_count() > 10, "Expected branches in colonization tree");
        assert!(tree.brick_count() > 0, "Expected bricks in colonization tree");
    }

    #[test]
    fn test_all_styles() {
        for style in [TreeStyle::Oak, TreeStyle::Willow, TreeStyle::Elm] {
            let mut generator = TreeGenerator::from_style(42, style);
            let tree = generator.generate(8.0, 5);
            assert!(tree.brick_count() > 0, "Style {:?} produced empty tree", style);
        }
    }

    #[test]
    fn test_different_crown_shapes() {
        for shape in [CrownShape::Sphere, CrownShape::Cone, CrownShape::Ellipsoid] {
            let mut params = TreeParams::oak();
            params.crown_shape = shape;
            let mut generator = TreeGenerator::with_params(42, params);
            let tree = generator.generate(8.0, 5);
            assert!(tree.brick_count() > 0, "Shape {:?} produced empty tree", shape);
        }
    }

    #[test]
    fn test_pipe_model_makes_larger_trees() {
        // The improved tree with pipe model + dense foliage should be bigger
        // than the bare minimum
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let tree = generator.generate(8.0, 6);
        assert!(tree.brick_count() > 10, "Oak tree should have substantial brick count, got {}", tree.brick_count());
    }
}
