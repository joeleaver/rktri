//! Camera trajectory prediction for prefetching chunks.

use std::collections::VecDeque;
use glam::Vec3;

use crate::math::Aabb;
use crate::voxel::chunk::ChunkCoord;
use crate::voxel::hierarchy::{world_to_chunk, CHUNK_SIZE_METERS};

/// Sample of camera position and time.
#[derive(Clone, Copy, Debug)]
struct PositionSample {
    position: Vec3,
    time: f32,
}

/// Configuration for frustum-based prefetching.
#[derive(Clone, Copy, Debug)]
pub struct FrustumPrefetchConfig {
    /// How much to expand the frustum for prefetching (1.0 = no expansion, 2.0 = double)
    pub expansion_factor: f32,
    /// Maximum depth to prefetch along frustum direction (meters)
    pub max_depth: f32,
    /// Field of view in radians
    pub fov: f32,
}

impl Default for FrustumPrefetchConfig {
    fn default() -> Self {
        Self {
            expansion_factor: 1.5,
            max_depth: 64.0,
            fov: std::f32::consts::FRAC_PI_3, // 60 degrees
        }
    }
}

/// Configuration for edit region prefetching.
#[derive(Clone, Copy, Debug)]
pub struct EditRegionPrefetchConfig {
    /// How many chunks to prefetch around edit regions
    pub margin_chunks: i32,
    /// Maximum number of edit regions to consider
    pub max_regions: usize,
}

impl Default for EditRegionPrefetchConfig {
    fn default() -> Self {
        Self {
            margin_chunks: 2,
            max_regions: 16,
        }
    }
}

/// Strategy weights for combining different prefetch methods.
#[derive(Clone, Copy, Debug)]
pub struct PrefetchStrategy {
    /// Weight for temporal (velocity-based) prediction
    pub temporal_weight: f32,
    /// Weight for frustum-based prediction
    pub frustum_weight: f32,
    /// Weight for edit region prediction
    pub edit_region_weight: f32,
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        Self {
            temporal_weight: 1.0,
            frustum_weight: 1.5,
            edit_region_weight: 2.0, // Highest priority
        }
    }
}

/// Predicts future camera positions for prefetching.
pub struct PrefetchPredictor {
    /// Camera position history
    history: VecDeque<PositionSample>,
    /// Maximum history size
    max_history: usize,
    /// Estimated velocity
    velocity: Vec3,
    /// Lookahead time for prefetch (seconds)
    lookahead_time: f32,
    /// Current time
    current_time: f32,
    /// Frustum prefetch configuration
    frustum_config: FrustumPrefetchConfig,
    /// Edit region prefetch configuration
    edit_region_config: EditRegionPrefetchConfig,
    /// Strategy weights
    strategy: PrefetchStrategy,
}

impl PrefetchPredictor {
    /// Create a new predictor.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(60),
            max_history: 60, // ~1 second at 60fps
            velocity: Vec3::ZERO,
            lookahead_time: 0.5, // 500ms lookahead
            current_time: 0.0,
            frustum_config: FrustumPrefetchConfig::default(),
            edit_region_config: EditRegionPrefetchConfig::default(),
            strategy: PrefetchStrategy::default(),
        }
    }

    /// Set lookahead time.
    pub fn with_lookahead(mut self, seconds: f32) -> Self {
        self.lookahead_time = seconds;
        self
    }

    /// Set frustum prefetch configuration.
    pub fn with_frustum_config(mut self, config: FrustumPrefetchConfig) -> Self {
        self.frustum_config = config;
        self
    }

    /// Set edit region prefetch configuration.
    pub fn with_edit_region_config(mut self, config: EditRegionPrefetchConfig) -> Self {
        self.edit_region_config = config;
        self
    }

    /// Set prefetch strategy weights.
    pub fn with_strategy(mut self, strategy: PrefetchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Update with new camera position.
    pub fn update(&mut self, position: Vec3, delta_time: f32) {
        self.current_time += delta_time;

        // Add new sample
        self.history.push_back(PositionSample {
            position,
            time: self.current_time,
        });

        // Remove old samples
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }

        // Estimate velocity from recent samples
        self.velocity = self.estimate_velocity();
    }

    /// Estimate velocity from position history.
    fn estimate_velocity(&self) -> Vec3 {
        if self.history.len() < 2 {
            return Vec3::ZERO;
        }

        // Use weighted average of recent velocities
        let mut total_velocity = Vec3::ZERO;
        let mut total_weight = 0.0;

        let samples: Vec<_> = self.history.iter().collect();
        for i in 1..samples.len() {
            let prev = samples[i - 1];
            let curr = samples[i];
            let dt = curr.time - prev.time;

            if dt > 0.0001 {
                let v = (curr.position - prev.position) / dt;
                // Weight recent samples more heavily
                let weight = (i as f32) / (samples.len() as f32);
                total_velocity += v * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            total_velocity / total_weight
        } else {
            Vec3::ZERO
        }
    }

    /// Get estimated velocity.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Get estimated speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// Predict position at future time.
    pub fn predict_position(&self, seconds_ahead: f32) -> Vec3 {
        if let Some(latest) = self.history.back() {
            latest.position + self.velocity * seconds_ahead
        } else {
            Vec3::ZERO
        }
    }

    /// Get current position.
    pub fn current_position(&self) -> Vec3 {
        self.history.back().map(|s| s.position).unwrap_or(Vec3::ZERO)
    }

    /// Predict chunks needed based on frustum expansion.
    ///
    /// # Arguments
    /// * `forward` - Camera forward direction (normalized)
    /// * `loaded` - Function to check if a chunk is already loaded
    pub fn predict_frustum_chunks(&self, forward: Vec3, loaded: &impl Fn(ChunkCoord) -> bool) -> Vec<(ChunkCoord, f32)> {
        let mut needed = Vec::new();
        let current_pos = self.current_position();
        let forward = forward.normalize();

        let config = &self.frustum_config;
        let half_fov = config.fov * 0.5 * config.expansion_factor;
        let tan_half_fov = half_fov.tan();

        // Sample along frustum depth
        let steps = 8;
        for i in 0..steps {
            let depth = (i as f32 / steps as f32) * config.max_depth;
            let center = current_pos + forward * depth;

            // Frustum radius at this depth
            let radius = depth * tan_half_fov;

            let chunks = self.chunks_around(center, radius, loaded);
            for (coord, base_priority) in chunks {
                // Priority decreases with depth
                let depth_factor = 1.0 - (depth / config.max_depth);
                let priority = base_priority * depth_factor * self.strategy.frustum_weight;
                needed.push((coord, priority));
            }
        }

        needed
    }

    /// Predict chunks needed around active edit regions.
    ///
    /// # Arguments
    /// * `edit_regions` - Slice of AABBs representing active edit areas
    /// * `loaded` - Function to check if a chunk is already loaded
    pub fn predict_edit_region_chunks(&self, edit_regions: &[Aabb], loaded: &impl Fn(ChunkCoord) -> bool) -> Vec<(ChunkCoord, f32)> {
        let mut needed = Vec::new();
        let config = &self.edit_region_config;

        // Process up to max_regions edit areas
        let regions_to_process = edit_regions.len().min(config.max_regions);

        for region in &edit_regions[..regions_to_process] {
            // Get chunk bounds for the AABB
            let (min_cx, min_cy, min_cz) = world_to_chunk(region.min);
            let (max_cx, max_cy, max_cz) = world_to_chunk(region.max);

            // Expand by margin
            let margin = config.margin_chunks;

            for cx in (min_cx - margin)..=(max_cx + margin) {
                for cy in (min_cy - margin)..=(max_cy + margin) {
                    for cz in (min_cz - margin)..=(max_cz + margin) {
                        let coord = ChunkCoord::new(cx, cy, cz);

                        if loaded(coord) {
                            continue;
                        }

                        // Calculate distance from region center for priority
                        let region_center = region.center();
                        let chunk_center = Vec3::new(
                            cx as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                            cy as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                            cz as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                        );
                        let distance = (chunk_center - region_center).length();

                        // High priority, decreases with distance
                        let priority = (1.0 / (1.0 + distance * 0.05)) * self.strategy.edit_region_weight;
                        needed.push((coord, priority));
                    }
                }
            }
        }

        needed
    }

    /// Predict chunks that will be needed, combining all strategies.
    ///
    /// # Arguments
    /// * `forward` - Optional camera forward direction for frustum prediction
    /// * `edit_regions` - Optional slice of active edit region AABBs
    /// * `loaded` - Function to check if a chunk is already loaded
    pub fn predict_needed_chunks_combined(
        &self,
        forward: Option<Vec3>,
        edit_regions: Option<&[Aabb]>,
        loaded: &impl Fn(ChunkCoord) -> bool,
    ) -> Vec<(ChunkCoord, f32)> {
        let mut all_chunks = Vec::new();

        // 1. Temporal prediction (velocity-based)
        if self.strategy.temporal_weight > 0.0 {
            let temporal = self.predict_needed_chunks_temporal(loaded);
            all_chunks.extend(temporal);
        }

        // 2. Frustum prediction
        if let Some(fwd) = forward {
            if self.strategy.frustum_weight > 0.0 {
                let frustum = self.predict_frustum_chunks(fwd, loaded);
                all_chunks.extend(frustum);
            }
        }

        // 3. Edit region prediction (highest priority)
        if let Some(regions) = edit_regions {
            if self.strategy.edit_region_weight > 0.0 && !regions.is_empty() {
                let edit = self.predict_edit_region_chunks(regions, loaded);
                all_chunks.extend(edit);
            }
        }

        // Deduplicate, keeping highest priority for each chunk
        self.deduplicate_and_sort(all_chunks)
    }

    /// Legacy method for backward compatibility - uses only temporal prediction.
    pub fn predict_needed_chunks(&self, loaded: &impl Fn(ChunkCoord) -> bool) -> Vec<(ChunkCoord, f32)> {
        self.predict_needed_chunks_temporal(loaded)
    }

    /// Predict chunks using temporal (velocity-based) prediction.
    fn predict_needed_chunks_temporal(&self, loaded: &impl Fn(ChunkCoord) -> bool) -> Vec<(ChunkCoord, f32)> {
        let mut needed = Vec::new();
        let current_pos = self.current_position();
        let speed = self.speed();

        if speed < 0.1 {
            // Camera is stationary, just load nearby
            return self.chunks_around(current_pos, 32.0, loaded);
        }

        // Predict along velocity vector
        let steps = 10;
        for i in 0..steps {
            let t = self.lookahead_time * (i as f32 / steps as f32);
            let predicted_pos = self.predict_position(t);

            // Add chunks around predicted position
            let radius = 16.0 + speed * t; // Expand radius with time
            let chunks = self.chunks_around(predicted_pos, radius, loaded);

            for (coord, base_priority) in chunks {
                // Priority decreases with prediction time
                let priority = base_priority * (1.0 - t / self.lookahead_time * 0.5) * self.strategy.temporal_weight;
                needed.push((coord, priority));
            }
        }

        needed
    }

    /// Deduplicate chunks and sort by priority.
    fn deduplicate_and_sort(&self, mut chunks: Vec<(ChunkCoord, f32)>) -> Vec<(ChunkCoord, f32)> {
        // Sort by chunk coordinate first to group duplicates
        chunks.sort_by(|a, b| {
            let coord_cmp = a.0.x.cmp(&b.0.x)
                .then(a.0.y.cmp(&b.0.y))
                .then(a.0.z.cmp(&b.0.z));
            coord_cmp.then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Deduplicate, keeping highest priority
        chunks.dedup_by(|a, b| {
            if a.0 == b.0 {
                // Keep the one with higher priority (b is kept)
                b.1 = b.1.max(a.1);
                true
            } else {
                false
            }
        });

        // Sort by priority (highest first)
        chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        chunks
    }

    /// Get chunks around a position that aren't loaded.
    fn chunks_around(&self, pos: Vec3, radius: f32, loaded: &impl Fn(ChunkCoord) -> bool) -> Vec<(ChunkCoord, f32)> {
        let (cx, cy, cz) = world_to_chunk(pos);
        let chunk_radius = (radius / CHUNK_SIZE_METERS).ceil() as i32;
        let mut result = Vec::new();

        for dx in -chunk_radius..=chunk_radius {
            for dy in -chunk_radius..=chunk_radius {
                for dz in -chunk_radius..=chunk_radius {
                    let coord = ChunkCoord::new(cx + dx, cy + dy, cz + dz);

                    if loaded(coord) {
                        continue;
                    }

                    // Priority based on distance
                    let chunk_center = Vec3::new(
                        (cx + dx) as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                        (cy + dy) as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                        (cz + dz) as f32 * CHUNK_SIZE_METERS + CHUNK_SIZE_METERS / 2.0,
                    );
                    let distance = (chunk_center - pos).length();
                    let priority = 1.0 / (1.0 + distance * 0.1);

                    result.push((coord, priority));
                }
            }
        }

        result
    }
}

impl Default for PrefetchPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_estimation() {
        let mut predictor = PrefetchPredictor::new();

        // Simulate moving at 10 m/s along X
        for i in 0..30 {
            let pos = Vec3::new(i as f32 * 0.166, 0.0, 0.0); // ~10 m/s at 60fps
            predictor.update(pos, 1.0 / 60.0);
        }

        let speed = predictor.speed();
        assert!(speed > 8.0 && speed < 12.0, "Speed should be ~10 m/s, got {}", speed);
    }

    #[test]
    fn test_position_prediction() {
        let mut predictor = PrefetchPredictor::new();

        // Set up with known velocity
        predictor.update(Vec3::ZERO, 0.0);
        predictor.update(Vec3::new(10.0, 0.0, 0.0), 1.0);

        let predicted = predictor.predict_position(0.5);
        // Should predict ~15m ahead (current 10m + 0.5s * 10m/s)
        assert!(predicted.x > 12.0 && predicted.x < 18.0);
    }

    #[test]
    fn test_stationary_camera() {
        let mut predictor = PrefetchPredictor::new();

        // Camera at origin, not moving
        for _ in 0..10 {
            predictor.update(Vec3::new(5.0, 0.0, 5.0), 1.0 / 60.0);
        }

        assert!(predictor.speed() < 0.1);
    }
}
