//! Streaming Manager - Coordinates brick streaming components
//!
//! Integrates BrickPool, BrickCache, and BrickRequestQueue to provide
//! on-demand brick loading based on ray-guided feedback from the GPU.

use super::{BrickPool, BrickCache, BrickRequestQueue};
use super::brick_cache::LoadPriority;
use crate::voxel::brick::VoxelBrick;
use crate::voxel::svo::Octree;
use glam::Vec3;

/// Maximum bricks to load per frame to avoid stalling
const MAX_LOADS_PER_FRAME: usize = 256;

/// Streaming manager coordinating GPU brick caching
pub struct StreamingManager {
    /// Brick pool for GPU storage
    pool: BrickPool,
    /// Cache state tracking
    cache: BrickCache,
    /// GPU request queue
    request_queue: BrickRequestQueue,
    /// Indirection buffer (brick_id -> pool_slot)
    indirection_buffer: wgpu::Buffer,
    /// Source brick data (kept for streaming)
    source_bricks: Vec<VoxelBrick>,
    /// Current frame number
    frame: u32,
    /// Camera position for distance-based priority
    camera_pos: Vec3,
}

impl StreamingManager {
    /// Create a new streaming manager
    pub fn new(device: &wgpu::Device, max_bricks: u32, pool_size: u32) -> Self {
        let pool = BrickPool::new(device, pool_size);
        let cache = BrickCache::new(max_bricks);
        let request_queue = BrickRequestQueue::new(device);

        // Create indirection buffer (brick_id -> pool_slot mapping)
        let indirection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_indirection"),
            size: (max_bricks as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pool,
            cache,
            request_queue,
            indirection_buffer,
            source_bricks: Vec::new(),
            frame: 0,
            camera_pos: Vec3::ZERO,
        }
    }

    /// Set source brick data (from octree)
    pub fn set_source_bricks(&mut self, bricks: Vec<VoxelBrick>) {
        self.source_bricks = bricks;
        // Resize cache if needed
        self.cache.resize(self.source_bricks.len() as u32);
    }

    /// Load source bricks from an octree
    pub fn load_from_octree(&mut self, octree: &Octree) {
        let bricks = octree.bricks_slice().to_vec();
        self.set_source_bricks(bricks);
        log::info!("StreamingManager: loaded {} source bricks", self.source_bricks.len());
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.pool.begin_frame();
        self.cache.begin_frame();
    }

    /// Set camera position for distance-based priority calculations
    pub fn set_camera_position(&mut self, pos: Vec3) {
        self.camera_pos = pos;
    }

    /// Reset request queue for new trace pass
    pub fn reset_requests(&self, encoder: &mut wgpu::CommandEncoder) {
        self.request_queue.reset(encoder);
    }

    /// Schedule readback of brick requests (call after trace pass)
    pub fn schedule_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.request_queue.schedule_readback(encoder);
    }

    /// Process brick requests and upload missing bricks
    /// Returns number of bricks loaded this frame
    pub fn process_requests(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> usize {
        // Get requested brick IDs from GPU
        let requests = self.request_queue.process_readback(device);

        if requests.is_empty() {
            return 0;
        }

        log::trace!("Processing {} brick requests", requests.len());

        // Add requests to cache with priority
        // Note: Individual brick world positions are not readily available from brick_id alone.
        // Distance calculation would require maintaining a brick_id -> world_pos mapping.
        // For now, all requests get distance=0.0, prioritized by request order.
        let priority = LoadPriority {
            distance: 0.0,
            lod_level: 0,
            request_frame: self.frame,
        };

        for brick_id in &requests {
            self.cache.request(*brick_id, priority);
        }

        // Load pending bricks (up to MAX_LOADS_PER_FRAME)
        let mut loaded = 0;
        let mut updated_indices = Vec::new();

        while loaded < MAX_LOADS_PER_FRAME {
            if let Some(brick_id) = self.cache.pop_pending() {
                if (brick_id as usize) < self.source_bricks.len() {
                    // Allocate slot in pool
                    if let Some(slot) = self.pool.allocate_slot(brick_id) {
                        // Upload brick data
                        let brick = &self.source_bricks[brick_id as usize];
                        self.pool.upload_brick(queue, slot, brick);

                        // Update cache
                        self.cache.mark_loaded(brick_id, slot);
                        updated_indices.push(brick_id);
                        loaded += 1;
                    }
                }
            } else {
                break;
            }
        }

        // Update indirection buffer if any bricks were loaded
        if !updated_indices.is_empty() {
            self.update_indirection(queue);
            log::debug!("Loaded {} bricks this frame, pool utilization: {:.1}%",
                loaded, self.pool.utilization());
        }

        loaded
    }

    /// Update indirection buffer on GPU
    fn update_indirection(&self, queue: &wgpu::Queue) {
        let data = self.cache.indirection_table();
        queue.write_buffer(&self.indirection_buffer, 0, bytemuck::cast_slice(data));
    }

    /// Pre-load all bricks (for non-streaming fallback)
    pub fn preload_all(&mut self, queue: &wgpu::Queue) {
        let total = self.source_bricks.len().min(self.pool.capacity() as usize);

        for brick_id in 0..total as u32 {
            if let Some(slot) = self.pool.allocate_slot(brick_id) {
                let brick = &self.source_bricks[brick_id as usize];
                self.pool.upload_brick(queue, slot, brick);
                self.cache.mark_loaded(brick_id, slot);
            }
        }

        self.update_indirection(queue);
        log::info!("Preloaded {} bricks, pool utilization: {:.1}%",
            total, self.pool.utilization());
    }

    /// Get indirection buffer for shader binding
    pub fn indirection_buffer(&self) -> &wgpu::Buffer {
        &self.indirection_buffer
    }

    /// Get brick pool buffer for shader binding
    pub fn brick_pool_buffer(&self) -> &wgpu::Buffer {
        self.pool.buffer()
    }

    /// Get request queue bind group layout
    pub fn request_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.request_queue.bind_group_layout()
    }

    /// Get request queue bind group
    pub fn request_bind_group(&self) -> &wgpu::BindGroup {
        self.request_queue.bind_group()
    }

    /// Get pool capacity
    pub fn pool_capacity(&self) -> u32 {
        self.pool.capacity()
    }

    /// Get loaded brick count
    pub fn loaded_count(&self) -> u32 {
        self.pool.loaded_count()
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f32 {
        self.cache.hit_rate()
    }

    /// Get pool utilization percentage
    pub fn utilization(&self) -> f32 {
        self.pool.utilization()
    }
}
