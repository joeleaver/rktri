//! Brick Request Queue - GPU->CPU feedback for missing bricks
//!
//! When rays encounter missing bricks during traversal, they write
//! the brick ID to a request buffer. The CPU reads this buffer and
//! schedules brick loading.
//!
//! Based on GigaVoxels ray-guided caching concept.

use std::collections::HashSet;

/// Maximum requests per frame to prevent overwhelming the system
pub const MAX_REQUESTS_PER_FRAME: usize = 1024;

/// GPU buffer for brick requests
pub struct BrickRequestQueue {
    /// GPU buffer where shaders write requested brick IDs
    request_buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback
    staging_buffer: wgpu::Buffer,
    /// Counter buffer (atomic counter for number of requests)
    counter_buffer: wgpu::Buffer,
    /// Counter staging for readback
    counter_staging: wgpu::Buffer,
    /// Bind group layout for shader access
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Collected requests from last frame
    pending_requests: HashSet<u32>,
    /// Whether a readback is in progress
    readback_pending: bool,
}

impl BrickRequestQueue {
    /// Create a new request queue
    pub fn new(device: &wgpu::Device) -> Self {
        // Buffer for brick IDs (u32 each)
        let request_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_request_buffer"),
            size: (MAX_REQUESTS_PER_FRAME * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_request_staging"),
            size: (MAX_REQUESTS_PER_FRAME * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Counter buffer (single u32)
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_request_counter"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_request_counter_staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout for shader access
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brick_request_layout"),
            entries: &[
                // Request buffer (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counter buffer (read-write storage for atomic ops)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick_request_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: request_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            request_buffer,
            staging_buffer,
            counter_buffer,
            counter_staging,
            bind_group_layout,
            bind_group,
            pending_requests: HashSet::new(),
            readback_pending: false,
        }
    }

    /// Reset the request counter for a new frame
    pub fn reset(&self, encoder: &mut wgpu::CommandEncoder) {
        // Clear the counter to 0
        encoder.clear_buffer(&self.counter_buffer, 0, None);
    }

    /// Schedule readback of requests (call after trace pass)
    pub fn schedule_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // Copy counter to staging
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        // Copy requests to staging
        encoder.copy_buffer_to_buffer(
            &self.request_buffer,
            0,
            &self.staging_buffer,
            0,
            (MAX_REQUESTS_PER_FRAME * std::mem::size_of::<u32>()) as u64,
        );

        self.readback_pending = true;
    }

    /// Process readback results (call after GPU work is done)
    /// Returns the set of requested brick IDs
    pub fn process_readback(&mut self, device: &wgpu::Device) -> HashSet<u32> {
        if !self.readback_pending {
            return HashSet::new();
        }

        self.pending_requests.clear();

        // Read counter
        let counter_slice = self.counter_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        counter_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        if rx.recv().unwrap().is_ok() {
            let data = counter_slice.get_mapped_range();
            let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            self.counter_staging.unmap();

            let count = count.min(MAX_REQUESTS_PER_FRAME as u32);

            if count > 0 {
                // Read requests
                let request_slice = self.staging_buffer.slice(..);
                let (tx2, rx2) = std::sync::mpsc::channel();
                request_slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx2.send(result).unwrap();
                });
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                if rx2.recv().unwrap().is_ok() {
                    let data = request_slice.get_mapped_range();
                    let requests: &[u32] = bytemuck::cast_slice(&data[..count as usize * 4]);

                    for &brick_id in requests {
                        if brick_id != u32::MAX {
                            self.pending_requests.insert(brick_id);
                        }
                    }

                    drop(data);
                    self.staging_buffer.unmap();
                }

                log::trace!(
                    "Brick requests: {} total, {} unique",
                    count,
                    self.pending_requests.len()
                );
            }
        }

        self.readback_pending = false;
        std::mem::take(&mut self.pending_requests)
    }

    /// Get bind group layout
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get bind group
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
