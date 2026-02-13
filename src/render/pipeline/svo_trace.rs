//! SVO ray tracing compute pipeline

use bytemuck::{Pod, Zeroable};
use crate::render::buffer::{OctreeBuffer, CameraBuffer};
use crate::grass::{GrassParams, GpuGrassProfile};

/// Maximum number of grass profiles supported on GPU.
pub const MAX_GRASS_PROFILES: u32 = 16;

/// Trace parameters uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TraceParams {
    pub width: u32,
    pub height: u32,
    pub chunk_count: u32,
    pub _pad0: u32,
    pub lod_distances: [f32; 4],     // First 4 LOD distances: 64, 128, 256, 512
    pub lod_distances_ext: [f32; 2], // Remaining: 1024, f32::MAX
    pub _pad: [f32; 2],              // Padding to 16-byte alignment
    // Chunk grid acceleration: DDA ray marching through a 3D grid of chunk indices
    pub grid_min_x: i32,
    pub grid_min_y: i32,
    pub grid_min_z: i32,
    pub chunk_size: f32,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub _pad2: u32,
}

/// SVO ray tracing compute pipeline
pub struct SvoTracePipeline {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    grass_buffer: wgpu::Buffer,
    profile_table_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group: wgpu::BindGroup,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl SvoTracePipeline {
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &CameraBuffer,
        octree_buffer: &OctreeBuffer,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svo_trace_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/svo_trace.wgsl").into()),
        });

        // Params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trace_params"),
            size: std::mem::size_of::<TraceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grass params buffer (initialized to zeroed = disabled)
        let grass_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_params"),
            size: std::mem::size_of::<GrassParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grass profile table (storage buffer, MAX_GRASS_PROFILES × 64 bytes)
        let profile_table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_profile_table"),
            size: (MAX_GRASS_PROFILES as u64) * (std::mem::size_of::<GpuGrassProfile>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: Camera + Params + Grass + Profile Table
        let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace_params_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: grass profile table (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace_params_bind_group"),
            layout: &params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grass_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: profile_table_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group 2: Output G-buffer textures
        let output_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace_output_layout"),
            entries: &[
                // Binding 0: Albedo (Rgba8Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 1: Normal (Rgba16Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 2: Depth (R32Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 3: Material (Rgba8Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 4: Motion (Rgba16Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svo_trace_pipeline_layout"),
            bind_group_layouts: &[
                &params_bind_group_layout,
                octree_buffer.bind_group_layout(),
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // Compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svo_trace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            params_buffer,
            grass_buffer,
            profile_table_buffer,
            params_bind_group_layout,
            params_bind_group,
            output_bind_group_layout,
        }
    }

    /// Create output bind group for G-buffer texture views
    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        albedo: &wgpu::TextureView,
        normal: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        material: &wgpu::TextureView,
        motion: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(motion),
                },
            ],
        })
    }

    /// Update trace parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &TraceParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update grass parameters
    pub fn update_grass_params(&self, queue: &wgpu::Queue, params: &GrassParams) {
        queue.write_buffer(&self.grass_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload grass profile table data
    pub fn update_profile_table(&self, queue: &wgpu::Queue, profiles: &[GpuGrassProfile]) {
        assert!(
            profiles.len() <= MAX_GRASS_PROFILES as usize,
            "Too many grass profiles: {} (max {})",
            profiles.len(),
            MAX_GRASS_PROFILES
        );
        if !profiles.is_empty() {
            queue.write_buffer(
                &self.profile_table_buffer,
                0,
                bytemuck::cast_slice(profiles),
            );
        }
    }

    /// Dispatch compute shader
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        octree_buffer: &OctreeBuffer,
        output_bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("svo_trace_pass"),
            timestamp_writes,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.params_bind_group, &[]);
        pass.set_bind_group(1, octree_buffer.bind_group(), &[]);
        pass.set_bind_group(2, output_bind_group, &[]);

        // Dispatch with 8x8 workgroups
        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
