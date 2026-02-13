//! Dynamic cloud rendering compute pipeline

use bytemuck::{Pod, Zeroable};
use crate::render::buffer::CameraBuffer;

/// Cloud rendering parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CloudParams {
    // Cloud layer geometry
    pub cloud_altitude: f32,
    pub cloud_thickness: f32,
    pub cloud_coverage: f32,
    pub cloud_density: f32,

    // Wind offset (accumulated)
    pub wind_offset: [f32; 3],
    pub noise_scale: f32,

    // Colors
    pub cloud_color: [f32; 3],
    pub detail_scale: f32,
    pub shadow_color: [f32; 3],
    pub edge_sharpness: f32,

    // Sun for lighting
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub time_of_day: f32,

    // Dimensions
    pub width: u32,
    pub height: u32,
    pub _pad: [u32; 2],
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            cloud_altitude: 80.0,
            cloud_thickness: 20.0,
            cloud_coverage: 0.3,
            cloud_density: 0.8,
            wind_offset: [0.0; 3],
            noise_scale: 0.02,
            cloud_color: [1.0, 1.0, 1.0],
            detail_scale: 0.1,
            shadow_color: [0.4, 0.4, 0.5],
            edge_sharpness: 3.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_intensity: 1.5,
            sun_color: [1.0, 0.98, 0.95],
            time_of_day: 10.0,
            width: 640,
            height: 360,
            _pad: [0; 2],
        }
    }
}

/// Dynamic cloud rendering compute pipeline
///
/// Raymarches through a cloud slab using 3D noise evaluated on the GPU.
/// Wind moves the noise domain over time for cloud movement.
pub struct CloudPipeline {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_params_bind_group_layout: wgpu::BindGroupLayout,
    camera_params_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl CloudPipeline {
    /// Create a new cloud pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `camera_buffer` - Camera uniform buffer
    pub fn new(device: &wgpu::Device, camera_buffer: &CameraBuffer) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/clouds.wgsl").into()),
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud_params"),
            size: std::mem::size_of::<CloudParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: Camera + Cloud params
        let camera_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clouds_camera_params_layout"),
                entries: &[
                    // Camera buffer
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
                    // Cloud params
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
                ],
            });

        let camera_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds_camera_params_bind_group"),
            layout: &camera_params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group 1: Input texture (depth)
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clouds_input_layout"),
                entries: &[
                    // Depth texture (R32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Bind group 2: Output cloud texture (Rgba16Float)
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clouds_output_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds_pipeline_layout"),
            bind_group_layouts: &[
                &camera_params_bind_group_layout,
                &input_bind_group_layout,
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clouds_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            params_buffer,
            camera_params_bind_group_layout,
            camera_params_bind_group,
            input_bind_group_layout,
            output_bind_group_layout,
        }
    }

    /// Create input bind group for depth texture
    pub fn create_input_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds_input_bind_group"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        })
    }

    /// Create output bind group for the cloud texture
    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        cloud_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(cloud_view),
            }],
        })
    }

    /// Update cloud parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &CloudParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the cloud compute shader
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clouds_pass"),
            timestamp_writes,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_params_bind_group, &[]);
        pass.set_bind_group(1, input_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);

        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
