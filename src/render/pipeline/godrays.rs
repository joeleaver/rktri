//! Screen-space volumetric light scattering (god rays) compute pipeline

use bytemuck::{Pod, Zeroable};
use crate::render::buffer::CameraBuffer;

/// God rays parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GodRaysParams {
    pub sun_screen_pos: [f32; 2],
    pub num_samples: u32,
    pub density: f32,
    pub decay: f32,
    pub exposure: f32,
    pub weight: f32,
    pub width: u32,
    pub height: u32,
    pub _pad: u32,
}

impl Default for GodRaysParams {
    fn default() -> Self {
        Self {
            sun_screen_pos: [0.5, 0.5],
            num_samples: 64,
            density: 0.5,
            decay: 0.97,
            exposure: 0.01,
            weight: 1.0,
            width: 1280,
            height: 720,
            _pad: 0,
        }
    }
}

/// Screen-space volumetric light scattering compute pipeline
///
/// Ray-marches through the shadow buffer toward the sun's screen position.
/// Where the shadow buffer reads "lit", scattered light accumulates, creating
/// light shafts through tree canopies and around objects.
pub struct GodRaysPipeline {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_params_bind_group_layout: wgpu::BindGroupLayout,
    camera_params_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl GodRaysPipeline {
    /// Create a new god rays pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `camera_buffer` - Camera uniform buffer
    pub fn new(device: &wgpu::Device, camera_buffer: &CameraBuffer) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godrays_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/godrays.wgsl").into()),
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("godrays_params"),
            size: std::mem::size_of::<GodRaysParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: Camera + GodRays params
        let camera_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("godrays_camera_params_layout"),
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
                    // GodRays params
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
            label: Some("godrays_camera_params_bind_group"),
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

        // Bind group 1: Input textures (shadow + depth)
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("godrays_input_layout"),
                entries: &[
                    // Shadow texture (R32Float)
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
                    // Depth texture (R32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        // Bind group 2: Output god rays texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("godrays_output_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godrays_pipeline_layout"),
            bind_group_layouts: &[
                &camera_params_bind_group_layout,
                &input_bind_group_layout,
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("godrays_pipeline"),
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

    /// Create input bind group for shadow and depth textures
    pub fn create_input_bind_group(
        &self,
        device: &wgpu::Device,
        shadow_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godrays_input_bind_group"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        })
    }

    /// Create output bind group for the god rays texture
    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        godrays_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godrays_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(godrays_view),
            }],
        })
    }

    /// Update god rays parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &GodRaysParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the god rays compute shader
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
            label: Some("godrays_pass"),
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
