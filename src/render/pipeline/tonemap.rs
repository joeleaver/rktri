//! Tone mapping post-process compute pipeline

use bytemuck::{Pod, Zeroable};

/// Tone mapping parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TonemapParams {
    pub width: u32,
    pub height: u32,
    pub exposure: f32,
    pub _pad: u32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            exposure: 1.0,
            _pad: 0,
        }
    }
}

/// Tone mapping compute pipeline
pub struct TonemapPipeline {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl TonemapPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/tonemap.wgsl").into()),
        });

        // Params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap_params"),
            size: std::mem::size_of::<TonemapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Group 0: params
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_params_bg"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Group 1: input texture (sampled)
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tonemap_input_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        // Group 2: output texture (storage)
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tonemap_output_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap_pipeline_layout"),
            bind_group_layouts: &[&params_layout, &input_bind_group_layout, &output_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            params_buffer,
            params_bind_group,
            input_bind_group_layout,
            output_bind_group_layout,
        }
    }

    pub fn create_input_bind_group(
        &self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_input_bg"),
            layout: &self.input_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input_view),
            }],
        })
    }

    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_output_bg"),
            layout: &self.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            }],
        })
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &TonemapParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tonemap_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.params_bind_group, &[]);
        pass.set_bind_group(1, input_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
