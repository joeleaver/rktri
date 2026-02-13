//! Water rendering pipeline

use bytemuck::{Pod, Zeroable};
use crate::render::buffer::{OctreeBuffer, CameraBuffer};

/// Water rendering parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WaterUniforms {
    /// Sea level in world space (meters)
    pub sea_level: f32,
    /// Time in seconds (for wave animation)
    pub time: f32,
    /// Index of refraction (water ~1.333)
    pub ior: f32,
    /// Surface roughness (0.0 = mirror, 1.0 = rough)
    pub roughness: f32,

    /// Camera position in world space
    pub camera_pos: [f32; 3],
    pub _pad1: f32,

    /// Water surface color (linear RGB)
    pub water_color: [f32; 3],
    pub _pad2: f32,

    /// Beer-Lambert absorption coefficients (RGB)
    pub absorption: [f32; 3],
    pub _pad3: f32,
}

impl Default for WaterUniforms {
    fn default() -> Self {
        Self {
            sea_level: 0.0,
            time: 0.0,
            ior: 1.333,
            roughness: 0.05,
            camera_pos: [0.0; 3],
            _pad1: 0.0,
            water_color: [0.0, 0.4, 0.6], // Blue-green
            _pad2: 0.0,
            absorption: [0.45, 0.15, 0.05], // Red absorbed most, blue least
            _pad3: 0.0,
        }
    }
}

/// Water rendering compute pipeline
///
/// Handles water surface tracing, refraction, reflection, and underwater volumetric effects.
/// Uses the water system from voxel/water/ for water body classification.
pub struct WaterPipeline {
    #[allow(dead_code)]
    trace_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    lighting_pipeline: wgpu::ComputePipeline,
    uniforms_buffer: wgpu::Buffer,
    camera_water_bind_group_layout: wgpu::BindGroupLayout,
    camera_water_bind_group: wgpu::BindGroup,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl WaterPipeline {
    /// Create a new water rendering pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `camera_buffer` - Camera uniform buffer (provides view-projection and camera position)
    /// * `octree_buffer` - SVO octree buffer (provides scene geometry including water voxels)
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &CameraBuffer,
        octree_buffer: &OctreeBuffer,
    ) -> Self {
        // Load shaders
        let trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water_trace_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/water_trace.wgsl").into()),
        });

        let lighting_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water_lighting_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/water_lighting.wgsl").into()),
        });

        // Create uniforms buffer
        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_uniforms"),
            size: std::mem::size_of::<WaterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: Camera + Water uniforms
        let camera_water_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_camera_uniforms_layout"),
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
                    // Water uniforms
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

        let camera_water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_camera_uniforms_bind_group"),
            layout: &camera_water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group 2: G-buffer input textures (color, depth, normal)
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_gbuffer_layout"),
                entries: &[
                    // Color texture (Rgba16Float)
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
                    // Normal texture (Rgba16Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        // Bind group 3: Output texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_output_layout"),
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

        // Create pipeline layouts
        let trace_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_trace_pipeline_layout"),
            bind_group_layouts: &[
                &camera_water_bind_group_layout,
                octree_buffer.bind_group_layout(),
                &gbuffer_bind_group_layout,
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let lighting_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_lighting_pipeline_layout"),
            bind_group_layouts: &[
                &camera_water_bind_group_layout,
                octree_buffer.bind_group_layout(),
                &gbuffer_bind_group_layout,
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // Create compute pipelines
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("water_trace_pipeline"),
            layout: Some(&trace_pipeline_layout),
            module: &trace_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let lighting_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("water_lighting_pipeline"),
            layout: Some(&lighting_pipeline_layout),
            module: &lighting_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            trace_pipeline,
            lighting_pipeline,
            uniforms_buffer,
            camera_water_bind_group_layout,
            camera_water_bind_group,
            gbuffer_bind_group_layout,
            output_bind_group_layout,
        }
    }

    /// Create G-buffer bind group for input textures
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `color_view` - Color texture view (lit scene color)
    /// * `depth_view` - Depth texture view (linear depth)
    /// * `normal_view` - Normal texture view (world-space normals)
    pub fn create_gbuffer_bind_group(
        &self,
        device: &wgpu::Device,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_gbuffer_bind_group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
            ],
        })
    }

    /// Create output bind group for the water-composited result texture
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `view` - Output texture view (Rgba16Float storage texture)
    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    /// Update water uniforms
    ///
    /// # Arguments
    /// * `queue` - WGPU queue
    /// * `uniforms` - New water parameters
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &WaterUniforms) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Dispatch the water rendering compute shader (stub - not implemented yet)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder
    /// * `octree_buffer` - SVO octree buffer (scene geometry)
    /// * `gbuffer_bind_group` - G-buffer input bind group
    /// * `output_bind_group` - Output texture bind group
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    #[allow(unused_variables)]
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        octree_buffer: &OctreeBuffer,
        gbuffer_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        // TODO: Implement water tracing and lighting dispatch
        // This will be implemented when the water shaders are ready

        // Expected implementation:
        // 1. Dispatch trace_pipeline to detect water surfaces and trace refraction/reflection
        // 2. Dispatch lighting_pipeline to apply water lighting and composite with scene
        //
        // let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //     label: Some("water_pass"),
        //     timestamp_writes: None,
        // });
        //
        // pass.set_pipeline(&self.trace_pipeline);
        // pass.set_bind_group(0, &self.camera_water_bind_group, &[]);
        // pass.set_bind_group(1, octree_buffer.bind_group(), &[]);
        // pass.set_bind_group(2, gbuffer_bind_group, &[]);
        // pass.set_bind_group(3, output_bind_group, &[]);
        //
        // let workgroups_x = (width + 7) / 8;
        // let workgroups_y = (height + 7) / 8;
        // pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Get the bind group layout for camera and water uniforms
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_water_bind_group_layout
    }

    /// Get the bind group for camera and water uniforms
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_water_bind_group
    }
}
