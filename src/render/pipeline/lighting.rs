//! PBR lighting compute pipeline

use bytemuck::{Pod, Zeroable};
use crate::render::buffer::CameraBuffer;

// Re-export SkyParams from skybox module for convenience
pub use super::skybox::SkyParams;

/// Debug visualization parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DebugParams {
    /// Debug mode: 0=Normal, 1=Albedo, 2=Normals, 3=Depth, 4=Material
    pub mode: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

impl Default for DebugParams {
    fn default() -> Self {
        Self {
            mode: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// Lighting uniforms for sun and ambient lighting
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightingUniforms {
    /// Sun direction (normalized vector towards the sun)
    pub sun_direction: [f32; 3],
    pub _pad1: f32,
    /// Sun color (linear RGB)
    pub sun_color: [f32; 3],
    /// Sun intensity multiplier
    pub sun_intensity: f32,
    /// Ambient light color (linear RGB)
    pub ambient_color: [f32; 3],
    pub _pad2: f32,
}

impl Default for LightingUniforms {
    fn default() -> Self {
        Self {
            sun_direction: [0.0, -1.0, 0.0], // Directly overhead
            _pad1: 0.0,
            sun_color: [1.0, 1.0, 1.0], // White light
            sun_intensity: 1.0,
            ambient_color: [0.03, 0.03, 0.03], // Low ambient
            _pad2: 0.0,
        }
    }
}

/// PBR lighting compute pipeline
///
/// Takes G-buffer inputs (albedo, normal, depth, material properties) and applies
/// Cook-Torrance BRDF lighting with sun and ambient contributions.
pub struct LightingPipeline {
    pipeline: wgpu::ComputePipeline,
    uniforms_buffer: wgpu::Buffer,
    sky_buffer: wgpu::Buffer,
    debug_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    camera_lighting_bind_group_layout: wgpu::BindGroupLayout,
    camera_lighting_bind_group: wgpu::BindGroup,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
}

impl LightingPipeline {
    /// Create a new lighting pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `camera_buffer` - Camera uniform buffer (provides view-projection and camera position)
    pub fn new(device: &wgpu::Device, camera_buffer: &CameraBuffer) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/lighting.wgsl").into()),
        });

        // Create uniforms buffer
        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting_uniforms"),
            size: std::mem::size_of::<LightingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create sky params buffer
        let sky_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting_sky_params"),
            size: std::mem::size_of::<SkyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create debug params buffer
        let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting_debug_params"),
            size: std::mem::size_of::<DebugParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group 0: Camera + Lighting uniforms + Sky params + Debug params
        let camera_lighting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_camera_lighting_layout"),
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
                    // Lighting uniforms
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
                    // Sky params
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
                    // Debug params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let camera_lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_camera_lighting_bind_group"),
            layout: &camera_lighting_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sky_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: debug_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group 1: G-buffer input textures + shadow mask
        // This will be created per-frame since the G-buffer textures may change
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_gbuffer_layout"),
                entries: &[
                    // Albedo texture (RGBA8)
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
                    // Normal texture (RGBA16Float)
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
                    // Depth texture (R32Float)
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
                    // Material texture (RGBA8: metallic, roughness, ao, unused)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Shadow mask texture (R8Unorm)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // God rays texture (R32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Cloud texture (Rgba16Float - color + alpha)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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

        // Bind group 2: Output texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_output_layout"),
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
            label: Some("lighting_pipeline_layout"),
            bind_group_layouts: &[
                &camera_lighting_bind_group_layout,
                &gbuffer_bind_group_layout,
                &output_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lighting_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            uniforms_buffer,
            sky_buffer,
            debug_buffer,
            camera_lighting_bind_group_layout,
            camera_lighting_bind_group,
            gbuffer_bind_group_layout,
            output_bind_group_layout,
        }
    }

    /// Create G-buffer bind group for input textures
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `albedo_view` - Albedo texture view (base color)
    /// * `normal_view` - Normal texture view (world-space normals)
    /// * `depth_view` - Depth texture view (linear depth)
    /// * `material_view` - Material properties texture view (metallic, roughness, AO)
    /// * `shadow_view` - Shadow mask texture view (shadow occlusion)
    /// * `godrays_view` - God rays texture view (volumetric light scattering)
    /// * `cloud_view` - Cloud texture view (Rgba16Float cloud color + alpha)
    pub fn create_gbuffer_bind_group(
        &self,
        device: &wgpu::Device,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        shadow_view: &wgpu::TextureView,
        godrays_view: &wgpu::TextureView,
        cloud_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_gbuffer_bind_group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(godrays_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(cloud_view),
                },
            ],
        })
    }

    /// Create output bind group for the lit result texture
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
            label: Some("lighting_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    /// Update lighting uniforms (sun direction, colors, intensities)
    ///
    /// # Arguments
    /// * `queue` - WGPU queue
    /// * `uniforms` - New lighting parameters
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &LightingUniforms) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Update sky parameters
    ///
    /// # Arguments
    /// * `queue` - WGPU queue
    /// * `params` - New sky parameters
    pub fn update_sky_params(&self, queue: &wgpu::Queue, params: &SkyParams) {
        queue.write_buffer(&self.sky_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update debug parameters
    ///
    /// # Arguments
    /// * `queue` - WGPU queue
    /// * `params` - New debug parameters
    pub fn update_debug_params(&self, queue: &wgpu::Queue, params: &DebugParams) {
        queue.write_buffer(&self.debug_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the lighting compute shader
    ///
    /// # Arguments
    /// * `encoder` - Command encoder
    /// * `gbuffer_bind_group` - G-buffer input bind group
    /// * `output_bind_group` - Output texture bind group
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lighting_pass"),
            timestamp_writes,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_lighting_bind_group, &[]);
        pass.set_bind_group(1, gbuffer_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);

        // Dispatch with 8x8 workgroups (matching the shader's @workgroup_size)
        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
