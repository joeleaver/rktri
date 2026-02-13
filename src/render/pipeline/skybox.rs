//! Skybox parameters for procedural sky rendering

use bytemuck::{Pod, Zeroable};

/// Sky rendering parameters
///
/// These parameters control the procedural sky appearance in the lighting shader.
/// The lighting shader checks for sky pixels (normal.w == 0) and uses these parameters
/// to generate a gradient sky with ground color, moon, and fog.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkyParams {
    // Sun (existing)
    /// Sun direction (normalized vector towards the sun)
    pub sun_direction: [f32; 3],
    pub _pad1: f32,
    /// Sun color (linear RGB)
    pub sun_color: [f32; 3],
    /// Sun intensity multiplier
    pub sun_intensity: f32,

    // Sky colors (from atmosphere color ramps)
    /// Sky zenith color (linear RGB)
    pub sky_zenith_color: [f32; 3],
    /// Sky intensity multiplier (affects gradient brightness)
    pub sky_intensity: f32,
    /// Sky horizon color (linear RGB)
    pub sky_horizon_color: [f32; 3],
    pub _pad2: f32,

    // Ground
    /// Ground/horizon color (linear RGB)
    pub ground_color: [f32; 3],
    pub _pad3: f32,

    // Moon
    /// Moon direction (normalized vector towards the moon)
    pub moon_direction: [f32; 3],
    /// Moon phase (0=new, 0.5=full, 1=new)
    pub moon_phase: f32,
    /// Moon color (linear RGB)
    pub moon_color: [f32; 3],
    /// Moon angular size
    pub moon_size: f32,
    /// Number of moons (0 or 1 for now)
    pub moon_count: u32,
    pub _pad4: [f32; 3],

    // Fog
    /// Fog color (linear RGB)
    pub fog_color: [f32; 3],
    /// Fog density (0 = no fog)
    pub fog_density: f32,
    /// Height fog falloff rate
    pub fog_height_falloff: f32,
    /// Height fog base altitude
    pub fog_height_base: f32,
    /// Fog inscattering intensity
    pub fog_inscattering: f32,
    pub _pad5: f32,

    // Ambient
    /// Ambient light color (linear RGB, from atmosphere ramps)
    pub ambient_color: [f32; 3],
    /// Ambient light intensity multiplier
    pub ambient_intensity: f32,
}

impl Default for SkyParams {
    fn default() -> Self {
        // Default to a pleasant daytime sky with sun at ~45 degrees
        let sun_dir = glam::Vec3::new(0.5, 0.8, 0.3).normalize();
        Self {
            sun_direction: sun_dir.to_array(),
            _pad1: 0.0,
            sun_color: [1.0, 0.95, 0.9],
            sun_intensity: 50.0,
            sky_zenith_color: [0.15, 0.35, 0.65],
            sky_intensity: 1.5,
            sky_horizon_color: [0.45, 0.55, 0.7],
            _pad2: 0.0,
            ground_color: [0.3, 0.25, 0.2],
            _pad3: 0.0,
            moon_direction: [0.0, -1.0, 0.0],
            moon_phase: 0.0,
            moon_color: [0.0; 3],
            moon_size: 0.0,
            moon_count: 0,
            _pad4: [0.0; 3],
            fog_color: [0.5, 0.55, 0.6],
            fog_density: 0.0,
            fog_height_falloff: 0.05,
            fog_height_base: 0.0,
            fog_inscattering: 0.0,
            _pad5: 0.0,
            ambient_color: [0.03, 0.04, 0.05],
            ambient_intensity: 1.0,
        }
    }
}

/// Skybox pipeline for managing sky rendering parameters
///
/// The actual sky rendering happens in the lighting shader (lighting.wgsl),
/// which includes skybox.wgsl for the procedural sky generation.
/// This pipeline just manages the parameter buffer and bind group.
pub struct SkyboxPipeline {
    params_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SkyboxPipeline {
    /// Create a new skybox pipeline with default parameters
    ///
    /// # Arguments
    /// * `device` - WGPU device
    pub fn new(device: &wgpu::Device) -> Self {
        // Create parameter buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skybox_params"),
            size: std::mem::size_of::<SkyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skybox_bind_group_layout"),
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        Self {
            params_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Update sky parameters
    ///
    /// # Arguments
    /// * `queue` - WGPU queue
    /// * `params` - New sky parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &SkyParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Get the bind group layout for integration with other pipelines
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group for binding in compute passes
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
