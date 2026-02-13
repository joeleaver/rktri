//! G-buffer texture management for deferred rendering

use wgpu::{Device, Extent3d, Texture, TextureView};

/// G-buffer textures for deferred rendering
///
/// Contains all render targets needed for the geometry pass:
/// - albedo: Base color from voxels
/// - normal: World-space normals + roughness
/// - depth: Linear depth values
/// - material: Material properties (metallic, roughness, ao, material_id)
/// - motion: Screen-space motion vectors for TAA/FSR
pub struct GBuffer {
    /// Base color texture (Rgba8Unorm)
    #[allow(dead_code)]
    albedo: Texture,
    /// Normal + roughness texture (Rgba16Float: xyz=normal, w=roughness)
    #[allow(dead_code)]
    normal: Texture,
    /// Linear depth texture (R32Float)
    #[allow(dead_code)]
    depth: Texture,
    /// Material properties texture (Rgba8Unorm: r=metallic, g=roughness, b=ao, a=material_id)
    #[allow(dead_code)]
    material: Texture,
    /// Motion vectors texture (Rgba16Float: xy=screen-space velocity, zw=unused)
    #[allow(dead_code)]
    motion: Texture,

    /// Texture views for rendering
    albedo_view: TextureView,
    normal_view: TextureView,
    depth_view: TextureView,
    material_view: TextureView,
    motion_view: TextureView,

    /// Bind group layout for reading G-buffer in lighting pass
    read_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for reading G-buffer
    read_bind_group: wgpu::BindGroup,
    /// Bind group layout for writing G-buffer from compute
    storage_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for writing G-buffer from compute
    storage_bind_group: wgpu::BindGroup,

    /// Current resolution
    width: u32,
    height: u32,
}

impl GBuffer {
    /// Create new G-buffer with specified dimensions
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Create albedo texture (Rgba8Unorm)
        let albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_albedo"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create normal texture (Rgba16Float)
        let normal = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create depth texture (R32Float)
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create material texture (Rgba8Unorm)
        let material = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_material"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create motion vectors texture (Rgba16Float)
        let motion = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_motion"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create texture views
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let material_view = material.create_view(&wgpu::TextureViewDescriptor::default());
        let motion_view = motion.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group layout for reading (texture sampling in lighting pass)
        let read_bind_group_layout = Self::create_read_bind_group_layout(device);

        // Create bind group for reading
        let read_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer_read_bind_group"),
            layout: &read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&motion_view),
                },
            ],
        });

        // Create bind group layout for writing (storage textures in compute)
        let storage_bind_group_layout = Self::create_storage_bind_group_layout(device);

        // Create bind group for writing
        let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer_storage_bind_group"),
            layout: &storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&motion_view),
                },
            ],
        });

        Self {
            albedo,
            normal,
            depth,
            material,
            motion,
            albedo_view,
            normal_view,
            depth_view,
            material_view,
            motion_view,
            read_bind_group_layout,
            read_bind_group,
            storage_bind_group_layout,
            storage_bind_group,
            width,
            height,
        }
    }

    /// Resize G-buffer to new dimensions
    ///
    /// Recreates all textures and bind groups with the new size.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return; // No resize needed
        }

        *self = Self::new(device, width, height);
    }

    /// Get all texture views
    pub fn views(&self) -> GBufferViews<'_> {
        GBufferViews {
            albedo: &self.albedo_view,
            normal: &self.normal_view,
            depth: &self.depth_view,
            material: &self.material_view,
            motion: &self.motion_view,
        }
    }

    /// Get bind group layout for reading G-buffer (texture sampling)
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.read_bind_group_layout
    }

    /// Get bind group for reading G-buffer
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.read_bind_group
    }

    /// Get bind group layout for writing G-buffer (storage textures)
    pub fn storage_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.storage_bind_group_layout
    }

    /// Get bind group for writing G-buffer
    pub fn storage_bind_group(&self) -> &wgpu::BindGroup {
        &self.storage_bind_group
    }

    /// Get current width
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get current height
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Create bind group layout for reading (texture sampling in fragment shaders)
    fn create_read_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuffer_read_bind_group_layout"),
            entries: &[
                // Binding 0: Albedo texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 1: Normal texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Material texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: Motion vectors texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for writing (storage textures in compute shaders)
    fn create_storage_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuffer_storage_bind_group_layout"),
            entries: &[
                // Binding 0: Albedo storage texture
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
                // Binding 1: Normal storage texture
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
                // Binding 2: Depth storage texture
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
                // Binding 3: Material storage texture
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
                // Binding 4: Motion vectors storage texture
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
        })
    }
}

/// References to G-buffer texture views
#[derive(Debug, Copy, Clone)]
pub struct GBufferViews<'a> {
    /// Albedo texture view
    pub albedo: &'a TextureView,
    /// Normal + roughness texture view
    pub normal: &'a TextureView,
    /// Linear depth texture view
    pub depth: &'a TextureView,
    /// Material properties texture view
    pub material: &'a TextureView,
    /// Motion vectors texture view
    pub motion: &'a TextureView,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbuffer_creation() {
        // Create a test device
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to find adapter");

        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_textures_per_shader_stage: 5,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            },
        ))
        .expect("Failed to create device");

        // Create G-buffer
        let gbuffer = GBuffer::new(&device, 1920, 1080);

        assert_eq!(gbuffer.width(), 1920);
        assert_eq!(gbuffer.height(), 1080);

        // Verify views are accessible
        let views = gbuffer.views();
        assert!(!std::ptr::eq(views.albedo, views.normal));
    }

    #[test]
    fn test_gbuffer_resize() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to find adapter");

        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_textures_per_shader_stage: 5,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            },
        ))
        .expect("Failed to create device");

        let mut gbuffer = GBuffer::new(&device, 1920, 1080);
        gbuffer.resize(&device, 2560, 1440);

        assert_eq!(gbuffer.width(), 2560);
        assert_eq!(gbuffer.height(), 1440);
    }
}
