//! GPU context management using wgpu

use std::sync::Arc;
use winit::window::Window;
use crate::core::error::Error;

#[cfg(feature = "dlss")]
use std::sync::Mutex;

/// DLSS support state (only available with dlss feature)
#[cfg(feature = "dlss")]
pub struct DlssSupport {
    /// The DLSS SDK instance
    pub sdk: Arc<Mutex<dlss_wgpu::DlssSdk>>,
    /// Which DLSS features are supported
    pub feature_support: dlss_wgpu::FeatureSupport,
}

/// GPU rendering context
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    /// DLSS support (None if not available or feature not enabled)
    #[cfg(feature = "dlss")]
    pub dlss: Option<DlssSupport>,
}

impl GpuContext {
    /// Create new GPU context from window
    pub async fn new(window: Arc<Window>) -> Result<Self, Error> {
        let instance_desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        };

        // Try DLSS-aware initialization when feature is enabled
        #[cfg(feature = "dlss")]
        let (instance, mut feature_support) = {
            let project_id = crate::render::upscale::dlss::RKTRI_PROJECT_ID;
            let mut fs = dlss_wgpu::FeatureSupport::default();
            match dlss_wgpu::create_instance(project_id, &instance_desc, &mut fs) {
                Ok(inst) => {
                    log::info!("DLSS: Instance created with extensions");
                    (inst, fs)
                }
                Err(e) => {
                    log::warn!("DLSS: Instance creation failed ({}), falling back to standard", e);
                    let mut fs = dlss_wgpu::FeatureSupport::default();
                    fs.super_resolution_supported = false;
                    fs.ray_reconstruction_supported = false;
                    (wgpu::Instance::new(&instance_desc), fs)
                }
            }
        };

        #[cfg(not(feature = "dlss"))]
        let instance = wgpu::Instance::new(&instance_desc);

        let surface = instance.create_surface(window.clone())
            .map_err(|e| Error::Gpu(e.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Gpu(format!("No suitable adapter found: {:?}", e)))?;

        let adapter_limits = adapter.limits();

        let device_desc = wgpu::DeviceDescriptor {
            label: Some("rktri_device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits {
                max_storage_textures_per_shader_stage: 8,
                max_storage_buffers_per_shader_stage: 10, // 9 in octree bind group + headroom
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: Default::default(),
            trace: Default::default(),
        };

        // Try DLSS-aware device creation
        #[cfg(feature = "dlss")]
        let (device, queue, dlss) = {
            let project_id = crate::render::upscale::dlss::RKTRI_PROJECT_ID;
            if feature_support.super_resolution_supported {
                match dlss_wgpu::request_device(project_id, &adapter, &device_desc, &mut feature_support) {
                    Ok((dev, q)) => {
                        log::info!("DLSS: Device created with extensions (SR supported: {})",
                            feature_support.super_resolution_supported);
                        // Create SDK
                        match dlss_wgpu::DlssSdk::new(project_id, dev.clone()) {
                            Ok(sdk) => {
                                log::info!("DLSS: SDK initialized successfully");
                                (dev, q, Some(DlssSupport {
                                    sdk,
                                    feature_support,
                                }))
                            }
                            Err(e) => {
                                log::warn!("DLSS: SDK creation failed ({}), DLSS disabled", e);
                                (dev, q, None)
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("DLSS: Device creation failed ({}), falling back to standard", e);
                        let (dev, q) = adapter.request_device(&device_desc).await
                            .map_err(|e| Error::Gpu(e.to_string()))?;
                        (dev, q, None)
                    }
                }
            } else {
                log::info!("DLSS: Super Resolution not supported on this system");
                let (dev, q) = adapter.request_device(&device_desc).await
                    .map_err(|e| Error::Gpu(e.to_string()))?;
                (dev, q, None)
            }
        };

        #[cfg(not(feature = "dlss"))]
        let (device, queue) = adapter
            .request_device(&device_desc)
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;

        log::info!("GPU buffer limits: max_buffer_size={}MB, max_storage_binding={}MB",
            adapter_limits.max_buffer_size / 1024 / 1024,
            adapter_limits.max_storage_buffer_binding_size / 1024 / 1024);

        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: capabilities.formats[0],
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            config,
            #[cfg(feature = "dlss")]
            dlss,
        })
    }

    /// Resize the surface
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Get current surface texture for rendering
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, Error> {
        self.surface
            .get_current_texture()
            .map_err(|e| Error::Gpu(e.to_string()))
    }

    /// Get surface size
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Get surface format
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Check if DLSS is available
    #[cfg(feature = "dlss")]
    pub fn dlss_available(&self) -> bool {
        self.dlss.is_some()
    }
}
