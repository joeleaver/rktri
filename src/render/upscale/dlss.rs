//! DLSS Super Resolution integration
//!
//! Wraps the dlss_wgpu crate to provide DLSS upscaling for the render pipeline.
//! Feature-gated behind the `dlss` cargo feature.

use std::sync::{Arc, Mutex};
use dlss_wgpu::{
    DlssSdk, DlssFeatureFlags, DlssPerfQualityMode, DlssError,
    super_resolution::{DlssSuperResolution, DlssSuperResolutionRenderParameters, DlssSuperResolutionExposure},
};
use uuid::Uuid;

/// Hardcoded project ID for rktri
pub const RKTRI_PROJECT_ID: Uuid = Uuid::from_bytes([
    0x72, 0x6b, 0x74, 0x72, 0x69, 0x2d, 0x76, 0x6f,
    0x78, 0x65, 0x6c, 0x2d, 0x64, 0x6c, 0x73, 0x73,
]);

/// DLSS quality mode for the upscaler
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DlssQuality {
    /// DLSS chooses optimal quality/performance balance
    Auto,
    /// Highest quality - 1.5x upscale
    Quality,
    /// Balanced quality/performance - 1.7x upscale
    Balanced,
    /// Performance mode - 2x upscale
    Performance,
    /// Ultra Performance - 3x upscale (best for photorealistic inference experiment)
    UltraPerformance,
    /// DLAA - no upscaling, just anti-aliasing at native res
    Dlaa,
}

impl DlssQuality {
    fn to_dlss_mode(self) -> DlssPerfQualityMode {
        match self {
            Self::Auto => DlssPerfQualityMode::Auto,
            Self::Quality => DlssPerfQualityMode::Quality,
            Self::Balanced => DlssPerfQualityMode::Balanced,
            Self::Performance => DlssPerfQualityMode::Performance,
            Self::UltraPerformance => DlssPerfQualityMode::UltraPerformance,
            Self::Dlaa => DlssPerfQualityMode::Dlaa,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Quality => "Quality",
            Self::Balanced => "Balanced",
            Self::Performance => "Performance",
            Self::UltraPerformance => "Ultra Performance",
            Self::Dlaa => "DLAA",
        }
    }
}

/// DLSS upscaler wrapping dlss_wgpu
pub struct DlssUpscaler {
    sdk: Arc<Mutex<DlssSdk>>,
    context: DlssSuperResolution,
    quality: DlssQuality,
    frame_number: u32,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    render_resolution: [u32; 2],
    upscaled_resolution: [u32; 2],
}

impl DlssUpscaler {
    /// Create a new DLSS upscaler
    ///
    /// # Arguments
    /// * `sdk` - The DLSS SDK (created during GPU context init)
    /// * `device` - wgpu device
    /// * `queue` - wgpu queue
    /// * `display_width` - Target display/window width
    /// * `display_height` - Target display/window height
    /// * `quality` - DLSS quality mode
    pub fn new(
        sdk: Arc<Mutex<DlssSdk>>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        display_width: u32,
        display_height: u32,
        quality: DlssQuality,
    ) -> Result<Self, DlssError> {
        let upscaled_resolution = [display_width, display_height];

        let context = DlssSuperResolution::new(
            upscaled_resolution,
            quality.to_dlss_mode(),
            DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::AutoExposure | DlssFeatureFlags::LowResolutionMotionVectors,
            sdk.clone(),
            device,
            queue,
        )?;

        let render_resolution = context.render_resolution();

        // Create output texture at display resolution (HDR for pre-tonemap DLSS)
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dlss_output"),
            size: wgpu::Extent3d {
                width: display_width,
                height: display_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        log::info!(
            "DLSS initialized: {} mode, render {}x{} -> upscale {}x{}",
            quality.name(),
            render_resolution[0], render_resolution[1],
            display_width, display_height,
        );

        Ok(Self {
            sdk,
            context,
            quality,
            frame_number: 0,
            output_texture,
            output_view,
            render_resolution,
            upscaled_resolution,
        })
    }

    /// Run the DLSS upscaling pass
    ///
    /// Returns a CommandBuffer that MUST be submitted immediately after the main encoder.
    /// Usage: `queue.submit([encoder.finish(), dlss_cmd_buf])`
    ///
    /// # Arguments
    /// * `encoder` - The main command encoder (DLSS will add resource transitions to it)
    /// * `adapter` - wgpu adapter reference
    /// * `color_view` - HDR lit texture view (at render resolution)
    /// * `depth_view` - Depth buffer view (at render resolution)
    /// * `motion_view` - Motion vectors view (at render resolution)
    /// * `jitter` - Sub-pixel jitter that was applied to the camera [x, y]
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        adapter: &wgpu::Adapter,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        motion_view: &wgpu::TextureView,
        jitter: [f32; 2],
    ) -> Result<wgpu::CommandBuffer, DlssError> {
        let params = DlssSuperResolutionRenderParameters {
            color: color_view,
            depth: depth_view,
            motion_vectors: motion_view,
            exposure: DlssSuperResolutionExposure::Automatic,
            bias: None,
            dlss_output: &self.output_view,
            reset: false,
            jitter_offset: jitter,
            partial_texture_size: Some(self.render_resolution),
            motion_vector_scale: None,
        };

        let cmd_buf = self.context.render(params, encoder, adapter)?;
        self.frame_number += 1;
        Ok(cmd_buf)
    }

    /// Get suggested jitter for the current frame
    pub fn suggested_jitter(&self) -> [f32; 2] {
        self.context.suggested_jitter(self.frame_number, self.render_resolution)
    }

    /// Get the render resolution DLSS wants (input resolution)
    pub fn render_resolution(&self) -> [u32; 2] {
        self.render_resolution
    }

    /// Get the upscaled resolution (output resolution)
    pub fn upscaled_resolution(&self) -> [u32; 2] {
        self.upscaled_resolution
    }

    /// Get the output texture view (for reading in subsequent passes)
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Get the output texture (for resource transitions)
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Get current quality mode
    pub fn quality(&self) -> DlssQuality {
        self.quality
    }

    /// Get current frame number
    pub fn frame_number(&self) -> u32 {
        self.frame_number
    }

    /// Reset temporal history (call on camera cuts or major scene changes)
    pub fn reset_history(&mut self) {
        self.frame_number = 0;
    }

    /// Recreate the DLSS context with new settings
    ///
    /// Call this when display resolution or quality mode changes.
    pub fn recreate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        display_width: u32,
        display_height: u32,
        quality: DlssQuality,
    ) -> Result<(), DlssError> {
        let upscaled_resolution = [display_width, display_height];

        let context = DlssSuperResolution::new(
            upscaled_resolution,
            quality.to_dlss_mode(),
            DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::AutoExposure | DlssFeatureFlags::LowResolutionMotionVectors,
            self.sdk.clone(),
            device,
            queue,
        )?;

        let render_resolution = context.render_resolution();

        // Recreate output texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dlss_output"),
            size: wgpu::Extent3d {
                width: display_width,
                height: display_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.context = context;
        self.quality = quality;
        self.output_texture = output_texture;
        self.output_view = output_view;
        self.render_resolution = render_resolution;
        self.upscaled_resolution = upscaled_resolution;
        self.frame_number = 0;

        log::info!(
            "DLSS recreated: {} mode, render {}x{} -> upscale {}x{}",
            quality.name(),
            render_resolution[0], render_resolution[1],
            display_width, display_height,
        );

        Ok(())
    }
}
