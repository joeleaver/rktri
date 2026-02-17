//! Rktri - Voxel Engine

use std::sync::{Arc, Mutex as StdMutex};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Window, WindowId},
};

use rktri::core::{
    camera::Camera,
    camera_controller::FpsCameraController,
    input::InputState,
    logging,
    time::FrameTimer,
};
use rktri::render::{
    context::GpuContext,
    buffer::{OctreeBuffer, CameraBuffer},
    pipeline::{SvoTracePipeline, DisplayPipeline, TraceParams, LightingPipeline, LightingUniforms, ShadowPipeline, ShadowParams, SkyParams, DebugParams, GodRaysPipeline, GodRaysParams, TonemapPipeline, TonemapParams, CloudPipeline, CloudParams},
    texture::GBuffer,
    culling::ChunkCuller,
};
use rktri::atmosphere::{AtmosphereSystem, AtmosphereConfig};
use rktri::grass::{GrassSystem, GrassConfig};
use rktri::grass::GrassCell;
use rktri::mask::MaskOctree;
use rktri::scene::SceneConfig;
use rktri::voxel::StreamingManager;
use rktri::streaming::disk_io;
use std::path::PathBuf;

#[cfg(feature = "dlss")]
use rktri::render::upscale::DlssUpscaler;
#[cfg(feature = "dlss")]
use rktri::render::upscale::dlss::DlssQuality;
#[cfg(feature = "dlss")]
use rktri::render::context::DlssSupport;

struct RenderResources {
    camera_buffer: CameraBuffer,
    octree_buffer: OctreeBuffer,
    svo_pipeline: SvoTracePipeline,
    shadow_pipeline: ShadowPipeline,
    godrays_pipeline: GodRaysPipeline,
    cloud_pipeline: CloudPipeline,
    lighting_pipeline: LightingPipeline,
    display_pipeline: DisplayPipeline,
    gbuffer: GBuffer,
    shadow_texture: wgpu::Texture,
    shadow_texture_view: wgpu::TextureView,
    godrays_texture: wgpu::Texture,
    godrays_texture_view: wgpu::TextureView,
    cloud_texture: wgpu::Texture,
    cloud_texture_view: wgpu::TextureView,
    lit_texture: wgpu::Texture,
    lit_texture_view: wgpu::TextureView,
    tonemap_pipeline: TonemapPipeline,
    post_texture: wgpu::Texture,
    post_texture_view: wgpu::TextureView,
    tonemap_input_bind_group: wgpu::BindGroup,
    tonemap_output_bind_group: wgpu::BindGroup,
    gbuffer_output_bind_group: wgpu::BindGroup,
    shadow_gbuffer_bind_group: wgpu::BindGroup,
    shadow_output_bind_group: wgpu::BindGroup,
    godrays_input_bind_group: wgpu::BindGroup,
    godrays_output_bind_group: wgpu::BindGroup,
    cloud_input_bind_group: wgpu::BindGroup,
    cloud_output_bind_group: wgpu::BindGroup,
    lighting_gbuffer_bind_group: wgpu::BindGroup,
    lighting_output_bind_group: wgpu::BindGroup,
    display_bind_group: wgpu::BindGroup,
    // DLSS upscaling (feature-gated)
    #[cfg(feature = "dlss")]
    dlss_upscaler: Option<DlssUpscaler>,
    // When DLSS is active, post_texture and bind groups are at window resolution
    #[cfg(feature = "dlss")]
    dlss_post_texture: Option<wgpu::Texture>,
    #[cfg(feature = "dlss")]
    dlss_post_texture_view: Option<wgpu::TextureView>,
    #[cfg(feature = "dlss")]
    dlss_tonemap_input_bind_group: Option<wgpu::BindGroup>,
    #[cfg(feature = "dlss")]
    dlss_tonemap_output_bind_group: Option<wgpu::BindGroup>,
    #[cfg(feature = "dlss")]
    dlss_display_bind_group: Option<wgpu::BindGroup>,
    #[cfg(feature = "dlss")]
    dlss_norm_depth_texture: Option<wgpu::Texture>,
    #[cfg(feature = "dlss")]
    dlss_norm_depth_view: Option<wgpu::TextureView>,
    #[cfg(feature = "dlss")]
    dlss_norm_depth_pipeline: Option<wgpu::ComputePipeline>,
    #[cfg(feature = "dlss")]
    dlss_norm_depth_bind_group: Option<wgpu::BindGroup>,
    chunk_count: u32,
    world_extent: f32,
    world_offset: glam::Vec3,
    chunk_bounds: Vec<[f32; 4]>, // [world_min_x, world_min_y, world_min_z, root_size]
    #[allow(dead_code)]
    chunk_culler: ChunkCuller,
    // Chunk grid acceleration params (uploaded once, used per frame)
    grid_min: [i32; 3],
    grid_size: [u32; 3],
    #[allow(dead_code)]
    streaming: StreamingManager,
    // Debug: loaded chunk info for GetChunkInfo command
    loaded_chunks: std::collections::HashMap<(i32, i32, i32), ChunkDebugInfo>,
    loaded_chunks_grass: std::collections::HashMap<(i32, i32, i32), GrassDebugInfo>,
}

impl RenderResources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat, width: u32, height: u32, render_scale: f32, world_path: &PathBuf, view_distance: f32) -> Self {
        use rktri::voxel::chunk::CHUNK_SIZE;

        // Only path: load pre-generated v3 world from disk (SVDAG pre-compressed)
        struct CompressedEntry {
            coord: (i32, i32, i32),
            world_min: [f32; 3],
            root_size: f32,
            layer_id: u32,
            octree: rktri::voxel::svo::Octree,
        }

        log::info!("Loading pre-generated v3 world from: {}", world_path.display());
        let chunks = Self::load_chunks_from_disk(world_path, view_distance);

        // Keep layers separate - each chunk is stored with its actual layer_id
        // The raycaster will check all layer indices at each grid cell
        let mut merged_entries: Vec<CompressedEntry> = Vec::with_capacity(chunks.len());

        for (coord, octree, layer_id) in &chunks {
            let world_min = [
                coord.x as f32 * CHUNK_SIZE as f32,
                coord.y as f32 * CHUNK_SIZE as f32,
                coord.z as f32 * CHUNK_SIZE as f32,
            ];
            let root_size = octree.root_size();

            merged_entries.push(CompressedEntry {
                coord: (coord.x, coord.y, coord.z),
                world_min,
                root_size,
                layer_id: *layer_id,
                octree: octree.clone(),
            });
        }

        let compressed_entries = merged_entries;

        // Sum up total nodes/bricks for buffer allocation
        let mut total_nodes = 0usize;
        let mut total_bricks = 0usize;
        let mut world_max = glam::Vec3::ZERO;
        for entry in &compressed_entries {
            total_nodes += entry.octree.node_count();
            total_bricks += entry.octree.brick_count();
            let entry_max = glam::Vec3::from(entry.world_min) + glam::Vec3::splat(entry.root_size);
            world_max = world_max.max(entry_max);
        }
        let chunk_count = compressed_entries.len() as u32;
        let world_extent = world_max.x.max(world_max.y).max(world_max.z);

        log::info!("World: {} chunks, {} total nodes, {} total bricks, extent={:.1}m",
            chunk_count, total_nodes, total_bricks, world_extent);

        // Create buffers
        let camera_buffer = CameraBuffer::new(device);
        let mut octree_buffer = OctreeBuffer::new(device,
            total_nodes.max(1) as u32,
            total_bricks.max(1) as u32);

        // Incrementally upload each chunk to GPU
        let mut all_chunk_infos: Vec<rktri::render::buffer::octree_buffer::GpuChunkInfo> = Vec::with_capacity(compressed_entries.len());
        let mut loaded_chunks: std::collections::HashMap<(i32, i32, i32), ChunkDebugInfo> = std::collections::HashMap::new();
        for entry in &compressed_entries {
            let info = octree_buffer.upload_chunk_incremental(
                queue,
                &entry.octree,
                entry.world_min,
                entry.root_size,
                entry.layer_id,
                0,
            );
            loaded_chunks.insert(entry.coord, ChunkDebugInfo {
                node_count: entry.octree.node_count() as u32,
                brick_count: entry.octree.brick_count() as u32,
                world_min: entry.world_min,
            });
            all_chunk_infos.push(info);
        }

        // Initialize feedback header
        let feedback_header: [u32; 4] = [0, rktri::voxel::streaming::feedback::MAX_FEEDBACK_REQUESTS, 0, 0];
        queue.write_buffer(octree_buffer.feedback_header_buffer(), 0, bytemuck::cast_slice(&feedback_header));

        // Write ALL chunk infos to buffer once (grid-based ray marching uses static indices)
        octree_buffer.update_chunk_infos(queue, &all_chunk_infos);

        // Build 3D chunk grid: maps chunk coordinates → LayerDescriptor for DDA ray marching
        let chunk_size_f = rktri::voxel::chunk::CHUNK_SIZE as f32;
        let (grid_min, grid_size, grid_data, layer_data) = Self::build_chunk_grid(&all_chunk_infos, chunk_size_f);
        octree_buffer.upload_chunk_grid(device, queue, &grid_data, &layer_data);

        // Debug: count cells with multiple layers
        let multi_layer_cells = grid_data.iter().filter(|d| d.layer_count > 1).count();
        let max_layers = grid_data.iter().map(|d| d.layer_count).max().unwrap_or(0);
        log::info!("Chunk grid: min={:?}, size={:?}, cells={}, layer_indices={}, multi_layer_cells={}, max_layers_per_cell={}",
            grid_min, grid_size,
            grid_size[0] as usize * grid_size[1] as usize * grid_size[2] as usize,
            layer_data.len(), multi_layer_cells, max_layers);

        // Load and upload grass masks
        let grass_masks_map = Self::load_grass_masks_from_disk(world_path);
        let mut loaded_chunks_grass: std::collections::HashMap<(i32, i32, i32), GrassDebugInfo> = std::collections::HashMap::new();
        if !grass_masks_map.is_empty() {
            // Build per-chunk mask array matching compressed_entries order
            let chunk_size_i = rktri::voxel::chunk::CHUNK_SIZE as f32;
            let masks_ordered: Vec<Option<&MaskOctree<GrassCell>>> = compressed_entries.iter()
                .map(|entry| {
                    let cx = (entry.world_min[0] / chunk_size_i).round() as i32;
                    let cy = (entry.world_min[1] / chunk_size_i).round() as i32;
                    let cz = (entry.world_min[2] / chunk_size_i).round() as i32;
                    grass_masks_map.get(&(cx, cy, cz))
                })
                .collect();
            let matched = masks_ordered.iter().filter(|m| m.is_some()).count();
            log::info!("Grass masks: {} loaded, {} matched to chunks", grass_masks_map.len(), matched);

            // Pack masks to GPU format and upload
            let (infos, nodes, values) = rktri::render::buffer::pack_grass_masks(&masks_ordered);

            // Track grass info per chunk
            for (i, entry) in compressed_entries.iter().enumerate() {
                if let Some(mask) = masks_ordered[i] {
                    loaded_chunks_grass.insert(entry.coord, GrassDebugInfo {
                        node_count: mask.node_count() as u32,
                    });
                }
            }

            octree_buffer.upload_grass_masks(device, queue, &infos, &nodes, &values);
        }

        // Store chunk bounds for terrain height queries
        let chunk_bounds: Vec<[f32; 4]> = all_chunk_infos.iter()
            .map(|c| [c.world_min[0], c.world_min[1], c.world_min[2], c.root_size])
            .collect();

        let chunk_culler = ChunkCuller::new(all_chunk_infos);

        // Create streaming manager (simplified - uses first chunk for now)
        let pool_size = (total_bricks as u32).min(1_000_000);
        let streaming = StreamingManager::new(device, total_bricks.max(1) as u32, pool_size);

        let world_offset = glam::Vec3::ZERO; // Chunks have world-space positions

        // Create pipelines
        let svo_pipeline = SvoTracePipeline::new(device, &camera_buffer, &octree_buffer);
        let shadow_pipeline = ShadowPipeline::new(device, &camera_buffer, &octree_buffer);
        let godrays_pipeline = GodRaysPipeline::new(device, &camera_buffer);
        let cloud_pipeline = CloudPipeline::new(device, &camera_buffer);
        let lighting_pipeline = LightingPipeline::new(device, &camera_buffer);
        let display_pipeline = DisplayPipeline::new(device, surface_format);
        let tonemap_pipeline = TonemapPipeline::new(device);

        // Internal render resolution (scaled from window size)
        let render_width = ((width as f32 * render_scale) as u32).max(1);
        let render_height = ((height as f32 * render_scale) as u32).max(1);

        // Create G-buffer at render resolution
        let gbuffer = GBuffer::new(device, render_width, render_height);

        // Create shadow texture at half of render resolution (R32Float shadow mask)
        let (shadow_texture, shadow_texture_view) = Self::create_shadow_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));

        // Create god rays texture at half of render resolution (R32Float scattered light intensity)
        let (godrays_texture, godrays_texture_view) = Self::create_godrays_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));

        // Create cloud texture at half of render resolution (Rgba16Float cloud color + alpha)
        let (cloud_texture, cloud_texture_view) = Self::create_cloud_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));

        // Create lit texture at render resolution (HDR output from lighting pass)
        let (lit_texture, lit_texture_view) = Self::create_lit_texture(device, render_width, render_height);

        // Create post-process texture at render resolution (LDR output from tone mapping)
        let (post_texture, post_texture_view) = Self::create_post_texture(device, render_width, render_height);

        // Create bind groups
        let gbuffer_views = gbuffer.views();
        let gbuffer_output_bind_group = svo_pipeline.create_output_bind_group(
            device,
            gbuffer_views.albedo,
            gbuffer_views.normal,
            gbuffer_views.depth,
            gbuffer_views.material,
            gbuffer_views.motion,
        );

        let shadow_gbuffer_bind_group = shadow_pipeline.create_gbuffer_bind_group(
            device,
            gbuffer_views.depth,
            gbuffer_views.normal,
            gbuffer_views.material,
        );

        let shadow_output_bind_group = shadow_pipeline.create_output_bind_group(device, &shadow_texture_view);

        let godrays_input_bind_group = godrays_pipeline.create_input_bind_group(
            device,
            &shadow_texture_view,
            gbuffer_views.depth,
        );
        let godrays_output_bind_group = godrays_pipeline.create_output_bind_group(device, &godrays_texture_view);

        let cloud_input_bind_group = cloud_pipeline.create_input_bind_group(
            device,
            gbuffer_views.depth,
        );
        let cloud_output_bind_group = cloud_pipeline.create_output_bind_group(device, &cloud_texture_view);

        let lighting_gbuffer_bind_group = lighting_pipeline.create_gbuffer_bind_group(
            device,
            gbuffer_views.albedo,
            gbuffer_views.normal,
            gbuffer_views.depth,
            gbuffer_views.material,
            &shadow_texture_view,
            &godrays_texture_view,
            &cloud_texture_view,
        );

        let lighting_output_bind_group = lighting_pipeline.create_output_bind_group(device, &lit_texture_view);

        let tonemap_input_bind_group = tonemap_pipeline.create_input_bind_group(device, &lit_texture_view);
        let tonemap_output_bind_group = tonemap_pipeline.create_output_bind_group(device, &post_texture_view);

        // Display reads from post_texture (tonemapped LDR) instead of lit_texture (HDR)
        let display_bind_group = display_pipeline.create_bind_group(device, &post_texture_view);

        Self {
            camera_buffer,
            octree_buffer,
            svo_pipeline,
            shadow_pipeline,
            godrays_pipeline,
            cloud_pipeline,
            lighting_pipeline,
            display_pipeline,
            gbuffer,
            shadow_texture,
            shadow_texture_view,
            godrays_texture,
            godrays_texture_view,
            cloud_texture,
            cloud_texture_view,
            lit_texture,
            lit_texture_view,
            tonemap_pipeline,
            post_texture,
            post_texture_view,
            tonemap_input_bind_group,
            tonemap_output_bind_group,
            gbuffer_output_bind_group,
            shadow_gbuffer_bind_group,
            shadow_output_bind_group,
            godrays_input_bind_group,
            godrays_output_bind_group,
            cloud_input_bind_group,
            cloud_output_bind_group,
            lighting_gbuffer_bind_group,
            lighting_output_bind_group,
            display_bind_group,
            #[cfg(feature = "dlss")]
            dlss_upscaler: None,
            #[cfg(feature = "dlss")]
            dlss_post_texture: None,
            #[cfg(feature = "dlss")]
            dlss_post_texture_view: None,
            #[cfg(feature = "dlss")]
            dlss_tonemap_input_bind_group: None,
            #[cfg(feature = "dlss")]
            dlss_tonemap_output_bind_group: None,
            #[cfg(feature = "dlss")]
            dlss_display_bind_group: None,
            #[cfg(feature = "dlss")]
            dlss_norm_depth_texture: None,
            #[cfg(feature = "dlss")]
            dlss_norm_depth_view: None,
            #[cfg(feature = "dlss")]
            dlss_norm_depth_pipeline: None,
            #[cfg(feature = "dlss")]
            dlss_norm_depth_bind_group: None,
            chunk_count,
            world_extent,
            world_offset,
            chunk_bounds,
            chunk_culler,
            grid_min,
            grid_size,
            streaming,
            loaded_chunks,
            loaded_chunks_grass,
        }
    }

    /// Build a 3D grid mapping chunk coordinates to LayerDescriptor + layer_data.
    /// Returns (grid_min, grid_size, grid_data, layer_data).
    /// Each grid cell stores a descriptor pointing into layer_data where octree indices are stored.
    fn build_chunk_grid(
        chunk_infos: &[rktri::render::buffer::octree_buffer::GpuChunkInfo],
        chunk_size: f32,
    ) -> ([i32; 3], [u32; 3], Vec<rktri::render::buffer::LayerDescriptor>, Vec<u32>) {
        use rktri::render::buffer::LayerDescriptor;

        if chunk_infos.is_empty() {
            return ([0; 3], [1, 1, 1], vec![LayerDescriptor { base_index: 0, layer_count: 0 }], vec![]);
        }

        // Find bounds of all chunk coordinates
        let mut min_coord = [i32::MAX; 3];
        let mut max_coord = [i32::MIN; 3];

        for info in chunk_infos {
            let cx = (info.world_min[0] / chunk_size).floor() as i32;
            let cy = (info.world_min[1] / chunk_size).floor() as i32;
            let cz = (info.world_min[2] / chunk_size).floor() as i32;

            min_coord[0] = min_coord[0].min(cx);
            min_coord[1] = min_coord[1].min(cy);
            min_coord[2] = min_coord[2].min(cz);
            max_coord[0] = max_coord[0].max(cx);
            max_coord[1] = max_coord[1].max(cy);
            max_coord[2] = max_coord[2].max(cz);
        }

        let size_x = (max_coord[0] - min_coord[0] + 1) as u32;
        let size_y = (max_coord[1] - min_coord[1] + 1) as u32;
        let size_z = (max_coord[2] - min_coord[2] + 1) as u32;
        let total_cells = size_x as usize * size_y as usize * size_z as usize;

        // First pass: count layers per cell
        let mut layer_counts = vec![0u32; total_cells];
        for info in chunk_infos {
            let cx = (info.world_min[0] / chunk_size).floor() as i32;
            let cy = (info.world_min[1] / chunk_size).floor() as i32;
            let cz = (info.world_min[2] / chunk_size).floor() as i32;

            let gx = (cx - min_coord[0]) as u32;
            let gy = (cy - min_coord[1]) as u32;
            let gz = (cz - min_coord[2]) as u32;

            let flat_idx = gx + gy * size_x + gz * size_x * size_y;
            layer_counts[flat_idx as usize] += 1;
        }

        // Build descriptors with base indices
        let mut descriptors = vec![LayerDescriptor { base_index: 0, layer_count: 0 }; total_cells];
        let mut current_base = 0u32;
        for i in 0..total_cells {
            descriptors[i].base_index = current_base;
            descriptors[i].layer_count = layer_counts[i];
            current_base += layer_counts[i];
        }

        // Second pass: fill layer data
        let mut layer_data = vec![0u32; current_base as usize];
        let mut cell_indices = vec![0u32; total_cells]; // tracks position within each cell's data

        for (idx, info) in chunk_infos.iter().enumerate() {
            let cx = (info.world_min[0] / chunk_size).floor() as i32;
            let cy = (info.world_min[1] / chunk_size).floor() as i32;
            let cz = (info.world_min[2] / chunk_size).floor() as i32;

            let gx = (cx - min_coord[0]) as u32;
            let gy = (cy - min_coord[1]) as u32;
            let gz = (cz - min_coord[2]) as u32;

            let flat_idx = gx + gy * size_x + gz * size_x * size_y;
            let offset = descriptors[flat_idx as usize].base_index + cell_indices[flat_idx as usize];
            layer_data[offset as usize] = idx as u32;
            cell_indices[flat_idx as usize] += 1;
        }

        (min_coord, [size_x, size_y, size_z], descriptors, layer_data)
    }

    fn create_shadow_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_godrays_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("godrays_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_cloud_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_lit_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lit_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_post_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, render_scale: f32) {
        // Internal render resolution (scaled from window size)
        let render_width = ((width as f32 * render_scale) as u32).max(1);
        let render_height = ((height as f32 * render_scale) as u32).max(1);

        // Resize G-buffer at render resolution
        self.gbuffer.resize(device, render_width, render_height);

        // Recreate shadow texture at half of render resolution
        let (shadow_texture, shadow_view) = Self::create_shadow_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        self.shadow_texture = shadow_texture;
        self.shadow_texture_view = shadow_view;

        // Recreate god rays texture at half of render resolution
        let (godrays_texture, godrays_view) = Self::create_godrays_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        self.godrays_texture = godrays_texture;
        self.godrays_texture_view = godrays_view;

        // Recreate cloud texture at half of render resolution
        let (cloud_texture, cloud_view) = Self::create_cloud_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        self.cloud_texture = cloud_texture;
        self.cloud_texture_view = cloud_view;

        // Recreate lit texture at render resolution
        let (texture, view) = Self::create_lit_texture(device, render_width, render_height);
        self.lit_texture = texture;
        self.lit_texture_view = view;

        // Recreate all bind groups
        let gbuffer_views = self.gbuffer.views();
        self.gbuffer_output_bind_group = self.svo_pipeline.create_output_bind_group(
            device,
            gbuffer_views.albedo,
            gbuffer_views.normal,
            gbuffer_views.depth,
            gbuffer_views.material,
            gbuffer_views.motion,
        );

        self.shadow_gbuffer_bind_group = self.shadow_pipeline.create_gbuffer_bind_group(
            device,
            gbuffer_views.depth,
            gbuffer_views.normal,
            gbuffer_views.material,
        );

        self.shadow_output_bind_group = self.shadow_pipeline.create_output_bind_group(device, &self.shadow_texture_view);

        self.godrays_input_bind_group = self.godrays_pipeline.create_input_bind_group(
            device,
            &self.shadow_texture_view,
            gbuffer_views.depth,
        );
        self.godrays_output_bind_group = self.godrays_pipeline.create_output_bind_group(device, &self.godrays_texture_view);

        self.cloud_input_bind_group = self.cloud_pipeline.create_input_bind_group(
            device,
            gbuffer_views.depth,
        );
        self.cloud_output_bind_group = self.cloud_pipeline.create_output_bind_group(device, &self.cloud_texture_view);

        self.lighting_gbuffer_bind_group = self.lighting_pipeline.create_gbuffer_bind_group(
            device,
            gbuffer_views.albedo,
            gbuffer_views.normal,
            gbuffer_views.depth,
            gbuffer_views.material,
            &self.shadow_texture_view,
            &self.godrays_texture_view,
            &self.cloud_texture_view,
        );

        self.lighting_output_bind_group = self.lighting_pipeline.create_output_bind_group(device, &self.lit_texture_view);

        // Recreate post-process texture at render resolution
        let (post_texture, post_view) = Self::create_post_texture(device, render_width, render_height);
        self.post_texture = post_texture;
        self.post_texture_view = post_view;

        self.tonemap_input_bind_group = self.tonemap_pipeline.create_input_bind_group(device, &self.lit_texture_view);
        self.tonemap_output_bind_group = self.tonemap_pipeline.create_output_bind_group(device, &self.post_texture_view);

        // Display reads from post_texture (tonemapped LDR) instead of lit_texture (HDR)
        self.display_bind_group = self.display_pipeline.create_bind_group(device, &self.post_texture_view);
    }


    /// Recreate all resolution-dependent textures and bind groups at the given render resolution.
    /// Used when DLSS changes the render resolution, or when DLSS is toggled off to restore
    /// render_scale resolution.
    #[cfg(feature = "dlss")]
    fn recreate_render_targets(&mut self, device: &wgpu::Device, render_width: u32, render_height: u32) {
        // Recreate GBuffer at new render resolution
        self.gbuffer = GBuffer::new(device, render_width, render_height);

        // Recreate intermediate textures
        let (shadow_tex, shadow_view) = Self::create_shadow_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        let (godrays_tex, godrays_view) = Self::create_godrays_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        let (cloud_tex, cloud_view) = Self::create_cloud_texture(device, (render_width / 2).max(1), (render_height / 2).max(1));
        let (lit_tex, lit_view) = Self::create_lit_texture(device, render_width, render_height);
        let (post_tex, post_view) = Self::create_post_texture(device, render_width, render_height);

        // Recreate all bind groups that reference these textures
        let gbuffer_views = self.gbuffer.views();

        self.gbuffer_output_bind_group = self.svo_pipeline.create_output_bind_group(
            device, gbuffer_views.albedo, gbuffer_views.normal, gbuffer_views.depth, gbuffer_views.material, gbuffer_views.motion,
        );
        self.shadow_gbuffer_bind_group = self.shadow_pipeline.create_gbuffer_bind_group(
            device, gbuffer_views.depth, gbuffer_views.normal, gbuffer_views.material,
        );
        self.shadow_output_bind_group = self.shadow_pipeline.create_output_bind_group(device, &shadow_view);
        self.godrays_input_bind_group = self.godrays_pipeline.create_input_bind_group(device, &shadow_view, gbuffer_views.depth);
        self.godrays_output_bind_group = self.godrays_pipeline.create_output_bind_group(device, &godrays_view);
        self.cloud_input_bind_group = self.cloud_pipeline.create_input_bind_group(device, gbuffer_views.depth);
        self.cloud_output_bind_group = self.cloud_pipeline.create_output_bind_group(device, &cloud_view);
        self.lighting_gbuffer_bind_group = self.lighting_pipeline.create_gbuffer_bind_group(
            device, gbuffer_views.albedo, gbuffer_views.normal, gbuffer_views.depth, gbuffer_views.material,
            &shadow_view, &godrays_view, &cloud_view,
        );
        self.lighting_output_bind_group = self.lighting_pipeline.create_output_bind_group(device, &lit_view);
        self.tonemap_input_bind_group = self.tonemap_pipeline.create_input_bind_group(device, &lit_view);
        self.tonemap_output_bind_group = self.tonemap_pipeline.create_output_bind_group(device, &post_view);
        self.display_bind_group = self.display_pipeline.create_bind_group(device, &post_view);

        // Store new textures
        self.shadow_texture = shadow_tex;
        self.shadow_texture_view = shadow_view;
        self.godrays_texture = godrays_tex;
        self.godrays_texture_view = godrays_view;
        self.cloud_texture = cloud_tex;
        self.cloud_texture_view = cloud_view;
        self.lit_texture = lit_tex;
        self.lit_texture_view = lit_view;
        self.post_texture = post_tex;
        self.post_texture_view = post_view;

        log::info!("Render targets recreated at {}x{}", render_width, render_height);
    }

    #[cfg(feature = "dlss")]
    fn create_dlss_depth_normalization(
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        render_width: u32,
        render_height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::ComputePipeline, wgpu::BindGroup) {
        // Create normalized depth texture at render resolution
        let norm_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dlss_normalized_depth"),
            size: wgpu::Extent3d {
                width: render_width.max(1),
                height: render_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let norm_depth_view = norm_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth_normalize_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_normalize.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth_normalize_bg_layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth_normalize_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("depth_normalize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth_normalize_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&norm_depth_view),
                },
            ],
        });

        (norm_depth_texture, norm_depth_view, pipeline, bind_group)
    }

    #[cfg(feature = "dlss")]
    fn init_dlss(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dlss_support: &DlssSupport,
        window_width: u32,
        window_height: u32,
    ) {
        // Drop existing DLSS resources BEFORE creating new upscaler.
        // NGX requires the old feature to be fully released before creating a new one.
        self.dlss_upscaler = None;
        self.dlss_norm_depth_bind_group = None;
        self.dlss_norm_depth_pipeline = None;
        self.dlss_norm_depth_view = None;
        self.dlss_norm_depth_texture = None;
        self.dlss_tonemap_input_bind_group = None;
        self.dlss_tonemap_output_bind_group = None;
        self.dlss_display_bind_group = None;
        self.dlss_post_texture_view = None;
        self.dlss_post_texture = None;

        match DlssUpscaler::new(
            dlss_support.sdk.clone(),
            device,
            queue,
            window_width,
            window_height,
            DlssQuality::Performance,
        ) {
            Ok(upscaler) => {
                let dlss_render_res = upscaler.render_resolution();

                // Only recreate render targets if sizes don't match current textures.
                // This avoids the texture recreation that can break DLSS output state.
                let current_w = self.gbuffer.width();
                let current_h = self.gbuffer.height();
                if current_w != dlss_render_res[0] || current_h != dlss_render_res[1] {
                    self.recreate_render_targets(device, dlss_render_res[0], dlss_render_res[1]);
                }

                // Create depth normalization pass for DLSS (converts linear depth to [0,1])
                let gbuffer_views = self.gbuffer.views();
                let (norm_tex, norm_view, norm_pipeline, norm_bg) = Self::create_dlss_depth_normalization(
                    device, gbuffer_views.depth, dlss_render_res[0], dlss_render_res[1],
                );
                self.dlss_norm_depth_texture = Some(norm_tex);
                self.dlss_norm_depth_view = Some(norm_view);
                self.dlss_norm_depth_pipeline = Some(norm_pipeline);
                self.dlss_norm_depth_bind_group = Some(norm_bg);

                // Create DLSS-specific resources at window resolution
                let (post_tex, post_view) = Self::create_post_texture(device, window_width, window_height);

                let tonemap_in = self.tonemap_pipeline.create_input_bind_group(device, upscaler.output_view());
                let tonemap_out = self.tonemap_pipeline.create_output_bind_group(device, &post_view);
                let display_bg = self.display_pipeline.create_bind_group(device, &post_view);

                self.dlss_post_texture = Some(post_tex);
                self.dlss_post_texture_view = Some(post_view);
                self.dlss_tonemap_input_bind_group = Some(tonemap_in);
                self.dlss_tonemap_output_bind_group = Some(tonemap_out);
                self.dlss_display_bind_group = Some(display_bg);
                self.dlss_upscaler = Some(upscaler);

                log::info!("DLSS: Upscaler initialized in render pipeline");
            }
            Err(e) => {
                log::warn!("DLSS: Failed to create upscaler: {}", e);
                self.dlss_upscaler = None;
                self.dlss_post_texture = None;
                self.dlss_post_texture_view = None;
                self.dlss_tonemap_input_bind_group = None;
                self.dlss_tonemap_output_bind_group = None;
                self.dlss_display_bind_group = None;
                self.dlss_norm_depth_texture = None;
                self.dlss_norm_depth_view = None;
                self.dlss_norm_depth_pipeline = None;
                self.dlss_norm_depth_bind_group = None;
            }
        }
    }

    /// Load individual chunk octrees from disk (v3 format only - SVDAG pre-compressed)
    /// Only loads chunks within view_distance of center (0 = load all)
    /// Returns (coord, octree, layer_id)
    fn load_chunks_from_disk(world_path: &PathBuf, view_distance: f32) -> Vec<(disk_io::ChunkCoord, rktri::voxel::svo::Octree, u32)> {
        use serde_json::Value;
        use rktri::voxel::chunk::CHUNK_SIZE;

        let manifest_path = world_path.join("manifest.json");
        if !manifest_path.exists() {
            log::error!("Manifest file not found: {}", manifest_path.display());
            log::info!("Generate a world first with: cargo run --release --bin generate_world -- --size <N> --name <name>");
            std::process::exit(1);
        }

        log::info!("Reading manifest from: {}", manifest_path.display());
        let manifest_data = std::fs::read_to_string(&manifest_path)
            .expect("Failed to read manifest file");
        let manifest: Value = serde_json::from_str(&manifest_data)
            .expect("Failed to parse manifest JSON");

        let version = manifest["version"].as_u64().unwrap_or(3);
        if version != 3 {
            log::error!("Only v3 world format is supported. Found version {}. Please regenerate the world.", version);
            std::process::exit(1);
        }
        log::info!("Manifest version: {}", version);

        let mut result = Vec::new();

        // V3: layer-based format — chunks stored under <layer_dir>/chunk_x_y_z.rkc
        let layers = manifest["layers"].as_array()
            .expect("V3 manifest missing 'layers' array");

        for layer in layers {
            let layer_name = layer["name"].as_str().unwrap_or("unknown");
            let layer_dir = layer["directory"].as_str().unwrap_or(layer_name);
            let layer_id = layer["id"].as_u64().unwrap_or(0) as u32;

            // Skip non-terrain layers (e.g. grass masks are loaded separately)
            let chunk_list = match layer["chunks"].as_array() {
                Some(list) => list,
                None => {
                    log::info!("Skipping layer '{}' (no chunks array)", layer_name);
                    continue;
                }
            };

            log::info!("Loading layer '{}' (id={}): {} chunks", layer_name, layer_id, chunk_list.len());
            eprintln!("DEBUG: Loading layer '{}' id={}", layer_name, layer_id);

            // Debug: print first few chunk coordinates
            for (i, chunk_info) in chunk_list.iter().take(3).enumerate() {
                let x = chunk_info["x"].as_i64().unwrap() as i32;
                let y = chunk_info["y"].as_i64().unwrap() as i32;
                let z = chunk_info["z"].as_i64().unwrap() as i32;
                log::debug!("Layer {} chunk[{}]: ({},{},{})", layer_name, i, x, y, z);
            }

            // Get world center from manifest (approximate)
            let world_center_x = manifest["size"].as_f64().unwrap_or(0.0) as f32 / 2.0;
            let world_center_z = manifest["size"].as_f64().unwrap_or(0.0) as f32 / 2.0;
            let chunk_size = CHUNK_SIZE as f32;

            for chunk_info in chunk_list {
                let x = chunk_info["x"].as_i64().unwrap() as i32;
                let y = chunk_info["y"].as_i64().unwrap() as i32;
                let z = chunk_info["z"].as_i64().unwrap() as i32;

                // Filter by view distance if specified
                if view_distance > 0.0 {
                    let chunk_world_x = x as f32 * chunk_size;
                    let chunk_world_z = z as f32 * chunk_size;
                    let dx = chunk_world_x - world_center_x;
                    let dz = chunk_world_z - world_center_z;
                    let dist = (dx * dx + dz * dz).sqrt();
                    if dist > view_distance {
                        continue;
                    }
                }

                let chunk_file = world_path
                    .join(layer_dir)
                    .join(format!("chunk_{}_{}_{}.rkc", x, y, z));

                if let Some((coord, octree)) = Self::load_chunk_file(&chunk_file, x, y, z, layer_id) {
                    result.push((coord, octree, layer_id));
                }
            }
        }

        log::info!("Loaded {} chunks total", result.len());
        result
    }

    fn load_chunk_file(path: &std::path::Path, x: i32, y: i32, z: i32, layer_id: u32)
        -> Option<(disk_io::ChunkCoord, rktri::voxel::svo::Octree)>
    {
        if !path.exists() {
            log::warn!("Chunk file not found: {}", path.display());
            return None;
        }

        let compressed = std::fs::read(path)
            .expect(&format!("Failed to read chunk file: {}", path.display()));

        // v3: SVDAG pre-compressed on disk
        let chunk = disk_io::decompress_svdag_chunk(&compressed)
            .expect(&format!("Failed to decompress SVDAG chunk: {}", path.display()));

        log::info!("Loaded chunk ({},{},{}) layer={}: {} nodes, {} bricks",
            x, y, z, layer_id, chunk.octree.node_count(), chunk.octree.brick_count());

        Some((disk_io::ChunkCoord::new(x, y, z), chunk.octree))
    }

    fn load_grass_masks_from_disk(world_path: &PathBuf) -> std::collections::HashMap<(i32, i32, i32), MaskOctree<GrassCell>> {
        let mut result = std::collections::HashMap::new();

        let grass_dir = world_path.join("grass");
        if !grass_dir.exists() {
            log::info!("No grass layer found at {}", grass_dir.display());
            return result;
        }

        // Read all .rkm files from grass directory
        let entries = match std::fs::read_dir(&grass_dir) {
            Ok(entries) => entries,
            Err(e) => {
                log::warn!("Failed to read grass directory: {}", e);
                return result;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rkm") {
                continue;
            }

            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("Failed to read grass mask {}: {}", path.display(), e);
                    continue;
                }
            };

            match disk_io::decompress_grass_mask(&data) {
                Ok((coord, mask)) => {
                    result.insert((coord.x, coord.y, coord.z), mask);
                }
                Err(e) => {
                    log::warn!("Failed to decompress grass mask {}: {}", path.display(), e);
                }
            }
        }

        log::info!("Loaded {} grass masks", result.len());
        result
    }
}

/// Debug info for a loaded chunk
#[derive(Debug, Clone)]
struct ChunkDebugInfo {
    node_count: u32,
    brick_count: u32,
    world_min: [f32; 3],
}

/// Debug info for a chunk's grass mask
#[derive(Debug, Clone)]
struct GrassDebugInfo {
    node_count: u32,
}

/// Shared state between the debug server and the render loop
struct SharedDebugState {
    // World path (set at startup, read by debug commands)
    world_path: Option<std::path::PathBuf>,

    // Camera overrides (set by debug handler, consumed by render loop)
    camera_move_to: Option<glam::Vec3>,
    camera_look_at: Option<glam::Vec3>,
    camera_move_relative: Option<glam::Vec3>,

    // Read-only camera state (updated by render loop each frame)
    camera_position: [f32; 3],
    camera_forward: [f32; 3],
    camera_fov_degrees: f32,

    // God rays overrides (set by debug, consumed by render)
    godrays_density: Option<f32>,
    godrays_decay: Option<f32>,
    godrays_exposure: Option<f32>,
    godrays_weight: Option<f32>,
    godrays_num_samples: Option<u32>,

    // Time override
    set_time_of_day: Option<f32>,

    // Key injection
    pending_keys: Vec<String>,

    // Current state (updated by render loop for reading)
    current_time_of_day: f32,
    current_debug_mode: u32,
    current_godrays: rktri_debug::GodRaysInfo,
    current_shadow: rktri_debug::ShadowInfo,
    chunk_count: u32,
    world_extent: f32,

    // FPS stats (updated by render loop each frame)
    current_fps: f32,
    frame_count: u64,
    fps_one_sec: [f32; 3],   // [avg, min, max]
    fps_five_sec: [f32; 3],
    fps_fifteen_sec: [f32; 3],

    // Screenshot (render loop captures, handler reads)
    screenshot_requested: bool,
    screenshot_data: Option<(u32, u32, String)>, // (width, height, png_base64)

    // GPU profiling (updated by render loop if profiling enabled)
    profile_timings: Option<rktri::render::profiler::GpuTimings>,

    // Render scale override
    set_render_scale: Option<f32>,

    // Atmosphere overrides (set by debug handler, consumed by render loop)
    set_weather: Option<String>,
    set_fog_enabled: Option<bool>,
    set_fog_density: Option<f32>,
    set_fog_height_falloff: Option<f32>,
    set_fog_height_base: Option<f32>,
    set_fog_inscattering: Option<f32>,
    set_wind_direction: Option<[f32; 2]>,
    set_wind_speed: Option<f32>,
    set_wind_gust: Option<f32>,

    // Current atmosphere state (updated by render loop for reading)
    current_weather_preset: String,
    current_wind_direction: [f32; 3],
    current_wind_speed: f32,
    current_fog_density: f32,
    current_fog_enabled: bool,
    current_cloud_coverage: f32,
    current_sun_color: [f32; 3],
    current_sun_intensity: f32,
    current_ambient_color: [f32; 3],
    current_sun_direction: [f32; 3],

    // DLSS overrides
    set_dlss_mode: Option<String>,

    // DLSS state (read-only, updated by render loop)
    dlss_enabled: bool,
    dlss_supported: bool,
    dlss_quality_mode: String,
    dlss_render_width: u32,
    dlss_render_height: u32,
    dlss_upscaled_width: u32,
    dlss_upscaled_height: u32,

    // Grass overrides (set by debug handler, consumed by render loop)
    set_grass_enabled: Option<bool>,
    set_grass_density: Option<f32>,
    set_grass_blade_height_min: Option<f32>,
    set_grass_blade_height_max: Option<f32>,
    set_grass_blade_width: Option<f32>,
    set_grass_sway_amount: Option<f32>,
    set_grass_max_distance: Option<f32>,
    set_grass_coverage_scale: Option<f32>,
    set_grass_coverage_amount: Option<f32>,

    // Chunk geometry data for terrain height queries
    chunk_bounds: Vec<[f32; 4]>, // [world_min_x, world_min_y, world_min_z, root_size]

    // Chunk info for GetChunkInfo (updated by render loop)
    loaded_chunks: std::collections::HashMap<(i32, i32, i32), ChunkDebugInfo>,
    loaded_chunks_grass: std::collections::HashMap<(i32, i32, i32), GrassDebugInfo>,

    // Current grass state (read-only, updated by render loop)
    current_grass_enabled: bool,
    current_grass_density: f32,
    current_grass_blade_height_min: f32,
    current_grass_blade_height_max: f32,
    current_grass_blade_width: f32,
    current_grass_sway_amount: f32,
    current_grass_max_distance: f32,
    current_grass_fade_start: f32,
    current_grass_coverage_scale: f32,
    current_grass_coverage_amount: f32,
}

impl Default for SharedDebugState {
    fn default() -> Self {
        Self {
            world_path: None,
            camera_move_to: None,
            camera_look_at: None,
            camera_move_relative: None,
            camera_position: [0.0; 3],
            camera_forward: [0.0; 3],
            camera_fov_degrees: 60.0,
            godrays_density: None,
            godrays_decay: None,
            godrays_exposure: None,
            godrays_weight: None,
            godrays_num_samples: None,
            set_time_of_day: None,
            pending_keys: Vec::new(),
            current_time_of_day: 10.0,
            current_debug_mode: 0,
            current_godrays: rktri_debug::GodRaysInfo {
                density: 0.5,
                decay: 0.97,
                exposure: 0.01,
                weight: 1.0,
                num_samples: 64,
                sun_screen_pos: [0.5, 0.5],
            },
            current_shadow: rktri_debug::ShadowInfo {
                soft_shadow_samples: 4,
                soft_shadow_angle: 0.06,
                shadow_bias: 0.01,
                leaf_opacity: 0.15,
            },
            chunk_bounds: Vec::new(),
            loaded_chunks: std::collections::HashMap::new(),
            loaded_chunks_grass: std::collections::HashMap::new(),
            chunk_count: 0,
            world_extent: 0.0,
            current_fps: 0.0,
            frame_count: 0,
            fps_one_sec: [0.0; 3],
            fps_five_sec: [0.0; 3],
            fps_fifteen_sec: [0.0; 3],
            screenshot_requested: false,
            screenshot_data: None,
            profile_timings: None,
            set_render_scale: None,
            set_weather: None,
            set_fog_enabled: None,
            set_fog_density: None,
            set_fog_height_falloff: None,
            set_fog_height_base: None,
            set_fog_inscattering: None,
            set_wind_direction: None,
            set_wind_speed: None,
            set_wind_gust: None,
            current_weather_preset: "Clear".to_string(),
            current_wind_direction: [0.0, 0.0, 0.0],
            current_wind_speed: 0.0,
            current_fog_density: 0.0,
            current_fog_enabled: true,
            current_cloud_coverage: 0.0,
            current_sun_color: [1.0, 1.0, 1.0],
            current_sun_intensity: 1.0,
            current_ambient_color: [0.3, 0.3, 0.3],
            current_sun_direction: [0.0, 1.0, 0.0],
            set_dlss_mode: None,
            dlss_enabled: false,
            dlss_supported: false,
            dlss_quality_mode: "Off".to_string(),
            dlss_render_width: 0,
            dlss_render_height: 0,
            dlss_upscaled_width: 0,
            dlss_upscaled_height: 0,
            set_grass_enabled: None,
            set_grass_density: None,
            set_grass_blade_height_min: None,
            set_grass_blade_height_max: None,
            set_grass_blade_width: None,
            set_grass_sway_amount: None,
            set_grass_max_distance: None,
            set_grass_coverage_scale: None,
            set_grass_coverage_amount: None,
            current_grass_enabled: true,
            current_grass_density: 0.8,
            current_grass_blade_height_min: 0.4,
            current_grass_blade_height_max: 1.0,
            current_grass_blade_width: 0.025,
            current_grass_sway_amount: 0.5,
            current_grass_max_distance: 30.0,
            current_grass_fade_start: 20.0,
            current_grass_coverage_scale: 5.0,
            current_grass_coverage_amount: 0.6,
        }
    }
}

struct AppDebugHandler {
    state: Arc<StdMutex<SharedDebugState>>,
}

impl rktri_debug::DebugHandler for AppDebugHandler {
    fn handle_command(&mut self, cmd: rktri_debug::DebugCommand) -> rktri_debug::DebugResponse {
        use rktri_debug::*;

        match cmd {
            DebugCommand::Ping => DebugResponse::pong(),

            DebugCommand::CameraMoveTo { x, y, z } => {
                let mut s = self.state.lock().unwrap();
                s.camera_move_to = Some(glam::Vec3::new(x, y, z));
                DebugResponse::none()
            }

            DebugCommand::CameraMoveRelative { dx, dy, dz } => {
                let mut s = self.state.lock().unwrap();
                s.camera_move_relative = Some(glam::Vec3::new(dx, dy, dz));
                DebugResponse::none()
            }

            DebugCommand::CameraLookAt { x, y, z } => {
                let mut s = self.state.lock().unwrap();
                s.camera_look_at = Some(glam::Vec3::new(x, y, z));
                DebugResponse::none()
            }

            DebugCommand::CameraGetState => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::CameraState {
                    position: s.camera_position,
                    forward: s.camera_forward,
                    fov_degrees: s.camera_fov_degrees,
                })
            }

            DebugCommand::SendKey { key } => {
                let mut s = self.state.lock().unwrap();
                s.pending_keys.push(key);
                DebugResponse::none()
            }

            DebugCommand::TakeScreenshot => {
                // Request screenshot
                {
                    let mut s = self.state.lock().unwrap();
                    s.screenshot_requested = true;
                    s.screenshot_data = None;
                }

                // Poll for result (up to 2 seconds)
                for _ in 0..200 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    let mut s = self.state.lock().unwrap();
                    if let Some((width, height, png_base64)) = s.screenshot_data.take() {
                        s.screenshot_requested = false;
                        return DebugResponse::ok(ResponseData::Screenshot {
                            width,
                            height,
                            png_base64,
                        });
                    }
                }

                DebugResponse::error("Screenshot timed out")
            }

            DebugCommand::GetWorldInfo => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::WorldInfo {
                    chunk_count: s.chunk_count,
                    world_extent: s.world_extent,
                    time_of_day: s.current_time_of_day,
                    camera_position: s.camera_position,
                })
            }

            DebugCommand::GetRenderParams => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::RenderParams {
                    time_of_day: s.current_time_of_day,
                    debug_mode: s.current_debug_mode,
                    godrays: s.current_godrays.clone(),
                    shadow: s.current_shadow.clone(),
                })
            }

            DebugCommand::SetGodRaysParams {
                density,
                decay,
                exposure,
                weight,
                num_samples,
            } => {
                let mut s = self.state.lock().unwrap();
                if let Some(v) = density {
                    s.godrays_density = Some(v);
                }
                if let Some(v) = decay {
                    s.godrays_decay = Some(v);
                }
                if let Some(v) = exposure {
                    s.godrays_exposure = Some(v);
                }
                if let Some(v) = weight {
                    s.godrays_weight = Some(v);
                }
                if let Some(v) = num_samples {
                    s.godrays_num_samples = Some(v);
                }
                let desc = format!(
                    "Updated: density={:?} decay={:?} exposure={:?} weight={:?} num_samples={:?}",
                    density, decay, exposure, weight, num_samples
                );
                DebugResponse::ok(ResponseData::ParamsUpdated { description: desc })
            }

            DebugCommand::SetTimeOfDay { hour } => {
                let mut s = self.state.lock().unwrap();
                s.set_time_of_day = Some(hour.clamp(0.0, 24.0));
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: format!("Time set to {:.1}", hour),
                })
            }

            DebugCommand::GetFpsStats => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::FpsStats {
                    current_fps: s.current_fps,
                    frame_count: s.frame_count,
                    one_sec: rktri_debug::FpsWindowInfo {
                        avg: s.fps_one_sec[0],
                        min: s.fps_one_sec[1],
                        max: s.fps_one_sec[2],
                    },
                    five_sec: rktri_debug::FpsWindowInfo {
                        avg: s.fps_five_sec[0],
                        min: s.fps_five_sec[1],
                        max: s.fps_five_sec[2],
                    },
                    fifteen_sec: rktri_debug::FpsWindowInfo {
                        avg: s.fps_fifteen_sec[0],
                        min: s.fps_fifteen_sec[1],
                        max: s.fps_fifteen_sec[2],
                    },
                })
            }

            DebugCommand::GetProfileStats => {
                let s = self.state.lock().unwrap();
                if let Some(timings) = &s.profile_timings {
                    DebugResponse::ok(ResponseData::ProfileStats {
                        enabled: true,
                        svo_trace_ms: timings.svo_trace_ms,
                        shadow_ms: timings.shadow_ms,
                        godrays_ms: timings.godrays_ms,
                        lighting_ms: timings.lighting_ms,
                        display_ms: timings.display_ms,
                        total_gpu_ms: timings.total_gpu_ms,
                    })
                } else {
                    DebugResponse::ok(ResponseData::ProfileStats {
                        enabled: false,
                        svo_trace_ms: 0.0,
                        shadow_ms: 0.0,
                        godrays_ms: 0.0,
                        lighting_ms: 0.0,
                        display_ms: 0.0,
                        total_gpu_ms: 0.0,
                    })
                }
            }

            DebugCommand::SetRenderScale { scale } => {
                let clamped = scale.clamp(0.25, 2.0);
                self.state.lock().unwrap().set_render_scale = Some(clamped);
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: format!("Render scale set to {:.2}", clamped),
                })
            }

            DebugCommand::SetWeather { preset } => {
                let mut s = self.state.lock().unwrap();
                s.set_weather = Some(preset.clone());
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: format!("Weather set to {}", preset),
                })
            }

            DebugCommand::SetFog {
                enabled,
                density,
                height_falloff,
                height_base,
                inscattering,
            } => {
                let mut s = self.state.lock().unwrap();
                if let Some(v) = enabled {
                    s.set_fog_enabled = Some(v);
                }
                if let Some(v) = density {
                    s.set_fog_density = Some(v);
                }
                if let Some(v) = height_falloff {
                    s.set_fog_height_falloff = Some(v);
                }
                if let Some(v) = height_base {
                    s.set_fog_height_base = Some(v);
                }
                if let Some(v) = inscattering {
                    s.set_fog_inscattering = Some(v);
                }
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: "Fog parameters updated".to_string(),
                })
            }

            DebugCommand::SetWind {
                direction_x,
                direction_z,
                speed,
                gust_strength,
            } => {
                let mut s = self.state.lock().unwrap();
                if direction_x.is_some() || direction_z.is_some() {
                    let cur_x = direction_x.unwrap_or(s.current_wind_direction[0]);
                    let cur_z = direction_z.unwrap_or(s.current_wind_direction[2]);
                    s.set_wind_direction = Some([cur_x, cur_z]);
                }
                if let Some(v) = speed {
                    s.set_wind_speed = Some(v);
                }
                if let Some(v) = gust_strength {
                    s.set_wind_gust = Some(v);
                }
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: "Wind parameters updated".to_string(),
                })
            }

            DebugCommand::GetAtmosphereState => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::AtmosphereInfo {
                    time_of_day: s.current_time_of_day,
                    sun_direction: s.current_sun_direction,
                    sun_color: s.current_sun_color,
                    sun_intensity: s.current_sun_intensity,
                    ambient_color: s.current_ambient_color,
                    fog_density: s.current_fog_density,
                    fog_enabled: s.current_fog_enabled,
                    weather_preset: s.current_weather_preset.clone(),
                    wind_direction: s.current_wind_direction,
                    wind_speed: s.current_wind_speed,
                    cloud_coverage: s.current_cloud_coverage,
                })
            }

            DebugCommand::SetGrassParams {
                enabled,
                density,
                blade_height_min,
                blade_height_max,
                blade_width,
                sway_amount,
                max_distance,
                coverage_scale,
                coverage_amount,
            } => {
                let mut s = self.state.lock().unwrap();
                if let Some(v) = enabled {
                    s.set_grass_enabled = Some(v);
                }
                if let Some(v) = density {
                    s.set_grass_density = Some(v);
                }
                if let Some(v) = blade_height_min {
                    s.set_grass_blade_height_min = Some(v);
                }
                if let Some(v) = blade_height_max {
                    s.set_grass_blade_height_max = Some(v);
                }
                if let Some(v) = blade_width {
                    s.set_grass_blade_width = Some(v);
                }
                if let Some(v) = sway_amount {
                    s.set_grass_sway_amount = Some(v);
                }
                if let Some(v) = max_distance {
                    s.set_grass_max_distance = Some(v);
                }
                if let Some(v) = coverage_scale {
                    s.set_grass_coverage_scale = Some(v);
                }
                if let Some(v) = coverage_amount {
                    s.set_grass_coverage_amount = Some(v);
                }
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: "Grass parameters updated".to_string(),
                })
            }

            DebugCommand::GetGrassState => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::GrassInfo {
                    enabled: s.current_grass_enabled,
                    density: s.current_grass_density,
                    blade_height_min: s.current_grass_blade_height_min,
                    blade_height_max: s.current_grass_blade_height_max,
                    blade_width: s.current_grass_blade_width,
                    sway_amount: s.current_grass_sway_amount,
                    max_distance: s.current_grass_max_distance,
                    fade_start: s.current_grass_fade_start,
                    coverage_scale: s.current_grass_coverage_scale,
                    coverage_amount: s.current_grass_coverage_amount,
                })
            }

            DebugCommand::SetDlssMode { mode } => {
                let mut s = self.state.lock().unwrap();
                s.set_dlss_mode = Some(mode.clone());
                DebugResponse::ok(ResponseData::ParamsUpdated {
                    description: format!("DLSS mode set to {}", mode),
                })
            }

            DebugCommand::GetDlssState => {
                let s = self.state.lock().unwrap();
                DebugResponse::ok(ResponseData::DlssState {
                    enabled: s.dlss_enabled,
                    supported: s.dlss_supported,
                    quality_mode: s.dlss_quality_mode.clone(),
                    render_width: s.dlss_render_width,
                    render_height: s.dlss_render_height,
                    upscaled_width: s.dlss_upscaled_width,
                    upscaled_height: s.dlss_upscaled_height,
                })
            }

            DebugCommand::RaycastDown { x, z } => {
                let s = self.state.lock().unwrap();
                let mut best_y = f32::NEG_INFINITY;
                let mut hit = false;
                for b in &s.chunk_bounds {
                    let min_x = b[0];
                    let min_y = b[1];
                    let min_z = b[2];
                    let size = b[3];
                    // Check if (x, z) is within this chunk's XZ bounds
                    if x >= min_x && x < min_x + size && z >= min_z && z < min_z + size {
                        let top_y = min_y + size;
                        if top_y > best_y {
                            best_y = top_y;
                            hit = true;
                        }
                    }
                }
                DebugResponse::ok(ResponseData::TerrainHeight {
                    x, z,
                    height: if hit { best_y } else { 0.0 },
                    hit,
                })
            }

            DebugCommand::GetWorldMetadata => {
                let s = self.state.lock().unwrap();
                let world_path = match &s.world_path {
                    Some(p) => p,
                    None => return DebugResponse::error("No world loaded"),
                };
                let manifest_path = world_path.join("manifest.json");
                let manifest_data = match std::fs::read_to_string(&manifest_path) {
                    Ok(d) => d,
                    Err(e) => return DebugResponse::error(format!("Failed to read manifest: {}", e)),
                };
                let manifest: serde_json::Value = match serde_json::from_str(&manifest_data) {
                    Ok(m) => m,
                    Err(e) => return DebugResponse::error(format!("Failed to parse manifest: {}", e)),
                };

                let name = manifest["name"].as_str().unwrap_or("unknown").to_string();
                let version = manifest["version"].as_u64().unwrap_or(0) as u32;
                let seed = manifest["seed"].as_u64().unwrap_or(0) as u32;
                let size = manifest["size"].as_f64().unwrap_or(0.0) as f32;
                let chunk_size = manifest["chunk_size"].as_u64().unwrap_or(4) as u32;

                let tp = &manifest["terrain_params"];
                let terrain_params = rktri_debug::TerrainParamsMeta {
                    scale: tp["scale"].as_f64().unwrap_or(150.0) as f32,
                    height_scale: tp["height_scale"].as_f64().unwrap_or(80.0) as f32,
                    octaves: tp["octaves"].as_u64().unwrap_or(5) as u32,
                    persistence: tp["persistence"].as_f64().unwrap_or(0.5) as f32,
                    lacunarity: tp["lacunarity"].as_f64().unwrap_or(2.0) as f32,
                    sea_level: tp["sea_level"].as_f64().unwrap_or(20.0) as f32,
                };

                let mut layers = Vec::new();
                if let Some(arr) = manifest["layers"].as_array() {
                    for layer in arr {
                        layers.push(rktri_debug::LayerMeta {
                            name: layer["name"].as_str().unwrap_or("unknown").to_string(),
                            id: layer["id"].as_u64().unwrap_or(0) as u32,
                            directory: layer["directory"].as_str().unwrap_or("").to_string(),
                            chunk_count: layer["chunk_count"].as_u64().unwrap_or(0) as u32,
                            total_bytes: layer["total_bytes"].as_u64().unwrap_or(0),
                        });
                    }
                }

                DebugResponse::ok(ResponseData::WorldMetadata {
                    name,
                    version,
                    seed,
                    size,
                    chunk_size,
                    terrain_params,
                    layers,
                })
            }

            DebugCommand::GetChunkInfo { x, y, z } => {
                let s = self.state.lock().unwrap();
                let key = (x, y, z);

                if let Some(chunk_info) = s.loaded_chunks.get(&key) {
                    let has_grass = s.loaded_chunks_grass.contains_key(&key);
                    let grass_nodes = s.loaded_chunks_grass.get(&key)
                        .map(|g| g.node_count)
                        .unwrap_or(0);

                    DebugResponse::ok(ResponseData::ChunkInfo {
                        x, y, z,
                        loaded: true,
                        node_count: chunk_info.node_count,
                        brick_count: chunk_info.brick_count,
                        world_min: Some(chunk_info.world_min),
                        has_grass,
                        grass_nodes,
                    })
                } else {
                    DebugResponse::ok(ResponseData::ChunkInfo {
                        x, y, z,
                        loaded: false,
                        node_count: 0,
                        brick_count: 0,
                        world_min: None,
                        has_grass: false,
                        grass_nodes: 0,
                    })
                }
            }
        }
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    resources: Option<RenderResources>,
    view_distance: f32,
    camera: Camera,
    controller: FpsCameraController,
    input: InputState,
    timer: FrameTimer,
    cursor_grabbed: bool,
    atmosphere: AtmosphereSystem,
    debug_mode: u32,  // 0=normal, 1=albedo, 2=normal, 3=depth, 4=material
    world_path: Option<PathBuf>,  // Path to pre-generated world
    debug_state: Arc<StdMutex<SharedDebugState>>,
    render_scale: f32,
    // Persistent god rays overrides (applied from debug state)
    godrays_density: f32,
    godrays_decay: f32,
    godrays_exposure: f32,
    godrays_weight: f32,
    godrays_num_samples: u32,
    #[cfg(feature = "dlss")]
    dlss_enabled: bool,
    grass: GrassSystem,
    grass_time: f32,
}

impl App {
    fn new(world_path: Option<PathBuf>, world_size: Option<f32>, debug_state: Arc<StdMutex<SharedDebugState>>) -> Self {
        let mut config = SceneConfig::default();
        if let Some(size) = world_size {
            config.view_distance = size / 2.0; // size is diameter, view_distance is radius
        }
        let view_distance = config.view_distance;
        let initial_pos = config.initial_camera_pos;
        // Look at terrain from above, angled toward horizon
        let target = initial_pos + glam::Vec3::new(10.0, -8.0, 10.0);
        Self {
            window: None,
            gpu: None,
            resources: None,
            view_distance,
            camera: Camera::look_at(initial_pos, target, glam::Vec3::Y),
            controller: FpsCameraController::new(2.0, 0.5),  // Slower for small world
            input: InputState::new(),
            timer: FrameTimer::new(),
            cursor_grabbed: false,
            atmosphere: {
                let mut atmo_config = AtmosphereConfig::default();
                atmo_config.start_time = config.time_of_day;
                atmo_config.time_paused = true;
                let mut sys = AtmosphereSystem::new(atmo_config);
                sys.set_time(config.time_of_day);
                sys
            },
            debug_mode: 0,
            world_path,
            debug_state,
            render_scale: 1.0,
            godrays_density: 0.3,
            godrays_decay: 0.96,
            godrays_exposure: 0.015,
            godrays_weight: 1.0,
            godrays_num_samples: 64,
            #[cfg(feature = "dlss")]
            dlss_enabled: true,  // Enable by default when feature is compiled
            grass: GrassSystem::new(GrassConfig::default()),
            grass_time: 0.0,
        }
    }

    fn toggle_cursor_grab(&mut self) {
        if let Some(window) = &self.window {
            self.cursor_grabbed = !self.cursor_grabbed;

            if self.cursor_grabbed {
                window.set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
                    .ok();
                window.set_cursor_visible(false);
            } else {
                window.set_cursor_grab(CursorGrabMode::None).ok();
                window.set_cursor_visible(true);
            }

            self.input.set_mouse_captured(self.cursor_grabbed);
        }
    }

    fn process_debug_commands(&mut self) {
        let mut state = self.debug_state.lock().unwrap();

        // Apply camera overrides
        if let Some(pos) = state.camera_move_to.take() {
            self.camera.position = pos;
        }
        if let Some(delta) = state.camera_move_relative.take() {
            self.camera.position += delta;
        }
        if let Some(target) = state.camera_look_at.take() {
            let forward = (target - self.camera.position).normalize();
            let right = forward.cross(glam::Vec3::Y).normalize();
            let up = right.cross(forward);
            self.camera.rotation =
                glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));
        }

        // Apply god rays overrides (persistent -- once set, stays until changed)
        if let Some(v) = state.godrays_density.take() {
            self.godrays_density = v;
        }
        if let Some(v) = state.godrays_decay.take() {
            self.godrays_decay = v;
        }
        if let Some(v) = state.godrays_exposure.take() {
            self.godrays_exposure = v;
        }
        if let Some(v) = state.godrays_weight.take() {
            self.godrays_weight = v;
        }
        if let Some(v) = state.godrays_num_samples.take() {
            self.godrays_num_samples = v;
        }

        // Apply time of day
        if let Some(t) = state.set_time_of_day.take() {
            self.atmosphere.set_time(t);
        }

        // Apply render scale
        if let Some(s) = state.set_render_scale.take() {
            self.render_scale = s;
            log::info!("Render scale set to {:.2}", s);
        }

        // Collect atmosphere overrides before dropping lock
        let weather_preset = state.set_weather.take();
        let fog_enabled = state.set_fog_enabled.take();
        let fog_density = state.set_fog_density.take();
        let fog_height_falloff = state.set_fog_height_falloff.take();
        let fog_height_base = state.set_fog_height_base.take();
        let fog_inscattering = state.set_fog_inscattering.take();
        let wind_direction = state.set_wind_direction.take();
        let wind_speed = state.set_wind_speed.take();
        let wind_gust = state.set_wind_gust.take();

        // Apply grass overrides
        if let Some(v) = state.set_grass_enabled.take() {
            self.grass.config_mut().enabled = v;
        }
        // Per-profile overrides: modify profile 1 (Tall Grass) as the primary debug target
        if let Some(v) = state.set_grass_density.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.density = v.clamp(0.0, 1.0);
            }
        }
        if let Some(v) = state.set_grass_blade_height_min.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.height_min = v.max(0.01);
            }
        }
        if let Some(v) = state.set_grass_blade_height_max.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.height_max = v.max(0.01);
            }
        }
        if let Some(v) = state.set_grass_blade_width.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.width = v.clamp(0.002, 0.05);
            }
        }
        if let Some(v) = state.set_grass_sway_amount.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.sway_amount = v.clamp(0.0, 2.0);
            }
        }
        if let Some(v) = state.set_grass_max_distance.take() {
            self.grass.config_mut().max_distance = v.clamp(1.0, 100.0);
            self.grass.config_mut().fade_start = (v * 0.667).min(v - 2.0);
        }
        if let Some(v) = state.set_grass_coverage_scale.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.coverage_scale = v.clamp(0.5, 50.0);
            }
        }
        if let Some(v) = state.set_grass_coverage_amount.take() {
            if let Some(p) = self.grass.profile_table_mut().get_mut(1) {
                p.gpu.coverage_amount = v.clamp(0.0, 1.0);
            }
        }

        // Collect DLSS override
        let dlss_mode = state.set_dlss_mode.take();

        // Process injected keys (simulate press via input state)
        let keys: Vec<String> = state.pending_keys.drain(..).collect();
        drop(state); // Release lock before processing keys

        // Apply weather preset
        if let Some(preset_str) = weather_preset {
            use rktri::atmosphere::WeatherPreset;
            let preset = match preset_str.to_lowercase().as_str() {
                "clear" => Some(WeatherPreset::Clear),
                "partlycloudy" | "partly_cloudy" => Some(WeatherPreset::PartlyCloudy),
                "overcast" => Some(WeatherPreset::Overcast),
                "foggy" => Some(WeatherPreset::Foggy),
                "rain" => Some(WeatherPreset::Rain),
                "snow" => Some(WeatherPreset::Snow),
                "storm" => Some(WeatherPreset::Storm),
                _ => {
                    log::warn!("Unknown weather preset: {}", preset_str);
                    None
                }
            };
            if let Some(p) = preset {
                self.atmosphere.set_weather(p);
                log::info!("Weather set to {:?}", p);
            }
        }

        // Apply fog overrides
        if let Some(v) = fog_enabled {
            self.atmosphere.config_mut().fog.enabled = v;
        }
        if let Some(v) = fog_density {
            self.atmosphere.config_mut().fog.distance_fog_density = v;
        }
        if let Some(v) = fog_height_falloff {
            self.atmosphere.config_mut().fog.height_fog_falloff = v;
        }
        if let Some(v) = fog_height_base {
            self.atmosphere.config_mut().fog.height_fog_base = v;
        }
        if let Some(v) = fog_inscattering {
            self.atmosphere.config_mut().fog.inscattering_intensity = v;
        }

        // Apply wind overrides
        if let Some([dx, dz]) = wind_direction {
            self.atmosphere.config_mut().wind.base_direction = [dx, dz];
        }
        if let Some(v) = wind_speed {
            self.atmosphere.config_mut().wind.base_speed = v;
        }
        if let Some(v) = wind_gust {
            self.atmosphere.config_mut().wind.gust_strength = v;
        }

        // Apply DLSS mode change
        if let Some(mode_str) = dlss_mode {
            #[cfg(feature = "dlss")]
            {
                match mode_str.to_lowercase().as_str() {
                    "off" | "disabled" => {
                        self.dlss_enabled = false;
                        // Restore render targets to render_scale resolution
                        if let (Some(gpu), Some(resources)) = (&self.gpu, &mut self.resources) {
                            let (w, h) = gpu.size();
                            let rw = ((w as f32 * self.render_scale) as u32).max(1);
                            let rh = ((h as f32 * self.render_scale) as u32).max(1);
                            resources.recreate_render_targets(&gpu.device, rw, rh);
                        }
                        log::info!("DLSS disabled via debug command");
                    }
                    "auto" | "quality" | "balanced" | "performance" | "ultraperformance" | "ultra_performance" | "dlaa" => {
                        self.dlss_enabled = true;
                        // Recreate DLSS with new quality if resources exist
                        if let (Some(gpu), Some(resources)) = (&self.gpu, &mut self.resources) {
                            let quality = match mode_str.to_lowercase().as_str() {
                                "auto" => rktri::render::upscale::dlss::DlssQuality::Auto,
                                "quality" => rktri::render::upscale::dlss::DlssQuality::Quality,
                                "balanced" => rktri::render::upscale::dlss::DlssQuality::Balanced,
                                "performance" => rktri::render::upscale::dlss::DlssQuality::Performance,
                                "ultraperformance" | "ultra_performance" => rktri::render::upscale::dlss::DlssQuality::UltraPerformance,
                                "dlaa" => rktri::render::upscale::dlss::DlssQuality::Dlaa,
                                _ => rktri::render::upscale::dlss::DlssQuality::Auto,
                            };
                            let (w, h) = gpu.size();
                            // Recreate upscaler with new quality, extract values before reborrowing resources
                            let recreate_result = if let Some(ref mut upscaler) = resources.dlss_upscaler {
                                match upscaler.recreate(&gpu.device, &gpu.queue, w, h, quality) {
                                    Ok(()) => Some(upscaler.render_resolution()),
                                    Err(e) => {
                                        log::error!("DLSS: Failed to change quality: {}", e);
                                        None
                                    }
                                }
                            } else { None };
                            if let Some(new_res) = recreate_result {
                                resources.recreate_render_targets(&gpu.device, new_res[0], new_res[1]);
                                let dlss_output_view = resources.dlss_upscaler.as_ref().unwrap().output_view();
                                let tonemap_in = resources.tonemap_pipeline.create_input_bind_group(&gpu.device, dlss_output_view);
                                let (post_tex, post_view) = RenderResources::create_post_texture(&gpu.device, w, h);
                                let tonemap_out = resources.tonemap_pipeline.create_output_bind_group(&gpu.device, &post_view);
                                let display_bg = resources.display_pipeline.create_bind_group(&gpu.device, &post_view);
                                resources.dlss_post_texture = Some(post_tex);
                                resources.dlss_post_texture_view = Some(post_view);
                                resources.dlss_tonemap_input_bind_group = Some(tonemap_in);
                                resources.dlss_tonemap_output_bind_group = Some(tonemap_out);
                                resources.dlss_display_bind_group = Some(display_bg);
                                // Recreate depth normalization resources at new DLSS render resolution
                                let gbuffer_views_for_norm = resources.gbuffer.views();
                                let (norm_tex, norm_view, norm_pipeline, norm_bg) = RenderResources::create_dlss_depth_normalization(
                                    &gpu.device, gbuffer_views_for_norm.depth, new_res[0], new_res[1],
                                );
                                resources.dlss_norm_depth_texture = Some(norm_tex);
                                resources.dlss_norm_depth_view = Some(norm_view);
                                resources.dlss_norm_depth_pipeline = Some(norm_pipeline);
                                resources.dlss_norm_depth_bind_group = Some(norm_bg);
                            }
                        }
                        log::info!("DLSS mode set to: {}", mode_str);
                    }
                    _ => {
                        log::warn!("Unknown DLSS mode: {}", mode_str);
                    }
                }
            }
            #[cfg(not(feature = "dlss"))]
            {
                log::warn!("DLSS not available (compiled without dlss feature): {}", mode_str);
            }
        }

        for key_name in &keys {
            match key_name.as_str() {
                "T" => {
                    let current = self.atmosphere.state().time_of_day;
                    self.atmosphere.set_time((current + 1.0) % 24.0);
                    log::info!("Debug: Time of day: {:.1}", self.atmosphere.state().time_of_day);
                }
                "ShiftT" => {
                    let current = self.atmosphere.state().time_of_day;
                    self.atmosphere.set_time((current - 1.0 + 24.0) % 24.0);
                    log::info!("Debug: Time of day: {:.1}", self.atmosphere.state().time_of_day);
                }
                "F1" => {
                    self.debug_mode = 0;
                }
                "F2" => {
                    self.debug_mode = 1;
                }
                "F3" => {
                    self.debug_mode = 2;
                }
                "F4" => {
                    self.debug_mode = 3;
                }
                "F5" => {
                    self.debug_mode = 4;
                }
                _ => {
                    log::warn!("Debug: Unknown key '{}'", key_name);
                }
            }
        }

        // Update read-only state for debug handler
        let mut state = self.debug_state.lock().unwrap();
        state.camera_position = self.camera.position.to_array();
        state.camera_forward = self.camera.forward().to_array();
        state.camera_fov_degrees = self.camera.fov_y.to_degrees();
        state.current_time_of_day = self.atmosphere.state().time_of_day;
        state.current_debug_mode = self.debug_mode;

        // Update atmosphere state
        let atmo = self.atmosphere.state();
        state.current_sun_direction = atmo.sun_direction;
        state.current_sun_color = atmo.sun_color;
        state.current_sun_intensity = atmo.sun_intensity;
        state.current_ambient_color = atmo.ambient_color;
        state.current_fog_density = atmo.fog_density;
        state.current_fog_enabled = self.atmosphere.config().fog.enabled;
        state.current_wind_direction = atmo.wind.direction;
        state.current_wind_speed = atmo.wind.speed;
        state.current_cloud_coverage = atmo.cloud_coverage;
        state.current_weather_preset = format!("{:?}", self.atmosphere.current_preset());

        // Update DLSS state
        #[cfg(feature = "dlss")]
        {
            state.dlss_supported = self.gpu.as_ref().map_or(false, |g| g.dlss.is_some());
            state.dlss_enabled = self.dlss_enabled && state.dlss_supported;
            if let Some(ref resources) = self.resources {
                if let Some(ref upscaler) = resources.dlss_upscaler {
                    state.dlss_quality_mode = upscaler.quality().name().to_string();
                    let rr = upscaler.render_resolution();
                    let ur = upscaler.upscaled_resolution();
                    state.dlss_render_width = rr[0];
                    state.dlss_render_height = rr[1];
                    state.dlss_upscaled_width = ur[0];
                    state.dlss_upscaled_height = ur[1];
                } else {
                    state.dlss_quality_mode = "Off".to_string();
                    state.dlss_render_width = 0;
                    state.dlss_render_height = 0;
                    state.dlss_upscaled_width = 0;
                    state.dlss_upscaled_height = 0;
                }
            }
        }
        #[cfg(not(feature = "dlss"))]
        {
            state.dlss_supported = false;
            state.dlss_enabled = false;
            state.dlss_quality_mode = "Not compiled".to_string();
        }

        // Update world info if resources available
        if let Some(res) = &self.resources {
            state.chunk_count = res.chunk_count;
            state.world_extent = res.world_extent;
            if state.chunk_bounds.is_empty() && !res.chunk_bounds.is_empty() {
                state.chunk_bounds = res.chunk_bounds.clone();
            }
            if state.loaded_chunks.is_empty() && !res.loaded_chunks.is_empty() {
                state.loaded_chunks = res.loaded_chunks.clone();
                state.loaded_chunks_grass = res.loaded_chunks_grass.clone();
            }
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(resources) = &mut self.resources else { return };

        let output = match gpu.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::error!("Failed to get surface texture: {}", e);
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = gpu.size();

        // Internal render resolution (lower than window for performance)
        // mut needed when dlss feature overrides render resolution
        #[allow(unused_mut)]
        let mut render_width = ((width as f32 * self.render_scale) as u32).max(1);
        #[allow(unused_mut)]
        let mut render_height = ((height as f32 * self.render_scale) as u32).max(1);

        // When DLSS is active, use its recommended render resolution
        #[cfg(feature = "dlss")]
        if self.dlss_enabled {
            if let Some(ref upscaler) = resources.dlss_upscaler {
                let res = upscaler.render_resolution();
                render_width = res[0];
                render_height = res[1];
            }
        }

        // Update camera - apply jitter when DLSS is active
        #[cfg(feature = "dlss")]
        {
            if self.dlss_enabled {
                if let Some(ref upscaler) = resources.dlss_upscaler {
                    let jitter = upscaler.suggested_jitter();
                    resources.camera_buffer.update_with_jitter(
                        &gpu.queue, &self.camera, resources.world_offset,
                        jitter, render_width, render_height,
                    );
                } else {
                    resources.camera_buffer.update_with_offset(&gpu.queue, &self.camera, resources.world_offset);
                }
            } else {
                resources.camera_buffer.update_with_offset(&gpu.queue, &self.camera, resources.world_offset);
            }
        }
        #[cfg(not(feature = "dlss"))]
        resources.camera_buffer.update_with_offset(&gpu.queue, &self.camera, resources.world_offset);

        // Grid-based ray marching: chunk_info_buffer is static, no per-frame rewrite needed.
        // The DDA in the shader uses the grid to find chunks along each ray.

        // Update trace params with grid acceleration
        let params = TraceParams {
            width: render_width,
            height: render_height,
            chunk_count: resources.chunk_count, // kept for fallback/debug
            _pad0: 0,
            lod_distances: [64.0, 128.0, 256.0, 512.0],
            lod_distances_ext: [1024.0, f32::MAX],
            _pad: [0.0; 2],
            grid_min_x: resources.grid_min[0],
            grid_min_y: resources.grid_min[1],
            grid_min_z: resources.grid_min[2],
            chunk_size: rktri::voxel::chunk::CHUNK_SIZE as f32,
            grid_size_x: resources.grid_size[0],
            grid_size_y: resources.grid_size[1],
            grid_size_z: resources.grid_size[2],
            _pad2: 0,
        };
        resources.svo_pipeline.update_params(&gpu.queue, &params);

        // Update grass params
        let atmo_for_grass = self.atmosphere.state();
        let grass_params = self.grass.build_params(&atmo_for_grass.wind, self.grass_time);
        resources.svo_pipeline.update_grass_params(&gpu.queue, &grass_params);

        // Upload grass profile table (may be modified by debug commands)
        let profile_gpu_data = self.grass.profile_table_gpu_data();
        resources.svo_pipeline.update_profile_table(&gpu.queue, &profile_gpu_data);

        // Get atmosphere state for this frame
        let atmo = self.atmosphere.state();
        let sun_dir = self.atmosphere.sun_direction();

        // Update lighting uniforms from atmosphere (uses primary light: sun or moon)
        let lighting_uniforms = LightingUniforms {
            sun_direction: atmo.primary_light_direction,
            _pad1: 0.0,
            sun_color: atmo.primary_light_color,
            sun_intensity: atmo.primary_light_intensity,
            ambient_color: atmo.ambient_color,
            _pad2: 0.0,
        };
        resources.lighting_pipeline.update_uniforms(&gpu.queue, &lighting_uniforms);

        // Update sky params from atmosphere
        // Sky needs brighter sun for scattering (ratio preserved from original 8.0/1.5)
        let sky_params = SkyParams {
            sun_direction: atmo.sun_direction,
            _pad1: 0.0,
            sun_color: atmo.sun_color,
            sun_intensity: atmo.sun_intensity * 5.333,
            sky_zenith_color: atmo.sky_zenith_color,
            sky_intensity: 1.2,
            sky_horizon_color: atmo.sky_horizon_color,
            _pad2: 0.0,
            ground_color: [0.35, 0.32, 0.28],
            _pad3: 0.0,
            moon_direction: atmo.moon_directions[0],
            moon_phase: atmo.moon_phases[0],
            moon_color: atmo.moon_colors[0],
            moon_size: atmo.moon_sizes[0],
            moon_count: atmo.moon_count,
            _pad4: [0.0; 3],
            fog_color: atmo.fog_color,
            fog_density: atmo.fog_density,
            fog_height_falloff: atmo.fog_height_falloff,
            fog_height_base: atmo.fog_height_base,
            fog_inscattering: atmo.fog_inscattering,
            _pad5: 0.0,
            ambient_color: atmo.ambient_color,
            ambient_intensity: atmo.ambient_intensity,
        };
        resources.lighting_pipeline.update_sky_params(&gpu.queue, &sky_params);

        // Update debug params
        let debug_params = DebugParams {
            mode: self.debug_mode,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        resources.lighting_pipeline.update_debug_params(&gpu.queue, &debug_params);

        // Update shadow params (uses primary light: sun or moon)
        let shadow_params = ShadowParams {
            light_dir: atmo.primary_light_direction,
            _pad1: 0.0,
            shadow_bias: 0.01,
            soft_shadow_samples: 4,
            soft_shadow_angle: 0.06, // ~3.5 degrees for dappled light
            chunk_count: resources.chunk_count,
            width: (render_width / 2).max(1),
            height: (render_height / 2).max(1),
            leaf_opacity: 0.15, // Light per-brick contribution; accumulates over many hits
            _pad2: 0.0,
            grid_min_x: resources.grid_min[0],
            grid_min_y: resources.grid_min[1],
            grid_min_z: resources.grid_min[2],
            chunk_size: rktri::voxel::chunk::CHUNK_SIZE as f32,
            grid_size_x: resources.grid_size[0],
            grid_size_y: resources.grid_size[1],
            grid_size_z: resources.grid_size[2],
            _pad3: 0,
        };
        resources.shadow_pipeline.update_params(&gpu.queue, &shadow_params);

        // Calculate sun screen position for god rays
        let view_proj = self.camera.view_projection();
        let camera_pos = self.camera.position;
        let sun_world = camera_pos + sun_dir * 1000.0;
        let sun_clip = view_proj * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);
        let sun_ndc = glam::Vec2::new(sun_clip.x / sun_clip.w, sun_clip.y / sun_clip.w);
        let sun_uv = glam::Vec2::new(sun_ndc.x * 0.5 + 0.5, 1.0 - (sun_ndc.y * 0.5 + 0.5)); // NDC to UV, flip Y

        let godrays_params = GodRaysParams {
            sun_screen_pos: sun_uv.to_array(),
            num_samples: self.godrays_num_samples,
            density: self.godrays_density,
            decay: self.godrays_decay,
            exposure: self.godrays_exposure,
            weight: self.godrays_weight,
            width: (render_width / 2).max(1),
            height: (render_height / 2).max(1),
            _pad: 0,
        };
        resources.godrays_pipeline.update_params(&gpu.queue, &godrays_params);

        // Update cloud params
        let cloud_width = (render_width / 2).max(1);
        let cloud_height = (render_height / 2).max(1);
        let cloud_params = CloudParams {
            cloud_altitude: self.atmosphere.config().clouds.altitude,
            cloud_thickness: self.atmosphere.config().clouds.thickness,
            cloud_coverage: atmo.cloud_coverage.max(self.atmosphere.config().clouds.coverage),
            cloud_density: self.atmosphere.config().clouds.density,
            wind_offset: atmo.wind.accumulated_offset,
            noise_scale: self.atmosphere.config().clouds.noise_scale,
            cloud_color: self.atmosphere.config().clouds.cloud_color,
            detail_scale: self.atmosphere.config().clouds.detail_scale,
            shadow_color: self.atmosphere.config().clouds.shadow_color,
            edge_sharpness: self.atmosphere.config().clouds.edge_sharpness,
            sun_direction: atmo.sun_direction,
            sun_intensity: atmo.sun_intensity,
            sun_color: atmo.sun_color,
            time_of_day: atmo.time_of_day,
            width: cloud_width,
            height: cloud_height,
            _pad: [0; 2],
        };
        resources.cloud_pipeline.update_params(&gpu.queue, &cloud_params);

        // Update shared debug state with current godrays info
        {
            let mut ds = self.debug_state.lock().unwrap();
            ds.current_godrays = rktri_debug::GodRaysInfo {
                density: self.godrays_density,
                decay: self.godrays_decay,
                exposure: self.godrays_exposure,
                weight: self.godrays_weight,
                num_samples: self.godrays_num_samples,
                sun_screen_pos: sun_uv.to_array(),
            };
            ds.current_shadow = rktri_debug::ShadowInfo {
                soft_shadow_samples: shadow_params.soft_shadow_samples,
                soft_shadow_angle: shadow_params.soft_shadow_angle,
                shadow_bias: shadow_params.shadow_bias,
                leaf_opacity: shadow_params.leaf_opacity,
            };
            // Update grass state (read per-profile values from profile 1 = Tall Grass)
            let gc = self.grass.config();
            ds.current_grass_enabled = gc.enabled;
            ds.current_grass_max_distance = gc.max_distance;
            ds.current_grass_fade_start = gc.fade_start;
            if let Some(p) = self.grass.profile_table().get(1) {
                ds.current_grass_density = p.gpu.density;
                ds.current_grass_blade_height_min = p.gpu.height_min;
                ds.current_grass_blade_height_max = p.gpu.height_max;
                ds.current_grass_blade_width = p.gpu.width;
                ds.current_grass_sway_amount = p.gpu.sway_amount;
                ds.current_grass_coverage_scale = p.gpu.coverage_scale;
                ds.current_grass_coverage_amount = p.gpu.coverage_amount;
            }
        }

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_encoder"),
        });

        // Pass 1: Run SVO trace at render resolution (outputs to G-buffer)
        resources.svo_pipeline.dispatch(
            &mut encoder,
            &resources.octree_buffer,
            &resources.gbuffer_output_bind_group,
            render_width,
            render_height,
            None,
        );

        // Pass 2: Run shadow tracing at half of render resolution
        resources.shadow_pipeline.dispatch(
            &mut encoder,
            &resources.octree_buffer,
            &resources.shadow_gbuffer_bind_group,
            &resources.shadow_output_bind_group,
            (render_width / 2).max(1),
            (render_height / 2).max(1),
            None,
        );

        // Pass 3: God rays at half of render resolution
        resources.godrays_pipeline.dispatch(
            &mut encoder,
            &resources.godrays_input_bind_group,
            &resources.godrays_output_bind_group,
            (render_width / 2).max(1),
            (render_height / 2).max(1),
            None,
        );

        // Pass 4: Clouds at half of render resolution
        resources.cloud_pipeline.dispatch(
            &mut encoder,
            &resources.cloud_input_bind_group,
            &resources.cloud_output_bind_group,
            cloud_width,
            cloud_height,
            None,
        );

        // Pass 5: Run lighting at render resolution (reads G-buffer + shadow + godrays + clouds, outputs to lit_texture)
        resources.lighting_pipeline.dispatch(
            &mut encoder,
            &resources.lighting_gbuffer_bind_group,
            &resources.lighting_output_bind_group,
            render_width,
            render_height,
            None,
        );

        // When DLSS is active: run DLSS, then tonemap+display in separate encoder
        #[cfg(feature = "dlss")]
        let dlss_active = self.dlss_enabled && resources.dlss_upscaler.is_some();
        #[cfg(not(feature = "dlss"))]
        let dlss_active = false;

        if dlss_active {
            #[cfg(feature = "dlss")]
            {
                // Defensive: verify all DLSS resources exist before using them
                let dlss_ready = resources.dlss_upscaler.is_some()
                    && resources.dlss_tonemap_input_bind_group.is_some()
                    && resources.dlss_tonemap_output_bind_group.is_some()
                    && resources.dlss_display_bind_group.is_some();

                if dlss_ready {
                    let upscaler = resources.dlss_upscaler.as_mut().unwrap();
                    let jitter = upscaler.suggested_jitter();
                    let gbuffer_views = resources.gbuffer.views();

                    // Normalize depth for DLSS (linear distance -> [0,1])
                    let dlss_depth_view = if let (Some(norm_pipeline), Some(norm_bg), Some(norm_view)) = (
                        &resources.dlss_norm_depth_pipeline,
                        &resources.dlss_norm_depth_bind_group,
                        &resources.dlss_norm_depth_view,
                    ) {
                        let render_res = upscaler.render_resolution();
                        {
                            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("depth_normalize_pass"),
                                timestamp_writes: None,
                            });
                            pass.set_pipeline(norm_pipeline);
                            pass.set_bind_group(0, norm_bg, &[]);
                            pass.dispatch_workgroups(
                                (render_res[0] + 7) / 8,
                                (render_res[1] + 7) / 8,
                                1,
                            );
                        }
                        norm_view
                    } else {
                        gbuffer_views.depth
                    };

                    // Run DLSS (adds barriers to encoder, returns separate cmd buffer)
                    match upscaler.render(
                        &mut encoder,
                        &gpu.adapter,
                        &resources.lit_texture_view,    // HDR color input
                        dlss_depth_view,                 // Normalized depth [0,1]
                        gbuffer_views.motion,            // Motion vectors
                        jitter,
                    ) {
                        Ok(dlss_cmd) => {
                            // Submit encoder1 with barriers, then DLSS command buffer
                            gpu.queue.submit([encoder.finish(), dlss_cmd]);

                            // Create encoder2 for post-DLSS passes
                            let mut encoder2 = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("post_dlss_encoder"),
                            });

                            // Pass 6: Tonemap (reads DLSS output at WINDOW resolution)
                            let tonemap_params = TonemapParams {
                                width,
                                height,
                                exposure: 1.0,
                                _pad: 0,
                            };
                            resources.tonemap_pipeline.update_params(&gpu.queue, &tonemap_params);
                            resources.tonemap_pipeline.dispatch(
                                &mut encoder2,
                                resources.dlss_tonemap_input_bind_group.as_ref().unwrap(),
                                resources.dlss_tonemap_output_bind_group.as_ref().unwrap(),
                                width,
                                height,
                            );

                            // Pass 7: Display
                            resources.display_pipeline.render(
                                &mut encoder2,
                                &view,
                                resources.dlss_display_bind_group.as_ref().unwrap(),
                                None,
                            );

                            gpu.queue.submit([encoder2.finish()]);
                        }
                        Err(e) => {
                            log::error!("DLSS render failed: {}, falling back to standard path", e);
                            self.dlss_enabled = false;
                            gpu.queue.submit([encoder.finish()]);
                        }
                    }
                } else {
                    log::warn!("DLSS: Resources incomplete, falling back to standard path");
                    self.dlss_enabled = false;
                    // Fall through handled below
                }
            }
        } else {
            // Standard path (no DLSS) - tonemap + display in same encoder

            // Pass 6: Tone mapping (reads lit_texture HDR, outputs to post_texture LDR)
            let tonemap_params = TonemapParams {
                width: render_width,
                height: render_height,
                exposure: 1.0,
                _pad: 0,
            };
            resources.tonemap_pipeline.update_params(&gpu.queue, &tonemap_params);
            resources.tonemap_pipeline.dispatch(
                &mut encoder,
                &resources.tonemap_input_bind_group,
                &resources.tonemap_output_bind_group,
                render_width,
                render_height,
            );

            // Pass 7: Display result to screen
            resources.display_pipeline.render(&mut encoder, &view, &resources.display_bind_group, None);

            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        output.present();

        // Handle screenshot request
        {
            let needs_screenshot = {
                let ds = self.debug_state.lock().unwrap();
                ds.screenshot_requested && ds.screenshot_data.is_none()
            };

            if needs_screenshot {
                if let (Some(gpu), Some(resources)) = (&self.gpu, &self.resources) {
                    // Screenshot reads from lit_texture which is at render resolution
                    let ss_width = render_width;
                    let ss_height = render_height;
                    // Create staging buffer for readback
                    let bytes_per_pixel = 8u32; // Rgba16Float = 4 channels * 2 bytes
                    let padded_bytes_per_row = ((ss_width * bytes_per_pixel + 255) / 256) * 256;
                    let buffer_size = (padded_bytes_per_row * ss_height) as u64;

                    let staging_buffer =
                        gpu.device
                            .create_buffer(&wgpu::BufferDescriptor {
                                label: Some("screenshot_staging"),
                                size: buffer_size,
                                usage: wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            });

                    let mut encoder =
                        gpu.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("screenshot_encoder"),
                            });

                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &resources.lit_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(ss_height),
                            },
                        },
                        wgpu::Extent3d {
                            width: ss_width,
                            height: ss_height,
                            depth_or_array_layers: 1,
                        },
                    );

                    gpu.queue.submit(std::iter::once(encoder.finish()));

                    let slice = staging_buffer.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).ok();
                    });
                    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

                    if rx.recv().ok().and_then(|r| r.ok()).is_some() {
                        let data = slice.get_mapped_range();

                        // Convert Rgba16Float to Rgba8 for PNG
                        let mut rgba8 = Vec::with_capacity((ss_width * ss_height * 4) as usize);
                        for y_row in 0..ss_height {
                            let row_start = (y_row * padded_bytes_per_row) as usize;
                            for x_col in 0..ss_width {
                                let pixel_start =
                                    row_start + (x_col * bytes_per_pixel) as usize;
                                // Read 4 f16 values
                                for c in 0..4u32 {
                                    let offset = pixel_start + (c * 2) as usize;
                                    if offset + 2 <= data.len() {
                                        let bits = u16::from_le_bytes([
                                            data[offset],
                                            data[offset + 1],
                                        ]);
                                        let f = half::f16::from_bits(bits).to_f32();
                                        // Simple tone mapping: clamp and gamma correct
                                        let mapped = f.clamp(0.0, 1.0).powf(1.0 / 2.2);
                                        rgba8.push((mapped * 255.0) as u8);
                                    } else {
                                        rgba8.push(0);
                                    }
                                }
                            }
                        }
                        drop(data);
                        staging_buffer.unmap();

                        // Encode as PNG
                        let mut png_data = Vec::new();
                        {
                            let encoder =
                                image::codecs::png::PngEncoder::new(&mut png_data);
                            use image::ImageEncoder;
                            encoder
                                .write_image(
                                    &rgba8,
                                    ss_width,
                                    ss_height,
                                    image::ExtendedColorType::Rgba8,
                                )
                                .ok();
                        }

                        use base64::Engine;
                        let base64_str =
                            base64::engine::general_purpose::STANDARD.encode(&png_data);

                        let mut ds = self.debug_state.lock().unwrap();
                        ds.screenshot_data = Some((ss_width, ss_height, base64_str));
                    }
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Rktri - Voxel Engine")
            .with_inner_size(PhysicalSize::new(640, 360));

        let window = Arc::new(event_loop.create_window(window_attrs).expect("Failed to create window"));

        // Create GPU context
        let gpu = pollster::block_on(GpuContext::new(window.clone()))
            .expect("Failed to create GPU context");

        let size = window.inner_size();
        self.camera.set_aspect(size.width as f32, size.height as f32);

        log::info!("Window created: {}x{}", size.width, size.height);
        log::info!("GPU: {}", gpu.adapter.get_info().name);

        // Create render resources - requires pre-generated v3 world
        let world_path = self.world_path.clone().expect("No world specified. Generate one with: cargo run --release --bin generate_world -- --size <N> --name <name>");
        let resources = RenderResources::new(&gpu.device, &gpu.queue, gpu.format(), size.width, size.height, self.render_scale, &world_path, self.view_distance);

        #[cfg(feature = "dlss")]
        {
            if let Some(dlss_support) = &gpu.dlss {
                resources.init_dlss(&gpu.device, &gpu.queue, dlss_support, size.width, size.height);
            }
        }

        // Position camera at highest terrain point (mountains) for pre-generated worlds
        if self.world_path.is_some() && !resources.chunk_bounds.is_empty() {
            // Find the highest terrain point in loaded chunks
            let mut highest_point: Option<([f32; 4])> = None;
            for b in &resources.chunk_bounds {
                let chunk_top = b[1] + b[3];
                match highest_point {
                    None => highest_point = Some(*b),
                    Some(high) if chunk_top > high[1] + high[3] => highest_point = Some(*b),
                    _ => {}
                }
            }

            if let Some(b) = highest_point {
                // Spawn at center of highest chunk
                let center_x = b[0] + b[3] / 2.0;
                let center_z = b[2] + b[3] / 2.0;
                let terrain_top = b[1] + b[3];

                // ~1.7m above terrain (adult human head height)
                let cam_pos = glam::Vec3::new(center_x, terrain_top + 1.7, center_z);
                let look_target = cam_pos + glam::Vec3::new(10.0, -5.0, 10.0);
                self.camera.position = cam_pos;
                let forward = (look_target - cam_pos).normalize();
                let right = forward.cross(glam::Vec3::Y).normalize();
                let up = right.cross(forward);
                self.camera.rotation = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));
                log::info!("Camera positioned at highest point: ({:.1}, {:.1}, {:.1}), terrain_top={:.1}",
                    center_x, cam_pos.y, center_z, terrain_top);
            }
        }

        self.window = Some(window);
        self.resources = Some(resources);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.input.process_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(gpu) = &mut self.gpu {
                        gpu.resize(size.width, size.height);
                        self.camera.set_aspect(size.width as f32, size.height as f32);

                        if let Some(resources) = &mut self.resources {
                            resources.resize(&gpu.device, size.width, size.height, self.render_scale);

                            #[cfg(feature = "dlss")]
                            {
                                if let Some(dlss_support) = &gpu.dlss {
                                    // Only re-init DLSS if window size actually changed
                                    let needs_reinit = match &resources.dlss_upscaler {
                                        Some(upscaler) => {
                                            let res = upscaler.upscaled_resolution();
                                            res[0] != size.width || res[1] != size.height
                                        }
                                        None => true,
                                    };
                                    if needs_reinit {
                                        resources.init_dlss(&gpu.device, &gpu.queue, dlss_support, size.width, size.height);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                        if self.cursor_grabbed {
                            self.toggle_cursor_grab();
                        } else {
                            event_loop.exit();
                        }
                    }
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::Tab) = event.physical_key {
                        self.toggle_cursor_grab();
                    }
                    // T key: advance/decrease time by 1 hour (Shift+T decreases)
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::KeyT) = event.physical_key {
                        let current = self.atmosphere.state().time_of_day;
                        if self.input.is_key_pressed(KeyCode::ShiftLeft) || self.input.is_key_pressed(KeyCode::ShiftRight) {
                            // Shift+T: decrease time
                            self.atmosphere.set_time((current - 1.0 + 24.0) % 24.0);
                        } else {
                            // T: increase time
                            self.atmosphere.set_time((current + 1.0) % 24.0);
                        }
                        log::info!("Time of day: {:.1}:00", self.atmosphere.state().time_of_day);
                    }
                    // F1-F5 for debug modes
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            KeyCode::F1 => { self.debug_mode = 0; log::info!("Debug mode: Normal"); }
                            KeyCode::F2 => { self.debug_mode = 1; log::info!("Debug mode: Albedo"); }
                            KeyCode::F3 => { self.debug_mode = 2; log::info!("Debug mode: Normals"); }
                            KeyCode::F4 => { self.debug_mode = 3; log::info!("Debug mode: Depth"); }
                            KeyCode::F5 => { self.debug_mode = 4; log::info!("Debug mode: Material"); }
                            KeyCode::F7 => {
                                // Cycle weather presets forward
                                use rktri::atmosphere::WeatherPreset;
                                let presets = [
                                    WeatherPreset::Clear, WeatherPreset::PartlyCloudy,
                                    WeatherPreset::Overcast, WeatherPreset::Foggy,
                                    WeatherPreset::Rain, WeatherPreset::Snow, WeatherPreset::Storm,
                                ];
                                let current = self.atmosphere.current_preset();
                                let idx = presets.iter().position(|p| *p == current).unwrap_or(0);
                                let next = presets[(idx + 1) % presets.len()];
                                self.atmosphere.set_weather(next);
                                log::info!("Weather: {:?}", next);
                            }
                            KeyCode::F8 => {
                                // Cycle weather presets backward
                                use rktri::atmosphere::WeatherPreset;
                                let presets = [
                                    WeatherPreset::Clear, WeatherPreset::PartlyCloudy,
                                    WeatherPreset::Overcast, WeatherPreset::Foggy,
                                    WeatherPreset::Rain, WeatherPreset::Snow, WeatherPreset::Storm,
                                ];
                                let current = self.atmosphere.current_preset();
                                let idx = presets.iter().position(|p| *p == current).unwrap_or(0);
                                let next = presets[(idx + presets.len() - 1) % presets.len()];
                                self.atmosphere.set_weather(next);
                                log::info!("Weather: {:?}", next);
                            }
                            #[cfg(feature = "dlss")]
                            KeyCode::F6 => {
                                self.dlss_enabled = !self.dlss_enabled;
                                if let (Some(gpu), Some(resources)) = (&self.gpu, &mut self.resources) {
                                    if self.dlss_enabled {
                                        // Toggling ON: recreate render targets at DLSS render resolution
                                        if let Some(ref upscaler) = resources.dlss_upscaler {
                                            let res = upscaler.render_resolution();
                                            resources.recreate_render_targets(&gpu.device, res[0], res[1]);
                                            // Recreate depth normalization resources
                                            let gbuffer_views_for_norm = resources.gbuffer.views();
                                            let (norm_tex, norm_view, norm_pipeline, norm_bg) = RenderResources::create_dlss_depth_normalization(
                                                &gpu.device, gbuffer_views_for_norm.depth, res[0], res[1],
                                            );
                                            resources.dlss_norm_depth_texture = Some(norm_tex);
                                            resources.dlss_norm_depth_view = Some(norm_view);
                                            resources.dlss_norm_depth_pipeline = Some(norm_pipeline);
                                            resources.dlss_norm_depth_bind_group = Some(norm_bg);
                                        }
                                    } else {
                                        // Toggling OFF: restore render targets to render_scale resolution
                                        let (w, h) = gpu.size();
                                        let rw = ((w as f32 * self.render_scale) as u32).max(1);
                                        let rh = ((h as f32 * self.render_scale) as u32).max(1);
                                        resources.recreate_render_targets(&gpu.device, rw, rh);
                                    }
                                }
                                log::info!("DLSS: {}", if self.dlss_enabled { "Enabled" } else { "Disabled" });
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state.is_pressed() && button == winit::event::MouseButton::Left && !self.cursor_grabbed {
                    self.toggle_cursor_grab();
                }
            }
            WindowEvent::RedrawRequested => {
                self.timer.tick();
                let dt = self.timer.delta_secs();

                // Update grass animation time
                self.grass_time += dt;

                // Update atmosphere (time-of-day, weather transitions)
                self.atmosphere.update(dt);

                // Update camera
                self.controller.update(&mut self.camera, &self.input, dt);

                // Update FPS stats in debug state
                {
                    let stats = self.timer.fps_stats();
                    let mut ds = self.debug_state.lock().unwrap();
                    ds.current_fps = stats.current_fps;
                    ds.frame_count = stats.frame_count;
                    ds.fps_one_sec = [stats.one_sec.avg, stats.one_sec.min, stats.one_sec.max];
                    ds.fps_five_sec = [stats.five_sec.avg, stats.five_sec.min, stats.five_sec.max];
                    ds.fps_fifteen_sec = [stats.fifteen_sec.avg, stats.fifteen_sec.min, stats.fifteen_sec.max];
                }

                // Process debug commands
                self.process_debug_commands();

                // Render
                self.render();

                // Update title with FPS
                if let Some(window) = &self.window {
                    let debug_str = match self.debug_mode {
                        0 => "",
                        1 => " | Albedo",
                        2 => " | Normals",
                        3 => " | Depth",
                        4 => " | Material",
                        _ => "",
                    };
                    #[cfg(feature = "dlss")]
                    let dlss_str = if self.dlss_enabled && self.resources.as_ref().map_or(false, |r| r.dlss_upscaler.is_some()) {
                        " | DLSS"
                    } else {
                        ""
                    };
                    #[cfg(not(feature = "dlss"))]
                    let dlss_str = "";

                    let weather_str = format!(" | {:?}", self.atmosphere.current_preset());
                    window.set_title(&format!("Rktri - {:.1} FPS | Tab=mouse, WASD=move, T=time, F1-F5=debug, F7/F8=weather{}{}{}",
                        self.timer.fps(), debug_str, dlss_str, weather_str));
                }

                // End frame
                self.input.end_frame();

                // Request next frame
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.process_mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    logging::init();
    log::info!("Rktri starting...");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let world_path = parse_world_arg(&args);
    let world_size = parse_size_arg(&args);

    if let Some(ref path) = world_path {
        log::info!("Loading world from: {}", path.display());
    }
    if let Some(size) = world_size {
        log::info!("World size: {}m x {}m", size, size);
    }

    // Create shared debug state
    let debug_state = Arc::new(StdMutex::new(SharedDebugState::default()));
    // Set world path for debug commands
    {
        let mut s = debug_state.lock().unwrap();
        s.world_path = world_path.clone();
    }

    // Start debug server in background thread with tokio runtime
    let debug_state_clone = debug_state.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
        rt.block_on(async {
            let handler = Arc::new(tokio::sync::Mutex::new(AppDebugHandler {
                state: debug_state_clone,
            }));
            let _server = rktri_debug::DebugServer::start(handler, rktri_debug::DEFAULT_PORT);
            log::info!("Debug server started on port {}", rktri_debug::DEFAULT_PORT);
            // Keep runtime alive forever
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
    });

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new(world_path, world_size, debug_state);

    event_loop.run_app(&mut app).expect("Event loop error");
}

/// Parse --world argument from command line
fn parse_world_arg(args: &[String]) -> Option<PathBuf> {
    for i in 0..args.len() {
        if args[i] == "--world" || args[i] == "-w" {
            if let Some(world_name) = args.get(i + 1) {
                return Some(PathBuf::from(format!("assets/worlds/{}", world_name)));
            }
        }
    }
    None
}

/// Parse --size argument from command line (world size in meters)
fn parse_size_arg(args: &[String]) -> Option<f32> {
    for i in 0..args.len() {
        if args[i] == "--size" || args[i] == "-s" {
            if let Some(size_str) = args.get(i + 1) {
                return size_str.parse().ok();
            }
        }
    }
    None
}
