//! Object Viewer - Minimal testing binary for viewing .rkc files
//!
//! Usage:
//!   cargo run --release --bin object_viewer -- --file <path>    # Load .rkc file
//!   cargo run --release --bin object_viewer -- --file <path> --bg-color <hex>
//!
//! Controls:
//!   WASD - Move camera
//!   Mouse - Look around
//!   B - Cycle background colors
//!   Escape - Release mouse / exit

use std::sync::{Arc, Mutex as StdMutex};
use std::path::PathBuf;

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
};
use rktri::render::{
    buffer::{OctreeBuffer, CameraBuffer},
    pipeline::{SvoTracePipeline, DisplayPipeline},
    pipeline::svo_trace::TraceParams,
    texture::GBuffer,
    context::GpuContext,
};
use bytemuck::Zeroable;
use rktri::grass::{GrassParams, GpuGrassProfile};
use rktri::streaming::disk_io;
use rktri::voxel::chunk::CHUNK_SIZE;
use rktri::voxel::svo::Octree;
use rktri::math::Aabb;
use rktri_debug::{DebugServer, DebugHandler, protocol::{DebugCommand, DebugResponse, ResponseData}};

const DEFAULT_BG_COLORS: [&str; 5] = [
    "#87CEEB", // Sky blue
    "#1a1a2e", // Dark night
    "#2d4a3e", // Forest green
    "#8b4513", // Saddle brown (earth)
    "#000000", // Black
];

/// Compute the axis-aligned bounding box of all occupied voxels in the octree
fn compute_auto_camera_bounds(octree: &Octree) -> Aabb {
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);
    let mut has_any = false;

    octree.iterate_voxels(|pos, _voxel| {
        has_any = true;
        min = min.min(pos);
        max = max.max(pos);
    });

    if has_any {
        // Add small padding
        let padding = 0.1;
        Aabb::new(min - padding, max + padding)
    } else {
        // Empty octree - use root_size as fallback
        let root_size = octree.root_size();
        Aabb::new(glam::Vec3::ZERO, glam::Vec3::splat(root_size))
    }
}

/// Compute optimal camera distance to fit the object in view
/// Based on the bounding box diagonal and viewing frustum
fn compute_optimal_camera_distance(aabb: &Aabb, fov: f32, aspect: f32, fill_factor: f32) -> f32 {
    let size = aabb.size();
    let max_dim = size.x.max(size.y).max(size.z);

    // Calculate frustum height at unit distance
    let frustum_height_at_1 = 2.0 * (fov / 2.0).tan();

    // Calculate required distance for each axis
    let dist_y = (max_dim / 2.0) / ((frustum_height_at_1 * fill_factor) / 2.0);
    let dist_x = (size.z.max(size.y) / 2.0) / ((frustum_height_at_1 * aspect * fill_factor) / 2.0);

    dist_y.max(dist_x)
}

struct ViewerState {
    camera: Camera,
    camera_controller: FpsCameraController,
    input: InputState,
    camera_buffer: CameraBuffer,
    octree_buffer: OctreeBuffer,
    svo_pipeline: SvoTracePipeline,
    display_pipeline: DisplayPipeline,
    gbuffer: GBuffer,
    gbuffer_output_bind_group: wgpu::BindGroup,
    display_bind_group: wgpu::BindGroup,
    bg_color_index: usize,
    pending_screenshot: bool,
    chunk_count: u32,
    grid_min: [i32; 3],
    grid_size: [u32; 3],
}

impl ViewerState {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        octree: rktri::voxel::svo::Octree,
    ) -> Self {
        // Compute auto camera bounds from octree
        let aabb = compute_auto_camera_bounds(&octree);
        let center = aabb.center();
        let size = aabb.size();

        // Calculate optimal camera distance
        let fov = 70.0_f32.to_radians();
        let aspect = width as f32 / height as f32;
        let distance = compute_optimal_camera_distance(&aabb, fov, aspect, 1.5);

        // Position camera at diagonal direction from center (pointing toward center)
        // Camera should be on OPPOSITE side looking AT center
        let from_center = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        let camera_pos = center + from_center * distance;

        // Create camera looking at object center
        let mut camera = Camera::look_at(camera_pos, center, glam::Vec3::Y);
        eprintln!("DEBUG: After look_at, forward = {:?}", camera.forward());

        // Create camera controller and set its orientation to match the camera
        let mut camera_controller = FpsCameraController::new(5.0, 0.5);
        let (yaw, pitch) = camera.euler_angles();
        camera_controller.set_orientation(yaw, pitch);
        eprintln!("DEBUG: Controller orientation set: yaw={}, pitch={}", yaw, pitch);

        let input = InputState::new();

        // Create buffers
        let camera_buffer = CameraBuffer::new(device);
        let mut octree_buffer = OctreeBuffer::new(
            device,
            octree.node_count().max(1) as u32,
            octree.brick_count().max(1) as u32,
        );

        // Upload octree - chunks are at origin in the file
        let world_min = [0.0, 0.0, 0.0];
        let root_size = octree.root_size();
        octree_buffer.upload_chunk_incremental(
            queue,
            &octree,
            world_min,
            root_size,
            0, // layer_id: terrain
            0, // flags: opaque
        );

        // Write chunk info for single chunk
        let chunk_info = rktri::render::buffer::octree_buffer::GpuChunkInfo {
            world_min,
            root_size,
            root_node: 0,
            max_depth: 8,
            layer_id: 0,
            flags: 0,
        };
        octree_buffer.update_chunk_infos(queue, &[chunk_info]);

        // Upload single-chunk grid
        let chunk_indices = [0u32];
        octree_buffer.upload_chunk_grid(device, queue, &[], &chunk_indices);

        // Single chunk at origin
        let chunk_count = 1;
        let grid_min = [0, 0, 0];
        let grid_size = [1, 1, 1];

        // Create pipelines
        let svo_pipeline = SvoTracePipeline::new(device, &camera_buffer, &octree_buffer);
        let display_pipeline = DisplayPipeline::new(device, surface_format);

        // Update trace params
        let params = TraceParams {
            width,
            height,
            chunk_count,
            _pad0: 0,
            lod_distances: [64.0, 128.0, 256.0, 512.0],
            lod_distances_ext: [1024.0, f32::MAX],
            _pad: [0.0; 2],
            grid_min_x: grid_min[0],
            grid_min_y: grid_min[1],
            grid_min_z: grid_min[2],
            chunk_size: CHUNK_SIZE as f32,
            grid_size_x: grid_size[0],
            grid_size_y: grid_size[1],
            grid_size_z: grid_size[2],
            _pad2: 0,
        };
        svo_pipeline.update_params(queue, &params);

        // Update grass params (disabled - no grass in object viewer)
        let grass_params = GrassParams {
            enabled: 0,
            max_distance: 80.0,
            fade_start: 40.0,
            time: 0.0,
            wind_direction: [1.0, 0.0, 0.0],
            wind_speed: 1.0,
            profile_count: 0,
            _pad: [0.0; 3],
        };
        svo_pipeline.update_grass_params(queue, &grass_params);

        // Update profile table (empty - max 16 profiles)
        let profiles = vec![GpuGrassProfile::zeroed(); 16];
        svo_pipeline.update_profile_table(queue, &profiles);

        // Create G-buffer at window resolution
        let gbuffer = GBuffer::new(device, width, height);

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

        let display_bind_group = display_pipeline.create_bind_group(device, &gbuffer_views.albedo);

        Self {
            camera,
            camera_controller,
            input,
            camera_buffer,
            octree_buffer,
            svo_pipeline,
            display_pipeline,
            gbuffer,
            gbuffer_output_bind_group,
            display_bind_group,
            bg_color_index: 0,
            pending_screenshot: false,
            chunk_count,
            grid_min,
            grid_size,
        }
    }

    fn reload_octree(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, octree: rktri::voxel::svo::Octree, width: u32, height: u32) {
        // Compute auto camera bounds from octree
        let aabb = compute_auto_camera_bounds(&octree);
        let center = aabb.center();

        // Calculate optimal camera distance
        let fov = 70.0_f32.to_radians();
        let aspect = width as f32 / height as f32;
        let distance = compute_optimal_camera_distance(&aabb, fov, aspect, 1.5);

        // Position camera at diagonal direction from center (pointing toward center)
        let from_center = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        let camera_pos = center + from_center * distance;

        // Update camera to look at object center
        self.camera = Camera::look_at(camera_pos, center, glam::Vec3::Y);

        // Recreate octree buffer with new size
        self.octree_buffer = OctreeBuffer::new(
            device,
            octree.node_count().max(1) as u32,
            octree.brick_count().max(1) as u32,
        );

        // Upload octree
        let world_min = [0.0, 0.0, 0.0];
        let root_size = octree.root_size();
        self.octree_buffer.upload_chunk_incremental(
            queue,
            &octree,
            world_min,
            root_size,
            0,
            0,
        );

        // Write chunk info
        let chunk_info = rktri::render::buffer::octree_buffer::GpuChunkInfo {
            world_min,
            root_size,
            root_node: 0,
            max_depth: 8,
            layer_id: 0,
            flags: 0,
        };
        self.octree_buffer.update_chunk_infos(queue, &[chunk_info]);
        let chunk_indices = [0u32];
        self.octree_buffer.upload_chunk_grid(device, queue, &[], &chunk_indices);

        // Recreate SVO pipeline with new buffer
        self.svo_pipeline = SvoTracePipeline::new(device, &self.camera_buffer, &self.octree_buffer);

        // Re-update params
        let params = TraceParams {
            width: self.chunk_count, // placeholder - will be updated in render
            height: 0,
            chunk_count: self.chunk_count,
            _pad0: 0,
            lod_distances: [64.0, 128.0, 256.0, 512.0],
            lod_distances_ext: [1024.0, f32::MAX],
            _pad: [0.0; 2],
            grid_min_x: self.grid_min[0],
            grid_min_y: self.grid_min[1],
            grid_min_z: self.grid_min[2],
            chunk_size: CHUNK_SIZE as f32,
            grid_size_x: self.grid_size[0],
            grid_size_y: self.grid_size[1],
            grid_size_z: self.grid_size[2],
            _pad2: 0,
        };
        self.svo_pipeline.update_params(queue, &params);

        log::info!("Reloaded octree: {} nodes, {} bricks", octree.node_count(), octree.brick_count());
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.gbuffer.resize(device, width, height);
        let gbuffer_views = self.gbuffer.views();
        self.gbuffer_output_bind_group = self.svo_pipeline.create_output_bind_group(
            device,
            gbuffer_views.albedo,
            gbuffer_views.normal,
            gbuffer_views.depth,
            gbuffer_views.material,
            gbuffer_views.motion,
        );
        self.display_bind_group = self.display_pipeline.create_bind_group(device, &gbuffer_views.albedo);
    }

    fn update_trace_params(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let params = TraceParams {
            width,
            height,
            chunk_count: self.chunk_count,
            _pad0: 0,
            lod_distances: [64.0, 128.0, 256.0, 512.0],
            lod_distances_ext: [1024.0, f32::MAX],
            _pad: [0.0; 2],
            grid_min_x: self.grid_min[0],
            grid_min_y: self.grid_min[1],
            grid_min_z: self.grid_min[2],
            chunk_size: CHUNK_SIZE as f32,
            grid_size_x: self.grid_size[0],
            grid_size_y: self.grid_size[1],
            grid_size_z: self.grid_size[2],
            _pad2: 0,
        };
        self.svo_pipeline.update_params(queue, &params);
    }
}

#[derive(Default)]
struct SharedDebugState {
    screenshot_requested: bool,
    camera_pos: Option<[f32; 3]>,
    camera_forward: Option<[f32; 3]>,
    camera_fov: Option<f32>,
}

struct AppDebugHandler {
    state: Arc<StdMutex<SharedDebugState>>,
}

impl DebugHandler for AppDebugHandler {
    fn handle_command(&mut self, cmd: DebugCommand) -> DebugResponse {
        match cmd {
            DebugCommand::TakeScreenshot => {
                self.state.lock().unwrap().screenshot_requested = true;
                DebugResponse::ok(ResponseData::None)
            }
            DebugCommand::CameraGetState => {
                let state = self.state.lock().unwrap();
                let pos = state.camera_pos.unwrap_or([0.0, 0.0, 0.0]);
                let forward = state.camera_forward.unwrap_or([0.0, 0.0, -1.0]);
                let fov = state.camera_fov.unwrap_or(70.0);
                DebugResponse::ok(ResponseData::CameraState {
                    position: pos,
                    forward,
                    fov_degrees: fov,
                })
            }
            DebugCommand::CameraMoveTo { x, y, z } => {
                self.state.lock().unwrap().camera_pos = Some([x, y, z]);
                // CameraMoveTo is stored in debug_state, applied in about_to_wait
                DebugResponse::ok(ResponseData::None)
            }
            _ => DebugResponse::error("Command not supported in object_viewer"),
        }
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    state: Option<ViewerState>,
    width: u32,
    height: u32,
    debug_state: Arc<StdMutex<SharedDebugState>>,
    loaded_file: Option<PathBuf>,
    cursor_grabbed: bool,
}

impl App {
    fn new(loaded_file: PathBuf) -> Self {
        Self {
            window: None,
            gpu: None,
            state: None,
            width: 1280,
            height: 720,
            debug_state: Arc::new(StdMutex::new(SharedDebugState::default())),
            loaded_file: Some(loaded_file),
            cursor_grabbed: false,
        }
    }

    fn toggle_cursor_grab(&mut self) {
        eprintln!("DEBUG: toggle_cursor_grab called, window.is_some={}", self.window.is_some());
        if let Some(window) = &self.window {
            let new_grab_state = !self.cursor_grabbed;
            eprintln!("DEBUG: Attempting cursor grab: current={} wanted={}", self.cursor_grabbed, new_grab_state);
            let result = if new_grab_state {
                // Try Confined first
                eprintln!("DEBUG: Trying Confined...");
                let r = window.set_cursor_grab(CursorGrabMode::Confined);
                if r.is_err() {
                    eprintln!("DEBUG: Confined failed, trying Locked...");
                    window.set_cursor_grab(CursorGrabMode::Locked)
                } else {
                    r
                }
            } else {
                eprintln!("DEBUG: Releasing cursor grab...");
                window.set_cursor_grab(CursorGrabMode::None)
            };
            match result {
                Ok(()) => {
                    self.cursor_grabbed = new_grab_state;
                    window.set_cursor_visible(!self.cursor_grabbed);
                    eprintln!("DEBUG: Cursor grab SUCCESS: {} (visible={})", self.cursor_grabbed, !self.cursor_grabbed);
                    if let Some(ref mut state) = self.state {
                        state.input.set_mouse_captured(self.cursor_grabbed);
                    }
                }
                Err(e) => {
                    eprintln!("DEBUG: Cursor grab FAILED: {:?} (wanted={})", e, new_grab_state);
                }
            }
        } else {
            eprintln!("DEBUG: toggle_cursor_grab called but window is None!");
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // Parse command line args
        let args: Vec<String> = std::env::args().collect();

        let bg_color = args.iter()
            .position(|a| a == "--bg-color")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
            .unwrap_or(DEFAULT_BG_COLORS[0]);

        let bg_color_index = DEFAULT_BG_COLORS.iter().position(|c| *c == bg_color).unwrap_or(0);

        // Load file path from args
        let file_path = args.iter()
            .position(|a| a == "--file")
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
            .expect("Usage: object_viewer --file <path>");

        log::info!("Loading: {}", file_path.display());

        // Load octree from file using the same approach as main.rs
        let compressed = std::fs::read(&file_path)
            .expect("Failed to read chunk file");
        let chunk = disk_io::decompress_chunk(&compressed)
            .expect("Failed to decompress chunk");
        let octree = chunk.octree;
        log::info!("Octree: {} nodes, {} bricks", octree.node_count(), octree.brick_count());

        // Create window
        let window = event_loop.create_window(Window::default_attributes()
            .with_title(format!("Object Viewer - {}", file_path.file_name().unwrap_or_default().to_string_lossy()))
            .with_inner_size(PhysicalSize::new(self.width, self.height)))
            .expect("Failed to create window");

        // Keep Arc reference to window for cursor grab
        let window_arc = Arc::new(window);
        self.window = Some(window_arc.clone());

        // Don't capture mouse automatically - wait for user click
        // User must click to capture, press Escape to release

        // Initialize GPU context
        let gpu = pollster::block_on(GpuContext::new(window_arc))
            .expect("Failed to create GPU context");
        self.gpu = Some(gpu);

        let surface_format = self.gpu.as_ref().unwrap().config.format;

        // Create viewer state
        let gpu_ref = self.gpu.as_ref().unwrap();
        let mut state = ViewerState::new(&gpu_ref.device, &gpu_ref.queue, surface_format, self.width, self.height, octree);
        state.bg_color_index = bg_color_index;
        // Don't capture mouse on startup - wait for user click
        state.input.set_mouse_captured(false);
        self.state = Some(state);

        // Start debug server
        let debug_state_clone = self.debug_state.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");
            rt.block_on(async {
                let handler = Arc::new(tokio::sync::Mutex::new(AppDebugHandler {
                    state: debug_state_clone,
                }));
                let _server = DebugServer::start(handler, 9742);
                log::info!("Debug server started on port 9742");
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                }
            });
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        // Debug log ALL events
        log::debug!("window_event: {:?}", event);

        // Process input through InputState
        if let Some(ref mut state) = self.state {
            state.input.process_event(&event);
        }

        // Debug log all keyboard events
        if let WindowEvent::KeyboardInput { event, .. } = &event {
            log::debug!("Keyboard: {:?} pressed={}", event.physical_key, event.state.is_pressed());
        }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.width = size.width;
                self.height = size.height;
                if let Some(ref mut gpu) = self.gpu {
                    if let Some(ref mut state) = self.state {
                        gpu.resize(self.width, self.height);
                        state.resize(&gpu.device, self.width, self.height);
                    }
                }
            }
            WindowEvent::Focused(focused) => {
                log::debug!("Window focus changed: {}", focused);
                // Auto-capture when window gains focus if not already captured
                if focused && !self.cursor_grabbed {
                    log::info!("Window gained focus - auto-capturing mouse");
                    self.toggle_cursor_grab();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                log::info!("MouseInput: button={:?} pressed={} grabbed={}", button, state.is_pressed(), self.cursor_grabbed);
                // Click to capture mouse
                if state.is_pressed() && button == winit::event::MouseButton::Left && !self.cursor_grabbed {
                    log::info!("Click detected - capturing mouse");
                    self.toggle_cursor_grab();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    // Escape to release cursor or exit
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                        if self.cursor_grabbed {
                            self.toggle_cursor_grab();
                        } else {
                            event_loop.exit();
                        }
                    }
                    // Tab to toggle cursor grab
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::Tab) = event.physical_key {
                        self.toggle_cursor_grab();
                    }
                    // 'B' key cycles background colors
                    if event.logical_key == winit::keyboard::Key::Character("b".into()) {
                        if let Some(ref mut state) = self.state {
                            state.bg_color_index = (state.bg_color_index + 1) % DEFAULT_BG_COLORS.len();
                            let color = DEFAULT_BG_COLORS[state.bg_color_index];
                            log::info!("Background color: {}", color);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        // Debug log all device events
        match &event {
            DeviceEvent::MouseMotion { delta } => {
                log::debug!("Mouse motion: {:?}", delta);
            }
            DeviceEvent::Key(key_event) => {
                log::debug!("Device key: {:?} state={:?}", key_event.physical_key, key_event.state);
            }
            _ => {}
        }

        // Process mouse motion through InputState
        if let Some(ref mut state) = self.state {
            if let DeviceEvent::MouseMotion { delta } = event {
                state.input.process_mouse_motion(delta);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request redraw to keep rendering
        if let Some(ref window) = self.window {
            window.request_redraw();
        }

        let Some(ref mut gpu) = self.gpu else { return };
        let Some(ref mut state) = self.state else { return };

        // Apply any pending camera move from debug commands
        {
            let mut debug_state = self.debug_state.lock().unwrap();
            if let Some(pos) = debug_state.camera_pos.take() {
                state.camera.position = glam::Vec3::new(pos[0], pos[1], pos[2]);
            }
        }

        // Debug: log input state
        let keys = [
            (KeyCode::KeyW, "W"),
            (KeyCode::KeyA, "A"),
            (KeyCode::KeyS, "S"),
            (KeyCode::KeyD, "D"),
            (KeyCode::Space, "Space"),
        ];
        let pressed: Vec<_> = keys.iter()
            .filter(|(k, _)| state.input.is_key_pressed(*k))
            .map(|(_, n)| *n)
            .collect();
        if !pressed.is_empty() || state.input.is_mouse_captured() {
            log::debug!("Input: keys={:?} mouse_captured={}", pressed, state.input.is_mouse_captured());
        }

        // Update camera using InputState
        let dt = 1.0 / 60.0;
        state.camera_controller.update(&mut state.camera, &state.input, dt);

        // End frame to reset per-frame input state
        state.input.end_frame();

        // Upload camera
        state.camera_buffer.update(&gpu.queue, &state.camera);

        // Update trace params for current resolution
        state.update_trace_params(&gpu.queue, self.width, self.height);

        // Process screenshot request and update camera state for debug
        {
            let mut debug_state = self.debug_state.lock().unwrap();
            if debug_state.screenshot_requested {
                state.pending_screenshot = true;
                debug_state.screenshot_requested = false;
            }
            // Update camera state for debug commands
            debug_state.camera_pos = Some(state.camera.position.into());
            debug_state.camera_forward = Some(state.camera.forward().into());
            debug_state.camera_fov = Some(state.camera.fov_y * 180.0 / std::f32::consts::PI);
        }

        // Render
        let frame = match gpu.get_current_texture() {
            Ok(frame) => frame,
            Err(e) => {
                log::error!("Failed to get frame: {}", e);
                return;
            }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Clear with background color
        let bg_hex = DEFAULT_BG_COLORS[state.bg_color_index];
        let bg_color = parse_hex_color(bg_hex);
        let clear_color = wgpu::Color {
            r: bg_color[0] as f64 / 255.0,
            g: bg_color[1] as f64 / 255.0,
            b: bg_color[2] as f64 / 255.0,
            a: 1.0,
        };

        // SVO trace pass - use dispatch method
        state.svo_pipeline.dispatch(
            &mut encoder,
            &state.octree_buffer,
            &state.gbuffer_output_bind_group,
            self.width,
            self.height,
            None,
        );

        // Display pass - use render method
        state.display_pipeline.render(
            &mut encoder,
            &view,
            &state.display_bind_group,
            None,
        );

        gpu.queue.submit([encoder.finish()]);
        frame.present();
    }
}

fn parse_hex_color(hex: &str) -> [u8; 3] {
    let hex = hex.trim_start_matches('#');
    if hex.len() >= 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        [r, g, b]
    } else {
        [0, 0, 0]
    }
}

fn main() {
    logging::init();
    log::info!("Object Viewer starting...");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let loaded_file = args.iter()
        .position(|a| a == "--file")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    let loaded_file = loaded_file.expect("Usage: object_viewer --file <path>");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new(loaded_file);

    event_loop.run_app(&mut app).expect("Event loop error");
}
