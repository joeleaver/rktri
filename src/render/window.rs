//! Window management using winit

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window as WinitWindow, WindowId, WindowAttributes},
};

use crate::core::error::Error;

/// Window configuration
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Rktri".to_string(),
            width: 640,
            height: 360,
        }
    }
}

/// Application handler that manages the window lifecycle
struct App<F>
where
    F: FnMut(&ActiveEventLoop, Arc<WinitWindow>),
{
    config: WindowConfig,
    window: Option<Arc<WinitWindow>>,
    app_fn: F,
}

impl<F> App<F>
where
    F: FnMut(&ActiveEventLoop, Arc<WinitWindow>),
{
    fn new(config: WindowConfig, app_fn: F) -> Self {
        Self {
            config,
            window: None,
            app_fn,
        }
    }
}

impl<F> ApplicationHandler for App<F>
where
    F: FnMut(&ActiveEventLoop, Arc<WinitWindow>),
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = WindowAttributes::default()
                .with_title(self.config.title.clone())
                .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    let window = Arc::new(window);
                    (self.app_fn)(event_loop, window.clone());
                    self.window = Some(window);
                }
                Err(e) => {
                    log::error!("Failed to create window: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Creates an event loop and runs the application
pub fn run<F>(config: WindowConfig, app_fn: F) -> Result<(), Error>
where
    F: FnMut(&ActiveEventLoop, Arc<WinitWindow>) + 'static,
{
    let event_loop = EventLoop::new()
        .map_err(|e| Error::Window(format!("Failed to create event loop: {}", e)))?;

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config, app_fn);

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Window(format!("Event loop error: {}", e)))?;

    Ok(())
}
