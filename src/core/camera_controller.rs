//! First-person camera controller

use crate::core::camera::Camera;
use crate::core::input::InputState;
use winit::keyboard::KeyCode;

/// FPS-style camera controller with WASD movement and mouse look
pub struct FpsCameraController {
    /// Movement speed in units per second
    pub speed: f32,
    /// Mouse sensitivity
    pub sensitivity: f32,
    /// Current yaw (rotation around Y axis) in radians
    yaw: f32,
    /// Current pitch (rotation around X axis) in radians
    pitch: f32,
    /// Sprint multiplier
    pub sprint_multiplier: f32,
}

impl FpsCameraController {
    /// Create new controller
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: 0.0,
            pitch: 0.0,
            sprint_multiplier: 2.0,
        }
    }

    /// Update camera based on input
    pub fn update(&mut self, camera: &mut Camera, input: &InputState, dt: f32) {
        // Mouse look (only when captured)
        if input.is_mouse_captured() {
            let (dx, dy) = input.mouse_delta();
            self.yaw -= dx * self.sensitivity * 0.001;
            self.pitch -= dy * self.sensitivity * 0.001;

            // Clamp pitch to prevent gimbal lock
            self.pitch = self.pitch.clamp(-1.5, 1.5);

            camera.set_rotation_euler(self.yaw, self.pitch);
        }

        // Movement
        let mut velocity = glam::Vec3::ZERO;
        let forward = camera.forward();
        let right = camera.right();

        // Forward/backward
        if input.is_key_pressed(KeyCode::KeyW) {
            velocity += forward;
        }
        if input.is_key_pressed(KeyCode::KeyS) {
            velocity -= forward;
        }

        // Strafe left/right
        if input.is_key_pressed(KeyCode::KeyA) {
            velocity -= right;
        }
        if input.is_key_pressed(KeyCode::KeyD) {
            velocity += right;
        }

        // Up/down
        if input.is_key_pressed(KeyCode::Space) {
            velocity.y += 1.0;
        }
        if input.is_key_pressed(KeyCode::ShiftLeft) || input.is_key_pressed(KeyCode::ShiftRight) {
            velocity.y -= 1.0;
        }

        // Normalize and apply speed
        if velocity.length_squared() > 0.0 {
            velocity = velocity.normalize();

            let mut speed = self.speed;
            if input.is_key_pressed(KeyCode::ControlLeft) {
                speed *= self.sprint_multiplier;
            }

            camera.position += velocity * speed * dt;
        }
    }

    /// Set initial orientation from angles (in radians)
    pub fn set_orientation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-1.5, 1.5);
    }

    /// Get current yaw
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Get current pitch
    pub fn pitch(&self) -> f32 {
        self.pitch
    }
}

impl Default for FpsCameraController {
    fn default() -> Self {
        Self::new(10.0, 1.0)
    }
}
