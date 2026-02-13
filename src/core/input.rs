//! Input state tracking

use std::collections::HashSet;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Tracks keyboard and mouse input state
pub struct InputState {
    /// Currently pressed keys
    keys_pressed: HashSet<KeyCode>,
    /// Keys pressed this frame
    keys_just_pressed: HashSet<KeyCode>,
    /// Keys released this frame
    keys_just_released: HashSet<KeyCode>,
    /// Mouse movement delta since last frame
    mouse_delta: (f32, f32),
    /// Accumulated mouse delta (for when cursor is grabbed)
    mouse_delta_accumulated: (f32, f32),
    /// Current mouse position
    mouse_position: (f32, f32),
    /// Currently pressed mouse buttons
    mouse_buttons: HashSet<MouseButton>,
    /// Whether mouse is captured
    mouse_captured: bool,
}

impl InputState {
    /// Create new input state
    pub fn new() -> Self {
        Self {
            keys_pressed: HashSet::new(),
            keys_just_pressed: HashSet::new(),
            keys_just_released: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            mouse_delta_accumulated: (0.0, 0.0),
            mouse_position: (0.0, 0.0),
            mouse_buttons: HashSet::new(),
            mouse_captured: false,
        }
    }

    /// Process a window event
    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key_code),
                    state,
                    ..
                },
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        if !self.keys_pressed.contains(key_code) {
                            self.keys_just_pressed.insert(*key_code);
                        }
                        self.keys_pressed.insert(*key_code);
                    }
                    ElementState::Released => {
                        self.keys_pressed.remove(key_code);
                        self.keys_just_released.insert(*key_code);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x as f32, position.y as f32);
                if !self.mouse_captured {
                    self.mouse_delta.0 += new_pos.0 - self.mouse_position.0;
                    self.mouse_delta.1 += new_pos.1 - self.mouse_position.1;
                }
                self.mouse_position = new_pos;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_buttons.insert(*button);
                    }
                    ElementState::Released => {
                        self.mouse_buttons.remove(button);
                    }
                }
            }
            _ => {}
        }
    }

    /// Process device event for raw mouse motion (when cursor is grabbed)
    pub fn process_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_delta_accumulated.0 += delta.0 as f32;
        self.mouse_delta_accumulated.1 += delta.1 as f32;
    }

    /// Call at end of frame to reset per-frame state
    pub fn end_frame(&mut self) {
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();

        if self.mouse_captured {
            self.mouse_delta = self.mouse_delta_accumulated;
        }
        self.mouse_delta_accumulated = (0.0, 0.0);

        if !self.mouse_captured {
            self.mouse_delta = (0.0, 0.0);
        }
    }

    /// Check if key is currently pressed
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Check if key was just pressed this frame
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys_just_pressed.contains(&key)
    }

    /// Check if key was just released this frame
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.keys_just_released.contains(&key)
    }

    /// Get mouse delta since last frame
    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    /// Get current mouse position
    pub fn mouse_position(&self) -> (f32, f32) {
        self.mouse_position
    }

    /// Check if mouse button is pressed
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }

    /// Set mouse captured state
    pub fn set_mouse_captured(&mut self, captured: bool) {
        self.mouse_captured = captured;
        if captured {
            self.mouse_delta = (0.0, 0.0);
            self.mouse_delta_accumulated = (0.0, 0.0);
        }
    }

    /// Check if mouse is captured
    pub fn is_mouse_captured(&self) -> bool {
        self.mouse_captured
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_press() {
        let mut input = InputState::new();

        assert!(!input.is_key_pressed(KeyCode::KeyW));

        // Simulate key press event would be tested with actual WindowEvent
        input.keys_pressed.insert(KeyCode::KeyW);
        input.keys_just_pressed.insert(KeyCode::KeyW);

        assert!(input.is_key_pressed(KeyCode::KeyW));
        assert!(input.is_key_just_pressed(KeyCode::KeyW));

        input.end_frame();

        assert!(input.is_key_pressed(KeyCode::KeyW));
        assert!(!input.is_key_just_pressed(KeyCode::KeyW));
    }
}
