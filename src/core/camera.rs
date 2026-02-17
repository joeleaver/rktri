//! Camera for 3D rendering

use crate::core::types::{Vec3, Mat4, Quat};

/// Camera with position, rotation, and projection parameters
pub struct Camera {
    /// World position
    pub position: Vec3,
    /// Rotation as quaternion
    pub rotation: Quat,
    /// Vertical field of view in radians
    pub fov_y: f32,
    /// Aspect ratio (width / height)
    pub aspect: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
    /// Whether camera is currently underwater
    pub is_underwater: bool,
    /// Depth below water surface (0.0 if above)
    pub water_depth: f32,
    /// Y coordinate of water surface above camera
    pub water_surface_y: f32,
}

impl Camera {
    /// Create a new camera
    pub fn new(position: Vec3, fov_y_degrees: f32, aspect: f32) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            fov_y: fov_y_degrees.to_radians(),
            aspect,
            near: 0.01,
            far: 1000.0,
            is_underwater: false,
            water_depth: 0.0,
            water_surface_y: 0.0,
        }
    }

    /// Create camera looking at a target
    pub fn look_at(position: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - position).normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);

        let rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));

        Self {
            position,
            rotation,
            fov_y: 60.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.01,
            far: 1000.0,
            is_underwater: false,
            water_depth: 0.0,
            water_surface_y: 0.0,
        }
    }

    /// Get view matrix (world to camera space)
    pub fn view_matrix(&self) -> Mat4 {
        let rotation_matrix = Mat4::from_quat(self.rotation.conjugate());
        let translation_matrix = Mat4::from_translation(-self.position);
        rotation_matrix * translation_matrix
    }

    /// Get projection matrix (camera to clip space)
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Get combined view-projection matrix
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get inverse view-projection matrix (for ray generation)
    pub fn view_projection_inverse(&self) -> Mat4 {
        self.view_projection().inverse()
    }

    /// Get forward direction (negative Z in camera space)
    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    /// Get right direction (positive X in camera space)
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Get up direction (positive Y in camera space)
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Set rotation from euler angles (yaw, pitch in radians)
    pub fn set_rotation_euler(&mut self, yaw: f32, pitch: f32) {
        self.rotation = Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
    }

    /// Get rotation as euler angles (yaw, pitch in radians)
    pub fn euler_angles(&self) -> (f32, f32) {
        let f = self.forward();
        // yaw: rotation around Y axis (horizontal)
        let yaw = f.z.atan2(f.x);
        // pitch: rotation around X axis (vertical)
        let pitch = (-f.y).asin();
        (yaw, pitch)
    }

    /// Update aspect ratio (call on window resize)
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
    }

    /// Update underwater state from water system data.
    /// Call each frame with the water surface height at the camera's XZ position.
    pub fn update_water_state(&mut self, water_surface_height: f32) {
        self.water_surface_y = water_surface_height;
        self.is_underwater = self.position.y < water_surface_height;
        self.water_depth = if self.is_underwater {
            water_surface_height - self.position.y
        } else {
            0.0
        };
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 0.0, 5.0), 60.0, 16.0 / 9.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directions() {
        let camera = Camera::default();

        // Default camera looks down -Z
        let forward = camera.forward();
        assert!((forward.z - (-1.0)).abs() < 0.001);

        let right = camera.right();
        assert!((right.x - 1.0).abs() < 0.001);

        let up = camera.up();
        assert!((up.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_view_matrix_translation() {
        let mut camera = Camera::default();
        camera.position = Vec3::new(10.0, 0.0, 0.0);

        let view = camera.view_matrix();
        // View matrix should translate world origin to (-10, 0, 0) in camera space
        let origin_in_camera = view.transform_point3(Vec3::ZERO);
        assert!((origin_in_camera.x - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_projection_inverse() {
        let camera = Camera::default();
        let vp = camera.view_projection();
        let vp_inv = camera.view_projection_inverse();

        // VP * VP^-1 should be identity
        let identity = vp * vp_inv;
        assert!((identity.w_axis.w - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_underwater_detection() {
        let mut camera = Camera::default();
        camera.position = Vec3::new(0.0, -5.0, 0.0);
        camera.update_water_state(10.0); // water at y=10
        assert!(camera.is_underwater);
        assert!((camera.water_depth - 15.0).abs() < 0.001);

        camera.position = Vec3::new(0.0, 15.0, 0.0);
        camera.update_water_state(10.0);
        assert!(!camera.is_underwater);
        assert_eq!(camera.water_depth, 0.0);
    }
}
