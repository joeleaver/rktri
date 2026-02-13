//! GPU uniform buffer for camera data

use bytemuck::{Pod, Zeroable};
use crate::core::camera::Camera;

/// Camera uniform data for GPU (must match shader struct exactly)
/// WGSL vec3 has 16-byte alignment, so we need explicit padding
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix (64 bytes, offset 0)
    pub view_proj: [[f32; 4]; 4],
    /// Inverse view-projection matrix (64 bytes, offset 64)
    pub view_proj_inv: [[f32; 4]; 4],
    /// Camera position in world space (12 bytes, offset 128)
    pub position: [f32; 3],
    /// Padding after position for vec3 alignment (4 bytes, offset 140)
    pub _pos_pad: f32,
    /// Near clip plane (4 bytes, offset 144)
    pub near: f32,
    /// Far clip plane (4 bytes, offset 148)
    pub far: f32,
    /// Padding to align world_offset to 16 bytes (8 bytes, offset 152)
    pub _pad2: [f32; 2],
    /// World offset - min world position of octree (12 bytes, offset 160)
    pub world_offset: [f32; 3],
    /// Final padding to 176 bytes (4 bytes, offset 172)
    pub _pad3: f32,
}

impl CameraUniform {
    /// Create uniform data from camera with world offset
    pub fn from_camera_with_offset(camera: &Camera, world_offset: glam::Vec3) -> Self {
        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
            view_proj_inv: camera.view_projection_inverse().to_cols_array_2d(),
            position: camera.position.to_array(),
            _pos_pad: 0.0,
            near: camera.near,
            far: camera.far,
            _pad2: [0.0; 2],
            world_offset: world_offset.to_array(),
            _pad3: 0.0,
        }
    }

    /// Create uniform data from camera (deprecated - use from_camera_with_offset)
    pub fn from_camera(camera: &Camera) -> Self {
        Self::from_camera_with_offset(camera, glam::Vec3::ZERO)
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            view_proj_inv: [[0.0; 4]; 4],
            position: [0.0; 3],
            _pos_pad: 0.0,
            near: 0.1,
            far: 1000.0,
            _pad2: [0.0; 2],
            world_offset: [0.0; 3],
            _pad3: 0.0,
        }
    }
}

/// GPU buffer for camera uniform
pub struct CameraBuffer {
    /// Uniform buffer
    buffer: wgpu::Buffer,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
}

impl CameraBuffer {
    /// Create new camera buffer
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Update buffer with camera data
    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera) {
        let uniform = CameraUniform::from_camera(camera);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Update buffer with camera data and world offset
    pub fn update_with_offset(&self, queue: &wgpu::Queue, camera: &Camera, world_offset: glam::Vec3) {
        let uniform = CameraUniform::from_camera_with_offset(camera, world_offset);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Get bind group layout
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get bind group
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Update buffer with camera data, world offset, and optional sub-pixel jitter
    ///
    /// When jitter is provided, it modifies the view-projection matrix to shift
    /// by the given sub-pixel offset. This is required for DLSS temporal accumulation.
    #[cfg(feature = "dlss")]
    pub fn update_with_jitter(
        &self,
        queue: &wgpu::Queue,
        camera: &Camera,
        world_offset: glam::Vec3,
        jitter: [f32; 2],
        render_width: u32,
        render_height: u32,
    ) {
        use crate::render::upscale::jitter::apply_jitter_to_projection;

        let jittered_proj = apply_jitter_to_projection(
            camera.projection_matrix(),
            (jitter[0], jitter[1]),
            render_width,
            render_height,
        );
        let view = camera.view_matrix();
        let jittered_vp = jittered_proj * view;

        let uniform = CameraUniform {
            view_proj: jittered_vp.to_cols_array_2d(),
            view_proj_inv: jittered_vp.inverse().to_cols_array_2d(),
            position: camera.position.to_array(),
            _pos_pad: 0.0,
            near: camera.near,
            far: camera.far,
            _pad2: [0.0; 2],
            world_offset: world_offset.to_array(),
            _pad3: 0.0,
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Get the raw buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_size() {
        // Must be exactly 176 bytes to match WGSL struct layout
        let size = std::mem::size_of::<CameraUniform>();
        assert_eq!(size, 176, "CameraUniform must be exactly 176 bytes, got {} bytes", size);
    }

    #[test]
    fn test_from_camera() {
        let camera = Camera::default();
        let uniform = CameraUniform::from_camera(&camera);

        assert_eq!(uniform.near, 0.01);
        assert_eq!(uniform.far, 1000.0);
        assert_eq!(uniform.position, camera.position.to_array());
    }
}
