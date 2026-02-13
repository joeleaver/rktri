//! GPU-side animation buffer for bone transforms

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// GPU-side bone transform (mat4 for skinning matrix)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBoneTransform {
    pub matrix: [[f32; 4]; 4],
}

impl GpuBoneTransform {
    /// Create from a glam Mat4
    pub fn from_mat4(matrix: Mat4) -> Self {
        Self {
            matrix: matrix.to_cols_array_2d(),
        }
    }

    /// Create identity transform
    pub fn identity() -> Self {
        Self::from_mat4(Mat4::IDENTITY)
    }
}

/// GPU buffer for bone transformation matrices
pub struct BoneTransformBuffer {
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    max_bones: usize,
}

impl BoneTransformBuffer {
    /// Create a new bone transform buffer
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `max_bones` - Maximum number of bones to support
    pub fn new(device: &wgpu::Device, max_bones: usize) -> Self {
        // Create buffer large enough to hold all bone transforms
        let buffer_size = (max_bones * std::mem::size_of::<GpuBoneTransform>()) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bone Transform Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bone Transform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bone Transform Bind Group"),
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
            max_bones,
        }
    }

    /// Update the bone transform buffer with new matrices
    ///
    /// # Arguments
    /// * `queue` - WGPU queue for writing data
    /// * `matrices` - Slice of bone transformation matrices
    ///
    /// # Panics
    /// Panics if the number of matrices exceeds max_bones
    pub fn update(&self, queue: &wgpu::Queue, matrices: &[Mat4]) {
        assert!(
            matrices.len() <= self.max_bones,
            "Matrix count {} exceeds max_bones {}",
            matrices.len(),
            self.max_bones
        );

        // Convert Mat4 to GPU format
        let gpu_transforms: Vec<GpuBoneTransform> = matrices
            .iter()
            .map(|m| GpuBoneTransform::from_mat4(*m))
            .collect();

        // Write to GPU buffer
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&gpu_transforms),
        );
    }

    /// Get the bind group layout for use in pipeline creation
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group for use in rendering
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Get the maximum number of bones this buffer supports
    pub fn max_bones(&self) -> usize {
        self.max_bones
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_bone_transform_from_mat4() {
        let mat = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let gpu_transform = GpuBoneTransform::from_mat4(mat);

        // Check that the matrix is correctly converted
        let reconstructed = Mat4::from_cols_array_2d(&gpu_transform.matrix);
        let diff = (mat - reconstructed).abs();

        // All elements should be very close to zero
        for col in 0..4 {
            for row in 0..4 {
                assert!(diff.col(col)[row] < 0.0001);
            }
        }
    }

    #[test]
    fn test_gpu_bone_transform_identity() {
        let gpu_transform = GpuBoneTransform::identity();
        let mat = Mat4::from_cols_array_2d(&gpu_transform.matrix);

        assert_eq!(mat, Mat4::IDENTITY);
    }

    #[test]
    fn test_gpu_bone_transform_pod() {
        // Ensure the type is correctly Pod/Zeroable
        let _: &[u8] = bytemuck::bytes_of(&GpuBoneTransform::identity());
        let _: GpuBoneTransform = bytemuck::Zeroable::zeroed();
    }
}
