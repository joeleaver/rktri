//! Skeletal animation bone hierarchy and skinning system

use glam::Mat4;
use std::collections::HashMap;

/// Maximum number of bones per skeleton (GPU uniform buffer limit)
pub const MAX_BONES: usize = 64;

/// A single bone in a skeletal hierarchy
#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent_index: Option<usize>,
    pub local_bind_pose: Mat4,
    pub inverse_bind_pose: Mat4,
}

impl Bone {
    /// Create a new bone with the given local transform
    /// The inverse_bind_pose will be calculated when the skeleton is built
    pub fn new(name: impl Into<String>, parent_index: Option<usize>, local_transform: Mat4) -> Self {
        Self {
            name: name.into(),
            parent_index,
            local_bind_pose: local_transform,
            inverse_bind_pose: Mat4::IDENTITY, // Will be calculated later
        }
    }
}

/// A hierarchical skeleton composed of bones
#[derive(Clone, Debug)]
pub struct Skeleton {
    bones: Vec<Bone>,
    bone_names: HashMap<String, usize>,
}

impl Skeleton {
    /// Create an empty skeleton
    pub fn new() -> Self {
        Self {
            bones: Vec::new(),
            bone_names: HashMap::new(),
        }
    }

    /// Add a bone to the skeleton
    /// Returns the bone index or an error if max bones exceeded
    pub fn add_bone(&mut self, mut bone: Bone) -> Result<usize, &'static str> {
        if self.bones.len() >= MAX_BONES {
            return Err("Maximum bone count exceeded");
        }

        // Validate parent index
        if let Some(parent) = bone.parent_index {
            if parent >= self.bones.len() {
                return Err("Invalid parent bone index");
            }
        }

        let index = self.bones.len();

        // Calculate world-space bind pose
        let world_bind_pose = if let Some(parent_idx) = bone.parent_index {
            // Get parent's world bind pose
            let parent_world = self.calculate_bone_world_bind_pose(parent_idx);
            parent_world * bone.local_bind_pose
        } else {
            bone.local_bind_pose
        };

        // Calculate inverse bind pose for skinning
        bone.inverse_bind_pose = world_bind_pose.inverse();

        // Check for name collision
        if self.bone_names.contains_key(&bone.name) {
            return Err("Bone name already exists");
        }

        self.bone_names.insert(bone.name.clone(), index);
        self.bones.push(bone);

        Ok(index)
    }

    /// Get the number of bones in the skeleton
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Get a bone by index
    pub fn get_bone(&self, index: usize) -> Option<&Bone> {
        self.bones.get(index)
    }

    /// Find a bone index by name
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.bone_names.get(name).copied()
    }

    /// Get the parent index of a bone
    pub fn parent_index(&self, bone_index: usize) -> Option<usize> {
        self.bones.get(bone_index)?.parent_index
    }

    /// Get all children of a bone
    pub fn children(&self, bone_index: usize) -> Vec<usize> {
        self.bones
            .iter()
            .enumerate()
            .filter_map(|(idx, bone)| {
                if bone.parent_index == Some(bone_index) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate world-space transforms for all bones given local transforms
    pub fn calculate_world_transforms(&self, local_transforms: &[Mat4]) -> Vec<Mat4> {
        assert_eq!(
            local_transforms.len(),
            self.bones.len(),
            "Local transforms array must match bone count"
        );

        let mut world_transforms = vec![Mat4::IDENTITY; self.bones.len()];

        // Process bones in order (parents before children)
        for (index, bone) in self.bones.iter().enumerate() {
            world_transforms[index] = if let Some(parent_idx) = bone.parent_index {
                world_transforms[parent_idx] * local_transforms[index]
            } else {
                local_transforms[index]
            };
        }

        world_transforms
    }

    /// Calculate skinning matrices (world_transform * inverse_bind_pose)
    pub fn calculate_skinning_matrices(&self, world_transforms: &[Mat4]) -> Vec<Mat4> {
        assert_eq!(
            world_transforms.len(),
            self.bones.len(),
            "World transforms array must match bone count"
        );

        world_transforms
            .iter()
            .zip(self.bones.iter())
            .map(|(world, bone)| *world * bone.inverse_bind_pose)
            .collect()
    }

    /// Calculate world-space bind pose for a single bone (used during construction)
    fn calculate_bone_world_bind_pose(&self, bone_index: usize) -> Mat4 {
        let mut transform = Mat4::IDENTITY;
        let mut current_index = Some(bone_index);

        // Walk up the hierarchy
        let mut chain = Vec::new();
        while let Some(idx) = current_index {
            chain.push(idx);
            current_index = self.bones[idx].parent_index;
        }

        // Apply transforms from root to bone
        for &idx in chain.iter().rev() {
            transform = transform * self.bones[idx].local_bind_pose;
        }

        transform
    }
}

impl Default for Skeleton {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for easier skeleton construction
pub struct SkeletonBuilder {
    skeleton: Skeleton,
    last_error: Option<&'static str>,
}

impl SkeletonBuilder {
    /// Create a new skeleton builder
    pub fn new() -> Self {
        Self {
            skeleton: Skeleton::new(),
            last_error: None,
        }
    }

    /// Add a root bone (no parent)
    pub fn add_root(mut self, name: &str, transform: Mat4) -> Self {
        if self.last_error.is_some() {
            return self;
        }

        let bone = Bone::new(name, None, transform);
        if let Err(e) = self.skeleton.add_bone(bone) {
            self.last_error = Some(e);
        }
        self
    }

    /// Add a bone with a parent
    pub fn add_bone(mut self, name: &str, parent: &str, transform: Mat4) -> Self {
        if self.last_error.is_some() {
            return self;
        }

        let parent_index = match self.skeleton.find_bone(parent) {
            Some(idx) => idx,
            None => {
                self.last_error = Some("Parent bone not found");
                return self;
            }
        };

        let bone = Bone::new(name, Some(parent_index), transform);
        if let Err(e) = self.skeleton.add_bone(bone) {
            self.last_error = Some(e);
        }
        self
    }

    /// Build the final skeleton
    pub fn build(self) -> Result<Skeleton, &'static str> {
        if let Some(error) = self.last_error {
            Err(error)
        } else if self.skeleton.bones.is_empty() {
            Err("Skeleton must have at least one bone")
        } else {
            Ok(self.skeleton)
        }
    }
}

impl Default for SkeletonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_bone_creation() {
        let bone = Bone::new("root", None, Mat4::IDENTITY);
        assert_eq!(bone.name, "root");
        assert_eq!(bone.parent_index, None);
    }

    #[test]
    fn test_skeleton_add_bone() {
        let mut skeleton = Skeleton::new();

        let root = Bone::new("root", None, Mat4::IDENTITY);
        let root_idx = skeleton.add_bone(root).unwrap();
        assert_eq!(root_idx, 0);
        assert_eq!(skeleton.bone_count(), 1);
    }

    #[test]
    fn test_skeleton_find_bone() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY)).unwrap();

        assert_eq!(skeleton.find_bone("root"), Some(0));
        assert_eq!(skeleton.find_bone("nonexistent"), None);
    }

    #[test]
    fn test_skeleton_hierarchy() {
        let mut skeleton = Skeleton::new();

        let root_idx = skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY)).unwrap();
        let child1_idx = skeleton.add_bone(Bone::new("child1", Some(root_idx), Mat4::IDENTITY)).unwrap();
        let child2_idx = skeleton.add_bone(Bone::new("child2", Some(root_idx), Mat4::IDENTITY)).unwrap();
        skeleton.add_bone(Bone::new("grandchild", Some(child1_idx), Mat4::IDENTITY)).unwrap();

        assert_eq!(skeleton.parent_index(child1_idx), Some(root_idx));

        let children = skeleton.children(root_idx);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&child1_idx));
        assert!(children.contains(&child2_idx));
    }

    #[test]
    fn test_world_transform_calculation() {
        let mut skeleton = Skeleton::new();

        // Root at origin
        let root_idx = skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY)).unwrap();

        // Child translated 1 unit along X
        let child_local = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let child_idx = skeleton.add_bone(Bone::new("child", Some(root_idx), child_local)).unwrap();

        // Grandchild translated 1 unit along Y
        let grandchild_local = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
        skeleton.add_bone(Bone::new("grandchild", Some(child_idx), grandchild_local)).unwrap();

        let local_transforms = vec![
            Mat4::IDENTITY,
            child_local,
            grandchild_local,
        ];

        let world_transforms = skeleton.calculate_world_transforms(&local_transforms);

        // Root should be at origin
        assert_eq!(world_transforms[0], Mat4::IDENTITY);

        // Child should be at (1, 0, 0)
        let child_pos = world_transforms[1].to_scale_rotation_translation().2;
        assert!((child_pos.x - 1.0).abs() < 0.001);

        // Grandchild should be at (1, 1, 0)
        let grandchild_pos = world_transforms[2].to_scale_rotation_translation().2;
        assert!((grandchild_pos.x - 1.0).abs() < 0.001);
        assert!((grandchild_pos.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_skinning_matrices() {
        let mut skeleton = Skeleton::new();

        let root_idx = skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY)).unwrap();
        let child_transform = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        skeleton.add_bone(Bone::new("child", Some(root_idx), child_transform)).unwrap();

        let local_transforms = vec![Mat4::IDENTITY, child_transform];
        let world_transforms = skeleton.calculate_world_transforms(&local_transforms);
        let skinning_matrices = skeleton.calculate_skinning_matrices(&world_transforms);

        assert_eq!(skinning_matrices.len(), 2);
        // Skinning matrix should be world * inverse_bind
        // Since we're using bind pose, should be close to identity
        for matrix in &skinning_matrices {
            let pos = matrix.to_scale_rotation_translation().2;
            assert!(pos.length() < 0.001);
        }
    }

    #[test]
    fn test_max_bones() {
        let mut skeleton = Skeleton::new();

        // Add MAX_BONES bones
        for i in 0..MAX_BONES {
            skeleton.add_bone(Bone::new(format!("bone_{}", i), None, Mat4::IDENTITY)).unwrap();
        }

        // Adding one more should fail
        let result = skeleton.add_bone(Bone::new("overflow", None, Mat4::IDENTITY));
        assert!(result.is_err());
    }

    #[test]
    fn test_skeleton_builder() {
        let skeleton = SkeletonBuilder::new()
            .add_root("root", Mat4::IDENTITY)
            .add_bone("spine", "root", Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)))
            .add_bone("head", "spine", Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)))
            .build()
            .unwrap();

        assert_eq!(skeleton.bone_count(), 3);
        assert_eq!(skeleton.find_bone("root"), Some(0));
        assert_eq!(skeleton.find_bone("spine"), Some(1));
        assert_eq!(skeleton.find_bone("head"), Some(2));
    }

    #[test]
    fn test_skeleton_builder_invalid_parent() {
        let result = SkeletonBuilder::new()
            .add_root("root", Mat4::IDENTITY)
            .add_bone("child", "nonexistent", Mat4::IDENTITY)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_bone_name() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY)).unwrap();
        let result = skeleton.add_bone(Bone::new("root", None, Mat4::IDENTITY));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parent_index() {
        let mut skeleton = Skeleton::new();
        let result = skeleton.add_bone(Bone::new("orphan", Some(999), Mat4::IDENTITY));
        assert!(result.is_err());
    }
}
