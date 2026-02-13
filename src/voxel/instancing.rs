//! Voxel model instancing system for vegetation and props

use super::svo::Octree;
use super::voxel::Voxel;
use super::svo::OctreeBuilder;
use super::procgen::{TreeGenerator, TreeStyle};
use std::collections::HashMap;

/// A small voxel model that can be instanced (tree, bush, rock, etc)
#[derive(Clone)]
pub struct VoxelModel {
    pub name: String,
    pub octree: Octree,
    pub size: glam::Vec3,      // Bounding box size
    pub origin: glam::Vec3,    // Offset from instance position
}

impl VoxelModel {
    /// Create a new voxel model
    pub fn new(name: impl Into<String>, octree: Octree, size: glam::Vec3) -> Self {
        Self {
            name: name.into(),
            octree,
            size,
            origin: glam::Vec3::ZERO,
        }
    }

    /// Set the origin offset
    pub fn with_origin(mut self, origin: glam::Vec3) -> Self {
        self.origin = origin;
        self
    }
}

/// Generate a simple tree model
pub fn generate_tree(height: u32, trunk_color: Voxel, leaf_color: Voxel) -> VoxelModel {
    let trunk_height = height * 2 / 3;
    let trunk_radius = height / 8;
    let crown_radius = height / 3;

    // Round up to nearest power of 2 for octree
    let size = (height.max(crown_radius * 2)).next_power_of_two();
    let builder = OctreeBuilder::new(size);

    // Create dense voxel array
    let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];

    // Helper to set voxel in dense array (Z-Y-X order)
    let set_voxel = |voxels: &mut Vec<Voxel>, x: u32, y: u32, z: u32, voxel: Voxel| {
        if x < size && y < size && z < size {
            let idx = (z * size * size + y * size + x) as usize;
            voxels[idx] = voxel;
        }
    };

    // Generate trunk
    for y in 0..trunk_height {
        for x in 0..=trunk_radius * 2 {
            for z in 0..=trunk_radius * 2 {
                let dx = x as i32 - trunk_radius as i32;
                let dz = z as i32 - trunk_radius as i32;
                let dist_sq = dx * dx + dz * dz;
                let radius_sq = trunk_radius as i32 * trunk_radius as i32;

                if dist_sq <= radius_sq {
                    set_voxel(&mut voxels, x, y, z, trunk_color);
                }
            }
        }
    }

    // Generate leaf crown (spherical)
    let crown_center = trunk_radius;
    let crown_center_y = trunk_height + crown_radius / 2;
    for y in trunk_height..height {
        for x in 0..crown_radius * 2 {
            for z in 0..crown_radius * 2 {
                let dx = x as i32 - crown_radius as i32;
                let dy = y as i32 - crown_center_y as i32;
                let dz = z as i32 - crown_radius as i32;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let radius_sq = crown_radius as i32 * crown_radius as i32;

                if dist_sq <= radius_sq {
                    // Add some randomness to make it look more natural
                    let hash = ((x * 73 + y * 179 + z * 283) % 100) as f32 / 100.0;
                    if hash > 0.3 { // 70% density
                        set_voxel(&mut voxels, x, y, z, leaf_color);
                    }
                }
            }
        }
    }

    let octree = builder.build(&voxels, size as f32);
    let model_size = glam::vec3(
        crown_radius as f32 * 2.0,
        height as f32,
        crown_radius as f32 * 2.0
    );

    VoxelModel::new("tree", octree, model_size)
        .with_origin(glam::vec3(crown_center as f32, 0.0, crown_center as f32))
}

/// Generate a bush model
pub fn generate_bush(bush_size: u32, color: Voxel) -> VoxelModel {
    let radius = bush_size / 2;

    // Round up to nearest power of 2 for octree
    let size = bush_size.next_power_of_two();
    let builder = OctreeBuilder::new(size);

    // Create dense voxel array
    let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];

    // Helper to set voxel in dense array (Z-Y-X order)
    let set_voxel = |voxels: &mut Vec<Voxel>, x: u32, y: u32, z: u32, voxel: Voxel| {
        if x < size && y < size && z < size {
            let idx = (z * size * size + y * size + x) as usize;
            voxels[idx] = voxel;
        }
    };

    // Generate spherical bush
    for y in 0..bush_size {
        for x in 0..bush_size {
            for z in 0..bush_size {
                let dx = x as i32 - radius as i32;
                let dy = y as i32 - radius as i32;
                let dz = z as i32 - radius as i32;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let radius_sq = radius as i32 * radius as i32;

                if dist_sq <= radius_sq {
                    // Add randomness for natural look
                    let hash = ((x * 73 + y * 179 + z * 283) % 100) as f32 / 100.0;
                    if hash > 0.2 { // 80% density
                        set_voxel(&mut voxels, x, y, z, color);
                    }
                }
            }
        }
    }

    let octree = builder.build(&voxels, size as f32);
    let size_vec = glam::vec3(bush_size as f32, bush_size as f32, bush_size as f32);

    VoxelModel::new("bush", octree, size_vec)
        .with_origin(glam::vec3(radius as f32, 0.0, radius as f32))
}

/// Generate a grass clump
pub fn generate_grass(height: u32, color: Voxel) -> VoxelModel {
    let width = height / 3;

    // Round up to nearest power of 2 for octree
    let size = height.max(width).next_power_of_two();
    let builder = OctreeBuilder::new(size);

    // Create dense voxel array
    let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];

    // Helper to set voxel in dense array (Z-Y-X order)
    let set_voxel = |voxels: &mut Vec<Voxel>, x: u32, y: u32, z: u32, voxel: Voxel| {
        if x < size && y < size && z < size {
            let idx = (z * size * size + y * size + x) as usize;
            voxels[idx] = voxel;
        }
    };

    // Generate thin vertical blades
    for blade in 0..3 {
        let offset_x = (blade as f32 - 1.0) * (width as f32 / 3.0);
        let offset_z = ((blade * 37) % 3) as f32 * (width as f32 / 3.0);

        for y in 0..height {
            // Slight curve to the blade
            let curve = (y as f32 / height as f32) * (width as f32 / 4.0);
            let x = (width as f32 / 2.0 + offset_x + curve) as u32;
            let z = (width as f32 / 2.0 + offset_z) as u32;

            if x < width && z < width {
                set_voxel(&mut voxels, x, y, z, color);
            }
        }
    }

    let octree = builder.build(&voxels, size as f32);
    let model_size = glam::vec3(width as f32, height as f32, width as f32);

    VoxelModel::new("grass", octree, model_size)
        .with_origin(glam::vec3(width as f32 / 2.0, 0.0, width as f32 / 2.0))
}

/// Generate a cactus
pub fn generate_cactus(height: u32, color: Voxel) -> VoxelModel {
    let trunk_radius = height / 6;
    let arm_height = height * 2 / 3;
    let arm_length = height / 3;
    let model_width = trunk_radius * 2 + arm_length;

    // Round up to nearest power of 2 for octree
    let size = height.max(model_width).next_power_of_two();
    let builder = OctreeBuilder::new(size);

    // Create dense voxel array
    let mut voxels = vec![Voxel::EMPTY; (size * size * size) as usize];

    // Helper to set voxel in dense array (Z-Y-X order)
    let set_voxel = |voxels: &mut Vec<Voxel>, x: u32, y: u32, z: u32, voxel: Voxel| {
        if x < size && y < size && z < size {
            let idx = (z * size * size + y * size + x) as usize;
            voxels[idx] = voxel;
        }
    };

    // Main trunk
    for y in 0..height {
        for x in 0..=trunk_radius * 2 {
            for z in 0..=trunk_radius * 2 {
                let dx = x as i32 - trunk_radius as i32;
                let dz = z as i32 - trunk_radius as i32;
                let dist_sq = dx * dx + dz * dz;
                let radius_sq = trunk_radius as i32 * trunk_radius as i32;

                if dist_sq <= radius_sq {
                    set_voxel(&mut voxels, x, y, z, color);
                }
            }
        }
    }

    // Left arm
    for x in 0..arm_length {
        for y in (arm_height.saturating_sub(trunk_radius))..(arm_height + trunk_radius) {
            for z in 0..=trunk_radius * 2 {
                let dz = z as i32 - trunk_radius as i32;
                let dy = y as i32 - arm_height as i32;
                let dist_sq = dz * dz + dy * dy;
                let radius_sq = (trunk_radius as i32 / 2) * (trunk_radius as i32 / 2);

                if dist_sq <= radius_sq {
                    set_voxel(&mut voxels, x, y, z, color);
                }
            }
        }
    }

    // Right arm
    for x in (trunk_radius * 2)..(trunk_radius * 2 + arm_length) {
        for y in (arm_height.saturating_sub(trunk_radius))..(arm_height + trunk_radius) {
            for z in 0..=trunk_radius * 2 {
                let dz = z as i32 - trunk_radius as i32;
                let dy = y as i32 - arm_height as i32;
                let dist_sq = dz * dz + dy * dy;
                let radius_sq = (trunk_radius as i32 / 2) * (trunk_radius as i32 / 2);

                if dist_sq <= radius_sq {
                    set_voxel(&mut voxels, x, y, z, color);
                }
            }
        }
    }

    let octree = builder.build(&voxels, size as f32);
    let model_size = glam::vec3(
        model_width as f32,
        height as f32,
        (trunk_radius * 2) as f32
    );

    VoxelModel::new("cactus", octree, model_size)
        .with_origin(glam::vec3(trunk_radius as f32, 0.0, trunk_radius as f32))
}

/// A single instance of a voxel model
#[derive(Clone, Debug)]
pub struct ModelInstance {
    pub model_index: usize,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: f32,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(model_index: usize, position: glam::Vec3) -> Self {
        Self {
            model_index,
            position,
            rotation: glam::Quat::IDENTITY,
            scale: 1.0,
        }
    }

    /// Set rotation
    pub fn with_rotation(mut self, rotation: glam::Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set scale
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Get transformation matrix for this instance
    pub fn transform(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(self.scale),
            self.rotation,
            self.position
        )
    }
}

/// Library of voxel models
pub struct ModelLibrary {
    models: Vec<VoxelModel>,
    names: HashMap<String, usize>,
}

impl ModelLibrary {
    /// Create a new empty model library
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            names: HashMap::new(),
        }
    }

    /// Add a model to the library and return its index
    pub fn add_model(&mut self, model: VoxelModel) -> usize {
        let index = self.models.len();
        self.names.insert(model.name.clone(), index);
        self.models.push(model);
        index
    }

    /// Get a model by index
    pub fn get_model(&self, index: usize) -> Option<&VoxelModel> {
        self.models.get(index)
    }

    /// Find a model by name
    pub fn find_model(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    /// Get the number of models in the library
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Generate a tree model using the TreeGenerator (space colonization algorithm)
    fn generate_tree_model(name: &str, seed: u64, style: TreeStyle, root_size: f32, max_depth: u8) -> VoxelModel {
        let mut generator = TreeGenerator::from_style(seed, style);
        let octree = generator.generate(root_size, max_depth);
        let half = root_size / 2.0;
        VoxelModel::new(name, octree, glam::Vec3::splat(root_size))
            .with_origin(glam::vec3(half, 0.0, half))
    }

    /// Create a library with procedurally generated vegetation models.
    /// Uses TreeGenerator for trees (space colonization) and grid models for bush/grass/cactus.
    pub fn with_procgen_vegetation() -> Self {
        let mut library = Self::new();

        // Tree models via TreeGenerator (space colonization)
        library.add_model(Self::generate_tree_model("small_oak", 1, TreeStyle::Oak, 8.0, 5));    // 0
        library.add_model(Self::generate_tree_model("large_oak", 2, TreeStyle::Oak, 8.0, 5));     // 1
        library.add_model(Self::generate_tree_model("willow", 3, TreeStyle::Willow, 8.0, 5));     // 2

        // Simple models (grid-based)
        let bush_color = Voxel::new(50, 155, 50, 2);
        let grass_color = Voxel::new(124, 252, 0, 2);
        let cactus_color = Voxel::new(0, 128, 0, 3);

        library.add_model(generate_bush(8, bush_color));       // 3
        library.add_model(generate_grass(4, grass_color));     // 4
        library.add_model(generate_cactus(12, cactus_color));  // 5

        library
    }

    /// Create a library with default vegetation models
    pub fn with_default_vegetation() -> Self {
        let mut library = Self::new();

        // Default colors
        let trunk_color = Voxel::new(101, 67, 33, 1); // Brown
        let leaf_color = Voxel::new(34, 139, 34, 2); // Forest green
        let bush_color = Voxel::new(50, 155, 50, 2); // Lighter green
        let grass_color = Voxel::new(124, 252, 0, 2); // Lawn green
        let cactus_color = Voxel::new(0, 128, 0, 3); // Dark green

        // Add default models
        library.add_model(generate_tree(16, trunk_color, leaf_color));
        library.add_model(generate_bush(8, bush_color));
        library.add_model(generate_grass(4, grass_color));
        library.add_model(generate_cactus(12, cactus_color));

        library
    }
}

impl Default for ModelLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_model_creation() {
        let octree = Octree::new(10.0, 8);
        let model = VoxelModel::new("test", octree, glam::vec3(5.0, 10.0, 5.0));

        assert_eq!(model.name, "test");
        assert_eq!(model.size, glam::vec3(5.0, 10.0, 5.0));
        assert_eq!(model.origin, glam::Vec3::ZERO);
    }

    #[test]
    fn test_voxel_model_with_origin() {
        let octree = Octree::new(10.0, 8);
        let model = VoxelModel::new("test", octree, glam::vec3(5.0, 10.0, 5.0))
            .with_origin(glam::vec3(2.5, 0.0, 2.5));

        assert_eq!(model.origin, glam::vec3(2.5, 0.0, 2.5));
    }

    #[test]
    fn test_generate_tree() {
        let trunk_color = Voxel::new(101, 67, 33, 1);
        let leaf_color = Voxel::new(34, 139, 34, 2);
        let tree = generate_tree(16, trunk_color, leaf_color);

        assert_eq!(tree.name, "tree");
        assert!(!tree.octree.is_empty());
        assert!(tree.size.y > 0.0);
    }

    #[test]
    fn test_generate_bush() {
        let color = Voxel::new(50, 155, 50, 2);
        let bush = generate_bush(8, color);

        assert_eq!(bush.name, "bush");
        assert!(!bush.octree.is_empty());
    }

    #[test]
    fn test_generate_grass() {
        let color = Voxel::new(124, 252, 0, 2);
        let grass = generate_grass(4, color);

        assert_eq!(grass.name, "grass");
        assert!(!grass.octree.is_empty());
    }

    #[test]
    fn test_generate_cactus() {
        let color = Voxel::new(0, 128, 0, 3);
        let cactus = generate_cactus(12, color);

        assert_eq!(cactus.name, "cactus");
        assert!(!cactus.octree.is_empty());
    }

    #[test]
    fn test_model_instance() {
        let instance = ModelInstance::new(0, glam::vec3(10.0, 0.0, 10.0));

        assert_eq!(instance.model_index, 0);
        assert_eq!(instance.position, glam::vec3(10.0, 0.0, 10.0));
        assert_eq!(instance.rotation, glam::Quat::IDENTITY);
        assert_eq!(instance.scale, 1.0);
    }

    #[test]
    fn test_model_instance_with_rotation() {
        let rotation = glam::Quat::from_rotation_y(std::f32::consts::PI / 2.0);
        let instance = ModelInstance::new(0, glam::Vec3::ZERO)
            .with_rotation(rotation);

        assert_eq!(instance.rotation, rotation);
    }

    #[test]
    fn test_model_instance_with_scale() {
        let instance = ModelInstance::new(0, glam::Vec3::ZERO)
            .with_scale(2.0);

        assert_eq!(instance.scale, 2.0);
    }

    #[test]
    fn test_model_instance_transform() {
        let instance = ModelInstance::new(0, glam::vec3(10.0, 5.0, 10.0))
            .with_scale(2.0);

        let transform = instance.transform();
        let expected = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::IDENTITY,
            glam::vec3(10.0, 5.0, 10.0)
        );

        assert_eq!(transform, expected);
    }

    #[test]
    fn test_model_library_new() {
        let library = ModelLibrary::new();
        assert_eq!(library.model_count(), 0);
    }

    #[test]
    fn test_model_library_add_model() {
        let mut library = ModelLibrary::new();
        let octree = Octree::new(10.0, 8);
        let model = VoxelModel::new("test", octree, glam::vec3(5.0, 10.0, 5.0));

        let index = library.add_model(model);
        assert_eq!(index, 0);
        assert_eq!(library.model_count(), 1);
    }

    #[test]
    fn test_model_library_get_model() {
        let mut library = ModelLibrary::new();
        let octree = Octree::new(10.0, 8);
        let model = VoxelModel::new("test", octree, glam::vec3(5.0, 10.0, 5.0));

        let index = library.add_model(model);
        let retrieved = library.get_model(index);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test");
    }

    #[test]
    fn test_model_library_find_model() {
        let mut library = ModelLibrary::new();
        let octree = Octree::new(10.0, 8);
        let model = VoxelModel::new("test", octree, glam::vec3(5.0, 10.0, 5.0));

        library.add_model(model);
        let index = library.find_model("test");

        assert_eq!(index, Some(0));
        assert_eq!(library.find_model("nonexistent"), None);
    }

    #[test]
    fn test_model_library_with_default_vegetation() {
        let library = ModelLibrary::with_default_vegetation();

        assert_eq!(library.model_count(), 4);
        assert!(library.find_model("tree").is_some());
        assert!(library.find_model("bush").is_some());
        assert!(library.find_model("grass").is_some());
        assert!(library.find_model("cactus").is_some());
    }

    #[test]
    fn test_model_library_with_procgen_vegetation() {
        let library = ModelLibrary::with_procgen_vegetation();

        assert_eq!(library.model_count(), 6);
        assert!(library.find_model("small_oak").is_some());
        assert!(library.find_model("large_oak").is_some());
        assert!(library.find_model("willow").is_some());
        assert!(library.find_model("bush").is_some());
        assert!(library.find_model("grass").is_some());
        assert!(library.find_model("cactus").is_some());
    }

    #[test]
    fn test_procgen_vegetation_models_not_empty() {
        let library = ModelLibrary::with_procgen_vegetation();

        for i in 0..library.model_count() {
            let model = library.get_model(i).unwrap();
            assert!(!model.octree.is_empty(), "Model {} should not be empty", model.name);
            assert!(model.size.x > 0.0 && model.size.y > 0.0 && model.size.z > 0.0);
        }
    }

    #[test]
    fn test_default_vegetation_models_not_empty() {
        let library = ModelLibrary::with_default_vegetation();

        for i in 0..library.model_count() {
            let model = library.get_model(i).unwrap();
            assert!(!model.octree.is_empty(), "Model {} should not be empty", model.name);
            assert!(model.size.x > 0.0 && model.size.y > 0.0 && model.size.z > 0.0);
        }
    }
}
