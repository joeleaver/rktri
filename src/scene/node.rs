//! Scene graph node types
//!
//! Core types for the CPU-side scene graph: node IDs, transforms, content variants, and nodes.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};

use crate::voxel::chunk::ChunkCoord;
use crate::voxel::layer::LayerId;
use crate::voxel::svo::Octree;

/// Unique identifier for a scene graph node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneNodeId(pub u64);

/// Local transform relative to the parent node.
#[derive(Clone, Debug)]
pub struct LocalTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for LocalTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

impl LocalTransform {
    /// Identity transform (no translation, rotation, or scaling).
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create a translation-only transform.
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Convert to a 4x4 matrix.
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.position,
        )
    }
}

/// What a scene node contains.
#[derive(Clone, Debug)]
pub enum NodeContent {
    /// A grouping node with no geometry of its own.
    Group,

    /// A region of chunks (e.g. terrain). Each chunk has its own octree.
    ChunkedRegion {
        chunks: HashMap<ChunkCoord, Octree>,
    },

    /// A single voxel model instance (e.g. a tree placed in the world).
    VoxelInstance {
        model: Arc<Octree>,
        /// Bounding size of the model in world units.
        bounds: Vec3,
    },
}

/// A single node in the scene graph.
#[derive(Clone, Debug)]
pub struct SceneNode {
    pub id: SceneNodeId,
    pub name: String,
    pub parent: Option<SceneNodeId>,
    pub children: Vec<SceneNodeId>,
    pub local_transform: LocalTransform,
    /// Cached world transform (recomputed during propagation).
    pub world_transform: Mat4,
    pub layer: LayerId,
    pub visible: bool,
    pub content: NodeContent,
}

impl SceneNode {
    /// Create a new scene node.
    pub fn new(
        id: SceneNodeId,
        name: impl Into<String>,
        layer: LayerId,
        content: NodeContent,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            parent: None,
            children: Vec::new(),
            local_transform: LocalTransform::identity(),
            world_transform: Mat4::IDENTITY,
            layer,
            visible: true,
            content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_node_id_equality() {
        let a = SceneNodeId(1);
        let b = SceneNodeId(1);
        let c = SceneNodeId(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_local_transform_identity() {
        let t = LocalTransform::identity();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, 1.0);
        assert_eq!(t.to_mat4(), Mat4::IDENTITY);
    }

    #[test]
    fn test_local_transform_from_position() {
        let pos = Vec3::new(10.0, 5.0, -3.0);
        let t = LocalTransform::from_position(pos);
        assert_eq!(t.position, pos);
        let m = t.to_mat4();
        let (_, _, translation) = m.to_scale_rotation_translation();
        assert!((translation - pos).length() < 1e-5);
    }

    #[test]
    fn test_local_transform_to_mat4_with_scale() {
        let t = LocalTransform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: 2.0,
        };
        let m = t.to_mat4();
        let (scale, _, translation) = m.to_scale_rotation_translation();
        assert!((scale - Vec3::splat(2.0)).length() < 1e-5);
        assert!((translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn test_scene_node_new() {
        let node = SceneNode::new(
            SceneNodeId(0),
            "root",
            LayerId::TERRAIN,
            NodeContent::Group,
        );
        assert_eq!(node.id, SceneNodeId(0));
        assert_eq!(node.name, "root");
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert!(node.visible);
        assert_eq!(node.layer, LayerId::TERRAIN);
    }

    #[test]
    fn test_node_content_group() {
        let content = NodeContent::Group;
        assert!(matches!(content, NodeContent::Group));
    }

    #[test]
    fn test_node_content_chunked_region() {
        let mut chunks = HashMap::new();
        chunks.insert(ChunkCoord::new(0, 0, 0), Octree::new(4.0, 8));
        let content = NodeContent::ChunkedRegion { chunks };
        assert!(matches!(content, NodeContent::ChunkedRegion { .. }));
    }

    #[test]
    fn test_node_content_voxel_instance() {
        let model = Arc::new(Octree::new(4.0, 8));
        let content = NodeContent::VoxelInstance {
            model,
            bounds: Vec3::new(2.0, 4.0, 2.0),
        };
        assert!(matches!(content, NodeContent::VoxelInstance { .. }));
    }
}
