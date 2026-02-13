//! Scene graph — CPU-side hierarchy of nodes.
//!
//! The scene graph organizes content into layers with parent/child relationships.
//! Each frame, `flatten()` walks the tree and produces a flat `Vec<FlatChunkEntry>`
//! that can be uploaded directly to the GPU.

use std::collections::HashMap;

use glam::{Mat4, Vec3};

use crate::voxel::chunk::{ChunkCoord, CHUNK_SIZE};
use crate::voxel::layer::{LayerCompositor, LayerId};

use super::flatten::FlatChunkEntry;
use super::node::{LocalTransform, NodeContent, SceneNode, SceneNodeId};

/// CPU-side scene graph that organizes voxel content into a hierarchy.
pub struct SceneGraph {
    nodes: HashMap<SceneNodeId, SceneNode>,
    root: SceneNodeId,
    next_id: u64,
    layers: LayerCompositor,
    dirty: bool,
}

impl SceneGraph {
    /// Create a new scene graph with a root Group node.
    pub fn new() -> Self {
        let root_id = SceneNodeId(0);
        let root_node = SceneNode::new(root_id, "root", LayerId::TERRAIN, NodeContent::Group);

        let mut nodes = HashMap::new();
        nodes.insert(root_id, root_node);

        Self {
            nodes,
            root: root_id,
            next_id: 1,
            layers: LayerCompositor::with_default_layers(),
            dirty: true,
        }
    }

    /// Get the root node ID.
    pub fn root(&self) -> SceneNodeId {
        self.root
    }

    /// Allocate a fresh node ID.
    fn alloc_id(&mut self) -> SceneNodeId {
        let id = SceneNodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a child node under `parent`. Returns the new node's ID.
    pub fn add_child(
        &mut self,
        parent: SceneNodeId,
        name: impl Into<String>,
        layer: LayerId,
        content: NodeContent,
    ) -> SceneNodeId {
        let id = self.alloc_id();
        let mut node = SceneNode::new(id, name, layer, content);
        node.parent = Some(parent);

        self.nodes.insert(id, node);

        // Register as child of parent
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.push(id);
        }

        self.dirty = true;
        id
    }

    /// Remove a node and its entire subtree. Cannot remove the root.
    pub fn remove(&mut self, id: SceneNodeId) {
        if id == self.root {
            return;
        }

        // Collect subtree IDs (BFS)
        let mut to_remove = vec![id];
        let mut i = 0;
        while i < to_remove.len() {
            let current = to_remove[i];
            if let Some(node) = self.nodes.get(&current) {
                to_remove.extend_from_slice(&node.children);
            }
            i += 1;
        }

        // Detach from parent
        if let Some(node) = self.nodes.get(&id) {
            if let Some(parent_id) = node.parent {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.children.retain(|c| *c != id);
                }
            }
        }

        // Remove all nodes in subtree
        for nid in to_remove {
            self.nodes.remove(&nid);
        }

        self.dirty = true;
    }

    /// Move a node to a new parent. Cannot reparent the root.
    pub fn reparent(&mut self, id: SceneNodeId, new_parent: SceneNodeId) {
        if id == self.root {
            return;
        }

        // Detach from old parent
        if let Some(node) = self.nodes.get(&id) {
            if let Some(old_parent_id) = node.parent {
                if let Some(old_parent) = self.nodes.get_mut(&old_parent_id) {
                    old_parent.children.retain(|c| *c != id);
                }
            }
        }

        // Attach to new parent
        if let Some(new_parent_node) = self.nodes.get_mut(&new_parent) {
            new_parent_node.children.push(id);
        }
        if let Some(node) = self.nodes.get_mut(&id) {
            node.parent = Some(new_parent);
        }

        self.dirty = true;
    }

    /// Set the local transform of a node.
    pub fn set_transform(&mut self, id: SceneNodeId, transform: LocalTransform) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.local_transform = transform;
            self.dirty = true;
        }
    }

    /// Set the visibility of a node.
    pub fn set_visible(&mut self, id: SceneNodeId, visible: bool) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.visible = visible;
            self.dirty = true;
        }
    }

    /// Get an immutable reference to a node.
    pub fn get(&self, id: SceneNodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: SceneNodeId) -> Option<&mut SceneNode> {
        self.dirty = true;
        self.nodes.get_mut(&id)
    }

    /// Iterate over the children of a node.
    pub fn children(&self, id: SceneNodeId) -> impl Iterator<Item = SceneNodeId> + '_ {
        self.nodes
            .get(&id)
            .map(|n| n.children.as_slice())
            .unwrap_or(&[])
            .iter()
            .copied()
    }

    /// Get a reference to the layer compositor.
    pub fn layer_compositor(&self) -> &LayerCompositor {
        &self.layers
    }

    /// Get a mutable reference to the layer compositor.
    pub fn layer_compositor_mut(&mut self) -> &mut LayerCompositor {
        &mut self.layers
    }

    /// Total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Walk the tree, propagate transforms, and collect all visible chunks/instances.
    pub fn flatten(&mut self) -> Vec<FlatChunkEntry> {
        // Propagate world transforms from root downward
        self.propagate_transforms(self.root, Mat4::IDENTITY);

        // Collect visible entries
        let mut out = Vec::new();
        self.collect_visible(self.root, &mut out);
        self.dirty = false;
        out
    }

    /// Recursively propagate world transforms.
    fn propagate_transforms(&mut self, node_id: SceneNodeId, parent_world: Mat4) {
        // Compute this node's world transform
        let (local_mat, children) = {
            let node = match self.nodes.get(&node_id) {
                Some(n) => n,
                None => return,
            };
            (node.local_transform.to_mat4(), node.children.clone())
        };

        let world = parent_world * local_mat;

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.world_transform = world;
        }

        for child_id in children {
            self.propagate_transforms(child_id, world);
        }
    }

    /// Recursively collect visible FlatChunkEntries.
    fn collect_visible(&self, node_id: SceneNodeId, out: &mut Vec<FlatChunkEntry>) {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return,
        };

        if !node.visible {
            return;
        }

        // Check if this layer is enabled in the compositor
        if let Some(layer_config) = self.layers.get_layer(node.layer) {
            if !layer_config.enabled {
                return;
            }
        }

        match &node.content {
            NodeContent::Group => {
                // Groups just recurse into children
            }
            NodeContent::ChunkedRegion { chunks } => {
                for (coord, octree) in chunks {
                    let world_min = Vec3::new(
                        coord.x as f32 * CHUNK_SIZE as f32,
                        coord.y as f32 * CHUNK_SIZE as f32,
                        coord.z as f32 * CHUNK_SIZE as f32,
                    );
                    // Apply the node's world transform to the chunk origin
                    let transformed = node.world_transform.transform_point3(world_min);

                    out.push(FlatChunkEntry {
                        coord: *coord,
                        octree: octree.clone(),
                        world_min: transformed,
                        root_size: octree.root_size() * node.local_transform.scale,
                        layer_id: node.layer,
                    });
                }
            }
            NodeContent::VoxelInstance { model, bounds: _ } => {
                // Extract position from the world transform
                let world_min = node.world_transform.transform_point3(Vec3::ZERO);
                let root_size = model.root_size() * node.local_transform.scale;

                // Synthetic chunk coord based on world position
                let coord = ChunkCoord::from_world_pos(world_min);

                out.push(FlatChunkEntry {
                    coord,
                    octree: (**model).clone(),
                    world_min,
                    root_size,
                    layer_id: node.layer,
                });
            }
        }

        // Recurse into children
        for &child_id in &node.children {
            self.collect_visible(child_id, out);
        }
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::voxel::svo::Octree;

    #[test]
    fn test_new_scene_graph() {
        let graph = SceneGraph::new();
        assert_eq!(graph.node_count(), 1); // root only
        assert!(graph.get(graph.root()).is_some());
        assert_eq!(graph.get(graph.root()).unwrap().name, "root");
    }

    #[test]
    fn test_add_child() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let child = graph.add_child(root, "terrain", LayerId::TERRAIN, NodeContent::Group);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.get(child).unwrap().parent, Some(root));
        assert!(graph.children(root).any(|c| c == child));
    }

    #[test]
    fn test_add_multiple_children() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let a = graph.add_child(root, "a", LayerId::TERRAIN, NodeContent::Group);
        let b = graph.add_child(root, "b", LayerId::STATIC_OBJECTS, NodeContent::Group);
        let c = graph.add_child(a, "c", LayerId::TERRAIN, NodeContent::Group);

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.children(root).count(), 2);
        assert_eq!(graph.children(a).count(), 1);
        assert!(graph.children(a).any(|x| x == c));
        assert_eq!(graph.children(b).count(), 0);
    }

    #[test]
    fn test_remove_leaf() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let child = graph.add_child(root, "child", LayerId::TERRAIN, NodeContent::Group);

        graph.remove(child);

        assert_eq!(graph.node_count(), 1);
        assert!(graph.get(child).is_none());
        assert_eq!(graph.children(root).count(), 0);
    }

    #[test]
    fn test_remove_subtree() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let parent = graph.add_child(root, "parent", LayerId::TERRAIN, NodeContent::Group);
        let child1 = graph.add_child(parent, "c1", LayerId::TERRAIN, NodeContent::Group);
        let child2 = graph.add_child(parent, "c2", LayerId::TERRAIN, NodeContent::Group);
        let _grandchild = graph.add_child(child1, "gc", LayerId::TERRAIN, NodeContent::Group);

        assert_eq!(graph.node_count(), 5);

        graph.remove(parent);

        assert_eq!(graph.node_count(), 1); // only root
        assert!(graph.get(parent).is_none());
        assert!(graph.get(child1).is_none());
        assert!(graph.get(child2).is_none());
    }

    #[test]
    fn test_cannot_remove_root() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        graph.remove(root);
        assert_eq!(graph.node_count(), 1); // root survives
    }

    #[test]
    fn test_reparent() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let a = graph.add_child(root, "a", LayerId::TERRAIN, NodeContent::Group);
        let b = graph.add_child(root, "b", LayerId::TERRAIN, NodeContent::Group);
        let c = graph.add_child(a, "c", LayerId::TERRAIN, NodeContent::Group);

        // Move c from under a to under b
        graph.reparent(c, b);

        assert_eq!(graph.children(a).count(), 0);
        assert!(graph.children(b).any(|x| x == c));
        assert_eq!(graph.get(c).unwrap().parent, Some(b));
    }

    #[test]
    fn test_set_visible() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let child = graph.add_child(root, "child", LayerId::TERRAIN, NodeContent::Group);

        graph.set_visible(child, false);
        assert!(!graph.get(child).unwrap().visible);

        graph.set_visible(child, true);
        assert!(graph.get(child).unwrap().visible);
    }

    #[test]
    fn test_set_transform() {
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let child = graph.add_child(root, "child", LayerId::TERRAIN, NodeContent::Group);

        let t = LocalTransform::from_position(Vec3::new(10.0, 0.0, 5.0));
        graph.set_transform(child, t);

        let node = graph.get(child).unwrap();
        assert_eq!(node.local_transform.position, Vec3::new(10.0, 0.0, 5.0));
    }

    #[test]
    fn test_flatten_empty_graph() {
        let mut graph = SceneGraph::new();
        let entries = graph.flatten();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_flatten_chunked_region() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let mut chunks = HashMap::new();
        chunks.insert(ChunkCoord::new(0, 0, 0), Octree::new(4.0, 8));
        chunks.insert(ChunkCoord::new(1, 0, 0), Octree::new(4.0, 8));

        graph.add_child(
            root,
            "terrain",
            LayerId::TERRAIN,
            NodeContent::ChunkedRegion { chunks },
        );

        let entries = graph.flatten();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.layer_id == LayerId::TERRAIN));
    }

    #[test]
    fn test_flatten_voxel_instance() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let model = Arc::new(Octree::new(2.0, 6));
        let id = graph.add_child(
            root,
            "tree",
            LayerId::STATIC_OBJECTS,
            NodeContent::VoxelInstance {
                model,
                bounds: Vec3::new(2.0, 4.0, 2.0),
            },
        );
        graph.set_transform(id, LocalTransform::from_position(Vec3::new(10.0, 0.0, 10.0)));

        let entries = graph.flatten();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].layer_id, LayerId::STATIC_OBJECTS);
        assert!((entries[0].world_min - Vec3::new(10.0, 0.0, 10.0)).length() < 1e-5);
    }

    #[test]
    fn test_flatten_hidden_node_excluded() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let mut chunks = HashMap::new();
        chunks.insert(ChunkCoord::new(0, 0, 0), Octree::new(4.0, 8));

        let child = graph.add_child(
            root,
            "terrain",
            LayerId::TERRAIN,
            NodeContent::ChunkedRegion { chunks },
        );

        graph.set_visible(child, false);

        let entries = graph.flatten();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_flatten_disabled_layer_excluded() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        let mut chunks = HashMap::new();
        chunks.insert(ChunkCoord::new(0, 0, 0), Octree::new(4.0, 8));

        graph.add_child(
            root,
            "water",
            LayerId::WATER,
            NodeContent::ChunkedRegion { chunks },
        );

        // Disable the water layer
        graph.layer_compositor_mut().set_layer_enabled(LayerId::WATER, false);

        let entries = graph.flatten();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_flatten_transform_propagation() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        // Parent offset (10, 0, 0)
        let parent = graph.add_child(root, "parent", LayerId::TERRAIN, NodeContent::Group);
        graph.set_transform(parent, LocalTransform::from_position(Vec3::new(10.0, 0.0, 0.0)));

        // Child has a voxel instance at local (5, 0, 0) — should end up at world (15, 0, 0)
        let model = Arc::new(Octree::new(1.0, 4));
        let child = graph.add_child(
            parent,
            "tree",
            LayerId::STATIC_OBJECTS,
            NodeContent::VoxelInstance {
                model,
                bounds: Vec3::splat(1.0),
            },
        );
        graph.set_transform(child, LocalTransform::from_position(Vec3::new(5.0, 0.0, 0.0)));

        let entries = graph.flatten();
        assert_eq!(entries.len(), 1);
        assert!((entries[0].world_min - Vec3::new(15.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn test_flatten_mixed_content() {
        let mut graph = SceneGraph::new();
        let root = graph.root();

        // Terrain with 2 chunks
        let mut chunks = HashMap::new();
        chunks.insert(ChunkCoord::new(0, 0, 0), Octree::new(4.0, 8));
        chunks.insert(ChunkCoord::new(1, 0, 0), Octree::new(4.0, 8));
        graph.add_child(
            root,
            "terrain",
            LayerId::TERRAIN,
            NodeContent::ChunkedRegion { chunks },
        );

        // Vegetation group with 2 tree instances
        let veg_group = graph.add_child(root, "vegetation", LayerId::STATIC_OBJECTS, NodeContent::Group);
        let model = Arc::new(Octree::new(2.0, 6));
        graph.add_child(
            veg_group,
            "tree1",
            LayerId::STATIC_OBJECTS,
            NodeContent::VoxelInstance {
                model: model.clone(),
                bounds: Vec3::new(2.0, 4.0, 2.0),
            },
        );
        graph.add_child(
            veg_group,
            "tree2",
            LayerId::STATIC_OBJECTS,
            NodeContent::VoxelInstance {
                model,
                bounds: Vec3::new(2.0, 4.0, 2.0),
            },
        );

        let entries = graph.flatten();
        assert_eq!(entries.len(), 4); // 2 terrain + 2 instances

        let terrain_count = entries.iter().filter(|e| e.layer_id == LayerId::TERRAIN).count();
        let object_count = entries.iter().filter(|e| e.layer_id == LayerId::STATIC_OBJECTS).count();
        assert_eq!(terrain_count, 2);
        assert_eq!(object_count, 2);
    }

    #[test]
    fn test_layer_compositor_access() {
        let graph = SceneGraph::new();
        assert_eq!(graph.layer_compositor().layer_count(), 5);
    }
}
