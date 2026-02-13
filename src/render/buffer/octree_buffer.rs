//! GPU storage buffers for octree data

use bytemuck::{Pod, Zeroable};

use crate::grass::profile::GrassCell;
use crate::mask::octree::MaskOctree;
use crate::streaming::disk_io::ChunkCoord;
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::svo::{Octree, OctreeNode};
use crate::voxel::brick::VoxelBrick;
use crate::voxel::streaming::feedback::MAX_FEEDBACK_REQUESTS;

/// Maximum number of chunks that can be uploaded
pub const MAX_CHUNKS: u32 = 8192;

/// Per-chunk metadata for the GPU (32 bytes, matches WGSL layout)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuChunkInfo {
    /// World-space min corner of this chunk
    pub world_min: [f32; 3],
    /// Size of this chunk in meters
    pub root_size: f32,
    /// Index of this chunk's root node in the shared nodes buffer
    pub root_node: u32,
    /// Maximum octree depth for this chunk
    pub max_depth: u32,
    /// Layer ID (LayerId::TERRAIN.0, etc.)
    pub layer_id: u32,
    /// Flags: 0=opaque, 1=alpha_test, 2=transparent, 3=volumetric
    pub flags: u32,
}

/// Per-chunk grass mask metadata for the GPU (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGrassMaskInfo {
    /// Offset into the shared grass_mask_nodes buffer
    pub node_offset: u32,
    /// Offset into the shared grass_mask_values buffer
    pub value_offset: u32,
    /// Number of nodes in this chunk's mask (0 = no grass mask)
    pub node_count: u32,
    /// Max depth of this mask octree
    pub max_depth: u32,
}

/// GPU representation of a mask octree node (16 bytes).
/// Must match `GrassMaskNode` in svo_trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGrassMaskNode {
    /// child_mask (low 8 bits) | leaf_mask (bits 8-15)
    pub masks: u32,
    /// Index of first internal child node (chunk-local)
    pub child_offset: u32,
    /// Index of first leaf value (chunk-local)
    pub value_offset: u32,
    /// LOD value index (chunk-local), 0xFFFFFFFF = none
    pub lod_value_idx: u32,
}

/// GPU buffers for octree node and brick data
pub struct OctreeBuffer {
    /// Storage buffer for octree nodes
    node_buffer: wgpu::Buffer,
    /// Storage buffer for voxel bricks
    brick_buffer: wgpu::Buffer,
    /// Storage buffer for per-chunk metadata
    chunk_info_buffer: wgpu::Buffer,
    /// Storage buffer for feedback header (count, max_requests, padding)
    feedback_header_buffer: wgpu::Buffer,
    /// Storage buffer for feedback requests (brick IDs)
    feedback_requests_buffer: wgpu::Buffer,
    /// Storage buffer for 3D chunk grid (maps chunk coords → chunk index)
    chunk_grid_buffer: wgpu::Buffer,
    /// Storage buffer for per-chunk grass mask metadata
    grass_mask_info_buffer: wgpu::Buffer,
    /// Storage buffer for packed grass mask nodes (all chunks)
    grass_mask_node_buffer: wgpu::Buffer,
    /// Storage buffer for packed grass mask values (all chunks)
    grass_mask_value_buffer: wgpu::Buffer,
    /// Bind group layout for accessing buffers in shaders
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for buffer access
    bind_group: wgpu::BindGroup,
    /// Max nodes this buffer can hold
    max_nodes: u32,
    /// Max bricks this buffer can hold
    max_bricks: u32,
    /// Next free node slot (tracks how many nodes have been uploaded)
    used_nodes: u32,
    /// Next free brick slot (tracks how many bricks have been uploaded)
    used_bricks: u32,
}

impl OctreeBuffer {
    /// Create new octree buffer with given capacity
    pub fn new(device: &wgpu::Device, max_nodes: u32, max_bricks: u32) -> Self {
        let node_size = (max_nodes as u64) * (std::mem::size_of::<OctreeNode>() as u64);
        let brick_size = (max_bricks as u64) * (std::mem::size_of::<VoxelBrick>() as u64);

        // wgpu max_buffer_size defaults to 256MB if not raised; log warnings
        if node_size > 512 * 1024 * 1024 {
            log::warn!("Node buffer size: {}MB - ensure device limits are sufficient", node_size / 1024 / 1024);
        }
        if brick_size > 512 * 1024 * 1024 {
            log::warn!("Brick buffer size: {}MB - ensure device limits are sufficient", brick_size / 1024 / 1024);
        }

        let node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("octree_nodes"),
            size: (max_nodes as usize * std::mem::size_of::<OctreeNode>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let brick_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("octree_bricks"),
            size: (max_bricks as usize * std::mem::size_of::<VoxelBrick>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_info"),
            size: (MAX_CHUNKS as usize * std::mem::size_of::<GpuChunkInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Feedback header: 4x u32 (count, max_requests, _pad1, _pad2)
        let feedback_header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("feedback_header"),
            size: 16, // 4 * u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Feedback requests: MAX_FEEDBACK_REQUESTS entries, 3x u32 each
        let feedback_requests_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("feedback_requests"),
            size: (MAX_FEEDBACK_REQUESTS * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chunk grid: 3D lookup table mapping chunk coordinates to chunk indices.
        // Preallocate for a reasonable world size; resized via upload_chunk_grid().
        // Minimum 4 bytes so the binding is valid even before grid data is uploaded.
        let chunk_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_grid"),
            size: 4, // Will be recreated when grid is uploaded
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grass mask buffers: per-chunk info (fixed size), nodes + values (resized on upload)
        let grass_mask_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_info"),
            size: (MAX_CHUNKS as u64) * (std::mem::size_of::<GpuGrassMaskInfo>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grass_mask_node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_nodes"),
            size: 16, // Minimum; recreated when masks are uploaded
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grass_mask_value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_values"),
            size: 4, // Minimum; recreated when masks are uploaded
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("octree_bind_group_layout"),
            entries: &[
                // binding 0: nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: bricks
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: feedback header
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: feedback requests
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: chunk info
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: chunk grid (3D lookup: chunk coord → chunk index)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: grass mask info (per-chunk metadata)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: grass mask nodes (packed from all chunks)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 8: grass mask values (packed from all chunks)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("octree_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: feedback_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: feedback_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: chunk_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: chunk_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grass_mask_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grass_mask_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: grass_mask_value_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            node_buffer,
            brick_buffer,
            chunk_info_buffer,
            feedback_header_buffer,
            feedback_requests_buffer,
            chunk_grid_buffer,
            grass_mask_info_buffer,
            grass_mask_node_buffer,
            grass_mask_value_buffer,
            bind_group_layout,
            bind_group,
            max_nodes,
            max_bricks,
            used_nodes: 0,
            used_bricks: 0,
        }
    }

    /// Upload multiple chunk octrees to GPU buffers.
    ///
    /// Packs all chunks' nodes and bricks into shared buffers with absolute offsets.
    /// Returns (chunk_count, Vec<GpuChunkInfo>) so the caller can cache chunk metadata
    /// for CPU-side frustum culling.
    pub fn upload_chunks(
        &mut self,
        queue: &wgpu::Queue,
        chunks: &[(ChunkCoord, Octree)],
    ) -> (u32, Vec<GpuChunkInfo>) {
        assert!(chunks.len() <= MAX_CHUNKS as usize, "Too many chunks (max {})", MAX_CHUNKS);

        let mut all_nodes: Vec<OctreeNode> = Vec::new();
        let mut all_bricks: Vec<VoxelBrick> = Vec::new();
        let mut chunk_infos: Vec<GpuChunkInfo> = Vec::new();

        for (coord, octree) in chunks {
            let node_base = all_nodes.len() as u32;
            let brick_base = all_bricks.len() as u32;

            // Copy nodes with adjusted offsets
            let src_nodes = octree.nodes_slice();
            for node in src_nodes {
                let mut adjusted = *node;
                let valid = node.child_valid_mask();
                let leaf = node.child_leaf_mask();

                // Offset child_offset if this node has any internal (non-leaf) children
                let has_internal = valid & !leaf;
                if has_internal != 0 {
                    adjusted.child_offset += node_base;
                }

                // Offset brick_offset if this node has any leaf children OR is a terminal leaf
                let has_leaves = valid & leaf;
                if has_leaves != 0 || node.is_terminal_leaf() {
                    adjusted.brick_offset += brick_base;
                }

                all_nodes.push(adjusted);
            }

            // Copy bricks as-is (no offsets needed)
            all_bricks.extend_from_slice(octree.bricks_slice());

            // Create chunk metadata
            let world_min = [
                coord.x as f32 * CHUNK_SIZE as f32,
                coord.y as f32 * CHUNK_SIZE as f32,
                coord.z as f32 * CHUNK_SIZE as f32,
            ];

            chunk_infos.push(GpuChunkInfo {
                world_min,
                root_size: octree.root_size(),
                root_node: node_base,
                max_depth: octree.max_depth() as u32,
                layer_id: 0, // TERRAIN by default
                flags: 0,   // opaque
            });
        }

        assert!(all_nodes.len() <= self.max_nodes as usize,
            "Too many total nodes: {} (max {})", all_nodes.len(), self.max_nodes);
        assert!(all_bricks.len() <= self.max_bricks as usize,
            "Too many total bricks: {} (max {})", all_bricks.len(), self.max_bricks);

        log::info!("Uploading {} chunks: {} nodes, {} bricks to GPU",
            chunks.len(), all_nodes.len(), all_bricks.len());

        queue.write_buffer(&self.node_buffer, 0, bytemuck::cast_slice(&all_nodes));
        queue.write_buffer(&self.brick_buffer, 0, bytemuck::cast_slice(&all_bricks));
        queue.write_buffer(&self.chunk_info_buffer, 0, bytemuck::cast_slice(&chunk_infos));

        // Initialize feedback header
        let feedback_header: [u32; 4] = [0, MAX_FEEDBACK_REQUESTS, 0, 0];
        queue.write_buffer(&self.feedback_header_buffer, 0, bytemuck::cast_slice(&feedback_header));

        // Update usage counters
        self.used_nodes = all_nodes.len() as u32;
        self.used_bricks = all_bricks.len() as u32;

        (chunks.len() as u32, chunk_infos)
    }

    /// Upload a single octree (backward compat - wraps as 1 chunk at origin)
    pub fn upload(&mut self, queue: &wgpu::Queue, octree: &Octree) -> (u32, Vec<GpuChunkInfo>) {
        let coord = ChunkCoord::new(0, 0, 0);
        self.upload_chunks(queue, &[(coord, octree.clone())])
    }

    /// Upload flattened scene graph entries to GPU buffers.
    ///
    /// Like `upload_chunks` but accepts `FlatChunkEntry` with layer metadata.
    pub fn upload_from_flat(
        &mut self,
        queue: &wgpu::Queue,
        entries: &[crate::scene::FlatChunkEntry],
    ) -> (u32, Vec<GpuChunkInfo>) {
        assert!(entries.len() <= MAX_CHUNKS as usize, "Too many chunks (max {})", MAX_CHUNKS);

        let mut all_nodes: Vec<OctreeNode> = Vec::new();
        let mut all_bricks: Vec<VoxelBrick> = Vec::new();
        let mut chunk_infos: Vec<GpuChunkInfo> = Vec::new();

        for entry in entries {
            let node_base = all_nodes.len() as u32;
            let brick_base = all_bricks.len() as u32;

            let octree = &entry.octree;

            // Copy nodes with adjusted offsets
            let src_nodes = octree.nodes_slice();
            for node in src_nodes {
                let mut adjusted = *node;
                let valid = node.child_valid_mask();
                let leaf = node.child_leaf_mask();

                let has_internal = valid & !leaf;
                if has_internal != 0 {
                    adjusted.child_offset += node_base;
                }

                let has_leaves = valid & leaf;
                if has_leaves != 0 || node.is_terminal_leaf() {
                    adjusted.brick_offset += brick_base;
                }

                all_nodes.push(adjusted);
            }

            all_bricks.extend_from_slice(octree.bricks_slice());

            chunk_infos.push(GpuChunkInfo {
                world_min: entry.world_min.into(),
                root_size: entry.root_size,
                root_node: node_base,
                max_depth: octree.max_depth() as u32,
                layer_id: entry.layer_id.0,
                flags: 0, // opaque by default
            });
        }

        assert!(all_nodes.len() <= self.max_nodes as usize,
            "Too many total nodes: {} (max {})", all_nodes.len(), self.max_nodes);
        assert!(all_bricks.len() <= self.max_bricks as usize,
            "Too many total bricks: {} (max {})", all_bricks.len(), self.max_bricks);

        log::info!("Uploading {} flat entries: {} nodes, {} bricks to GPU",
            entries.len(), all_nodes.len(), all_bricks.len());

        queue.write_buffer(&self.node_buffer, 0, bytemuck::cast_slice(&all_nodes));
        queue.write_buffer(&self.brick_buffer, 0, bytemuck::cast_slice(&all_bricks));
        queue.write_buffer(&self.chunk_info_buffer, 0, bytemuck::cast_slice(&chunk_infos));

        // Initialize feedback header
        let feedback_header: [u32; 4] = [0, MAX_FEEDBACK_REQUESTS, 0, 0];
        queue.write_buffer(&self.feedback_header_buffer, 0, bytemuck::cast_slice(&feedback_header));

        // Update usage counters
        self.used_nodes = all_nodes.len() as u32;
        self.used_bricks = all_bricks.len() as u32;

        (entries.len() as u32, chunk_infos)
    }

    /// Upload a single chunk incrementally without touching the chunk_info_buffer.
    ///
    /// Appends nodes and bricks at the current `used_nodes`/`used_bricks` offsets,
    /// advances the counters, and returns a `GpuChunkInfo` the caller can feed to
    /// `ChunkCuller`. The per-frame cull path writes the chunk_info_buffer.
    pub fn upload_chunk_incremental(
        &mut self,
        queue: &wgpu::Queue,
        octree: &Octree,
        world_min: [f32; 3],
        root_size: f32,
        layer_id: u32,
        flags: u32,
    ) -> GpuChunkInfo {
        let node_base = self.used_nodes;
        let brick_base = self.used_bricks;

        let src_nodes = octree.nodes_slice();
        let src_bricks = octree.bricks_slice();

        let new_node_count = src_nodes.len() as u32;
        let new_brick_count = src_bricks.len() as u32;

        assert!(
            node_base + new_node_count <= self.max_nodes,
            "Incremental upload would exceed node capacity: {} + {} > {}",
            node_base, new_node_count, self.max_nodes
        );
        assert!(
            brick_base + new_brick_count <= self.max_bricks,
            "Incremental upload would exceed brick capacity: {} + {} > {}",
            brick_base, new_brick_count, self.max_bricks
        );

        // Rebase child_offset/brick_offset by current used counts
        let mut adjusted_nodes: Vec<OctreeNode> = Vec::with_capacity(src_nodes.len());
        for node in src_nodes {
            let mut adjusted = *node;
            let valid = node.child_valid_mask();
            let leaf = node.child_leaf_mask();

            let has_internal = valid & !leaf;
            if has_internal != 0 {
                adjusted.child_offset += node_base;
            }

            let has_leaves = valid & leaf;
            if has_leaves != 0 || node.is_terminal_leaf() {
                adjusted.brick_offset += brick_base;
            }

            adjusted_nodes.push(adjusted);
        }

        // Write nodes at byte offset for the current used_nodes position
        let node_byte_offset = (node_base as u64) * (std::mem::size_of::<OctreeNode>() as u64);
        queue.write_buffer(&self.node_buffer, node_byte_offset, bytemuck::cast_slice(&adjusted_nodes));

        // Write bricks at byte offset for the current used_bricks position
        let brick_byte_offset = (brick_base as u64) * (std::mem::size_of::<VoxelBrick>() as u64);
        queue.write_buffer(&self.brick_buffer, brick_byte_offset, bytemuck::cast_slice(src_bricks));

        // Advance counters
        self.used_nodes += new_node_count;
        self.used_bricks += new_brick_count;

        log::debug!(
            "Incremental upload: nodes {}..{}, bricks {}..{} (world_min={:?})",
            node_base, self.used_nodes, brick_base, self.used_bricks, world_min
        );

        GpuChunkInfo {
            world_min,
            root_size,
            root_node: node_base,
            max_depth: octree.max_depth() as u32,
            layer_id,
            flags,
        }
    }

    /// Reset usage counters (for full reload scenarios).
    pub fn reset_usage(&mut self) {
        self.used_nodes = 0;
        self.used_bricks = 0;
    }

    /// Number of node slots currently in use.
    pub fn used_nodes(&self) -> u32 {
        self.used_nodes
    }

    /// Number of brick slots currently in use.
    pub fn used_bricks(&self) -> u32 {
        self.used_bricks
    }

    /// Re-upload chunk info data (for per-frame frustum culling).
    /// Only updates the chunk_info_buffer, not nodes or bricks.
    pub fn update_chunk_infos(&self, queue: &wgpu::Queue, chunk_infos: &[GpuChunkInfo]) {
        queue.write_buffer(&self.chunk_info_buffer, 0, bytemuck::cast_slice(chunk_infos));
    }

    /// Get bind group layout for pipeline creation
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get bind group for rendering
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Get node buffer
    pub fn node_buffer(&self) -> &wgpu::Buffer {
        &self.node_buffer
    }

    /// Get brick buffer
    pub fn brick_buffer(&self) -> &wgpu::Buffer {
        &self.brick_buffer
    }

    /// Get node count capacity
    pub fn max_nodes(&self) -> u32 {
        self.max_nodes
    }

    /// Get brick count capacity
    pub fn max_bricks(&self) -> u32 {
        self.max_bricks
    }

    /// Get feedback header buffer
    pub fn feedback_header_buffer(&self) -> &wgpu::Buffer {
        &self.feedback_header_buffer
    }

    /// Get feedback requests buffer
    pub fn feedback_requests_buffer(&self) -> &wgpu::Buffer {
        &self.feedback_requests_buffer
    }

    /// Upload a 3D chunk grid mapping chunk coordinates to chunk indices.
    ///
    /// `grid_data` is a flat array of u32 indexed as:
    ///   `(x - min_x) + (y - min_y) * size_x + (z - min_z) * size_x * size_y`
    /// where 0xFFFFFFFF means empty (no chunk at that coordinate).
    pub fn upload_chunk_grid(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_data: &[u32],
    ) {
        let grid_size = (grid_data.len() as u64) * 4;
        let grid_size = grid_size.max(4); // Minimum 4 bytes

        // Recreate grid buffer with correct size
        self.chunk_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_grid"),
            size: grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !grid_data.is_empty() {
            queue.write_buffer(&self.chunk_grid_buffer, 0, bytemuck::cast_slice(grid_data));
        }

        self.rebuild_bind_group(device);
    }

    /// Upload grass mask data for all chunks.
    ///
    /// Takes pre-packed GPU data produced by `pack_grass_masks()`.
    /// Recreates node/value buffers to fit the data and rebuilds the bind group.
    pub fn upload_grass_masks(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        infos: &[GpuGrassMaskInfo],
        nodes: &[GpuGrassMaskNode],
        values: &[u32],
    ) {
        // Recreate info buffer sized to actual data (MAX_CHUNKS entries, with real data in first N)
        // Must cover MAX_CHUNKS so any chunk_idx is a valid read (returns zero for unset entries).
        let info_count = MAX_CHUNKS as usize;
        let info_bytes = (info_count as u64) * (std::mem::size_of::<GpuGrassMaskInfo>() as u64);
        self.grass_mask_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_info"),
            size: info_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !infos.is_empty() {
            queue.write_buffer(
                &self.grass_mask_info_buffer,
                0,
                bytemuck::cast_slice(infos),
            );
        }

        // Recreate node buffer sized to actual data
        let node_bytes = (nodes.len() as u64) * (std::mem::size_of::<GpuGrassMaskNode>() as u64);
        let node_bytes = node_bytes.max(16); // Minimum for valid binding
        self.grass_mask_node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_nodes"),
            size: node_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !nodes.is_empty() {
            queue.write_buffer(
                &self.grass_mask_node_buffer,
                0,
                bytemuck::cast_slice(nodes),
            );
        }

        // Recreate value buffer sized to actual data
        let value_bytes = (values.len() as u64) * 4;
        let value_bytes = value_bytes.max(4); // Minimum for valid binding
        self.grass_mask_value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_mask_values"),
            size: value_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !values.is_empty() {
            queue.write_buffer(
                &self.grass_mask_value_buffer,
                0,
                bytemuck::cast_slice(values),
            );
        }

        log::info!(
            "Uploaded grass masks: {} chunk infos, {} nodes, {} values",
            infos.len(),
            nodes.len(),
            values.len(),
        );

        self.rebuild_bind_group(device);
    }

    /// Rebuild the bind group from all current buffers.
    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("octree_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.brick_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.feedback_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.feedback_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.chunk_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.chunk_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.grass_mask_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.grass_mask_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.grass_mask_value_buffer.as_entire_binding(),
                },
            ],
        });
    }
}

/// Pack grass mask octrees into GPU-ready data.
///
/// `masks` is indexed by chunk index (matching terrain upload order).
/// Returns (per-chunk infos, packed nodes, packed values).
pub fn pack_grass_masks(
    masks: &[Option<&MaskOctree<GrassCell>>],
) -> (Vec<GpuGrassMaskInfo>, Vec<GpuGrassMaskNode>, Vec<u32>) {
    let mut infos = Vec::with_capacity(masks.len());
    let mut all_nodes: Vec<GpuGrassMaskNode> = Vec::new();
    let mut all_values: Vec<u32> = Vec::new();

    for mask_opt in masks {
        match mask_opt {
            Some(mask) if !mask.is_empty() => {
                let node_offset = all_nodes.len() as u32;
                let value_offset = all_values.len() as u32;

                // Convert MaskNodes to GPU format
                for node in mask.nodes_slice() {
                    all_nodes.push(GpuGrassMaskNode {
                        masks: (node.child_mask as u32) | ((node.leaf_mask as u32) << 8),
                        child_offset: node.child_offset,
                        value_offset: node.value_offset,
                        lod_value_idx: node.lod_value_idx,
                    });
                }

                // Convert GrassCell values to u32 (profile in low 8 bits, density in bits 8-15)
                for val in mask.values_slice() {
                    all_values.push(val.0 as u32);
                }

                infos.push(GpuGrassMaskInfo {
                    node_offset,
                    value_offset,
                    node_count: mask.node_count() as u32,
                    max_depth: mask.max_depth() as u32,
                });
            }
            _ => {
                infos.push(GpuGrassMaskInfo {
                    node_offset: 0,
                    value_offset: 0,
                    node_count: 0,
                    max_depth: 0,
                });
            }
        }
    }

    (infos, all_nodes, all_values)
}
