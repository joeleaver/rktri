// RAY-SPACE SKELETAL ANIMATION SHADER SUPPORT
//
// This shader provides utilities for ray-space skeletal animation, which is the key
// innovation that allows skeletal animation of voxel octrees without deforming the octree.
//
// CORE CONCEPT:
// Instead of deforming the voxel octree (expensive), we transform rays:
// 1. For each ray, determine which bone(s) affect the region being traced
// 2. Transform the ray into bone-local space using inverse skinning matrix
// 3. Trace the octree in this transformed space
// 4. Transform hit results back to world space
//
// BENEFITS:
// - Octree data remains static (no per-frame rebuilds)
// - GPU-friendly (ray transformation is cheap)
// - Supports standard skeletal animation data
// - Memory efficient (no vertex deformation)
//
// LIMITATIONS:
// - Works best with rigid skinning (1 bone per region)
// - Smooth skinning requires multiple traces and blending
// - Requires bone assignment metadata for voxel regions

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Bone transformation matrix (world-to-bone or bone-to-world)
struct BoneTransform {
    matrix: mat4x4<f32>,
}

/// Ray structure (if not already defined in including shader)
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

/// Hit information in world or local space
struct HitInfo {
    position: vec3<f32>,
    normal: vec3<f32>,
}

// ============================================================================
// STORAGE BUFFERS
// ============================================================================

// Storage buffer for bone transforms (skinning matrices)
// This will be bound when rendering animated entities
// The binding group/index should be defined by the including shader
//
// Example binding:
// @group(2) @binding(0) var<storage, read> bone_transforms: array<BoneTransform>;
//
// Note: Transforms are typically pre-inverted on CPU for ray transformation

// ============================================================================
// RAY TRANSFORMATION FUNCTIONS
// ============================================================================

/// Transform a ray from world space into bone-local space
///
/// This function applies the inverse skinning transform to move a ray from world
/// space into the local coordinate system of a bone. The octree is then traced
/// in this local space.
///
/// @param ray_origin - Ray origin in world space
/// @param ray_dir - Ray direction in world space (should be normalized)
/// @param bone_transform - World-to-bone transformation matrix (pre-inverted)
/// @return Ray in bone-local space
fn transform_ray_to_bone_space(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    bone_transform: mat4x4<f32>
) -> Ray {
    // bone_transform is already the inverse of the skinning matrix (world_to_bone)
    // Transform origin as a point (w=1), direction as a vector (w=0)
    let local_origin = (bone_transform * vec4<f32>(ray_origin, 1.0)).xyz;
    let local_dir = normalize((bone_transform * vec4<f32>(ray_dir, 0.0)).xyz);

    return Ray(local_origin, local_dir);
}

/// Transform a ray from bone-local space back to world space
///
/// This is the inverse operation of transform_ray_to_bone_space, used when
/// you need to convert a locally-traced ray back to world coordinates.
///
/// @param ray_origin - Ray origin in bone-local space
/// @param ray_dir - Ray direction in bone-local space (should be normalized)
/// @param bone_transform_inv - Bone-to-world transformation matrix
/// @return Ray in world space
fn transform_ray_to_world_space(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    bone_transform_inv: mat4x4<f32>
) -> Ray {
    let world_origin = (bone_transform_inv * vec4<f32>(ray_origin, 1.0)).xyz;
    let world_dir = normalize((bone_transform_inv * vec4<f32>(ray_dir, 0.0)).xyz);

    return Ray(world_origin, world_dir);
}

// ============================================================================
// HIT RESULT TRANSFORMATION
// ============================================================================

/// Transform hit position and normal back to world space
///
/// After tracing in bone-local space, hit results must be transformed back
/// to world space for shading and output.
///
/// @param local_position - Hit position in bone-local space
/// @param local_normal - Hit normal in bone-local space (should be normalized)
/// @param bone_to_world - Bone-to-world transformation matrix
/// @return Hit information in world space
fn transform_hit_to_world(
    local_position: vec3<f32>,
    local_normal: vec3<f32>,
    bone_to_world: mat4x4<f32>
) -> HitInfo {
    // Transform position as point (w=1)
    let world_pos = (bone_to_world * vec4<f32>(local_position, 1.0)).xyz;

    // Transform normal as vector (w=0) and renormalize
    // Note: For correct normal transformation, should use transpose of inverse,
    // but for orthonormal transforms (typical in skeletal animation), this simplifies
    let world_normal = normalize((bone_to_world * vec4<f32>(local_normal, 0.0)).xyz);

    return HitInfo(world_pos, world_normal);
}

// ============================================================================
// BONE SELECTION FUNCTIONS
// ============================================================================

/// Find the dominant bone for a given position using simple spatial heuristics
///
/// This is a placeholder implementation. Real implementations should:
/// - Store bone indices in voxel metadata
/// - Use spatial acceleration structures
/// - Support multi-bone blending for smooth skinning
///
/// @param position - Position in model space
/// @param bone_count - Total number of bones
/// @return Index of the dominant bone
fn get_dominant_bone(position: vec3<f32>, bone_count: u32) -> u32 {
    // Placeholder: Simple position-based assignment
    // Real implementation depends on how bone assignments are stored
    // Options:
    // 1. Store bone index per voxel in octree metadata
    // 2. Use bounding volumes per bone
    // 3. Use distance fields or spatial hashing
    return 0u;
}

/// Find bones affecting a position with blend weights (for smooth skinning)
///
/// Returns up to 4 bone indices and weights. This is needed for smooth skinning
/// where multiple bones influence a region.
///
/// @param position - Position in model space
/// @param bone_count - Total number of bones
/// @return Array of (bone_index, weight) pairs, weight=0 for unused slots
fn get_bone_weights(
    position: vec3<f32>,
    bone_count: u32
) -> array<vec2<f32>, 4> {
    // Placeholder: Single bone with full weight
    // Real implementation would:
    // 1. Look up bone weights from voxel metadata
    // 2. Use spatial queries to find nearby bones
    // 3. Normalize weights to sum to 1.0

    var weights: array<vec2<f32>, 4>;
    weights[0] = vec2<f32>(0.0, 1.0); // bone 0, weight 1.0
    weights[1] = vec2<f32>(0.0, 0.0); // unused
    weights[2] = vec2<f32>(0.0, 0.0); // unused
    weights[3] = vec2<f32>(0.0, 0.0); // unused

    return weights;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Invert a 4x4 transformation matrix
///
/// Note: This is computationally expensive. Prefer pre-inverting matrices on CPU.
/// Provided for completeness but should only be used when necessary.
///
/// @param m - Matrix to invert
/// @return Inverted matrix (identity if singular)
fn invert_transform(m: mat4x4<f32>) -> mat4x4<f32> {
    // This is a simplified inverse for affine transformations
    // For full inverse, use CPU or more sophisticated GPU method

    // Extract 3x3 rotation and scale
    var inv: mat4x4<f32>;

    // Simple transpose for rotation part (assumes orthonormal)
    inv[0] = vec4<f32>(m[0][0], m[1][0], m[2][0], 0.0);
    inv[1] = vec4<f32>(m[0][1], m[1][1], m[2][1], 0.0);
    inv[2] = vec4<f32>(m[0][2], m[1][2], m[2][2], 0.0);

    // Invert translation
    let t = -vec3<f32>(m[3][0], m[3][1], m[3][2]);
    inv[3] = vec4<f32>(
        dot(vec3<f32>(inv[0].xyz), t),
        dot(vec3<f32>(inv[1].xyz), t),
        dot(vec3<f32>(inv[2].xyz), t),
        1.0
    );

    return inv;
}

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

// Example of how to use these functions in a ray tracing shader:
//
// @group(2) @binding(0) var<storage, read> bone_transforms: array<BoneTransform>;
//
// fn trace_animated_entity(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitInfo {
//     // 1. Determine which bone affects this ray's region
//     let bone_idx = get_dominant_bone(ray_origin, arrayLength(&bone_transforms));
//
//     // 2. Transform ray into bone-local space
//     let bone_transform = bone_transforms[bone_idx].matrix;  // world_to_bone
//     let local_ray = transform_ray_to_bone_space(ray_origin, ray_dir, bone_transform);
//
//     // 3. Trace the octree in bone-local space
//     let local_hit = trace_octree(local_ray.origin, local_ray.direction);
//
//     // 4. Transform hit back to world space
//     let bone_to_world = invert_transform(bone_transform);
//     let world_hit = transform_hit_to_world(
//         local_hit.position,
//         local_hit.normal,
//         bone_to_world
//     );
//
//     return world_hit;
// }
