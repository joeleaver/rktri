//! Skeletal animation system

pub mod skeleton;
pub mod clip;
pub mod animator;
// pub mod blend;
// pub mod skinning;
// pub mod gpu_animation;

pub use skeleton::{Bone, Skeleton, SkeletonBuilder, MAX_BONES};
pub use clip::{AnimationClip, BoneTrack, TransformKeyframe};
pub use animator::{AnimationState, Animator};
