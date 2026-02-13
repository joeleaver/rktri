//! Runtime animation playback and blending system

use super::{AnimationClip, Skeleton};
use glam::Mat4;

/// Animation state for tracking playback of a single clip
#[derive(Clone, Debug)]
pub struct AnimationState {
    pub clip_index: usize,
    pub time: f32,
    pub speed: f32,
    pub weight: f32, // For blending (0.0-1.0)
    pub playing: bool,
}

impl AnimationState {
    /// Create a new animation state for the given clip
    pub fn new(clip_index: usize) -> Self {
        Self {
            clip_index,
            time: 0.0,
            speed: 1.0,
            weight: 1.0,
            playing: false,
        }
    }

    /// Start playing the animation
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause the animation (keeps current time)
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop the animation (reset to beginning)
    pub fn stop(&mut self) {
        self.playing = false;
        self.time = 0.0;
    }

    /// Set the playback speed multiplier
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }
}

/// Runtime animator for skeletal animation playback and blending
#[derive(Clone)]
pub struct Animator {
    skeleton: Skeleton,
    clips: Vec<AnimationClip>,
    states: Vec<AnimationState>,
    current_local_transforms: Vec<Mat4>,
    current_skinning_matrices: Vec<Mat4>,
}

impl Animator {
    /// Create a new animator with the given skeleton
    pub fn new(skeleton: Skeleton) -> Self {
        let bone_count = skeleton.bone_count();
        Self {
            skeleton,
            clips: Vec::new(),
            states: Vec::new(),
            current_local_transforms: vec![Mat4::IDENTITY; bone_count],
            current_skinning_matrices: vec![Mat4::IDENTITY; bone_count],
        }
    }

    /// Add an animation clip to this animator
    /// Returns the index of the added clip
    pub fn add_clip(&mut self, clip: AnimationClip) -> usize {
        let index = self.clips.len();
        self.clips.push(clip);
        index
    }

    /// Get a reference to a clip by index
    pub fn get_clip(&self, index: usize) -> Option<&AnimationClip> {
        self.clips.get(index)
    }

    /// Get the number of clips in this animator
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Start playing a clip with full weight (stops other clips)
    pub fn play(&mut self, clip_index: usize) {
        // Stop all current animations first
        for state in self.states.iter_mut() {
            state.stop();
        }

        // Now start the new animation
        self.play_with_weight(clip_index, 1.0);
    }

    /// Start playing a clip with the given weight (for blending)
    pub fn play_with_weight(&mut self, clip_index: usize, weight: f32) {
        if clip_index >= self.clips.len() {
            return;
        }

        let mut state = AnimationState::new(clip_index);
        state.weight = weight.clamp(0.0, 1.0);
        state.play();
        self.states.push(state);
    }

    /// Stop a specific animation state by clip index
    pub fn stop(&mut self, clip_index: usize) {
        self.states.retain(|state| state.clip_index != clip_index || !state.playing);
    }

    /// Stop all playing animations
    pub fn stop_all(&mut self) {
        self.states.clear();
    }

    /// Update all playing animations by the given time delta
    pub fn update(&mut self, delta_time: f32) {
        if self.states.is_empty() {
            // No animations playing, use bind pose
            self.current_local_transforms = vec![Mat4::IDENTITY; self.skeleton.bone_count()];
            let world_transforms = self.skeleton.calculate_world_transforms(&self.current_local_transforms);
            self.current_skinning_matrices = self.skeleton.calculate_skinning_matrices(&world_transforms);
            return;
        }

        let bone_count = self.skeleton.bone_count();

        // Advance time for all playing animations
        for state in &mut self.states {
            if state.playing {
                if let Some(clip) = self.clips.get(state.clip_index) {
                    state.time += delta_time * state.speed;

                    // Handle looping
                    if clip.looping && clip.duration > 0.0 {
                        state.time = state.time % clip.duration;
                    } else if state.time >= clip.duration {
                        // Non-looping animation finished
                        state.time = clip.duration;
                        state.playing = false;
                    }
                }
            }
        }

        // Sample and blend all playing animations
        self.current_local_transforms = self.blend_animations(bone_count);

        // Calculate world transforms and skinning matrices
        let world_transforms = self.skeleton.calculate_world_transforms(&self.current_local_transforms);
        self.current_skinning_matrices = self.skeleton.calculate_skinning_matrices(&world_transforms);

        // Clean up stopped animations
        self.states.retain(|state| state.playing);
    }

    /// Blend all playing animations together
    fn blend_animations(&self, bone_count: usize) -> Vec<Mat4> {
        if self.states.is_empty() {
            return vec![Mat4::IDENTITY; bone_count];
        }

        // Calculate total weight
        let total_weight: f32 = self.states.iter()
            .map(|s| s.weight)
            .sum();

        if total_weight <= 0.0 {
            return vec![Mat4::IDENTITY; bone_count];
        }

        // If only one animation, no blending needed
        if self.states.len() == 1 {
            let state = &self.states[0];
            if let Some(clip) = self.clips.get(state.clip_index) {
                return clip.sample(state.time, bone_count);
            }
            return vec![Mat4::IDENTITY; bone_count];
        }

        // Blend multiple animations
        let mut blended_transforms = vec![Mat4::IDENTITY; bone_count];

        for bone_idx in 0..bone_count {
            // Decompose and blend each bone's transform
            let mut blended_translation = glam::Vec3::ZERO;
            let mut blended_rotation = glam::Quat::IDENTITY;
            let mut blended_scale = glam::Vec3::ZERO;
            let mut accumulated_weight = 0.0f32;

            for state in &self.states {
                if let Some(clip) = self.clips.get(state.clip_index) {
                    let sampled = clip.sample(state.time, bone_count);
                    let transform = sampled[bone_idx];

                    let (scale, rotation, translation) = transform.to_scale_rotation_translation();

                    let normalized_weight = state.weight / total_weight;

                    blended_translation += translation * normalized_weight;
                    blended_scale += scale * normalized_weight;

                    // Quaternion blending with first animation as base
                    if accumulated_weight == 0.0 {
                        blended_rotation = rotation;
                    } else {
                        // Use slerp for smooth rotation blending
                        let t = normalized_weight / (accumulated_weight + normalized_weight);
                        blended_rotation = blended_rotation.slerp(rotation, t);
                    }

                    accumulated_weight += normalized_weight;
                }
            }

            blended_transforms[bone_idx] =
                Mat4::from_scale_rotation_translation(blended_scale, blended_rotation, blended_translation);
        }

        blended_transforms
    }

    /// Get the skeleton reference
    pub fn skeleton(&self) -> &Skeleton {
        &self.skeleton
    }

    /// Get the current local transforms for all bones
    pub fn local_transforms(&self) -> &[Mat4] {
        &self.current_local_transforms
    }

    /// Get the current skinning matrices for all bones
    pub fn skinning_matrices(&self) -> &[Mat4] {
        &self.current_skinning_matrices
    }

    /// Get the number of bones in the skeleton
    pub fn bone_count(&self) -> usize {
        self.skeleton.bone_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{BoneTrack, TransformKeyframe, SkeletonBuilder};
    use glam::{Quat, Vec3};

    fn create_test_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .add_root("root", Mat4::IDENTITY)
            .add_bone("child", "root", Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)))
            .build()
            .unwrap()
    }

    fn create_test_clip(name: &str, looping: bool) -> AnimationClip {
        let mut clip = AnimationClip::new(name);
        clip.looping = looping;

        // Root bone moves from (0,0,0) to (10,0,0)
        let mut track0 = BoneTrack::new(0);
        track0.add_keyframe(TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        track0.add_keyframe(TransformKeyframe::new(
            1.0,
            Vec3::new(10.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));

        // Child bone moves from (0,0,0) to (0,5,0)
        let mut track1 = BoneTrack::new(1);
        track1.add_keyframe(TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        track1.add_keyframe(TransformKeyframe::new(
            1.0,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));

        clip.add_track(track0);
        clip.add_track(track1);
        clip.calculate_duration();

        clip
    }

    #[test]
    fn test_animation_state_creation() {
        let state = AnimationState::new(0);
        assert_eq!(state.clip_index, 0);
        assert_eq!(state.time, 0.0);
        assert_eq!(state.speed, 1.0);
        assert_eq!(state.weight, 1.0);
        assert!(!state.playing);
    }

    #[test]
    fn test_animation_state_playback_control() {
        let mut state = AnimationState::new(0);

        state.play();
        assert!(state.playing);

        state.pause();
        assert!(!state.playing);

        state.time = 5.0;
        state.stop();
        assert!(!state.playing);
        assert_eq!(state.time, 0.0);
    }

    #[test]
    fn test_animation_state_speed() {
        let mut state = AnimationState::new(0);
        state.set_speed(2.0);
        assert_eq!(state.speed, 2.0);
    }

    #[test]
    fn test_animator_creation() {
        let skeleton = create_test_skeleton();
        let animator = Animator::new(skeleton);

        assert_eq!(animator.bone_count(), 2);
        assert_eq!(animator.clip_count(), 0);
    }

    #[test]
    fn test_animator_add_clip() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("walk", false);
        let index = animator.add_clip(clip);

        assert_eq!(index, 0);
        assert_eq!(animator.clip_count(), 1);
        assert!(animator.get_clip(0).is_some());
        assert!(animator.get_clip(1).is_none());
    }

    #[test]
    fn test_animator_playback() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("walk", false);
        let clip_index = animator.add_clip(clip);

        animator.play(clip_index);
        assert_eq!(animator.states.len(), 1);
        assert!(animator.states[0].playing);

        animator.stop(clip_index);
        assert_eq!(animator.states.len(), 0);
    }

    #[test]
    fn test_animator_update_non_looping() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("walk", false);
        let clip_index = animator.add_clip(clip);

        animator.play(clip_index);

        // Update to midpoint
        animator.update(0.5);
        assert_eq!(animator.states.len(), 1);
        assert!((animator.states[0].time - 0.5).abs() < 0.001);

        // Check root bone position at midpoint (should be at 5.0, 0.0, 0.0)
        let local_transforms = animator.local_transforms();
        let pos = local_transforms[0].to_scale_rotation_translation().2;
        assert!((pos.x - 5.0).abs() < 0.001);

        // Update past end (duration is 1.0)
        animator.update(1.0);
        assert_eq!(animator.states.len(), 0); // Should stop when complete

        // Position should be at end (10.0, 0.0, 0.0)
        let local_transforms = animator.local_transforms();
        let pos = local_transforms[0].to_scale_rotation_translation().2;
        assert!((pos.x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_update_looping() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("idle", true);
        let clip_index = animator.add_clip(clip);

        animator.play(clip_index);

        // Update beyond duration
        animator.update(2.5);
        assert_eq!(animator.states.len(), 1);
        assert!(animator.states[0].playing); // Should still be playing

        // Time should wrap to 0.5 (2.5 % 1.0 = 0.5)
        assert!((animator.states[0].time - 0.5).abs() < 0.001);

        // Position should be at midpoint (5.0, 0.0, 0.0)
        let local_transforms = animator.local_transforms();
        let pos = local_transforms[0].to_scale_rotation_translation().2;
        assert!((pos.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_speed_control() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("run", false);
        let clip_index = animator.add_clip(clip);

        animator.play(clip_index);

        // Access the state and set speed
        assert_eq!(animator.states.len(), 1);
        animator.states[0].set_speed(2.0);

        // Update with double speed
        animator.update(0.5);

        // Time should advance by 1.0 (0.5 * 2.0), but animation completes at 1.0
        // Since the animation is non-looping and duration is 1.0, it will stop
        // We need to check before it completes
        let local_transforms = animator.local_transforms();
        let pos = local_transforms[0].to_scale_rotation_translation().2;
        // Should be at end position (10.0, 0.0, 0.0)
        assert!((pos.x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_stop_all() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip1 = create_test_clip("walk", false);
        let clip2 = create_test_clip("run", false);

        let idx1 = animator.add_clip(clip1);
        let idx2 = animator.add_clip(clip2);

        animator.play_with_weight(idx1, 0.5);
        animator.play_with_weight(idx2, 0.5);

        assert_eq!(animator.states.len(), 2);

        animator.stop_all();
        assert_eq!(animator.states.len(), 0);
    }

    #[test]
    fn test_animator_basic_blending() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        // Clip 1: moves root to (10, 0, 0)
        let clip1 = create_test_clip("walk", false);

        // Clip 2: moves root to (0, 10, 0)
        let mut clip2 = AnimationClip::new("strafe");
        let mut track0 = BoneTrack::new(0);
        track0.add_keyframe(TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        track0.add_keyframe(TransformKeyframe::new(
            1.0,
            Vec3::new(0.0, 10.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        clip2.add_track(track0);
        clip2.calculate_duration();

        let idx1 = animator.add_clip(clip1);
        let idx2 = animator.add_clip(clip2);

        // Play both with equal weight
        animator.play_with_weight(idx1, 0.5);
        animator.play_with_weight(idx2, 0.5);

        // Update to end
        animator.update(1.0);

        // Blended result should be at (5, 5, 0)
        let local_transforms = animator.local_transforms();
        let pos = local_transforms[0].to_scale_rotation_translation().2;

        assert!((pos.x - 5.0).abs() < 0.1);
        assert!((pos.y - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_animator_skinning_matrices() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip = create_test_clip("test", false);
        let clip_index = animator.add_clip(clip);

        animator.play(clip_index);
        animator.update(0.0);

        let skinning_matrices = animator.skinning_matrices();
        assert_eq!(skinning_matrices.len(), 2);

        // Skinning matrices should be computed (world * inverse_bind)
        // These should be valid transforms
        for matrix in skinning_matrices {
            // Check that the matrix is not degenerate
            let det = matrix.determinant();
            assert!(det.abs() > 0.001, "Matrix should not be degenerate");

            // Check that translation is reasonable (not NaN or infinite)
            let pos = matrix.to_scale_rotation_translation().2;
            assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
        }
    }

    #[test]
    fn test_animator_play_stops_others() {
        let skeleton = create_test_skeleton();
        let mut animator = Animator::new(skeleton);

        let clip1 = create_test_clip("walk", false);
        let clip2 = create_test_clip("run", false);

        let idx1 = animator.add_clip(clip1);
        let idx2 = animator.add_clip(clip2);

        animator.play(idx1);
        assert_eq!(animator.states.len(), 1);

        // Playing another clip should stop the first
        animator.play(idx2);

        // Update to clean up stopped states
        animator.update(0.0);

        // Should only have the second clip playing
        assert_eq!(animator.states.len(), 1);
        assert_eq!(animator.states[0].clip_index, idx2);
    }
}
