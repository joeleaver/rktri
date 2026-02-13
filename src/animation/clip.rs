//! Animation clip and keyframe system

use glam::{Mat4, Quat, Vec3};

/// A single transform keyframe at a specific time
#[derive(Clone, Debug)]
pub struct TransformKeyframe {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl TransformKeyframe {
    /// Create a new keyframe with the given transform
    pub fn new(time: f32, position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            time,
            position,
            rotation,
            scale,
        }
    }

    /// Create an identity transform keyframe at the given time
    pub fn identity(time: f32) -> Self {
        Self {
            time,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Convert this keyframe to a transformation matrix
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Linearly interpolate between two keyframes
    /// t should be in range [0, 1] where 0 = keyframe a, 1 = keyframe b
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            time: a.time + (b.time - a.time) * t,
            position: a.position.lerp(b.position, t),
            rotation: a.rotation.slerp(b.rotation, t),
            scale: a.scale.lerp(b.scale, t),
        }
    }
}

/// Animation track for a single bone
#[derive(Clone, Debug)]
pub struct BoneTrack {
    pub bone_index: usize,
    pub keyframes: Vec<TransformKeyframe>,
}

impl BoneTrack {
    /// Create a new empty bone track
    pub fn new(bone_index: usize) -> Self {
        Self {
            bone_index,
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe to this track (maintains sorted order by time)
    pub fn add_keyframe(&mut self, keyframe: TransformKeyframe) {
        // Insert keyframe in sorted order
        let pos = self
            .keyframes
            .binary_search_by(|k| k.time.partial_cmp(&keyframe.time).unwrap())
            .unwrap_or_else(|e| e);
        self.keyframes.insert(pos, keyframe);
    }

    /// Sample the animation at a given time, interpolating between keyframes
    pub fn sample(&self, time: f32) -> Mat4 {
        if self.keyframes.is_empty() {
            return Mat4::IDENTITY;
        }

        // If before first keyframe, use first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].to_matrix();
        }

        // If after last keyframe, use last keyframe
        if time >= self.keyframes[self.keyframes.len() - 1].time {
            return self.keyframes[self.keyframes.len() - 1].to_matrix();
        }

        // Find the two keyframes to interpolate between
        for i in 0..self.keyframes.len() - 1 {
            let current = &self.keyframes[i];
            let next = &self.keyframes[i + 1];

            if time >= current.time && time <= next.time {
                // Calculate interpolation factor
                let duration = next.time - current.time;
                let t = if duration > 0.0 {
                    (time - current.time) / duration
                } else {
                    0.0
                };

                // Interpolate and return matrix
                return TransformKeyframe::lerp(current, next, t).to_matrix();
            }
        }

        // Fallback (should never reach here)
        Mat4::IDENTITY
    }

    /// Get the duration of this track (time of last keyframe)
    pub fn duration(&self) -> f32 {
        self.keyframes
            .last()
            .map(|k| k.time)
            .unwrap_or(0.0)
    }
}

/// A complete animation clip containing tracks for multiple bones
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<BoneTrack>,
    pub looping: bool,
}

impl AnimationClip {
    /// Create a new empty animation clip
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            duration: 0.0,
            tracks: Vec::new(),
            looping: false,
        }
    }

    /// Add a bone track to this animation
    pub fn add_track(&mut self, track: BoneTrack) {
        self.tracks.push(track);
    }

    /// Get a track for a specific bone index
    pub fn get_track(&self, bone_index: usize) -> Option<&BoneTrack> {
        self.tracks.iter().find(|t| t.bone_index == bone_index)
    }

    /// Sample all bone transforms at a given time
    /// Returns local transforms indexed by bone index (missing bones get identity)
    pub fn sample(&self, time: f32, bone_count: usize) -> Vec<Mat4> {
        let mut transforms = vec![Mat4::IDENTITY; bone_count];

        // Wrap time if looping
        let sample_time = if self.looping && self.duration > 0.0 {
            time % self.duration
        } else {
            time.min(self.duration)
        };

        // Sample each track
        for track in &self.tracks {
            if track.bone_index < bone_count {
                transforms[track.bone_index] = track.sample(sample_time);
            }
        }

        transforms
    }

    /// Calculate the duration from all tracks
    pub fn calculate_duration(&mut self) {
        self.duration = self
            .tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0f32, |a, b| a.max(b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_identity() {
        let kf = TransformKeyframe::identity(1.0);
        assert_eq!(kf.time, 1.0);
        assert_eq!(kf.position, Vec3::ZERO);
        assert_eq!(kf.rotation, Quat::IDENTITY);
        assert_eq!(kf.scale, Vec3::ONE);
    }

    #[test]
    fn test_keyframe_to_matrix() {
        let kf = TransformKeyframe::new(
            0.0,
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        let matrix = kf.to_matrix();

        // Check translation component
        let translation = matrix.w_axis.truncate();
        assert!((translation - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
    }

    #[test]
    fn test_keyframe_lerp() {
        let kf_a = TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        let kf_b = TransformKeyframe::new(
            1.0,
            Vec3::new(10.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );

        let kf_mid = TransformKeyframe::lerp(&kf_a, &kf_b, 0.5);
        assert!((kf_mid.position - Vec3::new(5.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_bone_track_add_keyframe() {
        let mut track = BoneTrack::new(0);

        track.add_keyframe(TransformKeyframe::identity(1.0));
        track.add_keyframe(TransformKeyframe::identity(0.0));
        track.add_keyframe(TransformKeyframe::identity(0.5));

        // Keyframes should be sorted by time
        assert_eq!(track.keyframes[0].time, 0.0);
        assert_eq!(track.keyframes[1].time, 0.5);
        assert_eq!(track.keyframes[2].time, 1.0);
    }

    #[test]
    fn test_bone_track_sample() {
        let mut track = BoneTrack::new(0);

        track.add_keyframe(TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        track.add_keyframe(TransformKeyframe::new(
            1.0,
            Vec3::new(10.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));

        // Sample at midpoint
        let matrix = track.sample(0.5);
        let translation = matrix.w_axis.truncate();
        assert!((translation - Vec3::new(5.0, 0.0, 0.0)).length() < 0.001);

        // Sample before start
        let matrix = track.sample(-1.0);
        let translation = matrix.w_axis.truncate();
        assert!((translation - Vec3::new(0.0, 0.0, 0.0)).length() < 0.001);

        // Sample after end
        let matrix = track.sample(2.0);
        let translation = matrix.w_axis.truncate();
        assert!((translation - Vec3::new(10.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_bone_track_duration() {
        let mut track = BoneTrack::new(0);

        track.add_keyframe(TransformKeyframe::identity(0.0));
        track.add_keyframe(TransformKeyframe::identity(2.5));
        track.add_keyframe(TransformKeyframe::identity(1.0));

        assert_eq!(track.duration(), 2.5);
    }

    #[test]
    fn test_animation_clip_sample() {
        let mut clip = AnimationClip::new("test");

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

        assert_eq!(clip.duration, 1.0);

        // Sample at midpoint
        let transforms = clip.sample(0.5, 3);
        assert_eq!(transforms.len(), 3);

        let pos0 = transforms[0].w_axis.truncate();
        let pos1 = transforms[1].w_axis.truncate();

        assert!((pos0 - Vec3::new(5.0, 0.0, 0.0)).length() < 0.001);
        assert!((pos1 - Vec3::new(0.0, 2.5, 0.0)).length() < 0.001);
        assert_eq!(transforms[2], Mat4::IDENTITY); // No track for bone 2
    }

    #[test]
    fn test_animation_clip_looping() {
        let mut clip = AnimationClip::new("loop_test");
        clip.looping = true;
        clip.duration = 2.0;

        let mut track = BoneTrack::new(0);
        track.add_keyframe(TransformKeyframe::new(
            0.0,
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        track.add_keyframe(TransformKeyframe::new(
            2.0,
            Vec3::new(10.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));

        clip.add_track(track);

        // Sample beyond duration should wrap
        let transforms = clip.sample(2.5, 1);
        let pos = transforms[0].w_axis.truncate();

        // At time 2.5 with looping, wraps to 0.5
        // At 0.5 we should be 25% of the way from start to end
        assert!((pos - Vec3::new(2.5, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_get_track() {
        let mut clip = AnimationClip::new("test");

        let track0 = BoneTrack::new(0);
        let track2 = BoneTrack::new(2);

        clip.add_track(track0);
        clip.add_track(track2);

        assert!(clip.get_track(0).is_some());
        assert!(clip.get_track(1).is_none());
        assert!(clip.get_track(2).is_some());
    }

    #[test]
    fn test_calculate_duration() {
        let mut clip = AnimationClip::new("test");

        let mut track0 = BoneTrack::new(0);
        track0.add_keyframe(TransformKeyframe::identity(0.0));
        track0.add_keyframe(TransformKeyframe::identity(1.5));

        let mut track1 = BoneTrack::new(1);
        track1.add_keyframe(TransformKeyframe::identity(0.0));
        track1.add_keyframe(TransformKeyframe::identity(2.5));

        clip.add_track(track0);
        clip.add_track(track1);
        clip.calculate_duration();

        assert_eq!(clip.duration, 2.5);
    }
}
