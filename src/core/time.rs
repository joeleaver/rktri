//! Frame timing utilities

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// FPS statistics for a time window
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FpsWindow {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
}

/// Rolling FPS statistics over multiple time windows
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FpsStats {
    pub one_sec: FpsWindow,
    pub five_sec: FpsWindow,
    pub fifteen_sec: FpsWindow,
    pub current_fps: f32,
    pub frame_count: u64,
}

/// Tracks frame timing and calculates FPS
pub struct FrameTimer {
    last_frame: Instant,
    delta: Duration,
    frame_count: u64,
    fps_timer: Instant,
    fps: f32,
    fps_frame_count: u32,
    /// Ring buffer of (timestamp, frame_time_secs) for rolling stats
    frame_history: VecDeque<(Instant, f32)>,
}

impl FrameTimer {
    /// Create a new frame timer
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last_frame: now,
            delta: Duration::ZERO,
            frame_count: 0,
            fps_timer: now,
            fps: 0.0,
            fps_frame_count: 0,
            frame_history: VecDeque::new(),
        }
    }

    /// Call once per frame to update timing
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = now - self.last_frame;
        self.last_frame = now;
        self.frame_count += 1;
        self.fps_frame_count += 1;

        // Add frame to history
        let frame_time = self.delta.as_secs_f32();
        self.frame_history.push_back((now, frame_time));

        // Prune frames older than 15 seconds
        let cutoff = now - Duration::from_secs(15);
        while let Some(&(timestamp, _)) = self.frame_history.front() {
            if timestamp < cutoff {
                self.frame_history.pop_front();
            } else {
                break;
            }
        }

        // Update FPS every second
        let fps_elapsed = now - self.fps_timer;
        if fps_elapsed >= Duration::from_secs(1) {
            self.fps = self.fps_frame_count as f32 / fps_elapsed.as_secs_f32();
            self.fps_frame_count = 0;
            self.fps_timer = now;
        }
    }

    /// Get delta time in seconds
    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    /// Get delta time as Duration
    pub fn delta(&self) -> Duration {
        self.delta
    }

    /// Get current FPS (updated every second)
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Get total frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get rolling FPS statistics over 1s, 5s, and 15s windows
    pub fn fps_stats(&self) -> FpsStats {
        let now = Instant::now();

        let one_sec = self.compute_window_stats(now, Duration::from_secs(1));
        let five_sec = self.compute_window_stats(now, Duration::from_secs(5));
        let fifteen_sec = self.compute_window_stats(now, Duration::from_secs(15));

        FpsStats {
            one_sec,
            five_sec,
            fifteen_sec,
            current_fps: self.fps,
            frame_count: self.frame_count,
        }
    }

    /// Compute FPS statistics for a given time window
    fn compute_window_stats(&self, now: Instant, window: Duration) -> FpsWindow {
        let cutoff = now - window;

        let mut frame_count = 0;
        let mut total_time = 0.0f32;
        let mut min_fps = f32::INFINITY;
        let mut max_fps = 0.0f32;

        for &(timestamp, frame_time) in self.frame_history.iter() {
            if timestamp >= cutoff {
                frame_count += 1;
                total_time += frame_time;

                // Calculate FPS for this frame
                let fps = if frame_time > 0.0 {
                    1.0 / frame_time
                } else {
                    0.0
                };

                min_fps = min_fps.min(fps);
                max_fps = max_fps.max(fps);
            }
        }

        // Calculate average FPS using frame_count / total_time
        let avg_fps = if total_time > 0.0 {
            frame_count as f32 / total_time
        } else {
            0.0
        };

        // Handle case where no frames in window
        if frame_count == 0 {
            min_fps = 0.0;
            max_fps = 0.0;
        }

        FpsWindow {
            avg: avg_fps,
            min: min_fps,
            max: max_fps,
        }
    }
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new()
    }
}
