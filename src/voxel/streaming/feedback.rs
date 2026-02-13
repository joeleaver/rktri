//! GPU feedback buffer for demand-driven brick streaming.
//!
//! During ray traversal, the GPU shader writes requested brick IDs
//! to a feedback buffer. The CPU reads these back to determine which
//! bricks to stream in next frame.

use std::collections::HashSet;
use crate::voxel::brick_handle::BrickId;
use crate::voxel::chunk::ChunkCoord;

/// Sentinel value indicating a brick is not resident in GPU memory.
pub const BRICK_NOT_RESIDENT: u32 = 0xFFFFFFFF;

/// Maximum brick requests per frame.
pub const MAX_FEEDBACK_REQUESTS: u32 = 4096;

/// A brick request from the GPU feedback buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickRequest {
    /// The requested brick ID
    pub brick_id: BrickId,
    /// Screen-space importance (how many pixels needed this brick)
    pub pixel_count: u32,
}

/// CPU-side representation of the GPU feedback buffer.
pub struct FeedbackBuffer {
    /// Maximum requests per frame
    max_requests: u32,
    /// Requests from last frame
    requests: Vec<BrickRequest>,
    /// Deduplicated set of requested brick IDs
    unique_requests: HashSet<BrickId>,
    /// Frame counter
    frame: u32,
    /// Whether double-buffering is active
    #[allow(dead_code)]
    double_buffered: bool,
}

impl FeedbackBuffer {
    /// Create a new feedback buffer.
    pub fn new(max_requests: u32) -> Self {
        Self {
            max_requests,
            requests: Vec::with_capacity(max_requests as usize),
            unique_requests: HashSet::new(),
            frame: 0,
            double_buffered: true,
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(MAX_FEEDBACK_REQUESTS)
    }

    /// Begin a new frame (clear previous requests).
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.requests.clear();
        self.unique_requests.clear();
    }

    /// Process raw feedback data from GPU readback.
    ///
    /// In practice, this reads from a staging buffer after GPU->CPU copy.
    /// The format is: [count: u32, brick_id_0: u64, pixel_count_0: u32, ...]
    pub fn process_raw_feedback(&mut self, data: &[u32]) {
        if data.is_empty() {
            return;
        }

        let count = data[0].min(self.max_requests) as usize;

        // Each request is 3 u32s: chunk_x_y_z packed, local_index, pixel_count
        let stride = 3;
        for i in 0..count {
            let offset = 1 + i * stride;
            if offset + stride > data.len() {
                break;
            }

            let chunk_packed = data[offset];
            let local_index = data[offset + 1];
            let pixel_count = data[offset + 2];

            // Unpack chunk coordinate from packed u32
            // Format: x:10 | y:10 | z:10 (signed, biased by 512)
            let cx = ((chunk_packed >> 20) & 0x3FF) as i32 - 512;
            let cy = ((chunk_packed >> 10) & 0x3FF) as i32 - 512;
            let cz = (chunk_packed & 0x3FF) as i32 - 512;

            let brick_id = BrickId::new(ChunkCoord::new(cx, cy, cz), local_index);
            let request = BrickRequest { brick_id, pixel_count };

            self.unique_requests.insert(brick_id);
            self.requests.push(request);
        }
    }

    /// Add a request manually (for CPU-side demand).
    pub fn add_request(&mut self, brick_id: BrickId, pixel_count: u32) {
        if self.requests.len() < self.max_requests as usize {
            self.unique_requests.insert(brick_id);
            self.requests.push(BrickRequest { brick_id, pixel_count });
        }
    }

    /// Get all requests from this frame.
    pub fn requests(&self) -> &[BrickRequest] {
        &self.requests
    }

    /// Get unique requested brick IDs.
    pub fn unique_brick_ids(&self) -> &HashSet<BrickId> {
        &self.unique_requests
    }

    /// Get requests sorted by priority (highest pixel count first).
    pub fn requests_by_priority(&self) -> Vec<BrickRequest> {
        let mut sorted = self.requests.clone();
        sorted.sort_by(|a, b| b.pixel_count.cmp(&a.pixel_count));
        sorted
    }

    /// Number of requests this frame.
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Number of unique brick IDs requested.
    pub fn unique_count(&self) -> usize {
        self.unique_requests.len()
    }

    /// Current frame number.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// Maximum requests per frame.
    pub fn max_requests(&self) -> u32 {
        self.max_requests
    }

    /// Check if buffer overflowed (more requests than capacity).
    pub fn overflowed(&self) -> bool {
        self.requests.len() >= self.max_requests as usize
    }

    /// Pack a chunk coordinate into a single u32.
    pub fn pack_chunk_coord(coord: ChunkCoord) -> u32 {
        let x = (coord.x + 512) as u32 & 0x3FF;
        let y = (coord.y + 512) as u32 & 0x3FF;
        let z = (coord.z + 512) as u32 & 0x3FF;
        (x << 20) | (y << 10) | z
    }

    /// Unpack a chunk coordinate from a single u32.
    pub fn unpack_chunk_coord(packed: u32) -> ChunkCoord {
        let x = ((packed >> 20) & 0x3FF) as i32 - 512;
        let y = ((packed >> 10) & 0x3FF) as i32 - 512;
        let z = (packed & 0x3FF) as i32 - 512;
        ChunkCoord::new(x, y, z)
    }
}

impl Default for FeedbackBuffer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_request() {
        let mut fb = FeedbackBuffer::with_defaults();
        fb.begin_frame();

        let brick_id = BrickId::new(ChunkCoord::new(0, 0, 0), 42);
        fb.add_request(brick_id, 100);

        assert_eq!(fb.request_count(), 1);
        assert_eq!(fb.unique_count(), 1);
    }

    #[test]
    fn test_pack_unpack_coord() {
        let coord = ChunkCoord::new(5, -3, 10);
        let packed = FeedbackBuffer::pack_chunk_coord(coord);
        let unpacked = FeedbackBuffer::unpack_chunk_coord(packed);
        assert_eq!(unpacked, coord);
    }

    #[test]
    fn test_pack_unpack_negative() {
        let coord = ChunkCoord::new(-100, -200, 300);
        let packed = FeedbackBuffer::pack_chunk_coord(coord);
        let unpacked = FeedbackBuffer::unpack_chunk_coord(packed);
        assert_eq!(unpacked, coord);
    }

    #[test]
    fn test_process_raw_feedback() {
        let mut fb = FeedbackBuffer::with_defaults();
        fb.begin_frame();

        // Simulate GPU feedback: 1 request
        let coord = ChunkCoord::new(1, 2, 3);
        let packed = FeedbackBuffer::pack_chunk_coord(coord);
        let data = vec![1u32, packed, 42, 500]; // count=1, packed_coord, local_index=42, pixels=500

        fb.process_raw_feedback(&data);
        assert_eq!(fb.request_count(), 1);

        let req = &fb.requests()[0];
        assert_eq!(req.brick_id.chunk, coord);
        assert_eq!(req.brick_id.local_index, 42);
        assert_eq!(req.pixel_count, 500);
    }

    #[test]
    fn test_priority_sort() {
        let mut fb = FeedbackBuffer::with_defaults();
        fb.begin_frame();

        fb.add_request(BrickId::new(ChunkCoord::new(0, 0, 0), 1), 10);
        fb.add_request(BrickId::new(ChunkCoord::new(0, 0, 0), 2), 1000);
        fb.add_request(BrickId::new(ChunkCoord::new(0, 0, 0), 3), 100);

        let sorted = fb.requests_by_priority();
        assert_eq!(sorted[0].pixel_count, 1000);
        assert_eq!(sorted[1].pixel_count, 100);
        assert_eq!(sorted[2].pixel_count, 10);
    }

    #[test]
    fn test_frame_reset() {
        let mut fb = FeedbackBuffer::with_defaults();
        fb.begin_frame();
        fb.add_request(BrickId::new(ChunkCoord::new(0, 0, 0), 0), 1);
        assert_eq!(fb.request_count(), 1);

        fb.begin_frame();
        assert_eq!(fb.request_count(), 0);
        assert_eq!(fb.frame(), 2);
    }
}
