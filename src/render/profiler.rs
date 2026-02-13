//! GPU profiling using wgpu timestamp queries

/// Per-pass GPU timing data (in milliseconds)
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct GpuTimings {
    pub svo_trace_ms: f32,
    pub shadow_ms: f32,
    pub godrays_ms: f32,
    pub lighting_ms: f32,
    pub display_ms: f32,
    pub total_gpu_ms: f32,
}

/// GPU profiler using timestamp queries
pub struct GpuProfiler {
    enabled: bool,
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    read_buffer: wgpu::Buffer,
    timestamp_period: f32,
    /// Stores the latest resolved timings
    latest_timings: GpuTimings,
    /// Rolling average over N frames
    frame_timings: std::collections::VecDeque<GpuTimings>,
    max_history: usize,
}

const NUM_PASSES: u32 = 5; // svo_trace, shadow, godrays, lighting, display
const TIMESTAMPS_PER_PASS: u32 = 2; // begin + end
const TOTAL_TIMESTAMPS: u32 = NUM_PASSES * TIMESTAMPS_PER_PASS;

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: TOTAL_TIMESTAMPS,
        });

        let buffer_size = (TOTAL_TIMESTAMPS as u64) * std::mem::size_of::<u64>() as u64;

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_resolve"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_read"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_period = queue.get_timestamp_period();

        Self {
            enabled,
            query_set,
            resolve_buffer,
            read_buffer,
            timestamp_period,
            latest_timings: GpuTimings::default(),
            frame_timings: std::collections::VecDeque::new(),
            max_history: 60,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get timestamp writes for a compute pass
    /// pass_index: 0=svo_trace, 1=shadow, 2=godrays, 3=lighting
    pub fn compute_pass_timestamp_writes(&self, pass_index: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.enabled {
            return None;
        }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_index * 2),
            end_of_pass_write_index: Some(pass_index * 2 + 1),
        })
    }

    /// Get timestamp writes for a render pass (display pass)
    /// pass_index: 4=display
    pub fn render_pass_timestamp_writes(&self, pass_index: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if !self.enabled {
            return None;
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_index * 2),
            end_of_pass_write_index: Some(pass_index * 2 + 1),
        })
    }

    /// Resolve queries and copy to readable buffer. Call after all passes, before submit.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..TOTAL_TIMESTAMPS, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer, 0,
            &self.read_buffer, 0,
            (TOTAL_TIMESTAMPS as u64) * std::mem::size_of::<u64>() as u64,
        );
    }

    /// Read back results from previous frame. Non-blocking - returns cached if not ready.
    pub fn read_results(&mut self, device: &wgpu::Device) {
        if !self.enabled {
            return;
        }

        let buffer_slice = self.read_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll device to process the map
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

        if let Ok(Ok(())) = rx.try_recv() {
            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            if timestamps.len() >= TOTAL_TIMESTAMPS as usize {
                let ns_per_tick = self.timestamp_period as f64;
                let ms = |begin: u64, end: u64| -> f32 {
                    ((end.wrapping_sub(begin)) as f64 * ns_per_tick / 1_000_000.0) as f32
                };

                let timings = GpuTimings {
                    svo_trace_ms: ms(timestamps[0], timestamps[1]),
                    shadow_ms: ms(timestamps[2], timestamps[3]),
                    godrays_ms: ms(timestamps[4], timestamps[5]),
                    lighting_ms: ms(timestamps[6], timestamps[7]),
                    display_ms: ms(timestamps[8], timestamps[9]),
                    total_gpu_ms: ms(timestamps[0], timestamps[9]),
                };

                self.frame_timings.push_back(timings);
                if self.frame_timings.len() > self.max_history {
                    self.frame_timings.pop_front();
                }
                self.latest_timings = timings;
            }

            drop(data);
            self.read_buffer.unmap();
        }
    }

    /// Get latest per-pass timings
    pub fn latest_timings(&self) -> GpuTimings {
        self.latest_timings
    }

    /// Get averaged timings over the history window
    pub fn average_timings(&self) -> GpuTimings {
        if self.frame_timings.is_empty() {
            return GpuTimings::default();
        }
        let n = self.frame_timings.len() as f32;
        let mut avg = GpuTimings::default();
        for t in &self.frame_timings {
            avg.svo_trace_ms += t.svo_trace_ms;
            avg.shadow_ms += t.shadow_ms;
            avg.godrays_ms += t.godrays_ms;
            avg.lighting_ms += t.lighting_ms;
            avg.display_ms += t.display_ms;
            avg.total_gpu_ms += t.total_gpu_ms;
        }
        avg.svo_trace_ms /= n;
        avg.shadow_ms /= n;
        avg.godrays_ms /= n;
        avg.lighting_ms /= n;
        avg.display_ms /= n;
        avg.total_gpu_ms /= n;
        avg
    }
}
