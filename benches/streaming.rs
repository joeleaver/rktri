use criterion::{criterion_group, criterion_main, Criterion, black_box};

use rktri::voxel::streaming::{
    PrefetchPredictor,
    FeedbackBuffer,
};
use rktri::voxel::brick_handle::BrickId;
use rktri::voxel::chunk::ChunkCoord;
use rktri::voxel::svo::builder::{OctreeBuilder, create_test_sphere};

use glam::Vec3;

fn bench_octree_build_64(c: &mut Criterion) {
    let size = 64u32;
    let voxels = create_test_sphere(size, 28.0);

    c.bench_function("octree_build_64", |b| {
        b.iter(|| {
            let builder = OctreeBuilder::new(black_box(size));
            builder.build(black_box(&voxels), size as f32)
        });
    });
}

fn bench_octree_build_32(c: &mut Criterion) {
    let size = 32u32;
    let voxels = create_test_sphere(size, 14.0);

    c.bench_function("octree_build_32", |b| {
        b.iter(|| {
            let builder = OctreeBuilder::new(black_box(size));
            builder.build(black_box(&voxels), size as f32)
        });
    });
}

fn bench_octree_build_128(c: &mut Criterion) {
    let size = 128u32;
    let voxels = create_test_sphere(size, 56.0);

    c.bench_function("octree_build_128", |b| {
        b.iter(|| {
            let builder = OctreeBuilder::new(black_box(size));
            builder.build(black_box(&voxels), size as f32)
        });
    });
}

fn bench_prefetch_update(c: &mut Criterion) {
    let mut predictor = PrefetchPredictor::new();

    c.bench_function("prefetch_update_smooth_motion", |b| {
        let mut frame = 0u32;
        b.iter(|| {
            frame += 1;
            let pos = Vec3::new(
                (frame as f32 * 0.1).sin() * 100.0,
                10.0,
                (frame as f32 * 0.1).cos() * 100.0,
            );
            predictor.update(black_box(pos), black_box(1.0 / 60.0));
        });
    });
}

fn bench_prefetch_velocity_estimation(c: &mut Criterion) {
    c.bench_function("prefetch_velocity_estimation", |b| {
        b.iter(|| {
            let mut predictor = PrefetchPredictor::new();

            // Simulate 30 frames of movement at 10 m/s along X axis
            for i in 0..30 {
                let pos = Vec3::new(i as f32 * 0.166, 0.0, 0.0);
                predictor.update(pos, black_box(1.0 / 60.0));
            }

            let speed = predictor.speed();
            black_box(speed);
        });
    });
}

fn bench_prefetch_predict_chunks(c: &mut Criterion) {
    let mut predictor = PrefetchPredictor::new();

    // Prime with some positions
    for i in 0..20 {
        let pos = Vec3::new(i as f32 * 0.5, 10.0, 0.0);
        predictor.update(pos, 1.0 / 60.0);
    }

    c.bench_function("prefetch_predict_needed_chunks", |b| {
        b.iter(|| {
            let chunks = predictor.predict_needed_chunks(black_box(&|_| false));
            black_box(chunks);
        });
    });
}

fn bench_feedback_add_request(c: &mut Criterion) {
    let mut feedback = FeedbackBuffer::new(1024);
    feedback.begin_frame();

    c.bench_function("feedback_add_request", |b| {
        let mut counter = 0u32;
        b.iter(|| {
            counter += 1;
            let brick_id = BrickId::new(
                ChunkCoord::new((counter % 4) as i32, 0, (counter / 4) as i32),
                counter as u32,
            );
            feedback.add_request(black_box(brick_id), black_box(100));
        });
    });
}

fn bench_feedback_process_requests(c: &mut Criterion) {
    c.bench_function("feedback_process_requests", |b| {
        b.iter(|| {
            let mut feedback = FeedbackBuffer::new(1024);
            feedback.begin_frame();

            // Simulate 100 brick requests
            for i in 0..100 {
                let brick_id = BrickId::new(
                    ChunkCoord::new((i % 4) as i32, 0, (i / 4) as i32),
                    i as u32,
                );
                feedback.add_request(brick_id, black_box(100 + i as u32));
            }

            let requests = feedback.requests();
            black_box(requests);
        });
    });
}

fn bench_feedback_priority_sort(c: &mut Criterion) {
    c.bench_function("feedback_requests_by_priority", |b| {
        b.iter(|| {
            let mut feedback = FeedbackBuffer::new(1024);
            feedback.begin_frame();

            // Add requests with varying priorities
            for i in 0..100 {
                let brick_id = BrickId::new(
                    ChunkCoord::new((i % 4) as i32, 0, (i / 4) as i32),
                    i as u32,
                );
                let pixel_count = 1000u32.saturating_sub(i as u32 * 9);
                feedback.add_request(brick_id, pixel_count);
            }

            let sorted = feedback.requests_by_priority();
            black_box(sorted);
        });
    });
}

fn bench_feedback_pack_unpack_chunk(c: &mut Criterion) {
    let coord = ChunkCoord::new(5, -3, 10);

    c.bench_function("feedback_pack_unpack_chunk_coord", |b| {
        b.iter(|| {
            let packed = FeedbackBuffer::pack_chunk_coord(black_box(coord));
            let unpacked = FeedbackBuffer::unpack_chunk_coord(black_box(packed));
            black_box(unpacked);
        });
    });
}

fn bench_feedback_raw_feedback_processing(c: &mut Criterion) {
    c.bench_function("feedback_process_raw_feedback", |b| {
        b.iter(|| {
            let mut feedback = FeedbackBuffer::new(1024);
            feedback.begin_frame();

            // Simulate GPU feedback buffer with 50 brick requests
            let mut data = vec![50u32];
            for i in 0..50 {
                let coord = ChunkCoord::new((i % 4) as i32, 0, (i / 4) as i32);
                let packed = FeedbackBuffer::pack_chunk_coord(coord);
                data.push(packed);
                data.push(i as u32);  // local_index
                data.push(100 + i as u32);  // pixel_count
            }

            feedback.process_raw_feedback(black_box(&data));
            black_box(feedback.request_count());
        });
    });
}

criterion_group!(
    benches,
    bench_octree_build_32,
    bench_octree_build_64,
    bench_octree_build_128,
    bench_prefetch_update,
    bench_prefetch_velocity_estimation,
    bench_prefetch_predict_chunks,
    bench_feedback_add_request,
    bench_feedback_process_requests,
    bench_feedback_priority_sort,
    bench_feedback_pack_unpack_chunk,
    bench_feedback_raw_feedback_processing,
);
criterion_main!(benches);
