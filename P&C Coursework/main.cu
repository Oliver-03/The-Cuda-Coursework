#include <iostream>
#include <cuda_runtime.h>
#include "bird_data.cuh"

#define MAX_SPEED 10.0f
const float3 WORLD_BOUNDS_MIN = make_float3(-100.0f, -100.0f, -100.0f);
const float3 WORLD_BOUNDS_MAX = make_float3(100.0f, 100.0f, 100.0f);
#define MAX_BIRDS 1000

// Function to check CUDA errors
void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__global__ void find_neighbors_kernel(const BirdFlock flock, float radius, int* d_neighbors_count, int* d_neighbors_list) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= flock.numBirds) return;

    int count = 0;
    for (int i = 0; i < flock.numBirds; ++i) {
        if (i != idx) {
            float3 pos1 = flock.d_positions[idx];
            float3 pos2 = flock.d_positions[i];

            float dist_sq = (pos1.x - pos2.x) * (pos1.x - pos2.x) +
                (pos1.y - pos2.y) * (pos1.y - pos2.y) +
                (pos1.z - pos2.z) * (pos1.z - pos2.z);

            if (dist_sq < radius * radius) {
                d_neighbors_list[idx * MAX_BIRDS + count] = i;
                count++;
            }
        }
    }
    d_neighbors_count[idx] = count;
}

__global__ void update_birds_kernel(BirdFlock flock, float radius, int* d_neighbors_count, int* d_neighbors_list) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= flock.numBirds) return;

    float3 pos = flock.d_positions[i];
    float3 vel = flock.d_velocities[i];
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    // Apply flocking behavior (separation, alignment, cohesion)
    float3 separation = make_float3(0.0f, 0.0f, 0.0f);
    float3 alignment = make_float3(0.0f, 0.0f, 0.0f);
    float3 cohesion = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over neighbors and calculate forces
    for (int j = 0; j < d_neighbors_count[i]; ++j) {
        int neighbor_idx = d_neighbors_list[i * MAX_BIRDS + j];
        if (neighbor_idx == i) continue;

        float3 neighbor_pos = flock.d_positions[neighbor_idx];
        float3 neighbor_vel = flock.d_velocities[neighbor_idx];

        // Separation force: steer away from neighbors
        float3 diff = make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y, pos.z - neighbor_pos.z);
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (dist < radius) {
            separation.x += diff.x / dist;
            separation.y += diff.y / dist;
            separation.z += diff.z / dist;
        }

        // Alignment force: steer towards the average velocity of neighbors
        alignment.x += neighbor_vel.x;
        alignment.y += neighbor_vel.y;
        alignment.z += neighbor_vel.z;

        // Cohesion force: steer towards the average position of neighbors
        cohesion.x += neighbor_pos.x;
        cohesion.y += neighbor_pos.y;
        cohesion.z += neighbor_pos.z;
    }

    // Apply weights to forces
    float separation_weight = 1.5f;
    float alignment_weight = 1.0f;
    float cohesion_weight = 1.0f;

    acc = make_float3(separation_weight * separation.x + alignment_weight * alignment.x + cohesion_weight * cohesion.x,
        separation_weight * separation.y + alignment_weight * alignment.y + cohesion_weight * cohesion.y,
        separation_weight * separation.z + alignment_weight * alignment.z + cohesion_weight * cohesion.z);

    // Update velocity and position
    vel = limit_velocity(make_float3(vel.x + acc.x, vel.y + acc.y, vel.z + acc.z), MAX_SPEED);
    pos = clamp_position(make_float3(pos.x + vel.x, pos.y + vel.y, pos.z + vel.z), flock.bounds_min, flock.bounds_max);

    flock.d_positions[i] = pos;
    flock.d_velocities[i] = vel;
    flock.d_accelerations[i] = acc;
}


/*int main() {
    int numBirds = 1000;

    // Create BirdFlock structure
    BirdFlock flock;
    flock.numBirds = numBirds;
    flock.bounds_min = WORLD_BOUNDS_MIN;
    flock.bounds_max = WORLD_BOUNDS_MAX;

    // Allocate memory for positions, velocities, and accelerations on the device
    cudaMalloc(&flock.d_positions, numBirds * sizeof(float3));
    check_cuda_error("cudaMalloc for d_positions failed");

    cudaMalloc(&flock.d_velocities, numBirds * sizeof(float3));
    check_cuda_error("cudaMalloc for d_velocities failed");

    cudaMalloc(&flock.d_accelerations, numBirds * sizeof(float3));
    check_cuda_error("cudaMalloc for d_accelerations failed");

    // Allocate memory for neighbors count and list on the device
    int* d_neighbors_count;
    int* d_neighbors_list;
    cudaMalloc(&d_neighbors_count, numBirds * sizeof(int));
    cudaMalloc(&d_neighbors_list, numBirds * MAX_BIRDS * sizeof(int));

    check_cuda_error("cudaMalloc for d_neighbors_count and d_neighbors_list failed");

    // Set up grid and block size for kernel execution
    dim3 blockSize(256); // 256 threads per block
    dim3 numBlocks((numBirds + blockSize.x - 1) / blockSize.x); // Ensures all threads are covered


    // Simulate a few steps
    for (int step = 0; step < 100; ++step) {
        // Find neighbors first
        float radius = 10.0f;
        find_neighbors_kernel << <numBlocks, blockSize >> > (flock, radius, d_neighbors_count, d_neighbors_list);
        cudaDeviceSynchronize(); // Ensures any errors are caught
        check_cuda_error("Kernel launch failed for find_neighbors_kernel");

        // Synchronize the device after kernel execution
        cudaDeviceSynchronize();
        check_cuda_error("cudaDeviceSynchronize failed");

        // Update birds' positions and velocities in each simulation step
        update_birds_kernel << <numBlocks, blockSize >> > (flock, 10.0f, d_neighbors_count, d_neighbors_list);
        check_cuda_error("Kernel launch failed for update_birds_kernel");

        // Synchronize the device after kernel execution
        cudaDeviceSynchronize();
        check_cuda_error("cudaDeviceSynchronize failed");
    }

    // Free device memory
    cudaFree(flock.d_positions);
    cudaFree(flock.d_velocities);
    cudaFree(flock.d_accelerations);
    cudaFree(d_neighbors_count);
    cudaFree(d_neighbors_list);

    return 0;
}*/