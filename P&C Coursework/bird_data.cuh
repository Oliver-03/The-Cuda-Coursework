#pragma once
#include <cuda_runtime.h>

// =========================
// Configuration Constants
// =========================
#define MAX_BIRDS 10000

#define VISION_RADIUS 1.5f
#define MAX_SPEED 1.0f

#define SEPARATION_WEIGHT 1.5f
#define ALIGNMENT_WEIGHT 1.0f
#define COHESION_WEIGHT 1.0f

#define WORLD_BOUNDS 10.0f  // +/- bounds in each axis

// =========================
// Bird Flock SoA Structure
// =========================
struct BirdFlock {
    int numBirds;
    float3* d_positions;
    float3* d_velocities;
    float3* d_accelerations;
    float3 bounds_min;   // Minimum bounds for the flock
    float3 bounds_max;   // Maximum bounds for the flock
};


// =========================
// Device-side Vector Utils
// =========================
__device__ float3 limit_velocity(float3 v, float max_speed) {
    // Calculate squared magnitude of velocity vector
    float speed_squared = v.x * v.x + v.y * v.y + v.z * v.z;
    float max_speed_squared = max_speed * max_speed;

    if (speed_squared > max_speed_squared) {
        float scale = max_speed / sqrtf(speed_squared);
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
    }

    return v;
}

__device__ float3 clamp_position(float3 pos, float3 bounds_min, float3 bounds_max) {
    pos.x = fminf(fmaxf(pos.x, bounds_min.x), bounds_max.x);
    pos.y = fminf(fmaxf(pos.y, bounds_min.y), bounds_max.y);
    pos.z = fminf(fmaxf(pos.z, bounds_min.z), bounds_max.z);
    return pos;
}


