#ifndef BIRD_H
#define BIRD_H

#include <helper_math.h>  // For float3 (3D vector)


// Bird data structure
struct __align__(16) Bird {
    float3 position;    // 12 bytes
    float3 velocity;    // 12 bytes
    float3 acceleration; // 12 bytes
    uchar4 color;       // 4 bytes
    // Total: 40 bytes (aligned to 16 bytes for CUDA compatibility)

    __host__ __device__
        Bird(float3 pos, float3 vel, float3 acc, uchar4 col)
        : position(pos), velocity(vel), acceleration(acc), color(col) {
    }
};

#endif // BIRD_H
#pragma once
