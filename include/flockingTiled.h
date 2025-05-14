#ifndef FLOCKING_TILED_H
#define FLOCKING_TILED_H

#include "bird.h"
#include <helper_math.h>

// Accumulate flocking forces for a single neighbor
__device__ void accumulateFlocking(
    int myIndex,
    int neighborIndex,
    const Bird& myBird,
    const Bird& neighbor,
    float3& sep, int& sepCount,
    float3& align, int& alignCount,
    float3& coh, int& cohCount,
    float sepRadius, float alignRadius, float cohRadius)
{
    if (myIndex == neighborIndex) return; // skip self


    float3 offset = neighbor.position - myBird.position;
    float dist = length(offset);

    // Separation
    if (dist < sepRadius && dist > 0.0f) {
        sep -= (offset / dist);
        sepCount++;
    }
    // Alignment
    if (dist < alignRadius) {
        align += neighbor.velocity;
        alignCount++;
    }
    // Cohesion
    if (dist < cohRadius) {
        coh += neighbor.position;
        cohCount++;
    }
}

#endif 
