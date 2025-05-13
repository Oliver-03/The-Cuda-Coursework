#ifndef FLOCKING_H
#define FLOCKING_H

#include "bird.h"
#include <helper_math.h> // for float3 and vector operations

// Separation: Steer to avoid crowding local flockmates
__device__ float3 separation(const Bird* birds, int birdIndex, int numBirds, float separationDistance) {
    float3 steer = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;

    for (int i = 0; i < numBirds; i++) {
        if (i != birdIndex) {
            float3 diff = birds[birdIndex].position - birds[i].position;
            float dist = length(diff);
            if (dist < separationDistance) {
                steer += normalize(diff) / dist;  // Add inverse of distance to steer away
                count++;
            }
        }
    }

    if (count > 0) {
        steer /= count;
        steer = normalize(steer);  // Normalize to prevent overwhelming force
    }

    return steer;
}

// Alignment: Steer to align with the average heading of local flockmates
__device__ float3 alignment(const Bird* birds, int birdIndex, int numBirds, float alignmentDistance) {
    float3 steer = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;

    for (int i = 0; i < numBirds; i++) {
        if (i != birdIndex) {
            float3 diff = birds[birdIndex].position - birds[i].position;
            float dist = length(diff);
            if (dist < alignmentDistance) {
                steer += birds[i].velocity;  // Add the velocity of nearby birds
                count++;
            }
        }
    }

    if (count > 0) {
        steer /= count;
        steer = normalize(steer);
    }

    return steer;
}

// Cohesion: Steer to move toward the average position of local flockmates
__device__ float3 cohesion(const Bird* birds, int birdIndex, int numBirds, float cohesionDistance) {
    float3 steer = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;

    for (int i = 0; i < numBirds; i++) {
        if (i != birdIndex) {
            float3 diff = birds[birdIndex].position - birds[i].position;
            float dist = length(diff);
            if (dist < cohesionDistance) {
                steer += birds[i].position;  // Add the position of nearby birds
                count++;
            }
        }
    }

    if (count > 0) {
        steer /= count;
        steer = steer - birds[birdIndex].position;  // Move towards the average position
        steer = normalize(steer);
    }

    return steer;
}

#endif // FLOCKING_H
#pragma once
