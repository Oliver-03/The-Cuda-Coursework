#ifndef BIRD_UPDATE_H
#define BIRD_UPDATE_H

#include "bird.h"
#include "flocking.h"  // For flocking behaviors (separation, alignment, cohesion)

// Update a single bird's position based on flocking behaviors
__device__ void updateBird(Bird* birds, int birdIndex, int numBirds, float deltaTime, int screenWidth, int screenHeight) {
    // Flocking behavior parameters (normalized distances)
    float separationDistance = 50.0f / screenWidth;  // Normalize based on screen width
    float alignmentDistance = 100.0f / screenWidth;  // Normalize based on screen width
    float cohesionDistance = 150.0f / screenWidth;   // Normalize based on screen width

    // Maximum speed and acceleration
    float maxSpeed = 0.005f;  // Normalized speed (relative to screen size)
    float maxAcceleration = 0.0005f;  // Normalized acceleration (relative to screen size)

    // Compute the flocking forces
    float3 sep = separation(birds, birdIndex, numBirds, separationDistance);
    float3 align = alignment(birds, birdIndex, numBirds, alignmentDistance);
    float3 coh = cohesion(birds, birdIndex, numBirds, cohesionDistance);

    // Combine forces to compute the desired acceleration
    float3 desiredAcceleration = sep + align + coh;

    // Limit the magnitude of the acceleration
    float accelerationMagnitude = length(desiredAcceleration);
    if (accelerationMagnitude > maxAcceleration) {
        desiredAcceleration = normalize(desiredAcceleration) * maxAcceleration;
    }

    // Update velocity using the smoothed acceleration
    birds[birdIndex].velocity += desiredAcceleration * deltaTime;

    // Limit the magnitude of the velocity
    float velocityMagnitude = length(birds[birdIndex].velocity);
    if (velocityMagnitude > maxSpeed) {
        birds[birdIndex].velocity = normalize(birds[birdIndex].velocity) * maxSpeed;
    }

    // Update position
    birds[birdIndex].position.x += birds[birdIndex].velocity.x * deltaTime;
    birds[birdIndex].position.y += birds[birdIndex].velocity.y * deltaTime;

    // Apply boundary conditions (wrap around the screen)
    if (birds[birdIndex].position.x < 0.0f) birds[birdIndex].position.x += 1.0f;
    if (birds[birdIndex].position.x > 1.0f) birds[birdIndex].position.x -= 1.0f;
    if (birds[birdIndex].position.y < 0.0f) birds[birdIndex].position.y += 1.0f;
    if (birds[birdIndex].position.y > 1.0f) birds[birdIndex].position.y -= 1.0f;
}

#endif // BIRD_UPDATE_H
