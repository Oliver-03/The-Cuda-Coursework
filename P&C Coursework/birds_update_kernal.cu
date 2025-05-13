// bird_update_kernels.cu
#include "bird_update.h"  // Include the header where updateBird is defined

// CUDA kernel to update the positions of all birds
__global__ void updateBirdsKernel(Bird* birds, int numBirds, float deltaTime) {
    int birdIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (birdIndex < numBirds) {
        updateBird(birds, birdIndex, numBirds, deltaTime);  // Update each bird's state
    }
}
