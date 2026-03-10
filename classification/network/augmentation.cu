#ifndef AUGMENTATION_CU
#define AUGMENTATION_CU

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

// In-place CUDA Kernel for Random Horizontal Flipping and Brightness Adjustment
__global__ void augmentBatchKernel(float* data, int batchSize, int channels, int height, int width, 
                                          int* flip_flags, float* brightness_factors) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // width (processing half width)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    int n = blockIdx.z; // image in batch

    // Process only up to half width to swap left and right pixels
    if (x < (width + 1) / 2 && y < height && n < batchSize) {
        int do_flip = flip_flags[n];
        float bright = brightness_factors[n];
        int x_right = width - 1 - x;
        
        for (int c = 0; c < channels; ++c) {
            int idx_left = n * (channels * height * width) + c * (height * width) + y * width + x;
            int idx_right = n * (channels * height * width) + c * (height * width) + y * width + x_right;

            float val_left = data[idx_left];
            float val_right = data[idx_right];

            if (do_flip && x != x_right) {
                // Swap purely left and right coordinates
                float temp = val_left;
                val_left = val_right;
                val_right = temp;
            }

            // Apply color jittering (brightness) and clamp to [0.0, 1.0]
            val_left += bright;
            if (val_left < 0.0f) val_left = 0.0f;
            else if (val_left > 1.0f) val_left = 1.0f;
            data[idx_left] = val_left;

            if (x != x_right) { // Prevent double-brightening the center pixel if width is odd
                val_right += bright;
                if (val_right < 0.0f) val_right = 0.0f;
                else if (val_right > 1.0f) val_right = 1.0f;
                data[idx_right] = val_right;
            }
        }
    }
}

class DataAugmenter {
public:
    DataAugmenter(int batchSize) : batchSize(batchSize) {
        cudaMalloc(&d_flip_flags, batchSize * sizeof(int));
        cudaMalloc(&d_bright_factors, batchSize * sizeof(float));
    }

    ~DataAugmenter() {
        cudaFree(d_flip_flags);
        cudaFree(d_bright_factors);
    }

    void apply(float* d_data, int channels, int height, int width) {
        // Generate random flip states and brightness shifts on the CPU
        std::vector<int> h_flip(batchSize);
        std::vector<float> h_bright(batchSize);

        for (int i = 0; i < batchSize; ++i) {
            h_flip[i] = (rand() % 2 == 0) ? 1 : 0; // 50% chance to flip horizontally
            
            // Random brightness variation between -0.15 and +0.15
            float b = ((float)rand() / RAND_MAX) * 0.30f - 0.15f;
            h_bright[i] = b;
        }

        // Copy random directives to GPU memory
        cudaMemcpy(d_flip_flags, h_flip.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bright_factors, h_bright.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice);

        // Configure threads and blocks
        dim3 threads(16, 16);
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

        // Launch Kernel
        augmentBatchKernel<<<blocks, threads>>>(d_data, batchSize, channels, height, width, 
                                                d_flip_flags, d_bright_factors);
        // Do not require heavy synchronization here because the CPU will sync automatically when
        // cuDNN convolutions are fired in the default stream.
    }

private:
    int batchSize;
    int* d_flip_flags;
    float* d_bright_factors;
};

#endif
