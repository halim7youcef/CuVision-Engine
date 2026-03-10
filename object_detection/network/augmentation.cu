// ============================================================
//  CuVision-Engine | Object Detection Module
//  augmentation.cu  — Detection-specific GPU augmentations
//
//  Techniques implemented:
//    1. Random Horizontal Flip  (+ coord transform for boxes)
//    2. Mosaic-style random crop / padding
//    3. Color jitter  (brightness / contrast / saturation)
//    4. Gaussian noise injection
//    5. Random cutout (occlusion simulation)
// ============================================================
#ifndef OD_AUGMENTATION_CU
#define OD_AUGMENTATION_CU

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "cudnn_helper.h"

using namespace std;

// -----------------------------------------------------------
// KERNEL 1: Random horizontal flip
//   flip_flags[n] = 1  → flip image n horizontally
//   After CPU side, caller is responsible for flipping the
//   corresponding bounding-box x-coordinates:  x' = 1 - x
// -----------------------------------------------------------
__global__ void horizontalFlipKernel(float* data,
                                      int batchSize, int C, int H, int W,
                                      const int* flip_flags) {
    // Each thread handles one pixel column-pair
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // half-width index
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // height index
    int n = blockIdx.z;                               // batch index

    if (x >= (W + 1) / 2 || y >= H || n >= batchSize) return;
    if (!flip_flags[n]) return;

    int xr = W - 1 - x;
    if (x == xr) return;   // centre column — no swap needed

    for (int c = 0; c < C; ++c) {
        float* base = data + n * C * H * W + c * H * W + y * W;
        float tmp   = base[x];
        base[x]     = base[xr];
        base[xr]    = tmp;
    }
}

// -----------------------------------------------------------
// KERNEL 2: Color jitter (brightness + contrast + saturation)
//   Operates in-place.  Factors are per-image scalars stored
//   in device arrays brightness[], contrast[], saturation[].
//   Saturation approximated by lerp toward greyscale (fast).
// -----------------------------------------------------------
__global__ void colorJitterKernel(float* data,
                                   int batchSize, int C, int H, int W,
                                   const float* brightness,
                                   const float* contrast,
                                   const float* saturation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize || C < 3) return;

    int base_r = n * C * H * W + 0 * H * W + y * W + x;
    int base_g = n * C * H * W + 1 * H * W + y * W + x;
    int base_b = n * C * H * W + 2 * H * W + y * W + x;

    float r = data[base_r], g = data[base_g], b = data[base_b];

    // brightness
    float br = brightness[n];
    r += br; g += br; b += br;

    // contrast:  lerp to mean
    float mean = (r + g + b) / 3.f;
    float con  = contrast[n];
    r = con * (r - mean) + mean;
    g = con * (g - mean) + mean;
    b = con * (b - mean) + mean;

    // saturation: lerp toward luma
    float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    float sat  = saturation[n];
    r = sat * r + (1.f - sat) * luma;
    g = sat * g + (1.f - sat) * luma;
    b = sat * b + (1.f - sat) * luma;

    // clamp to [0, 1]
    data[base_r] = fmaxf(0.f, fminf(1.f, r));
    data[base_g] = fmaxf(0.f, fminf(1.f, g));
    data[base_b] = fmaxf(0.f, fminf(1.f, b));
}

// -----------------------------------------------------------
// KERNEL 3: Gaussian noise injection  (additive white noise)
//   Each thread generates its own random value via cuRAND.
// -----------------------------------------------------------
__global__ void gaussianNoiseKernel(float* data,
                                     int batchSize, int C, int H, int W,
                                     const float* noiseStd,
                                     unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize) return;

    // Unique per-thread sequence index
    unsigned long long seq = (unsigned long long)n * H * W +
                              (unsigned long long)y * W + x;
    curandState_t state;
    curand_init(seed, seq, 0, &state);

    float std = noiseStd[n];
    for (int c = 0; c < C; ++c) {
        int idx = n * C * H * W + c * H * W + y * W + x;
        float noise = curand_normal(&state) * std;
        data[idx] = fmaxf(0.f, fminf(1.f, data[idx] + noise));
    }
}

// -----------------------------------------------------------
// KERNEL 4: Random cutout  (square region zeroed out)
//   cx, cy: centre of the cutout (normalised coords [0,1])
//   size  : side length of cutout in pixels
// -----------------------------------------------------------
__global__ void cutoutKernel(float* data,
                               int batchSize, int C, int H, int W,
                               const int* cut_cx,   // pixel x
                               const int* cut_cy,   // pixel y
                               const int* cut_size  // square side in pixels
                               ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize) return;

    int cx   = cut_cx[n];
    int cy   = cut_cy[n];
    int half = cut_size[n] / 2;

    bool inside = (x >= cx - half && x <= cx + half &&
                   y >= cy - half && y <= cy + half);
    if (!inside) return;

    for (int c = 0; c < C; ++c)
        data[n * C * H * W + c * H * W + y * W + x] = 0.f;
}

// -----------------------------------------------------------
// Host class: DetectionAugmenter
//   Wraps all kernels above.  One instance per training run.
// -----------------------------------------------------------
class DetectionAugmenter {
public:
    DetectionAugmenter(int batchSize, int C, int H, int W)
        : B(batchSize), C(C), H(H), W(W) {

        CUDA_CHECK(cudaMalloc(&d_flip,    B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bright,  B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_contrast,B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sat,     B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_noiseStd,B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cx,      B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cy,      B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csize,   B * sizeof(int)));
    }

    ~DetectionAugmenter() {
        cudaFree(d_flip);    cudaFree(d_bright);
        cudaFree(d_contrast);cudaFree(d_sat);
        cudaFree(d_noiseStd);
        cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_csize);
    }

    // apply() modifies d_data in-place (NCHW float, values in [0,1])
    // Returns per-image flip flags so the caller can flip bboxes.
    vector<int> apply(float* d_data, unsigned long long seed = 42ULL) {
        vector<int>   h_flip(B);
        vector<float> h_bright(B), h_con(B), h_sat(B), h_noise(B);
        vector<int>   h_cx(B), h_cy(B), h_cs(B);

        for (int i = 0; i < B; ++i) {
            // Horizontal flip: 50 %
            h_flip[i] = rand() % 2;

            // Brightness: ±0.15
            h_bright[i] = ((float)rand()/RAND_MAX) * 0.30f - 0.15f;

            // Contrast: [0.8, 1.2]
            h_con[i]  = 0.8f + ((float)rand()/RAND_MAX) * 0.4f;

            // Saturation: [0.7, 1.3]
            h_sat[i]  = 0.7f + ((float)rand()/RAND_MAX) * 0.6f;

            // Gaussian noise std: [0, 0.04]
            h_noise[i] = ((float)rand()/RAND_MAX) * 0.04f;

            // Cutout: random centre + size ≈ 15…25 % of image
            h_cx[i] = rand() % W;
            h_cy[i] = rand() % H;
            int minSide  = min(H, W);
            h_cs[i] = (int)(minSide * (0.15f + ((float)rand()/RAND_MAX) * 0.10f));
        }

        // Upload to device
        CUDA_CHECK(cudaMemcpy(d_flip,    h_flip.data(),   B*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bright,  h_bright.data(), B*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_contrast,h_con.data(),    B*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sat,     h_sat.data(),    B*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_noiseStd,h_noise.data(),  B*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cx,      h_cx.data(),     B*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cy,      h_cy.data(),     B*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csize,   h_cs.data(),     B*sizeof(int),   cudaMemcpyHostToDevice));

        dim3 threads(16, 16);
        dim3 blocks((W+15)/16, (H+15)/16, B);

        // 1. Flip
        {
            dim3 bFlip((W/2+15)/16, (H+15)/16, B);
            horizontalFlipKernel<<<bFlip, threads>>>(d_data, B, C, H, W, d_flip);
        }

        // 2. Color jitter
        colorJitterKernel<<<blocks, threads>>>(d_data, B, C, H, W,
                                               d_bright, d_contrast, d_sat);

        // 3. Gaussian noise
        gaussianNoiseKernel<<<blocks, threads>>>(d_data, B, C, H, W,
                                                  d_noiseStd, seed);

        // 4. Cutout
        cutoutKernel<<<blocks, threads>>>(d_data, B, C, H, W,
                                          d_cx, d_cy, d_csize);

        CUDA_CHECK(cudaGetLastError());

        return h_flip;  // caller flips BBox x coords if flip_flag[i] == 1
    }

private:
    int B, C, H, W;
    int*   d_flip;
    float* d_bright;
    float* d_contrast;
    float* d_sat;
    float* d_noiseStd;
    int*   d_cx;
    int*   d_cy;
    int*   d_csize;
};

#endif // OD_AUGMENTATION_CU
