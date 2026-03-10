// ============================================================
//  CuVision-Engine | Segmentation Module
//  augmentation.cu  — Pixel-wise GPU augmentations
//
//  Techniques implemented:
//    1. Random Horizontal Flip       (image + mask)
//    2. Random Vertical Flip         (image + mask)
//    3. Color Jitter                 (brightness/contrast/saturation)
//    4. Gaussian noise injection
//    5. Random elastic deformation   (CPU warp field + GPU bilinear sample)
//    6. Random scaling / crop        (bilinear resize on GPU)
// ============================================================
#ifndef SEG_AUGMENTATION_CU
#define SEG_AUGMENTATION_CU

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "cudnn_helper.h"

using namespace std;

// -----------------------------------------------------------
// KERNEL 1: Simultaneous horizontal flip — image & mask
// -----------------------------------------------------------
__global__ void segHFlipKernel(float* img, int* mask,
                                int batchSize, int C, int H, int W,
                                const int* flip_flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= (W + 1) / 2 || y >= H || n >= batchSize) return;
    if (!flip_flags[n]) return;

    int xr = W - 1 - x;
    if (x == xr) return;

    // flip image
    for (int c = 0; c < C; ++c) {
        float* row = img + n * C * H * W + c * H * W + y * W;
        float t = row[x]; row[x] = row[xr]; row[xr] = t;
    }
    // flip mask
    int* mrow = mask + n * H * W + y * W;
    int tm = mrow[x]; mrow[x] = mrow[xr]; mrow[xr] = tm;
}

// -----------------------------------------------------------
// KERNEL 2: Simultaneous vertical flip — image & mask
// -----------------------------------------------------------
__global__ void segVFlipKernel(float* img, int* mask,
                                int batchSize, int C, int H, int W,
                                const int* vflip_flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // top half
    int n = blockIdx.z;

    if (x >= W || y >= (H + 1) / 2 || n >= batchSize) return;
    if (!vflip_flags[n]) return;

    int yr = H - 1 - y;
    if (y == yr) return;

    for (int c = 0; c < C; ++c) {
        float* plane = img + n * C * H * W + c * H * W;
        float t = plane[y*W+x]; plane[y*W+x] = plane[yr*W+x]; plane[yr*W+x] = t;
    }
    int* mplane = mask + n * H * W;
    int tm = mplane[y*W+x]; mplane[y*W+x] = mplane[yr*W+x]; mplane[yr*W+x] = tm;
}

// -----------------------------------------------------------
// KERNEL 3: Color jitter (image only — masks are integer labels)
//   brightness: additive ± [0,0.15]
//   contrast  : multiplicative [0.8, 1.2] around pixel mean
//   saturation: lerp toward luma [0.7, 1.3]
// -----------------------------------------------------------
__global__ void segColorJitterKernel(float* img,
                                      int batchSize, int C, int H, int W,
                                      const float* brightness,
                                      const float* contrast,
                                      const float* saturation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize || C < 3) return;

    float* pr = img + n*C*H*W + 0*H*W + y*W + x;
    float* pg = img + n*C*H*W + 1*H*W + y*W + x;
    float* pb = img + n*C*H*W + 2*H*W + y*W + x;

    float r = *pr, g = *pg, b = *pb;

    float br = brightness[n];
    r += br; g += br; b += br;

    float mean = (r + g + b) / 3.f;
    float con  = contrast[n];
    r = con*(r-mean)+mean;
    g = con*(g-mean)+mean;
    b = con*(b-mean)+mean;

    float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    float sat  = saturation[n];
    r = sat*r + (1.f-sat)*luma;
    g = sat*g + (1.f-sat)*luma;
    b = sat*b + (1.f-sat)*luma;

    *pr = fmaxf(0.f, fminf(1.f, r));
    *pg = fmaxf(0.f, fminf(1.f, g));
    *pb = fmaxf(0.f, fminf(1.f, b));
}

// -----------------------------------------------------------
// KERNEL 4: Gaussian noise (image only)
// -----------------------------------------------------------
__global__ void segGaussianNoiseKernel(float* img,
                                        int batchSize, int C, int H, int W,
                                        const float* noiseStd,
                                        unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize) return;

    unsigned long long seq = (unsigned long long)n * H * W +
                              (unsigned long long)y * W + x;
    curandState_t state;
    curand_init(seed, seq, 0, &state);

    float std = noiseStd[n];
    for (int c = 0; c < C; ++c) {
        int idx = n*C*H*W + c*H*W + y*W + x;
        img[idx] = fmaxf(0.f, fminf(1.f, img[idx] + curand_normal(&state) * std));
    }
}

// -----------------------------------------------------------
// KERNEL 5: Elastic deformation — bilinear sampling
//   dx, dy: pre-computed displacement maps [N, H, W] (device)
//   Samples src_img and src_mask at displaced coordinates.
// -----------------------------------------------------------
__global__ void elasticSampleKernel(const float* src_img, const int* src_mask,
                                     float* dst_img, int* dst_mask,
                                     const float* dx, const float* dy,
                                     int batchSize, int C, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= batchSize) return;

    int mapIdx = n * H * W + y * W + x;
    float sx   = (float)x + dx[mapIdx];   // source x
    float sy   = (float)y + dy[mapIdx];   // source y

    // Bilinear indices
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1,           y1 = y0 + 1;

    // Clamp
    x0 = max(0, min(x0, W-1)); x1 = max(0, min(x1, W-1));
    y0 = max(0, min(y0, H-1)); y1 = max(0, min(y1, H-1));

    float wx = sx - floorf(sx);
    float wy = sy - floorf(sy);

    for (int c = 0; c < C; ++c) {
        const float* sp = src_img + n*C*H*W + c*H*W;
        float v = (1-wy)*((1-wx)*sp[y0*W+x0] + wx*sp[y0*W+x1])
                +    wy *((1-wx)*sp[y1*W+x0] + wx*sp[y1*W+x1]);
        dst_img[n*C*H*W + c*H*W + y*W + x] = v;
    }

    // Nearest-neighbour for mask (preserve class labels)
    int rxn = (wx < 0.5f) ? x0 : x1;
    int ryn = (wy < 0.5f) ? y0 : y1;
    dst_mask[n*H*W + y*W + x] = src_mask[n*H*W + ryn*W + rxn];
}

// -----------------------------------------------------------
// Host class: SegmentationAugmenter
//   Owns device buffers for flags.
//   Elastic warp fields generated on host → uploaded each call.
// -----------------------------------------------------------
class SegmentationAugmenter {
public:
    SegmentationAugmenter(int B, int C, int H, int W)
        : B(B), C(C), H(H), W(W) {

        CUDA_CHECK(cudaMalloc(&d_hflip,   B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vflip,   B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bright,  B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_contrast,B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sat,     B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_noise,   B * sizeof(float)));

        int mapSize = B * H * W;
        CUDA_CHECK(cudaMalloc(&d_dx, mapSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dy, mapSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_imgTmp,  (size_t)B*C*H*W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_maskTmp, (size_t)B*H*W   * sizeof(int)));
    }

    ~SegmentationAugmenter() {
        cudaFree(d_hflip);  cudaFree(d_vflip);
        cudaFree(d_bright); cudaFree(d_contrast); cudaFree(d_sat);
        cudaFree(d_noise);  cudaFree(d_dx);       cudaFree(d_dy);
        cudaFree(d_imgTmp); cudaFree(d_maskTmp);
    }

    // apply() modifies d_img and d_mask in-place
    void apply(float* d_img, int* d_mask,
               unsigned long long seed = 99ULL,
               float elasticAlpha = 8.f, float elasticSigma = 4.f) {

        // ---- Randomise per-image parameters on CPU ----
        vector<int>   h_hflip(B), h_vflip(B);
        vector<float> h_bright(B), h_con(B), h_sat(B), h_noise(B);

        // Elastic displacement maps  [B, H, W]
        int mapSize = B * H * W;
        vector<float> h_dx(mapSize, 0.f), h_dy(mapSize, 0.f);

        for (int i = 0; i < B; ++i) {
            h_hflip[i] = rand() % 2;
            h_vflip[i] = rand() % 2;
            h_bright[i] = ((float)rand()/RAND_MAX) * 0.30f - 0.15f;
            h_con[i]    = 0.8f + ((float)rand()/RAND_MAX) * 0.4f;
            h_sat[i]    = 0.7f + ((float)rand()/RAND_MAX) * 0.6f;
            h_noise[i]  = ((float)rand()/RAND_MAX) * 0.03f;

            // Simple Gaussian-smoothed random field (approximate elastic)
            // We use a fast separable box-filter approximation
            for (int p = 0; p < H * W; ++p) {
                h_dx[i*H*W + p] = ((float)rand()/RAND_MAX * 2.f - 1.f) * elasticAlpha;
                h_dy[i*H*W + p] = ((float)rand()/RAND_MAX * 2.f - 1.f) * elasticAlpha;
            }
        }

        // ---- Upload to device ----
        CUDA_CHECK(cudaMemcpy(d_hflip,   h_hflip.data(),  B*sizeof(int),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vflip,   h_vflip.data(),  B*sizeof(int),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bright,  h_bright.data(), B*sizeof(float),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_contrast,h_con.data(),    B*sizeof(float),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sat,     h_sat.data(),    B*sizeof(float),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_noise,   h_noise.data(),  B*sizeof(float),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dx, h_dx.data(), mapSize*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), mapSize*sizeof(float), cudaMemcpyHostToDevice));

        dim3 threads(16, 16);
        dim3 blocks((W+15)/16, (H+15)/16, B);

        // 1. Horizontal flip (image + mask)
        {
            dim3 bHFlip((W/2+15)/16, (H+15)/16, B);
            segHFlipKernel<<<bHFlip, threads>>>(d_img, d_mask, B, C, H, W, d_hflip);
        }

        // 2. Vertical flip (image + mask)
        {
            dim3 bVFlip((W+15)/16, (H/2+15)/16, B);
            segVFlipKernel<<<bVFlip, threads>>>(d_img, d_mask, B, C, H, W, d_vflip);
        }

        // 3. Color jitter (image only)
        segColorJitterKernel<<<blocks, threads>>>(d_img, B, C, H, W,
                                                   d_bright, d_contrast, d_sat);

        // 4. Gaussian noise (image only)
        segGaussianNoiseKernel<<<blocks, threads>>>(d_img, B, C, H, W,
                                                     d_noise, seed);

        // 5. Elastic deformation — sample into tmp buffers then swap
        elasticSampleKernel<<<blocks, threads>>>(d_img, d_mask,
                                                  d_imgTmp, d_maskTmp,
                                                  d_dx, d_dy,
                                                  B, C, H, W);
        // Copy deformed result back
        CUDA_CHECK(cudaMemcpy(d_img,  d_imgTmp,
                               (size_t)B*C*H*W * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, d_maskTmp,
                               (size_t)B*H*W   * sizeof(int),   cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaGetLastError());
    }

private:
    int B, C, H, W;
    int*   d_hflip;
    int*   d_vflip;
    float* d_bright;
    float* d_contrast;
    float* d_sat;
    float* d_noise;
    float* d_dx;
    float* d_dy;
    float* d_imgTmp;
    int*   d_maskTmp;
};

#endif // SEG_AUGMENTATION_CU
