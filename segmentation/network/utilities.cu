// ============================================================
//  CuVision-Engine | Segmentation Module
//  utilities.cu  — GPU utility helpers for pixel-wise tasks
// ============================================================
#ifndef SEG_UTILITIES_CU
#define SEG_UTILITIES_CU

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "cudnn_helper.h"

using namespace std;

// -------------------------------------------------------
// 1.  Device information banner
// -------------------------------------------------------
void printDeviceInformation() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cout << "[SEG] No CUDA-compatible device found." << endl;
        return;
    }
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        cout << "[SEG] Device " << i << ": " << p.name << endl;
        cout << "      Compute Capability : " << p.major << "." << p.minor << endl;
        cout << "      Global Memory      : " << p.totalGlobalMem / (1 << 20) << " MB" << endl;
    }
}

// -------------------------------------------------------
// 2.  GPU kernel timer
// -------------------------------------------------------
class GpuTimer {
    cudaEvent_t t0, t1;
public:
    GpuTimer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~GpuTimer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void start() { cudaEventRecord(t0, 0); }
    void stop()  { cudaEventRecord(t1, 0); cudaEventSynchronize(t1); }
    float elapsed_ms() {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms;
    }
};

// -------------------------------------------------------
// 3.  He (Kaiming) weight initialisation — host-side
// -------------------------------------------------------
static void initHe(void* d_ptr, size_t n, int fanIn) {
    float std = sqrtf(2.0f / fanIn);
    vector<float> h(n);
    for (size_t i = 0; i < n; ++i) {
        float u1 = max((float)rand() / RAND_MAX, 1e-7f);
        float u2 = (float)rand() / RAND_MAX;
        h[i] = sqrtf(-2.f * logf(u1)) * cosf(2.f * M_PI * u2) * std;
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// -------------------------------------------------------
// 4.  GPU kernel: fill with constant
// -------------------------------------------------------
__global__ void setConstKernel(float* arr, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

static void fillConst(void* d, int n, float v) {
    setConstKernel<<<(n + 255) / 256, 256>>>((float*)d, n, v);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------
// 5.  Momentum-SGD parameter update kernel
// -------------------------------------------------------
__global__ void momentumSGDKernel(float* w, float* dw, float* vel,
                                   float lr, float mom, float decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = dw[i] + decay * w[i];
        float v = mom * vel[i] + lr * g;
        vel[i]  = v;
        w[i]   -= v;
        dw[i]   = 0.f;
    }
}

static void updateParam(void* w, void* dw, void* vel,
                         int n, float lr, float mom, float decay) {
    momentumSGDKernel<<<(n + 255) / 256, 256>>>(
        (float*)w, (float*)dw, (float*)vel, lr, mom, decay, n);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------
// 6.  Pixel-wise cross-entropy loss kernel (for segmentation)
//     Each pixel has a one-hot ground-truth class.
//     logits: [N, C, H, W]  (NCHW layout)
//     labels: [N, H, W]     (int, class index per pixel)
//     loss_out / grad_out:  [N, C, H, W]
// -------------------------------------------------------
__global__ void pixelCrossEntropyLossKernel(const float* logits,  // [N,C,H,W]
                                              const int*   labels,  // [N,H,W]
                                              float*       loss_out,// [N,H,W]
                                              float*       grad_out,// [N,C,H,W]
                                              int N, int C, int H, int W) {
    // Each thread handles one (n,h,w) pixel
    int pw = blockIdx.x * blockDim.x + threadIdx.x;   // pixel x
    int ph = blockIdx.y * blockDim.y + threadIdx.y;   // pixel y
    int n  = blockIdx.z;

    if (pw >= W || ph >= H || n >= N) return;

    int pixelIdx = n * H * W + ph * W + pw;
    int gt       = labels[pixelIdx];

    // Compute softmax over C channels for this pixel
    float maxVal = -1e30f;
    for (int c = 0; c < C; ++c) {
        float v = logits[n * C * H * W + c * H * W + ph * W + pw];
        if (v > maxVal) maxVal = v;
    }
    float sumExp = 0.f;
    for (int c = 0; c < C; ++c) {
        sumExp += expf(logits[n * C * H * W + c * H * W + ph * W + pw] - maxVal);
    }

    // Loss = -log(softmax[gt])
    float logitGt  = logits[n * C * H * W + gt * H * W + ph * W + pw];
    float softmaxGt = expf(logitGt - maxVal) / sumExp;
    loss_out[pixelIdx] = -logf(max(softmaxGt, 1e-7f));

    // Gradient: (softmax_c - one_hot_c) / (N*H*W)
    float scale = 1.f / (N * H * W);
    for (int c = 0; c < C; ++c) {
        float sc = expf(logits[n * C * H * W + c * H * W + ph * W + pw] - maxVal) / sumExp;
        float gt_onehot = (c == gt) ? 1.f : 0.f;
        grad_out[n * C * H * W + c * H * W + ph * W + pw] = (sc - gt_onehot) * scale;
    }
}

// -------------------------------------------------------
// 7.  Dice loss kernel (complementary to CE for class imbalance)
//     Operates on the probability maps (post-softmax).
//     dice = 1 - (2 * |P ∩ G|) / (|P| + |G|)
// -------------------------------------------------------
__global__ void diceLossFwdKernel(const float* probs,  // [N,C,H,W]
                                   const int*   labels, // [N,H,W]
                                   float*       intersect, // [N,C]
                                   float*       psum,      // [N,C]
                                   float*       gsum,      // [N,C]
                                   int N, int C, int H, int W) {
    int pw = blockIdx.x * blockDim.x + threadIdx.x;
    int ph = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;   // index into N*C
    int n  = nc / C;
    int c  = nc % C;

    if (pw >= W || ph >= H || n >= N) return;
    int pixIdx = n * H * W + ph * W + pw;
    int logIdx = n * C * H * W + c * H * W + ph * W + pw;

    float p = probs[logIdx];
    float g = (labels[pixIdx] == c) ? 1.f : 0.f;

    atomicAdd(&intersect[nc], p * g);
    atomicAdd(&psum[nc],      p);
    atomicAdd(&gsum[nc],      g);
}

// -------------------------------------------------------
// 8.  Weight serialisation helpers
// -------------------------------------------------------
static void saveWeightBlob(ofstream& f, void* d_ptr, size_t n) {
    vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    f.write(reinterpret_cast<char*>(h.data()), n * sizeof(float));
}

static void loadWeightBlob(ifstream& f, void* d_ptr, size_t n) {
    vector<float> h(n);
    f.read(reinterpret_cast<char*>(h.data()), n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// -------------------------------------------------------
// 9.  mIoU metric (host, post-inference evaluation)
// -------------------------------------------------------
static float computeMeanIoU(const vector<int>& pred,
                              const vector<int>& gt,
                              int numClasses) {
    vector<float> inter(numClasses, 0.f);
    vector<float> uni  (numClasses, 0.f);
    for (size_t i = 0; i < pred.size(); ++i) {
        int p = pred[i], g = gt[i];
        if (p == g) { inter[p]++; uni[p]++; }
        else        { uni[p]++;   uni[g]++; }
    }
    float miou = 0.f; int cnt = 0;
    for (int c = 0; c < numClasses; ++c) {
        if (uni[c] > 0.f) { miou += inter[c] / uni[c]; ++cnt; }
    }
    return cnt > 0 ? miou / cnt : 0.f;
}

#endif // SEG_UTILITIES_CU
