// ============================================================
//  CuVision-Engine | Object Detection Module
//  utilities.cu  — GPU utility helpers and anchor generation
// ============================================================
#ifndef OD_UTILITIES_CU
#define OD_UTILITIES_CU

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
        cout << "[OD] No CUDA-compatible device found." << endl;
        return;
    }
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        cout << "[OD] Device " << i << ": " << p.name << endl;
        cout << "     Compute Capability : " << p.major << "." << p.minor << endl;
        cout << "     Global Memory      : " << p.totalGlobalMem / (1 << 20) << " MB" << endl;
        cout << "     Multiprocessors    : " << p.multiProcessorCount << endl;
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
// 3.  He (Kaiming) weight initialisation  — host-side
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
// 4.  GPU kernel: fill array with constant
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
        float g   = dw[i] + decay * w[i];
        float v   = mom * vel[i] + lr * g;
        vel[i]    = v;
        w[i]     -= v;
        dw[i]     = 0.f;          // zero gradient in-place
    }
}

static void updateParam(void* w, void* dw, void* vel,
                         int n, float lr, float mom, float decay) {
    momentumSGDKernel<<<(n + 255) / 256, 256>>>(
        (float*)w, (float*)dw, (float*)vel, lr, mom, decay, n);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------
// 6.  Smooth-L1 loss kernel  (bounding-box regression)
//     delta_i = | pred_i - gt_i |
//     L = 0.5*delta^2   if |delta| < 1
//         |delta| - 0.5  otherwise
// -------------------------------------------------------
__global__ void smoothL1LossKernel(const float* pred, const float* target,
                                    float* loss_out, float* grad_out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = pred[i] - target[i];
        float abs_d = fabsf(d);
        if (abs_d < 1.f) {
            loss_out[i] = 0.5f * d * d;
            grad_out[i] = d;
        } else {
            loss_out[i] = abs_d - 0.5f;
            grad_out[i] = (d > 0.f) ? 1.f : -1.f;
        }
    }
}

// -------------------------------------------------------
// 7.  Sigmoid activation + focal loss for class head
//     FL(p) = -alpha * (1-p)^gamma * log(p)
//     used by modern one-stage detectors (RetinaNet / FCOS)
// -------------------------------------------------------
__global__ void focalLossKernel(const float* logits, const int* labels,
                                  float* loss_out, float* grad_out,
                                  int n, int numClasses,
                                  float alpha, float gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cls = labels[i];
        float x   = logits[i];
        float p   = 1.f / (1.f + expf(-x));         // sigmoid
        float pt  = (cls == 1) ? p : (1.f - p);
        float at  = (cls == 1) ? alpha : (1.f - alpha);
        float pw  = powf(1.f - pt, gamma);

        loss_out[i] = -at * pw * logf(max(pt, 1e-7f));

        // gradient w.r.t. logit
        float g = -at * pw * (gamma * pt * logf(max(pt, 1e-7f)) + pt - cls);
        grad_out[i] = g;
    }
}

// -------------------------------------------------------
// 8.  NMS — host-side (post-processing, not on critical path)
//     Uses IoU threshold; returns indices of kept boxes.
// -------------------------------------------------------
struct BBox { float x1, y1, x2, y2, score; int cls; };

static float iou(const BBox& a, const BBox& b) {
    float ix1 = max(a.x1, b.x1), iy1 = max(a.y1, b.y1);
    float ix2 = min(a.x2, b.x2), iy2 = min(a.y2, b.y2);
    float inter = max(0.f, ix2 - ix1) * max(0.f, iy2 - iy1);
    float ua  = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return (ua > 0.f) ? inter / ua : 0.f;
}

static vector<int> nonMaxSuppression(vector<BBox>& boxes, float iouThresh) {
    // sort descending by score
    vector<int> idx(boxes.size());
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    sort(idx.begin(), idx.end(),
         [&](int a, int b){ return boxes[a].score > boxes[b].score; });

    vector<bool> suppressed(boxes.size(), false);
    vector<int>  keep;
    for (int i : idx) {
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (int j : idx) {
            if (!suppressed[j] && j != i && iou(boxes[i], boxes[j]) > iouThresh)
                suppressed[j] = true;
        }
    }
    return keep;
}

// -------------------------------------------------------
// 9.  Default anchor generator  (SSD / RetinaNet style)
//     Returns host vector of (cx, cy, w, h) per feature map cell.
// -------------------------------------------------------
struct Anchor { float cx, cy, w, h; };

static vector<Anchor> generateAnchors(int fmapH, int fmapW,
                                       int strideH, int strideW,
                                       const vector<float>& baseScales,
                                       const vector<float>& aspectRatios) {
    vector<Anchor> anchors;
    for (int y = 0; y < fmapH; ++y) {
        for (int x = 0; x < fmapW; ++x) {
            float cx = (x + 0.5f) * strideW;
            float cy = (y + 0.5f) * strideH;
            for (float s : baseScales) {
                for (float ar : aspectRatios) {
                    float w = s * sqrtf(ar);
                    float h = s / sqrtf(ar);
                    anchors.push_back({cx, cy, w, h});
                }
            }
        }
    }
    return anchors;
}

// -------------------------------------------------------
// 10. Weight serialisation helpers
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

#endif // OD_UTILITIES_CU
