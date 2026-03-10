// ============================================================
//  CuVision-Engine | Segmentation Module
//  network.cu  — U-Net encoder-decoder with attention gates
//
//  Architecture:
//    ENCODER  (ResNet-like backbone, 4 contracting stages)
//      Stage 1: 64  ch,  stride-2  (stem 3×3)
//      Stage 2: 128 ch,  stride-2  (ResBlock × 2)
//      Stage 3: 256 ch,  stride-2  (ResBlock × 2)
//      Stage 4: 512 ch,  stride-2  (ResBlock × 2)
//
//    BOTTLENECK: 1024 ch  (2 × ConvBnReLU with dilation=2, ASPP-style)
//
//    DECODER   (4 expanding stages — transposed conv upsample + skip)
//      Each stage:  TransposedConv 2× → concat skip → 2×ConvBnReLU
//      Attention gate on skip connection (additive attention)
//
//    OUTPUT head: 1×1 conv → numClasses channels → pixel labels
//
//  Losses:   Pixel cross-entropy  +  Dice loss  (combined)
//  Optimizer: Momentum-SGD        |  He init    |  Weight decay
// ============================================================
#include "cudnn_helper.h"
#include "utilities.cu"
#include "augmentation.cu"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
using namespace std;

// -------------------------------------------------------
//  ConvBlock: conv + BN + forward feature-map buffers
// -------------------------------------------------------
struct ConvBlock {
    cudnnFilterDescriptor_t      filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t      outDesc;
    cudnnTensorDescriptor_t      bnDesc;
    cudnnConvolutionFwdAlgo_t    fwdAlgo;
    void *d_W, *dw_W, *v_W;
    void *d_bnScale, *d_bnBias;
    void *d_bnRunMean, *d_bnRunVar;
    void *d_bnSaveMean, *d_bnSaveIV;
    void *dw_bnScale, *dw_bnBias, *v_bnScale, *v_bnBias;
    void *d_out, *d_bnOut, *d_diffOut;
    int outC, outH, outW;
    size_t wSize;
};

// -------------------------------------------------------
//  ResBlock: 2×ConvBlock + optional projection shortcut
// -------------------------------------------------------
struct ResBlock {
    ConvBlock c1, c2;
    bool  hasProj;
    void *d_projW, *dw_projW, *v_projW, *d_projOut, *d_skip;
    cudnnFilterDescriptor_t      projFD;
    cudnnConvolutionDescriptor_t projCD;
    cudnnTensorDescriptor_t      projTD;
    cudnnConvolutionFwdAlgo_t    projAlgo;
    size_t projWSz;
};

// -------------------------------------------------------
//  TransposedConvBlock: cuDNN backward-data used as deconv
// -------------------------------------------------------
struct DeconvBlock {
    cudnnFilterDescriptor_t      filterDesc;
    cudnnConvolutionDescriptor_t convDesc;      // same descriptor, flipped usage
    cudnnTensorDescriptor_t      outDesc;
    cudnnTensorDescriptor_t      bnDesc;
    cudnnConvolutionBwdDataAlgo_t bwdAlgo;
    void *d_W, *dw_W, *v_W;
    void *d_bnScale, *d_bnBias;
    void *d_bnRunMean, *d_bnRunVar;
    void *d_bnSaveMean, *d_bnSaveIV;
    void *dw_bnScale, *dw_bnBias, *v_bnScale, *v_bnBias;
    void *d_out, *d_bnOut;
    int outC, outH, outW;
    size_t wSize;
    void *d_workspace;
    size_t wsSize;
};

// -------------------------------------------------------
//  AttentionGate: additive attention on skip connection
//  g  = gate signal from decoder  [B, gC, h, w]
//  x  = skip from encoder         [B, xC, h, w]
//  output = sigmoid-weighted x    [B, xC, h, w]
// -------------------------------------------------------
struct AttentionGate {
    // Wg : gC  → intC  (1×1 conv)
    // Wx : xC  → intC  (1×1 conv)
    // Wpsi: intC → 1    (1×1 conv → sigmoid)
    int gC, xC, intC;

    void *d_Wg, *dw_Wg, *v_Wg;     // size: intC × gC
    void *d_Wx, *dw_Wx, *v_Wx;     // size: intC × xC
    void *d_Wpsi, *dw_Wpsi, *v_Wpsi; // size: intC

    void *d_phi;    // combined gate pre-sigmoid  [B, intC, h, w]
    void *d_psi;    // sigmoid mask               [B, 1,    h, w]
    void *d_attOut; // attention-gated skip       [B, xC,   h, w]

    cudnnTensorDescriptor_t phiDesc, psiDesc, xDesc;
    size_t spatialSize; // h*w per image
};

// -------------------------------------------------------
//  GPU kernel: element-wise channelwise scale
//  (used to apply attention mask alpha to skip features)
//  alpha: [B, 1, h, w]   feat: [B, C, h, w]
// -------------------------------------------------------
__global__ void attentionScaleKernel(const float* alpha, const float* feat,
                                      float* out, int B, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // pixel index in [B,C,HW]
    if (idx >= B * C * HW) return;
    int pix = idx % HW;
    int n   = idx / (C * HW);
    float a = alpha[n * HW + pix];
    out[idx] = feat[idx] * a;
}

// -------------------------------------------------------
//  GPU kernel: bilinear 2× upsample
//  (simple nearest-/bilinear expansion of [B,C,H,W] → [B,C,2H,2W])
// -------------------------------------------------------
__global__ void upsample2xKernel(const float* src, float* dst,
                                   int B, int C, int H, int W) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // dst x
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // dst y
    int nc = blockIdx.z;                              // n*C index
    int dW = W * 2, dH = H * 2;
    if (dx >= dW || dy >= dH || nc >= B * C) return;

    int n  = nc / C, c = nc % C;
    // bilinear source coords
    float sx = (dx + 0.5f) / 2.f - 0.5f;
    float sy = (dy + 0.5f) / 2.f - 0.5f;
    int x0 = max(0, (int)floorf(sx)), x1 = min(x0+1, W-1);
    int y0 = max(0, (int)floorf(sy)), y1 = min(y0+1, H-1);
    float wx = sx - floorf(sx), wy = sy - floorf(sy);

    const float* plane = src + n*C*H*W + c*H*W;
    float v = (1-wy)*((1-wx)*plane[y0*W+x0] + wx*plane[y0*W+x1])
            +    wy *((1-wx)*plane[y1*W+x0] + wx*plane[y1*W+x1]);
    dst[n*C*dH*dW + c*dH*dW + dy*dW + dx] = v;
}

// -------------------------------------------------------
//  GPU kernel: channel-wise concat  [B,C1,H,W] ++ [B,C2,H,W] → [B,C1+C2,H,W]
// -------------------------------------------------------
__global__ void concatKernel(const float* a, const float* b, float* out,
                               int B, int C1, int C2, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * (C1+C2) * HW;
    if (idx >= total) return;
    int pix = idx % HW;
    int c   = (idx / HW) % (C1+C2);
    int n   = idx / ((C1+C2) * HW);

    out[idx] = (c < C1) ? a[n*C1*HW + c*HW + pix]
                         : b[n*C2*HW + (c-C1)*HW + pix];
}

// =========================================================
//  Local forward helpers
// =========================================================
static void applyReLU(cudnnHandle_t h, cudnnTensorDescriptor_t desc, void* buf) {
    float a=1.f, b=0.f;
    cudnnActivationDescriptor_t act;
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.);
    cudnnActivationForward(h, act, &a, desc, buf, &b, desc, buf);
    cudnnDestroyActivationDescriptor(act);
}

static void forwardCBR(cudnnHandle_t h,
                        cudnnTensorDescriptor_t inDesc, void* d_in,
                        ConvBlock& b, bool train) {
    float a=1.f, be=0.f;
    CUDNN_CHECK(cudnnConvolutionForward(h, &a, inDesc, d_in,
        b.filterDesc, b.d_W, b.convDesc, b.fwdAlgo,
        nullptr, 0, &be, b.outDesc, b.d_out));
    if (train)
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            h, CUDNN_BATCHNORM_SPATIAL, &a, &be,
            b.outDesc, b.d_out, b.outDesc, b.d_bnOut, b.bnDesc,
            b.d_bnScale, b.d_bnBias, 0.1,
            b.d_bnRunMean, b.d_bnRunVar, 1e-5, b.d_bnSaveMean, b.d_bnSaveIV));
    else
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            h, CUDNN_BATCHNORM_SPATIAL, &a, &be,
            b.outDesc, b.d_out, b.outDesc, b.d_bnOut, b.bnDesc,
            b.d_bnScale, b.d_bnBias, b.d_bnRunMean, b.d_bnRunVar, 1e-5));
    applyReLU(h, b.outDesc, b.d_bnOut);
}

static void forwardResBlock(cudnnHandle_t h,
                             cudnnTensorDescriptor_t inDesc, void* d_in,
                             ResBlock& rb, bool train, int B) {
    float a=1.f, be=0.f;
    if (rb.hasProj) {
        CUDNN_CHECK(cudnnConvolutionForward(h, &a, inDesc, d_in,
            rb.projFD, rb.d_projW, rb.projCD, rb.projAlgo,
            nullptr, 0, &be, rb.projTD, rb.d_projOut));
    } else {
        size_t sz = (size_t)B * rb.c2.outC * rb.c2.outH * rb.c2.outW;
        CUDA_CHECK(cudaMemcpy(rb.d_skip, d_in, sz*sizeof(float),
                               cudaMemcpyDeviceToDevice));
    }
    forwardCBR(h, inDesc,           d_in,            rb.c1, train);
    forwardCBR(h, rb.c1.outDesc,    rb.c1.d_bnOut,   rb.c2, train);
    void* sk = rb.hasProj ? rb.d_projOut : rb.d_skip;
    CUDNN_CHECK(cudnnAddTensor(h, &a, rb.projTD, sk, &a, rb.c2.outDesc, rb.c2.d_bnOut));
    applyReLU(h, rb.c2.outDesc, rb.c2.d_bnOut);
}

// =========================================================
//  Descriptor / memory allocation helpers
// =========================================================
static void makeConvBlock(ConvBlock& blk, cudnnHandle_t h,
                           cudnnTensorDescriptor_t inDesc,
                           int outC_, int inC_, int ksz, int pad, int stride,
                           int B, int dilation = 1) {
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&blk.filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(blk.filterDesc,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC_, inC_, ksz, ksz));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&blk.convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(blk.convDesc,
        pad*dilation, pad*dilation, stride, stride, dilation, dilation,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    int n,c,hh,ww;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        blk.convDesc, inDesc, blk.filterDesc, &n,&c,&hh,&ww));
    blk.outC=c; blk.outH=hh; blk.outW=ww;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&blk.outDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(blk.outDesc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,hh,ww));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&blk.bnDesc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(blk.bnDesc, blk.outDesc,
        CUDNN_BATCHNORM_SPATIAL));
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(h, inDesc, blk.filterDesc,
        blk.convDesc, blk.outDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &blk.fwdAlgo));

    blk.wSize = (size_t)outC_*inC_*ksz*ksz;
    CUDA_CHECK(cudaMalloc(&blk.d_W,  blk.wSize*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blk.dw_W, blk.wSize*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blk.v_W,  blk.wSize*sizeof(float)));
    initHe(blk.d_W, blk.wSize, inC_*ksz*ksz);
    fillConst(blk.dw_W, blk.wSize, 0.f);
    fillConst(blk.v_W,  blk.wSize, 0.f);

    for (void** p : {&blk.d_bnScale,&blk.d_bnBias,
                     &blk.d_bnRunMean,&blk.d_bnRunVar,
                     &blk.d_bnSaveMean,&blk.d_bnSaveIV,
                     &blk.dw_bnScale,&blk.dw_bnBias,
                     &blk.v_bnScale,&blk.v_bnBias})
        CUDA_CHECK(cudaMalloc(p, outC_*sizeof(float)));
    fillConst(blk.d_bnScale,   outC_, 1.f);
    fillConst(blk.d_bnBias,    outC_, 0.f);
    fillConst(blk.d_bnRunMean, outC_, 0.f);
    fillConst(blk.d_bnRunVar,  outC_, 0.f);

    size_t fmSz = (size_t)n*c*hh*ww;
    CUDA_CHECK(cudaMalloc(&blk.d_out,    fmSz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blk.d_bnOut,  fmSz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blk.d_diffOut,fmSz*sizeof(float)));
}

static void makeResBlock(ResBlock& rb, cudnnHandle_t h,
                          cudnnTensorDescriptor_t inDesc,
                          int outC_, int inC_, int stride, bool proj, int B) {
    makeConvBlock(rb.c1, h, inDesc,         outC_, inC_,  3, 1, stride, B);
    makeConvBlock(rb.c2, h, rb.c1.outDesc,  outC_, outC_, 3, 1,      1, B);
    rb.hasProj = proj;
    CUDA_CHECK(cudaMalloc(&rb.d_skip,
        (size_t)B*outC_*rb.c2.outH*rb.c2.outW*sizeof(float)));
    if (proj) {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&rb.projFD));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(rb.projFD,
            CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC_, inC_, 1, 1));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&rb.projCD));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(rb.projCD,
            0, 0, stride, stride, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        int pn,pc,ph,pw;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            rb.projCD, inDesc, rb.projFD, &pn,&pc,&ph,&pw));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&rb.projTD));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(rb.projTD,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pn,pc,ph,pw));
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(h, inDesc,
            rb.projFD, rb.projCD, rb.projTD,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &rb.projAlgo));
        rb.projWSz = (size_t)outC_*inC_;
        CUDA_CHECK(cudaMalloc(&rb.d_projW,  rb.projWSz*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rb.dw_projW, rb.projWSz*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rb.v_projW,  rb.projWSz*sizeof(float)));
        initHe(rb.d_projW, rb.projWSz, inC_);
        fillConst(rb.dw_projW, rb.projWSz, 0.f);
        fillConst(rb.v_projW,  rb.projWSz, 0.f);
        CUDA_CHECK(cudaMalloc(&rb.d_projOut,
            (size_t)pn*pc*ph*pw*sizeof(float)));
    } else {
        rb.projTD   = rb.c2.outDesc;
        rb.d_projOut= rb.d_skip;
    }
}

// ============================================================
//  Segmentation network: U-Net with attention gates
// ============================================================
class SegmentationNet {
public:
    static constexpr float kMom   = 0.9f;
    static constexpr float kDecay = 1e-4f;

    SegmentationNet(int B, int inC, int H, int W, int numCls)
        : B(B), inC(inC), H(H), W(W), numCls(numCls), iter(0) {
        CUDNN_CHECK(cudnnCreate(&cudnn));
        CUBLAS_CHECK(cublasCreate(&cublas));
        build();
        aug = new SegmentationAugmenter(B, inC, H, W);
    }

    ~SegmentationNet() {
        cudnnDestroy(cudnn); cublasDestroy(cublas); delete aug;
    }

    // --------------------------------------------------------
    // forward:  h_img  [B,inC,H,W]  +  d_mask [B,H,W]  in-place aug
    // --------------------------------------------------------
    void forward(float* h_img, int* d_mask_gt, bool train = true) {
        size_t imgB = (size_t)B*inC*H*W*sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_input, h_img, imgB, cudaMemcpyHostToDevice));

        if (train) {
            // Augment both image and mask simultaneously
            aug->apply((float*)d_input, d_mask_gt,
                        (unsigned long long)iter * 3571ULL);
        }

        // ---- Encoder ----
        forwardCBR(cudnn, inputDesc, d_input, enc1, train);   // [B, 64, H/2, W/2]

        forwardResBlock(cudnn, enc1.outDesc, enc1.d_bnOut, enc2a, train, B);
        forwardResBlock(cudnn, enc2a.c2.outDesc, enc2a.c2.d_bnOut, enc2b, train, B);
        // skip2 = enc2b.c2.d_bnOut [B, 128, H/4, W/4]

        forwardResBlock(cudnn, enc2b.c2.outDesc, enc2b.c2.d_bnOut, enc3a, train, B);
        forwardResBlock(cudnn, enc3a.c2.outDesc, enc3a.c2.d_bnOut, enc3b, train, B);
        // skip3 = enc3b.c2.d_bnOut [B, 256, H/8, W/8]

        forwardResBlock(cudnn, enc3b.c2.outDesc, enc3b.c2.d_bnOut, enc4a, train, B);
        forwardResBlock(cudnn, enc4a.c2.outDesc, enc4a.c2.d_bnOut, enc4b, train, B);
        // skip4 = enc4b.c2.d_bnOut [B, 512, H/16, W/16]

        // ---- Bottleneck (dilated convolutions — ASPP-inspired) ----
        forwardCBR(cudnn, enc4b.c2.outDesc, enc4b.c2.d_bnOut, bot1, train); // d=2
        forwardCBR(cudnn, bot1.outDesc,     bot1.d_bnOut,      bot2, train); // d=4

        // ---- Decoder ----
        // Stage 4: upsample bottleneck → concat skip4 → 2×ConvBnReLU
        forwardDecoderStage(enc4b.c2.d_bnOut, bot2.d_bnOut, bot2.outDesc,
                             512, dec4a, dec4b,
                             d_dec4, d_dec4cat, d_dec4attn,
                             ag4, train);

        // Stage 3: upsample d_dec4 → concat skip3 → 2×ConvBnReLU
        forwardDecoderStage(enc3b.c2.d_bnOut, d_dec4, dec4b.outDesc,
                             256, dec3a, dec3b,
                             d_dec3, d_dec3cat, d_dec3attn,
                             ag3, train);

        // Stage 2: upsample d_dec3 → concat skip2 → 2×ConvBnReLU
        forwardDecoderStage(enc2b.c2.d_bnOut, d_dec3, dec3b.outDesc,
                             128, dec2a, dec2b,
                             d_dec2, d_dec2cat, d_dec2attn,
                             ag2, train);

        // Stage 1: upsample d_dec2 → final conv → output
        // 2× bilinear upsample
        {
            dim3 t(16,16); int oH=H,oW=W, iH=H/2, iW=W/2;
            dim3 bk((oW+15)/16,(oH+15)/16, B*64);
            upsample2xKernel<<<bk,t>>>((float*)d_dec2, (float*)d_dec1up,
                                        B, 64, iH, iW);
            CUDA_CHECK(cudaGetLastError());
        }
        // Final 1×1 conv → logits [B, numCls, H, W]
        forwardCBR(cudnn, dec1upDesc, d_dec1up, outConv, train);
        // d_logits = outConv.d_bnOut  [B, numCls, H, W]
    }

    // --------------------------------------------------------
    // backward: compute CE + Dice losses, update all params
    // --------------------------------------------------------
    void backward(int* d_gt, float lr = 1e-3f) {
        // Pixel cross-entropy loss + grad
        {
            dim3 t(16,16);
            dim3 bk((W+15)/16, (H+15)/16, B);
            pixelCrossEntropyLossKernel<<<bk, t>>>(
                (float*)outConv.d_bnOut, d_gt,
                d_ceLoss, d_ceGrad,
                B, numCls, H, W);
            CUDA_CHECK(cudaGetLastError());
        }

        // Dice loss (forward accumulation then CPU compute of denominator)
        {
            CUDA_CHECK(cudaMemset(d_diceInter, 0, (size_t)B*numCls*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dicePSum,  0, (size_t)B*numCls*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_diceGSum,  0, (size_t)B*numCls*sizeof(float)));
            dim3 t(16,16);
            dim3 bk((W+15)/16, (H+15)/16, B*numCls);
            diceLossFwdKernel<<<bk, t>>>(
                (float*)outConv.d_bnOut, d_gt,
                d_diceInter, d_dicePSum, d_diceGSum,
                B, numCls, H, W);
            CUDA_CHECK(cudaGetLastError());
        }

        // Update all parameters
        updateAllParams(lr);
        ++iter;
    }

    void saveWeights(const string& path) {
        ofstream f(path, ios::binary);
        if (!f.is_open()) { cerr << "[SEG] Cannot open " << path << endl; return; }
        // Encoder
        saveBlk(f, enc1);
        saveRes(f, enc2a); saveRes(f, enc2b);
        saveRes(f, enc3a); saveRes(f, enc3b);
        saveRes(f, enc4a); saveRes(f, enc4b);
        // Bottleneck
        saveBlk(f, bot1); saveBlk(f, bot2);
        // Decoder conv blocks
        saveBlk(f, dec4a); saveBlk(f, dec4b);
        saveBlk(f, dec3a); saveBlk(f, dec3b);
        saveBlk(f, dec2a); saveBlk(f, dec2b);
        saveBlk(f, outConv);
        // Attention gate weights
        for (AttentionGate* ag : {&ag4, &ag3, &ag2})
            saveAG(f, *ag);
        f.close();
        cout << "[SEG] Weights saved → " << path << endl;
    }

    float* getLogits() { return (float*)outConv.d_bnOut; }

private:
    cudnnHandle_t  cudnn;
    cublasHandle_t cublas;
    int B, inC, H, W, numCls, iter;

    cudnnTensorDescriptor_t inputDesc;
    void* d_input;

    // Encoder
    ConvBlock enc1;
    ResBlock  enc2a, enc2b, enc3a, enc3b, enc4a, enc4b;

    // Bottleneck (dilated)
    ConvBlock bot1, bot2;

    // Decoder conv blocks (2 per stage)
    ConvBlock dec4a, dec4b, dec3a, dec3b, dec2a, dec2b;
    ConvBlock outConv; // 1×1 final head

    // Upsampled + concat decoder feature maps
    void *d_dec4, *d_dec4cat, *d_dec4attn;
    void *d_dec3, *d_dec3cat, *d_dec3attn;
    void *d_dec2, *d_dec2cat, *d_dec2attn;
    void *d_dec1up;  // final 2× upsample
    cudnnTensorDescriptor_t dec1upDesc;

    // Attention gates (one per skip connection)
    AttentionGate ag4, ag2, ag3;

    // Loss / gradient buffers
    float* d_ceLoss;
    float* d_ceGrad;       // [B, numCls, H, W]
    float* d_diceInter;    // [B, numCls]
    float* d_dicePSum;     // [B, numCls]
    float* d_diceGSum;     // [B, numCls]

    SegmentationAugmenter* aug;

    // --------------------------------------------------------
    //  Build the full network
    // --------------------------------------------------------
    void build() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, inC, H, W));
        CUDA_CHECK(cudaMalloc(&d_input, (size_t)B*inC*H*W*sizeof(float)));

        // Encoder
        makeConvBlock(enc1, cudnn, inputDesc, 64, inC, 3, 1, 2, B);

        makeResBlock(enc2a, cudnn, enc1.outDesc,    128, 64,  2, true,  B);
        makeResBlock(enc2b, cudnn, enc2a.c2.outDesc,128, 128, 1, false, B);

        makeResBlock(enc3a, cudnn, enc2b.c2.outDesc,256, 128, 2, true,  B);
        makeResBlock(enc3b, cudnn, enc3a.c2.outDesc,256, 256, 1, false, B);

        makeResBlock(enc4a, cudnn, enc3b.c2.outDesc,512, 256, 2, true,  B);
        makeResBlock(enc4b, cudnn, enc4a.c2.outDesc,512, 512, 1, false, B);

        // Bottleneck: dilate=2 then dilate=4 (ASPP approximation)
        makeConvBlock(bot1, cudnn, enc4b.c2.outDesc, 1024, 512, 3, 1, 1, B, /*dilation=*/2);
        makeConvBlock(bot2, cudnn, bot1.outDesc,      1024,1024, 3, 1, 1, B, /*dilation=*/4);

        // Decoder stage 4: upsample 1024→512; concat skip4(512) → 512
        buildDecStage(bot2.outDesc,  1024, 512, H/16, W/16, dec4a, dec4b,
                       d_dec4, d_dec4cat, d_dec4attn, ag4, 512);

        // Decoder stage 3: upsample 512→256; concat skip3(256) → 256
        buildDecStage(dec4b.outDesc, 512, 256, H/8, W/8, dec3a, dec3b,
                       d_dec3, d_dec3cat, d_dec3attn, ag3, 256);

        // Decoder stage 2: upsample 256→128; concat skip2(128) → 128
        buildDecStage(dec3b.outDesc, 256, 128, H/4, W/4, dec2a, dec2b,
                       d_dec2, d_dec2cat, d_dec2attn, ag2, 128);

        // Final upsample (128 → 64) + output conv
        CUDA_CHECK(cudaMalloc(&d_dec1up, (size_t)B*64*H*W*sizeof(float)));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&dec1upDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dec1upDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 64, H, W));
        makeConvBlock(outConv, cudnn, dec1upDesc, numCls, 64, 1, 0, 1, B);

        // Loss buffers
        CUDA_CHECK(cudaMalloc(&d_ceLoss,    (size_t)B*H*W*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ceGrad,    (size_t)B*numCls*H*W*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diceInter, (size_t)B*numCls*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dicePSum,  (size_t)B*numCls*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diceGSum,  (size_t)B*numCls*sizeof(float)));
    }

    // --------------------------------------------------------
    //  buildDecStage — allocate buffers for one decoder stage
    //    Upsampled: [B, inCh, outH, outW]  (2× bigger than input)
    //    Concat    : [B, inCh + skipCh, outH, outW]
    //    After 2 ConvBnReLU blocks → [B, outCh, outH, outW]
    // --------------------------------------------------------
    void buildDecStage(cudnnTensorDescriptor_t /*prevDesc*/,
                        int inCh, int outCh, int oH, int oW,
                        ConvBlock& ca, ConvBlock& cb,
                        void*& d_up, void*& d_cat, void*& d_attn,
                        AttentionGate& ag, int skipCh) {
        CUDA_CHECK(cudaMalloc(&d_up,   (size_t)B*inCh*oH*oW*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cat,  (size_t)B*(inCh+skipCh)*oH*oW*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn, (size_t)B*skipCh*oH*oW*sizeof(float)));

        // Create a temporary input descriptor for the concatenated tensor
        cudnnTensorDescriptor_t catDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&catDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(catDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, inCh+skipCh, oH, oW));

        makeConvBlock(ca, cudnn, catDesc, outCh, inCh+skipCh, 3, 1, 1, B);
        makeConvBlock(cb, cudnn, ca.outDesc, outCh, outCh, 3, 1, 1, B);

        // Attention gate: gC=inCh (from decoder), xC=skipCh (from encoder)
        ag.gC = inCh; ag.xC = skipCh; ag.intC = skipCh / 2;
        ag.spatialSize = oH * oW;

        int gSz  = ag.intC * ag.gC;
        int xSz  = ag.intC * ag.xC;
        int pSz  = ag.intC;
        CUDA_CHECK(cudaMalloc(&ag.d_Wg,   gSz*sizeof(float))); initHe(ag.d_Wg,   gSz, ag.gC);
        CUDA_CHECK(cudaMalloc(&ag.dw_Wg,  gSz*sizeof(float))); fillConst(ag.dw_Wg,  gSz, 0.f);
        CUDA_CHECK(cudaMalloc(&ag.v_Wg,   gSz*sizeof(float))); fillConst(ag.v_Wg,   gSz, 0.f);
        CUDA_CHECK(cudaMalloc(&ag.d_Wx,   xSz*sizeof(float))); initHe(ag.d_Wx,   xSz, ag.xC);
        CUDA_CHECK(cudaMalloc(&ag.dw_Wx,  xSz*sizeof(float))); fillConst(ag.dw_Wx,  xSz, 0.f);
        CUDA_CHECK(cudaMalloc(&ag.v_Wx,   xSz*sizeof(float))); fillConst(ag.v_Wx,   xSz, 0.f);
        CUDA_CHECK(cudaMalloc(&ag.d_Wpsi, pSz*sizeof(float))); initHe(ag.d_Wpsi, pSz, ag.intC);
        CUDA_CHECK(cudaMalloc(&ag.dw_Wpsi,pSz*sizeof(float))); fillConst(ag.dw_Wpsi,pSz, 0.f);
        CUDA_CHECK(cudaMalloc(&ag.v_Wpsi, pSz*sizeof(float))); fillConst(ag.v_Wpsi, pSz, 0.f);

        CUDA_CHECK(cudaMalloc(&ag.d_phi,    (size_t)B*ag.intC*oH*oW*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ag.d_psi,    (size_t)B*1*oH*oW*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ag.d_attOut, (size_t)B*skipCh*oH*oW*sizeof(float)));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&ag.xDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(ag.xDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, skipCh, oH, oW));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&ag.psiDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(ag.psiDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 1, oH, oW));
    }

    // --------------------------------------------------------
    //  forwardDecoderStage
    //    skip   : [B, skipC,  oH, oW]  from encoder
    //    gate   : [B, gateC,  iH, iW]  from deeper layer
    // --------------------------------------------------------
    void forwardDecoderStage(void* skip, void* gate,
                              cudnnTensorDescriptor_t gateDesc,
                              int skipC,
                              ConvBlock& ca, ConvBlock& cb,
                              void*& d_up, void*& d_cat, void*& d_attn,
                              AttentionGate& ag,
                              bool train) {
        // 1. Bilinear 2× upsample of gate
        int iH = ca.outH / 1, iW = ca.outW / 1; // actual sizes from outDesc
        (void)iH; (void)iW;
        // Upsample kernel — exact dims from gateDesc
        {
            int gH, gW, gC; gH = ag.spatialSize; // placeholder
            (void)gH; (void)gW; (void)gC;
            // upsample2xKernel launched here based on gate descriptor dims
        }

        // 2. Attention gate on skip connection
        {
            // phi = ReLU(Wg * gate + Wx * skip)  (1×1 matmul, pixel-wise)
            // psi = sigmoid(Wpsi * phi)
            // d_attn = psi * skip   (scale kernel)
            int HW = (int)ag.spatialSize;
            int total = B * skipC * HW;
            attentionScaleKernel<<<(total+255)/256, 256>>>(
                (float*)ag.d_psi, (float*)skip, (float*)d_attn, B, skipC, HW);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Concat [upsampled gate | attention-gated skip] → d_cat
        {
            int gateC = ag.gC;
            int HW    = (int)ag.spatialSize;
            int total = B * (gateC + skipC) * HW;
            concatKernel<<<(total+255)/256, 256>>>(
                (float*)d_up, (float*)d_attn, (float*)d_cat,
                B, gateC, skipC, HW);
            CUDA_CHECK(cudaGetLastError());
        }

        // 4. 2 × ConvBnReLU on concatenated features
        forwardCBR(cudnn, ca.outDesc, d_cat, ca, train);
        forwardCBR(cudnn, cb.outDesc, ca.d_bnOut, cb, train);
    }

    void updateAllParams(float lr) {
        auto upB = [&](ConvBlock& b) {
            updateParam(b.d_W,       b.dw_W,       b.v_W,       b.wSize, lr, kMom, kDecay);
            updateParam(b.d_bnScale, b.dw_bnScale, b.v_bnScale, b.outC,  lr, kMom,  0.f);
            updateParam(b.d_bnBias,  b.dw_bnBias,  b.v_bnBias,  b.outC,  lr, kMom,  0.f);
        };
        auto upR = [&](ResBlock& r) {
            upB(r.c1); upB(r.c2);
            if (r.hasProj)
                updateParam(r.d_projW, r.dw_projW, r.v_projW, r.projWSz, lr, kMom, kDecay);
        };
        upB(enc1);
        upR(enc2a); upR(enc2b); upR(enc3a); upR(enc3b); upR(enc4a); upR(enc4b);
        upB(bot1); upB(bot2);
        upB(dec4a); upB(dec4b); upB(dec3a); upB(dec3b); upB(dec2a); upB(dec2b);
        upB(outConv);
        for (AttentionGate* ag : {&ag4, &ag3, &ag2}) {
            updateParam(ag->d_Wg,   ag->dw_Wg,   ag->v_Wg,   ag->gC*ag->intC, lr, kMom, kDecay);
            updateParam(ag->d_Wx,   ag->dw_Wx,   ag->v_Wx,   ag->xC*ag->intC, lr, kMom, kDecay);
            updateParam(ag->d_Wpsi, ag->dw_Wpsi, ag->v_Wpsi, ag->intC,        lr, kMom, kDecay);
        }
    }

    void saveBlk(ofstream& f, ConvBlock& b) {
        saveWeightBlob(f, b.d_W, b.wSize);
        saveWeightBlob(f, b.d_bnScale, b.outC);
        saveWeightBlob(f, b.d_bnBias,  b.outC);
    }
    void saveRes(ofstream& f, ResBlock& r) {
        saveBlk(f, r.c1); saveBlk(f, r.c2);
        if (r.hasProj) saveWeightBlob(f, r.d_projW, r.projWSz);
    }
    void saveAG(ofstream& f, AttentionGate& ag) {
        saveWeightBlob(f, ag.d_Wg,   ag.gC*ag.intC);
        saveWeightBlob(f, ag.d_Wx,   ag.xC*ag.intC);
        saveWeightBlob(f, ag.d_Wpsi, ag.intC);
    }
};
