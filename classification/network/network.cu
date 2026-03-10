#include "cudnn_helper.h"
#include "utilities.cu"
#include "augmentation.cu"
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

#define CUBLAS_CHECK(status)                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        cerr << "cuBLAS Error at line " << __LINE__ << ": " << status << endl; \
        exit(EXIT_FAILURE);                                                    \
    }

__global__ void applyMomentumSGD(float* weights, float* grad, float* velocity, float lr, float momentum, float decay, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float g = grad[i] + decay * weights[i];
        float v = momentum * velocity[i] + lr * g;
        velocity[i] = v;
        weights[i] -= v;
        grad[i] = 0.0f; // Reset for next iteration
    }
}

__global__ void setConst(float* arr, int size, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) arr[i] = val;
}

void fillConst(void* d_ptr, int size, float val) {
    setConst<<<(size+255)/256, 256>>>((float*)d_ptr, size, val);
    CUDA_CHECK(cudaGetLastError());
}

void updateParam(void* w, void* dw, void* v, int size, float lr, float momentum, float decay) {
    applyMomentumSGD<<<(size+255)/256, 256>>>((float*)w, (float*)dw, (float*)v, lr, momentum, decay, size);
    CUDA_CHECK(cudaGetLastError());
}

class ImageClassifier {
public:
    ImageClassifier(int batchSize, int channels, int height, int width, int numClasses = 10)
        : batchSize(batchSize), channels(channels), height(height), width(width), numClasses(numClasses) {
        CUDNN_CHECK(cudnnCreate(&cudnnHandle));
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        setupDescriptors();
        setupMemory();
    }

    ~ImageClassifier() {
        // Assume context destruction handles cleanup for brevity
        cudnnDestroy(cudnnHandle);
        cublasDestroy(cublasHandle);
        if (augmenter) delete augmenter;
    }

    void forward(float* h_input, bool isTraining = true) {
        size_t inputSize = batchSize * channels * height * width * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

        if (isTraining) {
            augmenter->apply((float*)d_input, channels, height, width);
        }

        float alpha = 1.0f, beta = 0.0f;

        // Layer 1: Conv -> BN -> ReLU -> Pool
        CUDNN_CHECK(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc, d_input, filter1Desc, d_filter1, conv1Desc, algo1, d_workspace, workspaceSize, &beta, conv1OutDesc, d_conv1Out));
        if (isTraining) {
            CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, conv1OutDesc, d_conv1Out, conv1OutDesc, d_bn1Out, bn1Desc, d_bn1Scale, d_bn1Bias, 0.1, d_bn1RunningMean, d_bn1RunningVar, 1e-5, d_bn1SaveMean, d_bn1SaveInvVar));
        } else {
            CUDNN_CHECK(cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, conv1OutDesc, d_conv1Out, conv1OutDesc, d_bn1Out, bn1Desc, d_bn1Scale, d_bn1Bias, d_bn1RunningMean, d_bn1RunningVar, 1e-5));
        }
        CUDNN_CHECK(cudnnActivationForward(cudnnHandle, actDesc, &alpha, conv1OutDesc, d_bn1Out, &beta, conv1OutDesc, d_bn1Out));
        CUDNN_CHECK(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1OutDesc, d_bn1Out, &beta, pool1OutDesc, d_pool1Out));

        // Layer 2: Conv -> BN -> ReLU -> Pool
        CUDNN_CHECK(cudnnConvolutionForward(cudnnHandle, &alpha, pool1OutDesc, d_pool1Out, filter2Desc, d_filter2, conv2Desc, algo2, d_workspace, workspaceSize, &beta, conv2OutDesc, d_conv2Out));
        if (isTraining) {
            CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, conv2OutDesc, d_conv2Out, conv2OutDesc, d_bn2Out, bn2Desc, d_bn2Scale, d_bn2Bias, 0.1, d_bn2RunningMean, d_bn2RunningVar, 1e-5, d_bn2SaveMean, d_bn2SaveInvVar));
        } else {
            CUDNN_CHECK(cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, conv2OutDesc, d_conv2Out, conv2OutDesc, d_bn2Out, bn2Desc, d_bn2Scale, d_bn2Bias, d_bn2RunningMean, d_bn2RunningVar, 1e-5));
        }
        CUDNN_CHECK(cudnnActivationForward(cudnnHandle, actDesc, &alpha, conv2OutDesc, d_bn2Out, &beta, conv2OutDesc, d_bn2Out));
        CUDNN_CHECK(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2OutDesc, d_bn2Out, &beta, pool2OutDesc, d_pool2Out));

        // Dropout
        if (isTraining) {
            CUDNN_CHECK(cudnnDropoutForward(cudnnHandle, dropDesc, pool2OutDesc, d_pool2Out, pool2OutDesc, d_pool2OutDrop, d_dropReserve, dropResSize));
        } else {
            CUDA_CHECK(cudaMemcpy(d_pool2OutDrop, d_pool2Out, batchSize * c2 * h2 * w2 * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // FC & Softmax
        int flatSize = c2 * h2 * w2;
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numClasses, batchSize, flatSize, &alpha, (float*)d_fcWeights, flatSize, (float*)d_pool2OutDrop, flatSize, &beta, (float*)d_fcOut, numClasses));
        CUDNN_CHECK(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, pool2OutDesc, d_fcOut, &beta, softmaxOutDesc, d_softmaxOut));
    }

    void backward(uint8_t* h_labels, float lr, float weightDecay = 0.0005f, float momentum = 0.9f) {
        float alpha = 1.0f, beta = 0.0f;
        int flatSize = c2 * h2 * w2;

        // 1. Loss Gradient
        vector<float> h_soft(batchSize * numClasses);
        CUDA_CHECK(cudaMemcpy(h_soft.data(), d_softmaxOut, h_soft.size() * sizeof(float), cudaMemcpyDeviceToHost));
        vector<float> h_grad(batchSize * numClasses);
        for(int i=0; i<batchSize; ++i) {
            for(int c=0; c<numClasses; ++c) h_grad[i*numClasses+c] = (h_soft[i*numClasses+c] - (h_labels[i]==c ? 1.0f : 0.0f)) / batchSize;
        }
        CUDA_CHECK(cudaMemcpy(d_diffLogits, h_grad.data(), h_grad.size() * sizeof(float), cudaMemcpyHostToDevice));

        // 2. FC Backprop
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, flatSize, numClasses, batchSize, &alpha, (float*)d_pool2OutDrop, flatSize, (float*)d_diffLogits, numClasses, &beta, (float*)dw_fcWeights, flatSize));
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, flatSize, batchSize, numClasses, &alpha, (float*)d_fcWeights, flatSize, (float*)d_diffLogits, numClasses, &beta, (float*)d_diffPool2Drop, flatSize));

        // 3. Dropout Backprop
        CUDNN_CHECK(cudnnDropoutBackward(cudnnHandle, dropDesc, pool2OutDesc, d_diffPool2Drop, pool2OutDesc, d_diffPool2, d_dropReserve, dropResSize));

        // 4. Layer 2 Backprop
        CUDNN_CHECK(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool2OutDesc, d_bn2Out, pool2OutDesc, d_diffPool2, conv2OutDesc, d_bn2Out, &beta, conv2OutDesc, d_diffBn2Out));
        CUDNN_CHECK(cudnnActivationBackward(cudnnHandle, actDesc, &alpha, conv2OutDesc, d_bn2Out, conv2OutDesc, d_diffBn2Out, conv2OutDesc, d_bn2Out, &beta, conv2OutDesc, d_diffBn2Out));
        CUDNN_CHECK(cudnnBatchNormalizationBackward(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, &alpha, &beta, conv2OutDesc, d_conv2Out, conv2OutDesc, d_diffBn2Out, bn2Desc, d_bn2Scale, dw_bn2Scale, dw_bn2Bias, 1e-5, d_bn2SaveMean, d_bn2SaveInvVar, conv2OutDesc, d_diffConv2Out));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1OutDesc, d_pool1Out, conv2OutDesc, d_diffConv2Out, conv2Desc, bwdF2Algo, d_workspace, workspaceSize, &beta, filter2Desc, dw_filter2));
        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnnHandle, &alpha, filter2Desc, d_filter2, conv2OutDesc, d_diffConv2Out, conv2Desc, bwdD2Algo, d_workspace, workspaceSize, &beta, pool1OutDesc, d_diffPool1));

        // 5. Layer 1 Backprop
        CUDNN_CHECK(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool1OutDesc, d_bn1Out, pool1OutDesc, d_diffPool1, conv1OutDesc, d_bn1Out, &beta, conv1OutDesc, d_diffBn1Out));
        CUDNN_CHECK(cudnnActivationBackward(cudnnHandle, actDesc, &alpha, conv1OutDesc, d_bn1Out, conv1OutDesc, d_diffBn1Out, conv1OutDesc, d_bn1Out, &beta, conv1OutDesc, d_diffBn1Out));
        CUDNN_CHECK(cudnnBatchNormalizationBackward(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, &alpha, &beta, conv1OutDesc, d_conv1Out, conv1OutDesc, d_diffBn1Out, bn1Desc, d_bn1Scale, dw_bn1Scale, dw_bn1Bias, 1e-5, d_bn1SaveMean, d_bn1SaveInvVar, conv1OutDesc, d_diffConv1Out));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, inputDesc, d_input, conv1OutDesc, d_diffConv1Out, conv1Desc, bwdF1Algo, d_workspace, workspaceSize, &beta, filter1Desc, dw_filter1));

        // 6. Parameter Updates (Stateful Optimizer)
        updateParam(d_fcWeights, dw_fcWeights, v_fcWeights, flatSize * numClasses, lr, momentum, weightDecay);
        updateParam(d_filter2, dw_filter2, v_filter2, 64 * 32 * 3 * 3, lr, momentum, weightDecay);
        updateParam(d_filter1, dw_filter1, v_filter1, 32 * channels * 3 * 3, lr, momentum, weightDecay);
        updateParam(d_bn2Scale, dw_bn2Scale, v_bn2Scale, 64, lr, momentum, 0.0f);
        updateParam(d_bn2Bias, dw_bn2Bias, v_bn2Bias, 64, lr, momentum, 0.0f);
        updateParam(d_bn1Scale, dw_bn1Scale, v_bn1Scale, 32, lr, momentum, 0.0f);
        updateParam(d_bn1Bias, dw_bn1Bias, v_bn1Bias, 32, lr, momentum, 0.0f);
    }

    void saveWeights(const string& fn) {
        ofstream f(fn, ios::binary);
        if(!f.is_open()) return;
        saveW(f, d_filter1, 32 * channels * 3 * 3);
        saveW(f, d_bn1Scale, 32); saveW(f, d_bn1Bias, 32);
        saveW(f, d_filter2, 64 * 32 * 3 * 3);
        saveW(f, d_bn2Scale, 64); saveW(f, d_bn2Bias, 64);
        saveW(f, d_fcWeights, (c2*h2*w2) * numClasses);
        f.close();
        cout << "Model + BN states saved to " << fn << endl;
    }

    float* getOutput() { return (float*)d_softmaxOut; }

private:
    cudnnHandle_t cudnnHandle; cublasHandle_t cublasHandle;
    int batchSize, channels, height, width, numClasses;
    int c1, h1, w1, c2, h2, w2;
    size_t workspaceSize, dropResSize;

    cudnnTensorDescriptor_t inputDesc, conv1OutDesc, pool1OutDesc, conv2OutDesc, pool2OutDesc, softmaxOutDesc;
    cudnnFilterDescriptor_t filter1Desc, filter2Desc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t actDesc;
    cudnnTensorDescriptor_t bn1Desc, bn2Desc;
    cudnnDropoutDescriptor_t dropDesc;

    cudnnConvolutionFwdAlgo_t algo1, algo2;
    cudnnConvolutionBwdFilterAlgo_t bwdF1Algo, bwdF2Algo;
    cudnnConvolutionBwdDataAlgo_t bwdD2Algo;

    void *d_input, *d_workspace;
    void *d_filter1, *v_filter1, *dw_filter1;
    void *d_filter2, *v_filter2, *dw_filter2;
    void *d_fcWeights, *v_fcWeights, *dw_fcWeights;
    
    void *d_bn1Scale, *d_bn1Bias, *d_bn1RunningMean, *d_bn1RunningVar, *d_bn1SaveMean, *d_bn1SaveInvVar;
    void *v_bn1Scale, *v_bn1Bias, *dw_bn1Scale, *dw_bn1Bias;
    void *d_bn2Scale, *d_bn2Bias, *d_bn2RunningMean, *d_bn2RunningVar, *d_bn2SaveMean, *d_bn2SaveInvVar;
    void *v_bn2Scale, *v_bn2Bias, *dw_bn2Scale, *dw_bn2Bias;
    void *d_dropStates, *d_dropReserve, *d_pool2OutDrop;

    void *d_pool1Out, *d_pool2Out, *d_fcOut, *d_softmaxOut;
    void *d_conv1Out, *d_bn1Out, *d_conv2Out, *d_bn2Out;
    void *d_diffLogits, *d_diffPool2Drop, *d_diffPool2, *d_diffBn2Out, *d_diffConv2Out, *d_diffPool1, *d_diffBn1Out, *d_diffConv1Out;

    DataAugmenter* augmenter;

    void setupDescriptors() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, channels, height, width));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter1Desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter1Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, channels, 3, 3));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv1Desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1Desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        
        int n;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1Desc, inputDesc, filter1Desc, &n, &c1, &h1, &w1));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c1, h1, w1));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn1Desc));
        CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn1Desc, conv1OutDesc, CUDNN_BATCHNORM_SPATIAL));

        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

        int pn, pc, ph, pw;
        CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(poolDesc, conv1OutDesc, &pn, &pc, &ph, &pw));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool1OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pn, pc, ph, pw));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter2Desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter2Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv2Desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2Desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2Desc, pool1OutDesc, filter2Desc, &n, &c2, &h2, &w2));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c2, h2, w2));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn2Desc));
        CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn2Desc, conv2OutDesc, CUDNN_BATCHNORM_SPATIAL));

        CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(poolDesc, conv2OutDesc, &pn, &pc, &ph, &pw));
        c2 = pc; h2 = ph; w2 = pw;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pn, pc, ph, pw));

        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropDesc));

        CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmaxOutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(softmaxOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, numClasses, 1, 1));
    }

    void setupMemory() {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc, filter1Desc, conv1Desc, conv1OutDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo1));
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, pool1OutDesc, filter2Desc, conv2Desc, conv2OutDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo2));
        
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, inputDesc, conv1OutDesc, conv1Desc, filter1Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwdF1Algo));
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, pool1OutDesc, conv2OutDesc, conv2Desc, filter2Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwdF2Algo));
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filter2Desc, conv2OutDesc, conv2Desc, pool1OutDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwdD2Algo));

        size_t ws1, ws2, ws3, ws4, ws5;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc, filter1Desc, conv1Desc, conv1OutDesc, algo1, &ws1));
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, pool1OutDesc, filter2Desc, conv2Desc, conv2OutDesc, algo2, &ws2));
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputDesc, conv1OutDesc, conv1Desc, filter1Desc, bwdF1Algo, &ws3));
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, pool1OutDesc, conv2OutDesc, conv2Desc, filter2Desc, bwdF2Algo, &ws4));
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filter2Desc, conv2OutDesc, conv2Desc, pool1OutDesc, bwdD2Algo, &ws5));
        
        workspaceSize = max({ws1, ws2, ws3, ws4, ws5});
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

        CUDA_CHECK(cudaMalloc(&d_input, batchSize * channels * height * width * sizeof(float)));
        
        // Setup Filter 1 Memory
        size_t f1s = 32 * channels * 3 * 3;
        CUDA_CHECK(cudaMalloc(&d_filter1, f1s * sizeof(float))); initHe(d_filter1, f1s, channels * 3 * 3);
        CUDA_CHECK(cudaMalloc(&dw_filter1, f1s * sizeof(float))); fillConst(dw_filter1, f1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&v_filter1, f1s * sizeof(float))); fillConst(v_filter1, f1s, 0.0f);
        
        // Setup Filter 2 Memory
        size_t f2s = 64 * 32 * 3 * 3;
        CUDA_CHECK(cudaMalloc(&d_filter2, f2s * sizeof(float))); initHe(d_filter2, f2s, 32 * 3 * 3);
        CUDA_CHECK(cudaMalloc(&dw_filter2, f2s * sizeof(float))); fillConst(dw_filter2, f2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&v_filter2, f2s * sizeof(float))); fillConst(v_filter2, f2s, 0.0f);

        // FC Memory
        int flatSize = c2 * h2 * w2;
        size_t fcs = flatSize * numClasses;
        CUDA_CHECK(cudaMalloc(&d_fcWeights, fcs * sizeof(float))); initHe(d_fcWeights, fcs, flatSize);
        CUDA_CHECK(cudaMalloc(&dw_fcWeights, fcs * sizeof(float))); fillConst(dw_fcWeights, fcs, 0.0f);
        CUDA_CHECK(cudaMalloc(&v_fcWeights, fcs * sizeof(float))); fillConst(v_fcWeights, fcs, 0.0f);

        // BN1 Memory
        int b1s = 32;
        CUDA_CHECK(cudaMalloc(&d_bn1Scale, b1s*4)); fillConst(d_bn1Scale, b1s, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_bn1Bias, b1s*4)); fillConst(d_bn1Bias, b1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn1RunningMean, b1s*4)); fillConst(d_bn1RunningMean, b1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn1RunningVar, b1s*4)); fillConst(d_bn1RunningVar, b1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn1SaveMean, b1s*4)); CUDA_CHECK(cudaMalloc(&d_bn1SaveInvVar, b1s*4));
        CUDA_CHECK(cudaMalloc(&v_bn1Scale, b1s*4)); fillConst(v_bn1Scale, b1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&v_bn1Bias, b1s*4)); fillConst(v_bn1Bias, b1s, 0.0f);
        CUDA_CHECK(cudaMalloc(&dw_bn1Scale, b1s*4)); CUDA_CHECK(cudaMalloc(&dw_bn1Bias, b1s*4));

        // BN2 Memory
        int b2s = 64;
        CUDA_CHECK(cudaMalloc(&d_bn2Scale, b2s*4)); fillConst(d_bn2Scale, b2s, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_bn2Bias, b2s*4)); fillConst(d_bn2Bias, b2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn2RunningMean, b2s*4)); fillConst(d_bn2RunningMean, b2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn2RunningVar, b2s*4)); fillConst(d_bn2RunningVar, b2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_bn2SaveMean, b2s*4)); CUDA_CHECK(cudaMalloc(&d_bn2SaveInvVar, b2s*4));
        CUDA_CHECK(cudaMalloc(&v_bn2Scale, b2s*4)); fillConst(v_bn2Scale, b2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&v_bn2Bias, b2s*4)); fillConst(v_bn2Bias, b2s, 0.0f);
        CUDA_CHECK(cudaMalloc(&dw_bn2Scale, b2s*4)); CUDA_CHECK(cudaMalloc(&dw_bn2Bias, b2s*4));

        // Dropout Memory
        size_t stateSize;
        CUDNN_CHECK(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
        CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(pool2OutDesc, &dropResSize));
        CUDA_CHECK(cudaMalloc(&d_dropStates, stateSize));
        CUDA_CHECK(cudaMalloc(&d_dropReserve, dropResSize));
        CUDNN_CHECK(cudnnSetDropoutDescriptor(dropDesc, cudnnHandle, 0.5f, d_dropStates, stateSize, 12345));

        // Layer Buffers
        CUDA_CHECK(cudaMalloc(&d_conv1Out, batchSize * 32 * h1 * w1 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bn1Out, batchSize * 32 * h1 * w1 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool1Out, batchSize * 32 * (h1/2) * (w1/2) * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_conv2Out, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bn2Out, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2Out, batchSize * 64 * (h2/2) * (w2/2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2OutDrop, batchSize * 64 * (h2/2) * (w2/2) * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_fcOut, batchSize * numClasses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_softmaxOut, batchSize * numClasses * sizeof(float)));

        // Backward Buffers
        CUDA_CHECK(cudaMalloc(&d_diffLogits, batchSize * numClasses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffPool2Drop, flatSize * batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffPool2, flatSize * batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffBn2Out, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffConv2Out, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffPool1, batchSize * 32 * (h1/2) * (w1/2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffBn1Out, batchSize * 32 * h1 * w1 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffConv1Out, batchSize * 32 * h1 * w1 * sizeof(float)));

        // Setup Data Augmenter
        augmenter = new DataAugmenter(batchSize);
    }

    void initHe(void* d_ptr, size_t size, int fanIn) {
        float stdDev = sqrt(2.0f / fanIn);
        float* h = new float[size];
        for(size_t i = 0; i < size; ++i) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float randStdNormal = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
            h[i] = randStdNormal * stdDev;
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h, size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h;
    }

    void saveW(ofstream& f, void* d_ptr, size_t size) {
        vector<float> h(size);
        CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
        f.write((char*)h.data(), size * sizeof(float));
    }
};
