// ============================================================
//  CuVision-Engine | Segmentation Module
//  cudnn_helper.h  — CUDA / cuDNN error-checking macros
// ============================================================
#ifndef SEG_CUDNN_HELPER_H
#define SEG_CUDNN_HELPER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

using namespace std;

#define CUDNN_CHECK(status)                                                         \
    if ((status) != CUDNN_STATUS_SUCCESS) {                                         \
        cerr << "cuDNN Error at line " << __LINE__                                  \
             << " [" << __FILE__ << "]: "                                           \
             << cudnnGetErrorString(status) << endl;                                \
        exit(EXIT_FAILURE);                                                         \
    }

#define CUDA_CHECK(err)                                                             \
    if ((err) != cudaSuccess) {                                                     \
        cerr << "CUDA Error at line " << __LINE__                                   \
             << " [" << __FILE__ << "]: "                                           \
             << cudaGetErrorString(err) << endl;                                    \
        exit(EXIT_FAILURE);                                                         \
    }

#define CUBLAS_CHECK(status)                                                        \
    if ((status) != CUBLAS_STATUS_SUCCESS) {                                        \
        cerr << "cuBLAS Error at line " << __LINE__                                 \
             << " [" << __FILE__ << "]: " << (status) << endl;                     \
        exit(EXIT_FAILURE);                                                         \
    }

#endif // SEG_CUDNN_HELPER_H
