#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// GPU Utility Functions
void printDeviceInformation() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        cout << "No CUDA compatible devices found." << endl;
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    }
}

// Timer for benchmarking CUDA kernels
class GpuTimer {
    cudaEvent_t startEvent, stopEvent;
public:
    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }
    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    void start() {
        cudaEventRecord(startEvent, 0);
    }
    void stop() {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }
    float elapsed_ms() {
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }
};
