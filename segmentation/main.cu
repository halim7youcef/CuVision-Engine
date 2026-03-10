// ============================================================
//  CuVision-Engine | Segmentation
//  main.cu  — Training loop for the Attention U-Net
//
//  Dataset:  Oxford-IIIT Pet  (seg_pets.bin from prepare_dataset.py)
//  Binary format per record:
//    [C × IMG_H × IMG_W uint8 pixels, CHW]
//    [IMG_H × IMG_W uint8 mask, class index per pixel]
// ============================================================
#include "network/network.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

// -------------------------------------------------------
//  Dataset constants (must match prepare_dataset.py)
// -------------------------------------------------------
static const int IMG_C       = 3;
static const int IMG_H       = 256;
static const int IMG_W       = 256;
static const int NUM_CLASSES = 3;    // 0=background, 1=pet, 2=boundary

// -------------------------------------------------------
//  Training hyper-parameters
// -------------------------------------------------------
static const int   BATCH_SIZE = 4;
static const float LR_INIT    = 5e-4f;
static const int   EPOCHS     = 15;

// -------------------------------------------------------
//  loadBatch — reads BATCH_SIZE (image, mask) pairs
//  Fills:
//    h_images [B, C, H, W] float32  normalised [0,1]
//    d_masks  [B, H, W]    int32    class indices  (on device)
//  Returns: actual records loaded
// -------------------------------------------------------
static int loadBatch(ifstream& file,
                      float*   h_images,
                      vector<int>& h_masks,
                      int batchSize) {
    int pixC  = IMG_C * IMG_H * IMG_W;
    int pixM  = IMG_H * IMG_W;
    int loaded = 0;

    for (int i = 0; i < batchSize; ++i) {
        // Image — uint8 CHW
        vector<uint8_t> rawImg(pixC);
        if (!file.read(reinterpret_cast<char*>(rawImg.data()), pixC)) break;
        float* dst = h_images + i * pixC;
        for (int p = 0; p < pixC; ++p)
            dst[p] = rawImg[p] / 255.0f;

        // Mask — uint8 HW → int32 HW
        vector<uint8_t> rawMask(pixM);
        if (!file.read(reinterpret_cast<char*>(rawMask.data()), pixM)) break;
        int* mDst = h_masks.data() + i * pixM;
        for (int p = 0; p < pixM; ++p)
            mDst[p] = (int)rawMask[p];

        ++loaded;
    }
    return loaded;
}

// -------------------------------------------------------
//  mIoU  — computed on host from predicted argmax mask
// -------------------------------------------------------
static float computeBatchMIoU(const float* h_logits,   // [B, C, H, W]
                                const int*   h_masks,   // [B, H, W]
                                int B) {
    int pixM   = IMG_H * IMG_W;
    int pixOut = NUM_CLASSES * pixM;

    vector<int> pred(B * pixM);
    for (int n = 0; n < B; ++n) {
        for (int p = 0; p < pixM; ++p) {
            float best = -1e30f; int bestC = 0;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float v = h_logits[n * pixOut + c * pixM + p];
                if (v > best) { best = v; bestC = c; }
            }
            pred[n * pixM + p] = bestC;
        }
    }

    // Per-class IoU then mean
    vector<float> inter(NUM_CLASSES, 0.f);
    vector<float> uni  (NUM_CLASSES, 0.f);
    for (int i = 0; i < B * pixM; ++i) {
        int p = pred[i], g = h_masks[i];
        if (p == g) { inter[p]++; uni[p]++; }
        else        { uni[p]++;   uni[g]++; }
    }
    float miou = 0.f; int cnt = 0;
    for (int c = 0; c < NUM_CLASSES; ++c)
        if (uni[c] > 0.f) { miou += inter[c] / uni[c]; ++cnt; }
    return cnt > 0 ? miou / cnt : 0.f;
}

// -------------------------------------------------------
//  main
// -------------------------------------------------------
int main() {
    const char* datasetPath = "dataset/seg_pets.bin";

    cout << "======================================================" << endl;
    cout << "  CuVision-Engine | Segmentation Training" << endl;
    cout << "  Dataset : Oxford-IIIT Pet  (3-class pixel labelling)" << endl;
    cout << "  Network : Attention U-Net  (ResNet encoder + ASPP)" << endl;
    cout << "======================================================" << endl;

    // -- Read dataset header --
    ifstream headerFile(datasetPath, ios::binary);
    if (!headerFile.is_open()) {
        cerr << "[ERROR] Cannot open " << datasetPath
             << "\n        Run dataset/prepare_dataset.py first." << endl;
        return 1;
    }
    int totalImages, fileH, fileW, fileCls;
    headerFile.read(reinterpret_cast<char*>(&totalImages), 4);
    headerFile.read(reinterpret_cast<char*>(&fileH),       4);
    headerFile.read(reinterpret_cast<char*>(&fileW),       4);
    headerFile.read(reinterpret_cast<char*>(&fileCls),     4);
    headerFile.close();

    cout << "Dataset: " << totalImages << " image-mask pairs | "
         << fileH << "×" << fileW << " | " << fileCls << " classes" << endl;

    if (fileH != IMG_H || fileW != IMG_W || fileCls != NUM_CLASSES) {
        cerr << "[ERROR] Dataset dimensions do not match compiled constants." << endl;
        return 1;
    }

    int numBatches = totalImages / BATCH_SIZE;
    cout << "Batch size: " << BATCH_SIZE
         << " | Batches/epoch: " << numBatches
         << " | Epochs: " << EPOCHS << endl;

    // -- Build network --
    printDeviceInformation();
    SegmentationNet segNet(BATCH_SIZE, IMG_C, IMG_H, IMG_W, NUM_CLASSES);

    // -- Device mask buffer (int32 [B, H, W]) --
    int* d_masks = nullptr;
    int maskSzBytes = BATCH_SIZE * IMG_H * IMG_W * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_masks, maskSzBytes));

    // -- Host buffers --
    vector<float> h_images(BATCH_SIZE * IMG_C * IMG_H * IMG_W);
    vector<int>   h_masks (BATCH_SIZE * IMG_H * IMG_W);
    vector<float> h_logits(BATCH_SIZE * NUM_CLASSES * IMG_H * IMG_W);

    float lr = LR_INIT;

    // -- Training loop --
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        ifstream file(datasetPath, ios::binary);
        if (!file.is_open()) {
            cerr << "[ERROR] Cannot open dataset file." << endl;
            return 1;
        }
        file.seekg(4 * sizeof(int));   // skip 16-byte header

        float epochMIoU     = 0.f;
        int   batchCount    = 0;
        int   processedImgs = 0;

        GpuTimer epochTimer;
        epochTimer.start();

        for (int b = 0; b < numBatches; ++b) {
            int loaded = loadBatch(file, h_images.data(), h_masks, BATCH_SIZE);
            if (loaded == 0) break;

            // Upload masks to device
            CUDA_CHECK(cudaMemcpy(d_masks, h_masks.data(),
                                   loaded * IMG_H * IMG_W * sizeof(int),
                                   cudaMemcpyHostToDevice));

            // Forward pass (with training augmentation)
            segNet.forward(h_images.data(), d_masks, /*train=*/true);

            // Retrieve logits for mIoU tracking
            CUDA_CHECK(cudaMemcpy(h_logits.data(), segNet.getLogits(),
                                   (size_t)loaded * NUM_CLASSES * IMG_H * IMG_W * sizeof(float),
                                   cudaMemcpyDeviceToHost));

            // Backward pass: CE + Dice loss → weight update
            segNet.backward(d_masks, lr);

            float batchMIoU = computeBatchMIoU(h_logits.data(), h_masks.data(), loaded);
            epochMIoU    += batchMIoU;
            processedImgs += loaded;
            ++batchCount;

            if (b % 20 == 0 || b == numBatches - 1) {
                cout << "\r  Epoch " << epoch+1 << "/" << EPOCHS
                     << "  Batch " << b+1 << "/" << numBatches
                     << "  Images: " << processedImgs
                     << "  mIoU: " << (epochMIoU / batchCount * 100.f) << "%"
                     << "     " << flush;
            }
        }
        epochTimer.stop();
        file.close();

        // Polynomial LR decay:  lr = lr_init * (1 - epoch/epochs)^0.9
        float decay = pow(1.f - (float)(epoch+1) / EPOCHS, 0.9f);
        lr = LR_INIT * decay;

        cout << "\n  Epoch " << epoch+1 << "/" << EPOCHS
             << " | mIoU: " << (epochMIoU / max(batchCount, 1) * 100.f) << "%"
             << " | LR: " << lr
             << " | Time: " << epochTimer.elapsed_ms() / 1000.f << "s" << endl;

        // Save checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            string ckpt = "seg_pets_ep" + to_string(epoch + 1) + ".bin";
            segNet.saveWeights(ckpt);
            cout << "  [Checkpoint] Saved → " << ckpt << endl;
        }
    }

    cout << "\nTraining complete." << endl;
    segNet.saveWeights("seg_pets_final.bin");

    cudaFree(d_masks);
    return 0;
}
