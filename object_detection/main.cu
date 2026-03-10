// ============================================================
//  CuVision-Engine | Object Detection
//  main.cu  — Training loop for the RetinaNet-FPN detector
//
//  Dataset:  Pascal VOC 2007  (od_voc2007.bin from prepare_dataset.py)
//  Binary format per record:
//    [num_boxes (int32)]
//    [num_boxes × 5 floats: cls x1 y1 x2 y2 (normalised)]
//    [C × IMG_H × IMG_W uint8 pixels, CHW]
// ============================================================
#include "network/network.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

// -------------------------------------------------------
//  Dataset constants (must match prepare_dataset.py)
// -------------------------------------------------------
static const int IMG_C       = 3;
static const int IMG_H       = 300;
static const int IMG_W       = 300;
static const int NUM_CLASSES = 20;   // Pascal VOC 20 classes
static const int MAX_BOXES   = 30;   // hard cap per image (from prepare_dataset.py)

// -------------------------------------------------------
//  Anchor matching parameters
// -------------------------------------------------------
static const int   BATCH_SIZE    = 4;
static const float LR_INIT       = 1e-3f;
static const int   EPOCHS        = 10;
static const float IOU_THRESH_POS = 0.5f;   // anchor is positive if IoU ≥ 0.5
static const float IOU_THRESH_NEG = 0.4f;   // anchor is negative if IoU < 0.4

// -------------------------------------------------------
//  Binary record
// -------------------------------------------------------
struct ImageRecord {
    int   numBoxes;
    float boxes[MAX_BOXES][5];   // [cls, x1, y1, x2, y2]
    vector<float>    pixels;     // [C, H, W] normalised float
};

// -------------------------------------------------------
//  loadBatch — read BATCH_SIZE records from the binary file
//  Fills:
//    h_images : [B, C, H, W] float32  (normalised)
//    records  : [B] parsed records
//  Returns: actual records read (< BATCH_SIZE at epoch boundary)
// -------------------------------------------------------
static int loadBatch(ifstream& file,
                      float*   h_images,
                      vector<ImageRecord>& records,
                      int batchSize) {
    int pixelCount = IMG_C * IMG_H * IMG_W;
    int loaded = 0;
    records.resize(batchSize);

    for (int i = 0; i < batchSize; ++i) {
        ImageRecord& rec = records[i];

        // read num_boxes
        if (!file.read(reinterpret_cast<char*>(&rec.numBoxes), sizeof(int)))
            break;
        rec.numBoxes = min(rec.numBoxes, MAX_BOXES);

        // read box data (always MAX_BOXES slots written — pad reads are harmless)
        for (int b = 0; b < rec.numBoxes; ++b)
            file.read(reinterpret_cast<char*>(rec.boxes[b]), 5 * sizeof(float));

        // skip any extra box slots if writer stored more than MAX_BOXES
        // (not needed here — prepare_dataset.py already caps at MAX_BOXES)

        // read pixel bytes
        vector<uint8_t> raw(pixelCount);
        file.read(reinterpret_cast<char*>(raw.data()), pixelCount);
        if (file.gcount() != pixelCount) break;

        float* dst = h_images + i * pixelCount;
        for (int p = 0; p < pixelCount; ++p)
            dst[p] = raw[p] / 255.0f;

        ++loaded;
    }
    return loaded;
}

// -------------------------------------------------------
//  Anchor-based target generation (host-side, for Smooth-L1)
//
//  Given a list of anchors [totalAnchors × 4 (cx,cy,w,h)]
//  and ground-truth boxes for the batch, build:
//    clsTargets[B × totalAnchors]  : -1=ignore, 0=bg, 1..K=class
//    regTargets[B × totalAnchors × 4] : (Δcx, Δcy, Δw, Δh)
//
//  We use a simplified greedy IoU match  (production would
//  use CUDA but here the anchor count is moderate).
// -------------------------------------------------------
static float boxIoU(float acx, float acy, float aw, float ah,
                     float bx1, float by1, float bx2, float by2) {
    float ax1 = acx - aw*0.5f, ay1 = acy - ah*0.5f;
    float ax2 = acx + aw*0.5f, ay2 = acy + ah*0.5f;
    float ix1 = max(ax1, bx1), iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2), iy2 = min(ay2, by2);
    float inter = max(0.f, ix2-ix1) * max(0.f, iy2-iy1);
    float ua    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
    return (ua > 0.f) ? inter / ua : 0.f;
}

static void buildTargets(const vector<Anchor>& anchors,
                           const vector<ImageRecord>& records,
                           int batchSize,
                           vector<int>& clsTargets,
                           vector<float>& regTargets) {
    int A = (int)anchors.size();
    clsTargets.assign(batchSize * A, -1);   // -1 = ignore
    regTargets.assign(batchSize * A * 4, 0.f);

    for (int n = 0; n < batchSize; ++n) {
        const ImageRecord& rec = records[n];
        for (int a = 0; a < A; ++a) {
            float acx = anchors[a].cx, acy = anchors[a].cy;
            float aw  = anchors[a].w,  ah  = anchors[a].h;

            float bestIoU = -1.f; int bestBoxIdx = -1;
            for (int b = 0; b < rec.numBoxes; ++b) {
                // GT stored as normalised [x1,y1,x2,y2]
                float gx1 = rec.boxes[b][1] * IMG_W;
                float gy1 = rec.boxes[b][2] * IMG_H;
                float gx2 = rec.boxes[b][3] * IMG_W;
                float gy2 = rec.boxes[b][4] * IMG_H;
                float iou = boxIoU(acx, acy, aw, ah, gx1, gy1, gx2, gy2);
                if (iou > bestIoU) { bestIoU = iou; bestBoxIdx = b; }
            }

            int& cls = clsTargets[n * A + a];
            if (bestIoU >= IOU_THRESH_POS) {
                cls = (int)rec.boxes[bestBoxIdx][0];   // class index
                // Regression delta encoding (RCNN-style)
                float gx1 = rec.boxes[bestBoxIdx][1] * IMG_W;
                float gy1 = rec.boxes[bestBoxIdx][2] * IMG_H;
                float gx2 = rec.boxes[bestBoxIdx][3] * IMG_W;
                float gy2 = rec.boxes[bestBoxIdx][4] * IMG_H;
                float gcx = (gx1+gx2)*0.5f, gcy = (gy1+gy2)*0.5f;
                float gw  = gx2-gx1,        gh  = gy2-gy1;

                float* r = &regTargets[(n*A + a)*4];
                r[0] = (gcx - acx) / aw;
                r[1] = (gcy - acy) / ah;
                r[2] = log(max(gw, 1.f)  / aw);
                r[3] = log(max(gh, 1.f)  / ah);
            } else if (bestIoU < IOU_THRESH_NEG) {
                cls = 0;   // background
            }
            // (-1) ignore zone: IOU_THRESH_NEG ≤ iou < IOU_THRESH_POS
        }
    }
}

// -------------------------------------------------------
//  main
// -------------------------------------------------------
int main() {
    const char* datasetPath = "dataset/od_voc2007.bin";
    cout << "======================================================" << endl;
    cout << "  CuVision-Engine | Object Detection Training" << endl;
    cout << "  Dataset : Pascal VOC 2007" << endl;
    cout << "  Network : RetinaNet-FPN (ResNet backbone)" << endl;
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

    cout << "Dataset: " << totalImages << " images | "
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
    ObjectDetector detector(BATCH_SIZE, IMG_C, IMG_H, IMG_W, NUM_CLASSES);

    // -- Pre-generate anchors (same logic as network's buildFPN) --
    vector<Anchor> anchors;
    struct LevelCfg { int h, w, sh, sw; };
    vector<LevelCfg> levels = {
        {IMG_H/4,  IMG_W/4,  4,  4},   // P2
        {IMG_H/8,  IMG_W/8,  8,  8},   // P3
        {IMG_H/16, IMG_W/16, 16, 16},  // P4
    };
    vector<float> scales = {32.f, 64.f, 128.f};
    vector<float> ratios = {0.5f,  1.f,   2.0f};
    for (auto& lv : levels) {
        auto lvAnchors = generateAnchors(lv.h, lv.w, lv.sh, lv.sw, scales, ratios);
        anchors.insert(anchors.end(), lvAnchors.begin(), lvAnchors.end());
    }
    cout << "Total anchors: " << anchors.size() << endl;

    // -- Training buffers --
    vector<float>        h_images(BATCH_SIZE * IMG_C * IMG_H * IMG_W);
    vector<ImageRecord>  records(BATCH_SIZE);
    vector<int>          h_clsTgt;
    vector<float>        h_regTgt;

    float lr = LR_INIT;

    // -- Training loop --
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        ifstream file(datasetPath, ios::binary);
        if (!file.is_open()) {
            cerr << "[ERROR] Cannot open dataset file." << endl;
            return 1;
        }
        // skip 16-byte header on each epoch
        file.seekg(4 * sizeof(int));

        float epochClsLoss = 0.f;
        float epochRegLoss = 0.f;
        int   processedImages = 0;

        GpuTimer epochTimer;
        epochTimer.start();

        for (int b = 0; b < numBatches; ++b) {
            int loaded = loadBatch(file, h_images.data(), records, BATCH_SIZE);
            if (loaded == 0) break;

            // 1. Build regression / classification targets from GT boxes
            buildTargets(anchors, records, loaded, h_clsTgt, h_regTgt);

            // 2. Forward pass (with training augmentation)
            detector.forward(h_images.data(), /*train=*/true);

            // 3. Backward pass: compute losses + update weights
            detector.backward(h_clsTgt.data(), h_regTgt.data(), lr);

            processedImages += loaded;

            if (b % 20 == 0 || b == numBatches - 1) {
                cout << "\r  Epoch " << epoch+1 << "/" << EPOCHS
                     << "  Batch " << b+1 << "/" << numBatches
                     << "  Images: " << processedImages
                     << "     " << flush;
            }
        }
        epochTimer.stop();
        file.close();

        // Learning-rate cosine decay step
        lr = LR_INIT * 0.5f * (1.f + cos((float)(epoch+1) / EPOCHS * M_PI));

        cout << "\n  Epoch " << epoch+1 << "/" << EPOCHS
             << " | LR: " << lr
             << " | Time: " << epochTimer.elapsed_ms() / 1000.f << "s" << endl;
    }

    cout << "\nTraining complete." << endl;
    detector.saveWeights("od_voc2007_model.bin");

    return 0;
}
