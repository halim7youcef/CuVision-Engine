#include "network/network.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

void loadBatch(ifstream& file, float* batchData, uint8_t* labels, int batchSize) {
    int imageSize = 3 * 32 * 32;
    vector<uint8_t> pixelBuffer(imageSize);

    for (int i = 0; i < batchSize; ++i) {
        if (!file.read((char*)&labels[i], 1)) return;
        file.read((char*)pixelBuffer.data(), imageSize);

        for (int j = 0; j < imageSize; ++j) {
            batchData[i * imageSize + j] = (float)pixelBuffer[j] / 255.0f;
        }
    }
}

int main() {
    // Correct paths for the dataset since main is in classification/ but data is in classification/dataset/
    const char* datasetPath = "dataset/flowers10.bin";
    int batchSize = 16;
    int channels = 3, height = 32, width = 32, numClasses = 10;
    float learningRate = 0.01f;
    int epochs = 5;

    cout << "Initializing Training for 10-class Classification..." << endl;
    ImageClassifier dnn(batchSize, channels, height, width, numClasses);

    vector<float> h_batch(batchSize * channels * height * width);
    vector<uint8_t> h_labels(batchSize);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        ifstream file(datasetPath, ios::binary);
        if (!file.is_open()) {
            cerr << "Error: Could not open " << datasetPath << ". Run dataset/prepare_dataset.py first." << endl;
            return 1;
        }

        int totalImages;
        file.read((char*)&totalImages, 4);
        int numBatches = totalImages / batchSize;

        float totalLoss = 0;
        int correct = 0;

        GpuTimer epochTimer; epochTimer.start();
        for (int b = 0; b < numBatches; ++b) {
            loadBatch(file, h_batch.data(), h_labels.data(), batchSize);

            // Forward Pass
            dnn.forward(h_batch.data());
            float* d_output = dnn.getOutput();

            vector<float> h_output(batchSize * numClasses);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost));

            // Calculate Loss (Cross Entropy) and Accuracy
            for (int i = 0; i < batchSize; ++i) {
                float prob = h_output[i * numClasses + h_labels[i]];
                totalLoss -= log(prob + 1e-7f);

                int predicted = 0;
                float maxProb = h_output[i * numClasses];
                for (int c = 1; c < numClasses; ++c) {
                    if (h_output[i * numClasses + c] > maxProb) {
                        maxProb = h_output[i * numClasses + c];
                        predicted = c;
                    }
                }
                if (predicted == h_labels[i]) correct++;
            }

            // Backward Pass (Training)
            dnn.backward(h_labels.data(), learningRate);

            if (b % 10 == 0) cout << "\rBatch " << b << "/" << numBatches << " processed..." << flush;
        }
        epochTimer.stop();

        // Learning Rate Decay (reduce by 20% every epoch)
        learningRate *= 0.8f;

        cout << "\rEpoch " << epoch + 1 << "/" << epochs
             << " - Loss: " << totalLoss / totalImages
             << " - Accuracy: " << (float)correct / totalImages * 100 << "%" 
             << " - LR: " << learningRate 
             << " - Time: " << epochTimer.elapsed_ms() / 1000.0f << "s" << endl;

        file.close();
    }

    cout << "\nTraining completed successfully." << endl;
    dnn.saveWeights("flower_model.bin");

    return 0;
}
