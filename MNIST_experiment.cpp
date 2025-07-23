#include <iostream>
#include <fstream>
#include <vector>
#include "ImageNew.h"
#include "NeuralNetworkNew.h"
#include <random>
#include <algorithm>
#include <omp.h>
#include <chrono> // Required for std::chrono::system_clock

using namespace std;

std::vector<unsigned int> getRGBFromInt(unsigned int x) {
    unsigned int red = (x & 0xff0000) >> 16;
    unsigned int green = (x & 0x00ff00) >> 8;
    unsigned int blue = (x & 0x0000ff);
    return { red, green, blue };
}

std::vector<ImageData> dataset = {};
std::vector<ImageData> trainingSet = {};
std::vector<ImageData> testingSet = {};

// Load MNIST dataset 
std::vector<std::string> fileVec{
    "C:\\DEV\\Datasets\\train-images.idx3-ubyte",  // Training images
    "C:\\DEV\\Datasets\\train-labels.idx1-ubyte",  // Training labels
    "C:\\DEV\\Datasets\\t10k-images.idx3-ubyte",   // Testing images
    "C:\\DEV\\Datasets\\t10k-labels.idx1-ubyte"    // Testing labels
};

int globalOutputNodesCount = 10;  // MNIST has 10 output classes (digits 0-9)
NeuralNetwork nn(784, 64, 32, globalOutputNodesCount);  // 784 input nodes for 28x28 MNIST images


void loadMNISTData() {
    // Load MNIST training data
    std::ifstream imageFile(fileVec[0], ios::binary);
    std::ifstream labelFile(fileVec[1], ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open()) {
        std::cerr << "Error: Cannot open MNIST files." << std::endl;
        return;
    }

    // Skip headers (16 bytes for images, 8 bytes for labels)
    imageFile.seekg(16);
    labelFile.seekg(8);

    const int imageSize = 28 * 28;
    std::vector<unsigned char> imageBuffer(imageSize);
    unsigned char label;

    while (imageFile.read(reinterpret_cast<char*>(imageBuffer.data()), imageSize) &&
        labelFile.read(reinterpret_cast<char*>(&label), 1)) {
        std::vector<float> normalizedImage;
        normalizedImage.reserve(imageSize);
        for (unsigned char pix : imageBuffer) {
            normalizedImage.push_back(pix / 255.0f);
        }
        dataset.emplace_back(ImageData(normalizedImage, static_cast<int>(label)));
    }

    imageFile.close();
    labelFile.close();

    // Group images by label
    std::vector<std::vector<ImageData>> labelBuckets(10);
    for (auto& img : dataset) {
        if (img.label >= 0 && img.label < 10) { // Ensure label is within range
            labelBuckets[img.label].emplace_back(img);
        }
    }

    // Define the number of images per label
    const int imagesPerLabel = 1000;
    std::vector<ImageData> balancedSubset;
    balancedSubset.reserve(imagesPerLabel * 10);

    for (int l = 0; l < 10; l++) {
        int count = 0;
        for (auto& im : labelBuckets[l]) {
            if (count >= imagesPerLabel) break;
            balancedSubset.emplace_back(im);
            count++;
        }
        // If a label has fewer than imagesPerLabel samples, fill the rest by repeating existing samples
        while (count < imagesPerLabel && !labelBuckets[l].empty()) {
            balancedSubset.emplace_back(labelBuckets[l][count % labelBuckets[l].size()]);
            count++;
        }
    }

    // Shuffle the balanced subset
    std::default_random_engine rng(0); // Use a fixed seed for reproducibility
    std::shuffle(balancedSubset.begin(), balancedSubset.end(), rng);

    // Split into training and testing sets (60% training, 40% testing)
    int loopto = static_cast<int>(balancedSubset.size());
    for (int i = 0; i < loopto; i++) {
        if (i < static_cast<int>(0.6 * loopto))
            trainingSet.emplace_back(balancedSubset[i]);
        else
            testingSet.emplace_back(balancedSubset[i]);
    }

    std::cout << "Training set size: " << trainingSet.size() << "\n";
    std::cout << "Testing Set Size: " << testingSet.size() << "\n";
}


void test() {
    int correct = 0;
    for (const auto& testSample : testingSet) {
        vector<float> inputs = testSample.vec;
        int label = testSample.label;
        vector<float> guess = nn.FeedForward(inputs);

        int predictedLabel = max_element(guess.begin(), guess.end()) - guess.begin();
        if (predictedLabel == label) {
            correct++;
        }
    }
    std::cout << "Accuracy: " << 100 * (float)correct / testingSet.size() << "%\n";
}

void train() {
    for (const auto& trainSample : trainingSet) {
        std::vector<float> inputs = trainSample.vec;
        int label = trainSample.label;
        std::vector<float> targets(globalOutputNodesCount, 0.0f);
        targets[label] = 1.0f;
        nn.BackPropagate(inputs, targets);
    }
}

void batchTrain(int batchSize, int numEpochs) {
    std::default_random_engine rng(0); // Use a fixed seed for reproducibility
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        std::shuffle(trainingSet.begin(), trainingSet.end(), rng);
        double start = omp_get_wtime();

        vector<vector<float>> batchInputs(batchSize, vector<float>(784));
        vector<vector<float>> batchOutputs(batchSize, vector<float>(globalOutputNodesCount));

        for (size_t i = 0; i < trainingSet.size(); i += batchSize) {
            // Adjust batch size for last incomplete batch
            int currentBatchSize = std::min(batchSize, static_cast<int>(trainingSet.size() - i));

            // Populate batch data
            for (size_t j = 0; j < currentBatchSize; j++) {
                batchInputs[j] = trainingSet[i + j].vec;

                std::fill(batchOutputs[j].begin(), batchOutputs[j].end(), 0.0f);
                batchOutputs[j][trainingSet[i + j].label] = 1.0f;
            }

            // Train on the current batch
            nn.BackPropagateBatch(batchInputs, batchOutputs);

            // Calculate and print loss for the current batch
            float loss = 0.0f;
            for (int j = 0; j < currentBatchSize; ++j) {
                vector<float> outputs_host = nn.FeedForward(batchInputs[j]);
                int trueLabel = trainingSet[i + j].label;
                // Ensure the index is within bounds
                if (trueLabel >= 0 && trueLabel < outputs_host.size()) {
                    loss -= std::log(outputs_host[trueLabel] + 1e-9f);
                }
                //if (i == 0 && j == 0) { // Only print for the very first sample of the first batch
                //    std::cout << "p(y=label0) for first sample: " << outputs_host[trueLabel] << "\n";
                //}
            }
            loss /= currentBatchSize;
            //std::cout << "Batch Loss: " << loss << "\n";
        }

        std::cout << "Epoch " << epoch + 1 << " completed.\n";
        std::cout << "Epoch " << epoch + 1 << " took " << omp_get_wtime() - start << " seconds\n";
    }
}

int main() {
    nn.setLearningRate(0.05f);
    std::srand(std::time(0));

    double start = omp_get_wtime();
    loadMNISTData();
    std::cout << "Loading MNIST data took " << omp_get_wtime() - start << " seconds\n";
    std::cout << "Training set size: " << trainingSet.size() << "\nTesting Set Size: " << testingSet.size() << "\n";

    nn.printDeviceInfo(); // Print SYCL device information

    // Train using batch processing
    int batchSize = 64;
    int numEpochs = 10;
    std::cout << "Training with batch size: " << batchSize << ", for " << numEpochs << " epochs.\n";
    batchTrain(batchSize, numEpochs); 

    test(); // Evaluate the model after training
    return 0;
}
