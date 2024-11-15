#include <iostream>
#include <fstream>
#include <vector>
#include "C:\Users\ncvn\source\repos\ANN-from-scratch\ImageNew.h"
#include "C:\Users\ncvn\source\repos\ANN-from-scratch\NeuralNetworkNew.h"
#include <random>
#include <algorithm>
#include <omp.h>
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
        for (auto& pixel : imageBuffer) {
            normalizedImage.push_back(pixel / 255.0f);
        }
        dataset.push_back(ImageData(normalizedImage, static_cast<int>(label)));
    }

    imageFile.close();
    labelFile.close();

    // Split dataset into training and testing sets
    std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine(0));
    for (int i = 0; i < dataset.size(); i++) {
        if (i < (int)(0.9 * dataset.size())) trainingSet.push_back(dataset.at(i));
        else testingSet.push_back(dataset.at(i));
    }
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
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        std::shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
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
        }

        std::cout << "Epoch " << epoch + 1 << " completed.\n";
        std::cout << "Epoch " << epoch + 1 << " took " << omp_get_wtime() - start << " seconds\n";
    }
}

int main() {
    nn.setLearningRate(0.001f);
    std::srand(std::time(0));

    double start = omp_get_wtime();
    loadMNISTData();
    std::cout << "Loading MNIST data took " << omp_get_wtime() - start << " seconds\n";
    std::cout << "Training set size: " << trainingSet.size() << "\nTesting Set Size: " << testingSet.size() << "\n";

    // Train using batch processing
    batchTrain(64, 10); // 10 epochs with batch size of 64

    test(); // Evaluate the model after training
    return 0;
}
