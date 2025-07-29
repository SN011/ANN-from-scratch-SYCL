#include <iostream>
#include <fstream>
#include <vector>
#include "ImageNew.h"
#include "NeuralNetworkNew.h"
#include <random>
#include <algorithm>
#include <omp.h>
#include <chrono> // Required for std::chrono::system_clock
#include <sstream>
#include <cstdint>
#include <cstring> // Required for strncmp

using namespace std;

std::vector<unsigned int> getRGBFromInt(unsigned int x) {
    unsigned int red = (x & 0xff0000) >> 16;
    unsigned int green = (x & 0x00ff00) >> 8;
    unsigned int blue = (x & 0x0000ff);
    return { red, green, blue };
}

std::vector<ImageData> trainingSet = {};
std::vector<ImageData> testingSet = {};

// Load MNIST dataset 
std::vector<std::string> MNISTfileVec{
    "C:\\DEV\\Datasets\\train-images.idx3-ubyte",  // Training images
    "C:\\DEV\\Datasets\\train-labels.idx1-ubyte",  // Training labels
    "C:\\DEV\\Datasets\\t10k-images.idx3-ubyte",   // Testing images
    "C:\\DEV\\Datasets\\t10k-labels.idx1-ubyte"    // Testing labels
};

std::vector<std::string> FashionMNISTfileVec{
    "C:\\DEV\\Datasets\\train-images-idx3-ubyte",  // Training images
    "C:\\DEV\\Datasets\\train-labels-idx1-ubyte",  // Training labels
    "C:\\DEV\\Datasets\\t10k-images-idx3-ubyte",   // Testing images
    "C:\\DEV\\Datasets\\t10k-labels-idx1-ubyte"    // Testing labels
};

// New CIFAR-10 .npy file vector (train images/labels, test images/labels)
std::vector<std::string> CIFAR10fileVec{
    "C:\\DEV\\Datasets\\train_images.npy",   // [N,32,32,3] uint8
    "C:\\DEV\\Datasets\\train_labels.npy",   // [N] int64
    "C:\\DEV\\Datasets\\test_images.npy",    // [M,32,32,3] uint8
    "C:\\DEV\\Datasets\\test_labels.npy"     // [M] int64
};

// ---------------- NPY parsing helpers -----------------
static bool parseNpyHeader(std::ifstream &ifs, std::vector<size_t> &shape, size_t &wordSize) {
    // Read magic string
    char magic[6];
    ifs.read(magic, 6);
    if (strncmp(magic, "\x93NUMPY", 6) != 0) return false;

    unsigned char major, minor;
    ifs.read(reinterpret_cast<char *>(&major), 1);
    ifs.read(reinterpret_cast<char *>(&minor), 1);

    uint16_t headerLen16 = 0;
    uint32_t headerLen32 = 0;
    size_t headerLen = 0;
    if (major == 1) {
        ifs.read(reinterpret_cast<char *>(&headerLen16), 2);
        headerLen = headerLen16;
    } else {
        ifs.read(reinterpret_cast<char *>(&headerLen32), 4);
        headerLen = headerLen32;
    }

    std::string header(headerLen, ' ');
    ifs.read(header.data(), headerLen);

    // Parse word size from 'descr'
    const std::string descrKey = "'descr':";
    auto posDescr = header.find(descrKey);
    if (posDescr == std::string::npos) return false;
    auto quote1 = header.find("'", posDescr + descrKey.size());
    auto quote2 = header.find("'", quote1 + 1);
    std::string descr = header.substr(quote1 + 1, quote2 - quote1 - 1);
    if (descr == "|u1") wordSize = 1;
    else if (descr == "<i8") wordSize = 8;
    else {
        std::cerr << "Unsupported dtype: " << descr << std::endl;
        return false;
    }

    // Parse shape tuple
    const std::string shapeKey = "'shape':";
    auto posShape = header.find(shapeKey);
    if (posShape == std::string::npos) return false;
    auto paren1 = header.find('(', posShape);
    auto paren2 = header.find(')', paren1);
    std::string shapeContent = header.substr(paren1 + 1, paren2 - paren1 - 1);

    std::stringstream ss(shapeContent);
    while (ss.good()) {
        size_t dim;
        char comma;
        ss >> dim;
        shape.push_back(dim);
        ss >> comma; // consume comma
    }

    // Move stream cursor to start of data (already there after header read)
    return true;
}

void loadCIFAR10Data() {
    using std::cerr; using std::endl;
    // ---------- Load labels first (int64) ----------
    std::ifstream lblTrainFile(CIFAR10fileVec[1], ios::binary);
    if (!lblTrainFile.is_open()) { cerr << "Cannot open " << CIFAR10fileVec[1] << endl; return; }
    std::vector<size_t> lblShape; size_t lblWord;
    if (!parseNpyHeader(lblTrainFile, lblShape, lblWord)) { cerr << "Label header parse error" << endl; return; }
    const size_t Ntrain = lblShape[0];
    std::vector<int64_t> trainLabels(Ntrain);
    lblTrainFile.read(reinterpret_cast<char *>(trainLabels.data()), Ntrain * sizeof(int64_t));
    lblTrainFile.close();

    std::ifstream lblTestFile(CIFAR10fileVec[3], ios::binary);
    if (!lblTestFile.is_open()) { cerr << "Cannot open " << CIFAR10fileVec[3] << endl; return; }
    std::vector<size_t> lblShapeTest; size_t lblWordTest;
    if (!parseNpyHeader(lblTestFile, lblShapeTest, lblWordTest)) { cerr << "Test label header parse error" << endl; return; }
    const size_t Ntest = lblShapeTest[0];
    std::vector<int64_t> testLabels(Ntest);
    lblTestFile.read(reinterpret_cast<char *>(testLabels.data()), Ntest * sizeof(int64_t));
    lblTestFile.close();

    // ---------- Load images (uint8) ----------
    std::ifstream imgTrainFile(CIFAR10fileVec[0], ios::binary);
    if (!imgTrainFile.is_open()) { cerr << "Cannot open " << CIFAR10fileVec[0] << endl; return; }
    std::vector<size_t> imgShape; size_t imgWord;
    if (!parseNpyHeader(imgTrainFile, imgShape, imgWord)) { cerr << "Train images header parse error" << endl; return; }
    const size_t H = imgShape[1], W = imgShape[2], C = imgShape[3];
    const size_t imageSize = H * W * C;  // should be 3072

    trainingSet.clear();
    trainingSet.reserve(Ntrain);
    std::vector<uint8_t> imgBuf(imageSize);
    for (size_t i = 0; i < Ntrain; ++i) {
        imgTrainFile.read(reinterpret_cast<char *>(imgBuf.data()), imageSize);
        std::vector<float> pixels(imageSize);
        for (size_t p = 0; p < imageSize; ++p) pixels[p] = imgBuf[p] / 255.0f;
        trainingSet.emplace_back(ImageData(std::move(pixels), static_cast<int>(trainLabels[i])));
    }
    imgTrainFile.close();

    // Test images
    std::ifstream imgTestFile(CIFAR10fileVec[2], ios::binary);
    if (!imgTestFile.is_open()) { cerr << "Cannot open " << CIFAR10fileVec[2] << endl; return; }
    std::vector<size_t> imgShapeT; size_t imgWordT;
    if (!parseNpyHeader(imgTestFile, imgShapeT, imgWordT)) { cerr << "Test images header parse error" << endl; return; }
    testingSet.clear();
    testingSet.reserve(Ntest);
    for (size_t i = 0; i < Ntest; ++i) {
        imgTestFile.read(reinterpret_cast<char *>(imgBuf.data()), imageSize);
        std::vector<float> pixels(imageSize);
        for (size_t p = 0; p < imageSize; ++p) pixels[p] = imgBuf[p] / 255.0f;
        testingSet.emplace_back(ImageData(std::move(pixels), static_cast<int>(testLabels[i])));
    }
    imgTestFile.close();

    std::cout << "CIFAR-10 loaded: " << trainingSet.size() << " train, " << testingSet.size() << " test samples\n";
}

// ---------------- END CIFAR-10 LOADER -----------------

int globalOutputNodesCount = 10;  // CIFAR-10 has 10 output classes
// Use CNN instead of fully connected network
NeuralNetwork nn(784,512,256,globalOutputNodesCount);

void loadMNISTData() {
    // Load MNIST training data
    std::ifstream imageFile(FashionMNISTfileVec[0], ios::binary);
    std::ifstream labelFile(FashionMNISTfileVec[1], ios::binary);

    std::ifstream imageFileTest(FashionMNISTfileVec[2], ios::binary);
    std::ifstream labelFileTest(FashionMNISTfileVec[3], ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open()) {
        std::cerr << "Error: Cannot open MNIST files." << std::endl;
        if (!imageFileTest.is_open() || !labelFileTest.is_open()) {
            std::cerr << "Error: Cannot open MNIST files." << std::endl;

        }
        return;
    }



    // Skip headers (16 bytes for images, 8 bytes for labels)
    imageFile.seekg(16);
    labelFile.seekg(8);

    imageFileTest.seekg(16);
    labelFileTest.seekg(8);

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
        trainingSet.emplace_back(ImageData(normalizedImage, static_cast<int>(label)));
    }

    imageFile.close();
    labelFile.close();

    int testCount = 0;
    while (imageFileTest.read(reinterpret_cast<char*>(imageBuffer.data()), imageSize) &&
        labelFileTest.read(reinterpret_cast<char*>(&label), 1)) {

        std::vector<float> normalizedImage;
        normalizedImage.reserve(imageSize);
        for (unsigned char pix : imageBuffer) {
            normalizedImage.push_back(pix / 255.0f);
        }
        testingSet.emplace_back(ImageData(normalizedImage, static_cast<int>(label)));
        testCount++;
    }
    std::cout << "Read " << testCount << " test samples." << std::endl;


    imageFileTest.close();
    labelFileTest.close();


    // Shuffle the balanced subset
    std::default_random_engine rng(0); // Use a fixed seed for reproducibility
    std::shuffle(trainingSet.begin(), trainingSet.end(), rng);


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
        std::vector<std::vector<float>> batchInputs = {inputs};
        std::vector<std::vector<float>> batchTargets = {targets};
        nn.BackPropagateBatch(batchInputs, batchTargets);
    }
}

void batchTrain(int batchSize, int numEpochs) {
    std::default_random_engine rng(0); // Use a fixed seed for reproducibility
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        std::shuffle(trainingSet.begin(), trainingSet.end(), rng);
        double start = omp_get_wtime();

        for (size_t i = 0; i < trainingSet.size(); i += batchSize) {
            // Adjust batch size for last incomplete batch
            int currentBatchSize = std::min(batchSize, static_cast<int>(trainingSet.size() - i));

            vector<vector<float>> batchInputs(currentBatchSize, vector<float>(784)); // MNIST is 784 (28x28)
            vector<vector<float>> batchOutputs(currentBatchSize, vector<float>(globalOutputNodesCount));

            // Populate batch data
            for (size_t j = 0; j < currentBatchSize; j++) {
                batchInputs[j] = trainingSet[i + j].vec;

                std::fill(batchOutputs[j].begin(), batchOutputs[j].end(), 0.0f);
                batchOutputs[j][trainingSet[i + j].label] = 1.0f;
            }

            // Train on the current batch and get outputs
            nn.BackPropagateBatchAndReturnOutputs(batchInputs, batchOutputs);

        }

        std::cout << "Epoch " << epoch + 1 << " completed.\n";
        std::cout << "Epoch " << epoch + 1 << " took " << omp_get_wtime() - start << " seconds\n";
    }
}
//
//int main() {
//    nn.setLearningRate(0.15f);
//    std::srand(std::time(0));
//
//    double start = omp_get_wtime();
//    loadMNISTData();
//    std::cout << "Loading MNIST data took " << omp_get_wtime() - start << " seconds\n";
//    std::cout << "Training set size: " << trainingSet.size() << "\nTesting Set Size: " << testingSet.size() << "\n";
//
//    nn.printDeviceInfo(); // Print SYCL device information
//
//    // Train using batch processing
//    int batchSize = 256;  // Reduced batch size for CNN
//    int numEpochs = 30;  // More epochs for CNN
//    std::cout << "Training ANN with batch size: " << batchSize << ", for " << numEpochs << " epochs.\n";
//    batchTrain(batchSize, numEpochs);
//
//    test(); // Evaluate the model after training
//    return 0;
//}

