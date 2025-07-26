#pragma once
#include "SYCLMatrixLib.h"
#include "ConvLayer.hpp"
#include "MaxPool2x2.hpp"
#include <vector>

class CNN {
private:
    float learningRate = 0.05f;
    sycl::queue q{sycl::gpu_selector{}, sycl::property_list{sycl::property::queue::in_order{}}};

    // CNN layers
    ConvLayer conv1;
    MaxPool2x2 pool1;
    
    // FC layers
    Matrix fc1W, fc1b;  // 32*15*15 -> 256
    Matrix fc2W, fc2b;  // 256 -> 128
    Matrix fc3W, fc3b;  // 128 -> 10

    // Cache for activations (needed for backprop)
    float* input_cache;
    Matrix conv1_out_cache;
    Matrix pool1_out_cache;
    Matrix fc1_out_cache;
    Matrix fc2_out_cache;
    int cached_batch_size;

public:
    CNN() : conv1(3, 32, 32, 32, q),           // 3 channels, 32x32 input, 32 filters
            pool1(32, 30, 30, q),               // 32 channels, 30x30 input (after 3x3 conv)
            fc1W(256, 32*15*15, q),             // Flatten 32*15*15 -> 256
            fc1b(256, 1, q),
            fc2W(128, 256, q),                  // 256 -> 128
            fc2b(128, 1, q),
            fc3W(10, 128, q),                   // 128 -> 10 classes
            fc3b(10, 1, q),
            input_cache(nullptr),
            conv1_out_cache(0, 0, q),
            pool1_out_cache(0, 0, q),
            fc1_out_cache(0, 0, q),
            fc2_out_cache(0, 0, q),
            cached_batch_size(0)
    {
        // Initialize FC weights with He initialization
        fc1W.RandInit(32*15*15, 256);
        fc2W.RandInit(256, 128);
        fc3W.RandInit(128, 10);
    }

    ~CNN() {
        if (input_cache) {
            sycl::free(input_cache, q);
        }
    }

    void setLearningRate(float lr) {
        learningRate = lr;
    }

    void printDeviceInfo() {
        std::cout << "SYCL Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    }

    // Forward pass for single batch
    std::vector<float> FeedForward(const std::vector<float>& input) {
        // Reshape input to device memory (N=1, C=3, H=32, W=32)
        float* x_device = sycl::malloc_device<float>(3 * 32 * 32, q);
        q.memcpy(x_device, input.data(), input.size() * sizeof(float));

        // Conv1 -> ReLU
        Matrix conv1_out = conv1.forward(x_device, 1);
        conv1_out.applyReLU();

        // Pool1
        Matrix pool1_out = pool1.forward(conv1_out, 1);

        // Flatten for FC layers: reshape from [C*H*W, N] to [C*H*W, N]
        // pool1_out is already in the right format

        // FC1 -> ReLU
        Matrix fc1_out = Matrix::multiply(fc1W, pool1_out, q);
        fc1_out.add(fc1b);
        fc1_out.applyReLU();

        // FC2 -> ReLU
        Matrix fc2_out = Matrix::multiply(fc2W, fc1_out, q);
        fc2_out.add(fc2b);
        fc2_out.applyReLU();

        // FC3 -> Softmax
        Matrix fc3_out = Matrix::multiply(fc3W, fc2_out, q);
        fc3_out.add(fc3b);
        Matrix output = Matrix::Softmax(fc3_out, q);

        sycl::free(x_device, q);
        return output.toArr();
    }

    // Batch training
    Matrix BackPropagateBatch(const std::vector<std::vector<float>>& batchInputs, 
                             const std::vector<std::vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return Matrix();

        cached_batch_size = batchSize;

        // Prepare input data in device memory [N, C, H, W] format
        if (!input_cache) {
            input_cache = sycl::malloc_device<float>(batchSize * 3 * 32 * 32, q);
        }

        // Copy batch inputs to device (convert from [N][CHW] to [N,C,H,W])
        std::vector<float> host_inputs_flat(batchSize * 3 * 32 * 32);
        for (int n = 0; n < batchSize; n++) {
            for (int i = 0; i < 3 * 32 * 32; i++) {
                host_inputs_flat[n * 3 * 32 * 32 + i] = batchInputs[n][i];
            }
        }
        q.memcpy(input_cache, host_inputs_flat.data(), batchSize * 3 * 32 * 32 * sizeof(float));

        // FORWARD PASS
        // Conv1 -> ReLU
        conv1_out_cache = conv1.forward(input_cache, batchSize);
        conv1_out_cache.applyReLU();

        // Pool1
        pool1_out_cache = pool1.forward(conv1_out_cache, batchSize);

        // FC1 -> ReLU
        fc1_out_cache = Matrix::multiply(fc1W, pool1_out_cache, q);
        fc1_out_cache.add(fc1b);
        fc1_out_cache.applyReLU();

        // FC2 -> ReLU
        fc2_out_cache = Matrix::multiply(fc2W, fc1_out_cache, q);
        fc2_out_cache.add(fc2b);
        fc2_out_cache.applyReLU();

        // FC3 -> Softmax
        Matrix outputs = Matrix::multiply(fc3W, fc2_out_cache, q);
        outputs.add(fc3b);
        outputs = Matrix::Softmax(outputs, q);

        // Prepare targets
        Matrix targets(10, batchSize, q);
        std::vector<float> host_targets_flat(10 * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 10; j++) {
                host_targets_flat[j * batchSize + i] = batchOutputs[i][j];
            }
        }
        q.memcpy(targets.data, host_targets_flat.data(), 10 * batchSize * sizeof(float));

        // BACKWARD PASS
        // Output layer error
        Matrix output_errors = Matrix::subtract(outputs, targets, q);
        output_errors.multiplyScalar(1.0f / batchSize);

        // FC3 gradients
        Matrix fc3_grad = output_errors;
        fc3_grad.multiplyScalar(-learningRate);
        Matrix fc3W_delta = Matrix::multiply(fc3_grad, fc2_out_cache.Transpose(), q);
        fc3W.add(fc3W_delta);
        fc3b.add(fc3_grad.sumAlongAxis(1));

        // FC2 errors and gradients
        Matrix fc2_errors = Matrix::multiply(fc3W.Transpose(), output_errors, q);
        Matrix fc2_grad_relu = Matrix::ApplyReLUDerivative(fc2_out_cache, q);
        fc2_grad_relu.elementWiseMult(fc2_errors);
        fc2_grad_relu.multiplyScalar(-learningRate);
        Matrix fc2W_delta = Matrix::multiply(fc2_grad_relu, fc1_out_cache.Transpose(), q);
        fc2W.add(fc2W_delta);
        fc2b.add(fc2_grad_relu.sumAlongAxis(1));

        // FC1 errors and gradients
        Matrix fc1_errors = Matrix::multiply(fc2W.Transpose(), fc2_errors, q);
        Matrix fc1_grad_relu = Matrix::ApplyReLUDerivative(fc1_out_cache, q);
        fc1_grad_relu.elementWiseMult(fc1_errors);
        fc1_grad_relu.multiplyScalar(-learningRate);
        Matrix fc1W_delta = Matrix::multiply(fc1_grad_relu, pool1_out_cache.Transpose(), q);
        fc1W.add(fc1W_delta);
        fc1b.add(fc1_grad_relu.sumAlongAxis(1));

        // Pool1 errors (no parameters to update)
        Matrix pool1_errors = Matrix::multiply(fc1W.Transpose(), fc1_errors, q);
        Matrix conv1_pool_errors = pool1.backward(pool1_errors, batchSize);

        // Conv1 errors and gradients (apply ReLU derivative first)
        Matrix conv1_grad_relu = Matrix::ApplyReLUDerivative(conv1_out_cache, q);
        conv1_grad_relu.elementWiseMult(conv1_pool_errors);
        conv1.backward(conv1_grad_relu, learningRate);

        return outputs;
    }
}; 