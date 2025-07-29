#pragma once
#include "SYCLMatrixLib_CPU.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace std;
using namespace sycl;

class NeuralNetwork_CPU {
private:
    float learningRate = 0.1f;

    vector<int> nn_topology;
    vector<Matrix_CPU> weightMatrices;
    vector<Matrix_CPU> biasMatrices;
    
    // CPU-optimized queue configuration
    sycl::queue q;

    void initializeQueue() {
        try {
            // Prefer CPU selector for dedicated CPU optimization
            q = sycl::queue{sycl::cpu_selector_v, 
                           sycl::property_list{sycl::property::queue::in_order{}}};
            
            // Fallback to default selector if CPU not available
            if (!q.get_device().is_cpu()) {
                std::cout << "Warning: CPU device not available, using default device\n";
                q = sycl::queue{sycl::default_selector_v, 
                               sycl::property_list{sycl::property::queue::in_order{}}};
            }
            
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL queue initialization failed: " << e.what() << std::endl;
            // Fallback to default selector
            q = sycl::queue{sycl::default_selector_v, 
                           sycl::property_list{sycl::property::queue::in_order{}}};
        }
    }

public:

    NeuralNetwork_CPU(int in, int hid1, int hid2, int out, unsigned int seed = 42) {
        initializeQueue();
        
        //Add the params to the topology array list --> list will be [in, hid1, hid2, out]
        nn_topology.push_back(in);
        nn_topology.push_back(hid1);
        nn_topology.push_back(hid2);
        nn_topology.push_back(out);

        // Initialize weight and bias matrices with CPU-optimized approach (in-place construction)
        weightMatrices.reserve(nn_topology.size() - 1);
        biasMatrices.reserve(nn_topology.size() - 1);

        for (size_t i = 0; i + 1 < nn_topology.size(); ++i) {
            // Construct weight matrix directly inside vector
            weightMatrices.emplace_back(nn_topology[i + 1], nn_topology[i], q);
            auto &W = weightMatrices.back();
            W.RandInit(nn_topology[i], nn_topology[i + 1], seed + static_cast<unsigned int>(i));

            // Construct bias matrix (zero-initialised by default)
            biasMatrices.emplace_back(nn_topology[i + 1], 1, q);
        }
    }

    void setLearningRate(float lr) {
        this->learningRate = lr;
    }

    vector<float> FeedForward(vector<float> targetInputs) {
        Matrix_CPU inputs = Matrix_CPU::fromArr(targetInputs, q);
        
        //inputL->hidden1L
        Matrix_CPU hidden = Matrix_CPU::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();
        
        //hidden1L -> hidden2L
        Matrix_CPU hidden2 = Matrix_CPU::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();
        
        //hiddenL->outputL
        Matrix_CPU outputs = Matrix_CPU::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix_CPU::Softmax(outputs, q);

        // Ensure all operations complete before returning
        q.wait();
        
        return outputs.toArr();
    }

    void printDeviceInfo() {
        auto device = q.get_device();
        std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Device Type: " << (device.is_cpu() ? "CPU" : 
                                        device.is_gpu() ? "GPU" : "Other") << std::endl;
        std::cout << "Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
        std::cout << "Max Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    }

    sycl::queue& getQueue() { return q; }

    void BackPropagate(vector<float> targetInputs, vector<float> targetOutputs) {
        //FeedForward phase with intermediate results stored for backprop
        Matrix_CPU inputs = Matrix_CPU::fromArr(targetInputs, q);
        
        //inputL->hidden1L
        Matrix_CPU hidden = Matrix_CPU::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();
        
        //hidden1L -> hidden2L
        Matrix_CPU hidden2 = Matrix_CPU::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();
        
        //hiddenL->outputL
        Matrix_CPU outputs = Matrix_CPU::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix_CPU::Softmax(outputs, q);

        //Target array --> converted to matrix
        Matrix_CPU targets = Matrix_CPU::fromArr(targetOutputs, q);

        //Output errors = outputs - targets (for Softmax + Cross-Entropy)
        Matrix_CPU output_errors = Matrix_CPU::subtract(outputs, targets, q);

        //Calculate the gradients for output layer (simplified for Softmax + Cross-Entropy)
        Matrix_CPU gradients = output_errors;
        gradients.multiplyScalar(-learningRate);

        //Update weights and biases for hidden2 -> output layer
        Matrix_CPU weight_h2o_deltas = Matrix_CPU::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));

        //Hidden layer 2 errors
        Matrix_CPU hidden2_errors = Matrix_CPU::multiply(weightMatrices[2].Transpose(), output_errors, q);

        //Calculate the gradients for hidden layer 2
        Matrix_CPU hidden2_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-learningRate);

        //Update weights and biases for hidden1 -> hidden2 layer
        Matrix_CPU weight_h1h2_deltas = Matrix_CPU::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        //Hidden layer 1 errors
        Matrix_CPU hidden_errors = Matrix_CPU::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        //Calculate the gradients for hidden layer 1
        Matrix_CPU hidden_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-learningRate);

        //Update weights and biases for input -> hidden1 layer
        Matrix_CPU weight_ih1_deltas = Matrix_CPU::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));

        // Ensure all operations complete
        q.wait();
    }

    void BackPropagateBatch(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return;

        // Combine inputs into a batch matrix - CPU optimized data layout
        Matrix_CPU inputs(nn_topology[0], batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                inputs.host_data[j * batchSize + i] = batchInputs[i][j];
            }
        }

        // FeedForward Step
        Matrix_CPU hidden = Matrix_CPU::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        Matrix_CPU hidden2 = Matrix_CPU::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();

        Matrix_CPU outputs = Matrix_CPU::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix_CPU::Softmax(outputs, q);

        // Combine targets into a batch matrix
        Matrix_CPU targets(nn_topology.back(), batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                targets.host_data[j * batchSize + i] = batchOutputs[i][j];
            }
        }

        // Output errors = outputs - targets (matrix subtraction)
        Matrix_CPU output_errors = Matrix_CPU::subtract(outputs, targets, q);
        output_errors.multiplyScalar(1.0f / batchSize);

        // Calculate the gradients for output layer
        Matrix_CPU gradients = output_errors;
        gradients.multiplyScalar(-learningRate);

        // Update weights and biases for hidden2 -> output layer
        Matrix_CPU weight_h2o_deltas = Matrix_CPU::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));

        // Hidden layer 2 errors
        Matrix_CPU hidden2_errors = Matrix_CPU::multiply(weightMatrices[2].Transpose(), output_errors, q);

        // Calculate the gradients for hidden layer 2
        Matrix_CPU hidden2_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-learningRate);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix_CPU weight_h1h2_deltas = Matrix_CPU::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        // Hidden layer 1 errors
        Matrix_CPU hidden_errors = Matrix_CPU::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        // Calculate the gradients for hidden layer 1
        Matrix_CPU hidden_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-learningRate);

        // Update weights and biases for input -> hidden1 layer
        Matrix_CPU weight_ih1_deltas = Matrix_CPU::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));

        // Ensure all operations complete
        q.wait();
    }

    Matrix_CPU BackPropagateBatchAndReturnOutputs(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return Matrix_CPU();

        // Combine inputs into a batch matrix
        Matrix_CPU inputs(nn_topology[0], batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                inputs.host_data[j * batchSize + i] = batchInputs[i][j];
            }
        }

        // FeedForward Step
        Matrix_CPU hidden = Matrix_CPU::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        Matrix_CPU hidden2 = Matrix_CPU::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();

        Matrix_CPU outputs = Matrix_CPU::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix_CPU::Softmax(outputs, q);

        // Combine targets into a batch matrix
        Matrix_CPU targets(nn_topology.back(), batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                targets.host_data[j * batchSize + i] = batchOutputs[i][j];
            }
        }

        // Output errors = outputs - targets (matrix subtraction)
        Matrix_CPU output_errors = Matrix_CPU::subtract(outputs, targets, q);
        output_errors.multiplyScalar(1.0f / batchSize);

        // Calculate the gradients for output layer
        Matrix_CPU gradients = output_errors;
        gradients.multiplyScalar(-learningRate);

        // Update weights and biases for hidden2 -> output layer
        Matrix_CPU weight_h2o_deltas = Matrix_CPU::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));

        // Hidden layer 2 errors
        Matrix_CPU hidden2_errors = Matrix_CPU::multiply(weightMatrices[2].Transpose(), output_errors, q);

        // Calculate the gradients for hidden layer 2
        Matrix_CPU hidden2_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-learningRate);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix_CPU weight_h1h2_deltas = Matrix_CPU::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        // Hidden layer 1 errors
        Matrix_CPU hidden_errors = Matrix_CPU::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        // Calculate the gradients for hidden layer 1
        Matrix_CPU hidden_gradient = Matrix_CPU::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-learningRate);

        // Update weights and biases for input -> hidden1 layer
        Matrix_CPU weight_ih1_deltas = Matrix_CPU::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));

        // Ensure all operations complete before returning
        q.wait();
        
        return outputs; // Return the outputs matrix
    }

    // CPU-specific optimization methods
    void enableCPUOptimizations() {
        // Set CPU-specific environment variables if needed
        // This could be used to set thread counts, etc.
        auto device = q.get_device();
        if (device.is_cpu()) {
            std::cout << "CPU optimizations enabled for device: " 
                      << device.get_info<sycl::info::device::name>() << std::endl;
        }
    }

    // Performance timing utilities for benchmarking
    struct TimingResult {
        double feedforward_time;
        double backprop_time;
        double total_time;
    };

    TimingResult timedTraining(const vector<float>& inputs, const vector<float>& targets) {
        TimingResult result;
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Time feedforward
        auto start_ff = std::chrono::high_resolution_clock::now();
        auto outputs = FeedForward(inputs);
        auto end_ff = std::chrono::high_resolution_clock::now();
        
        // Time backpropagation
        auto start_bp = std::chrono::high_resolution_clock::now();
        BackPropagate(inputs, targets);
        auto end_bp = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        result.feedforward_time = std::chrono::duration<double>(end_ff - start_ff).count();
        result.backprop_time = std::chrono::duration<double>(end_bp - start_bp).count();
        result.total_time = std::chrono::duration<double>(end_total - start_total).count();
        
        return result;
    }

    TimingResult timedBatchTraining(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchTargets) {
        TimingResult result;
        
        auto start_total = std::chrono::high_resolution_clock::now();
        BackPropagateBatch(batchInputs, batchTargets);
        auto end_total = std::chrono::high_resolution_clock::now();
        
        result.total_time = std::chrono::duration<double>(end_total - start_total).count();
        result.feedforward_time = 0.0; // Not separated in batch training
        result.backprop_time = result.total_time;
        
        return result;
    }

    // Method to get current loss for monitoring training
    float calculateLoss(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchTargets) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return 0.0f;

        // Combine inputs into a batch matrix
        Matrix_CPU inputs(nn_topology[0], batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                inputs.host_data[j * batchSize + i] = batchInputs[i][j];
            }
        }

        // Forward pass to get outputs
        Matrix_CPU hidden = Matrix_CPU::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        Matrix_CPU hidden2 = Matrix_CPU::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();

        Matrix_CPU outputs = Matrix_CPU::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix_CPU::Softmax(outputs, q);

        // Combine targets into a batch matrix
        Matrix_CPU targets(nn_topology.back(), batchSize, q);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                targets.host_data[j * batchSize + i] = batchTargets[i][j];
            }
        }

        // Calculate cross-entropy loss
        float loss = Matrix_CPU::crossEntropy(outputs, targets, q);
        
        q.wait(); // Ensure computation completes
        
        return loss;
    }
}; 