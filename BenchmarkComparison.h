#pragma once
#include "NeuralNetworkNew.h"           // GPU version
#include "NeuralNetworkNew_CPU.h"       // CPU version
#include "ImageNew.h"                   // For ImageData structure
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <map>
#include <random>
#include <algorithm>

using namespace std;

struct BenchmarkConfig {
    int input_size = 784;    // MNIST 28x28
    int hidden1_size = 128;
    int hidden2_size = 64;
    int output_size = 10;
    int num_epochs = 10;
    int batch_size = 32;
    int num_samples = 60000; // Full MNIST training dataset (60,000 images)
    float learning_rate = 0.01f;
    bool verbose = true;
    string dataset_name = "MNIST";
    bool use_fashion_mnist = false;  // true for Fashion-MNIST, false for regular MNIST
    unsigned int random_seed = 42;   // Fixed seed for reproducible results

};
class BenchmarkComparison {
public:
    struct PerformanceMetrics {
        double total_time;
        double avg_feedforward_time;
        double avg_backprop_time;
        double throughput_samples_per_sec;
        double accuracy;
        double final_loss;
        size_t memory_usage_bytes;
        string device_name;
        string device_type;
        int max_compute_units;
        size_t max_work_group_size;
    };

    

private:
    BenchmarkConfig config;
    
    // MNIST file paths - same as used in MNIST_experiment.cpp
    vector<string> MNISTfileVec{
        "C:\\DEV\\Datasets\\train-images.idx3-ubyte",  // Training images
        "C:\\DEV\\Datasets\\train-labels.idx1-ubyte",  // Training labels
        "C:\\DEV\\Datasets\\t10k-images.idx3-ubyte",   // Testing images
        "C:\\DEV\\Datasets\\t10k-labels.idx1-ubyte"    // Testing labels
    };

    vector<string> FashionMNISTfileVec{
        "C:\\DEV\\Datasets\\train-images-idx3-ubyte",  // Training images
        "C:\\DEV\\Datasets\\train-labels-idx1-ubyte",  // Training labels
        "C:\\DEV\\Datasets\\t10k-images-idx3-ubyte",   // Testing images
        "C:\\DEV\\Datasets\\t10k-labels-idx1-ubyte"    // Testing labels
    };
    
    // Load real MNIST dataset using the same approach as MNIST_experiment.cpp
    pair<vector<vector<float>>, vector<vector<float>>> loadMNISTDataset() {
        vector<vector<float>> inputs;
        vector<vector<float>> targets;
        
        // Choose which dataset to load
        vector<string>& fileVec = config.use_fashion_mnist ? FashionMNISTfileVec : MNISTfileVec;
        
        // Load training data
        ifstream imageFile(fileVec[0], ios::binary);
        ifstream labelFile(fileVec[1], ios::binary);
        
        if (!imageFile.is_open() || !labelFile.is_open()) {
            cerr << "ERROR: Cannot open MNIST files!" << std::endl;
            cerr << "Image file: " << fileVec[0] << std::endl;
            cerr << "Label file: " << fileVec[1] << std::endl;
            cerr << "Make sure MNIST dataset files are in the correct location." << std::endl;
            throw runtime_error("MNIST files not found");
        }
        
        // Skip headers (16 bytes for images, 8 bytes for labels)
        imageFile.seekg(16);
        labelFile.seekg(8);
        
        const int imageSize = 28 * 28;
        vector<unsigned char> imageBuffer(imageSize);
        unsigned char label;
        
        int samplesLoaded = 0;
        while (imageFile.read(reinterpret_cast<char*>(imageBuffer.data()), imageSize) &&
               labelFile.read(reinterpret_cast<char*>(&label), 1) &&
               samplesLoaded < config.num_samples) {
            
            // Normalize image data from 0-255 to 0-1
            vector<float> normalizedImage;
            normalizedImage.reserve(imageSize);
            for (unsigned char pix : imageBuffer) {
                normalizedImage.push_back(pix / 255.0f);
            }
            inputs.push_back(normalizedImage);
            
            // Create one-hot encoded target
            vector<float> target(config.output_size, 0.0f);
            target[static_cast<int>(label)] = 1.0f;
            targets.push_back(target);
            
            samplesLoaded++;
        }
        
        imageFile.close();
        labelFile.close();
        
        // Shuffle the data for better training using fixed seed for reproducibility
        vector<size_t> indices(inputs.size());
        iota(indices.begin(), indices.end(), 0);
        
        mt19937 gen(config.random_seed);  // Use fixed seed from config
        shuffle(indices.begin(), indices.end(), gen);
        
        vector<vector<float>> shuffled_inputs, shuffled_targets;
        for (size_t idx : indices) {
            shuffled_inputs.push_back(inputs[idx]);
            shuffled_targets.push_back(targets[idx]);
        }
        
        if (config.verbose) {
            cout << "Loaded " << samplesLoaded << " MNIST samples from "
                 << (config.use_fashion_mnist ? "Fashion-MNIST" : "MNIST") << " dataset" << std::endl;
        }
        
        return make_pair(shuffled_inputs, shuffled_targets);
    }
    
    double calculateAccuracy(const vector<vector<float>>& predictions, 
                           const vector<vector<float>>& targets) {
        int correct = 0;
        int total = predictions.size();
        
        for (size_t i = 0; i < predictions.size(); i++) {
            // Find predicted class (max value index)
            int pred_class = 0;
            for (int j = 1; j < config.output_size; j++) {
                if (predictions[i][j] > predictions[i][pred_class]) {
                    pred_class = j;
                }
            }
            
            // Find true class
            int true_class = 0;
            for (int j = 1; j < config.output_size; j++) {
                if (targets[i][j] > targets[i][true_class]) {
                    true_class = j;
                }
            }
            
            if (pred_class == true_class) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / total;
    }
    
    template<typename NetworkType>
    PerformanceMetrics benchmarkNetwork(NetworkType& network, const string& device_type_name) {
        PerformanceMetrics metrics = {};
        metrics.device_type = device_type_name;
        
        // Get device information
        auto device = network.getQueue().get_device();
        metrics.device_name = device.get_info<sycl::info::device::name>();
        metrics.max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
        metrics.max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
        
        if (config.verbose) {
            std::cout << "\n=== Benchmarking " << device_type_name << " Implementation ===" << std::endl;
            std::cout << "Device: " << metrics.device_name << std::endl;
            std::cout << "Max Compute Units: " << metrics.max_compute_units << std::endl;
            std::cout << "Max Work Group Size: " << metrics.max_work_group_size << std::endl;
        }
        
        // Load MNIST dataset
        auto [inputs, targets] = loadMNISTDataset();
        
        // Set learning rate
        network.setLearningRate(config.learning_rate);
        
        auto start_time = chrono::high_resolution_clock::now();
        
        double total_feedforward_time = 0.0;
        double total_backprop_time = 0.0;
        int total_batches = 0;
        
        // Training loop
        for (int epoch = 0; epoch < config.num_epochs; epoch++) {
            if (config.verbose) {
                std::cout << "Epoch " << (epoch + 1) << "/" << config.num_epochs << ": ";
            }
            
            for (int batch_start = 0; batch_start < config.num_samples; batch_start += config.batch_size) {
                int batch_end = min(batch_start + config.batch_size, config.num_samples);
                int actual_batch_size = batch_end - batch_start;
                
                // Create batch
                vector<vector<float>> batch_inputs(actual_batch_size);
                vector<vector<float>> batch_targets(actual_batch_size);
                
                for (int i = 0; i < actual_batch_size; i++) {
                    batch_inputs[i] = inputs[batch_start + i];
                    batch_targets[i] = targets[batch_start + i];
                }
                
                // Time the training step
                auto batch_start_time = chrono::high_resolution_clock::now();
                
                if constexpr (std::is_same_v<NetworkType, NeuralNetwork_CPU>) {
                    auto timing = network.timedBatchTraining(batch_inputs, batch_targets);
                    total_feedforward_time += timing.feedforward_time;
                    total_backprop_time += timing.backprop_time;
                } else {
                    network.BackPropagateBatch(batch_inputs, batch_targets);
                    auto batch_end_time = chrono::high_resolution_clock::now();
                    double batch_time = chrono::duration<double>(batch_end_time - batch_start_time).count();
                    total_backprop_time += batch_time; // Combined time for GPU version
                }
                
                total_batches++;
            }
            
            // Calculate loss for this epoch (using a sample of data)
            int sample_size = min(config.batch_size * 4, config.num_samples);
            vector<vector<float>> sample_inputs(sample_size);
            vector<vector<float>> sample_targets(sample_size);
            
            for (int i = 0; i < sample_size; i++) {
                sample_inputs[i] = inputs[i];
                sample_targets[i] = targets[i];
            }
            
            float epoch_loss = 0.0f;
            // Both CPU and GPU versions now have calculateLoss implemented
            epoch_loss = network.calculateLoss(sample_inputs, sample_targets);
            
            if (config.verbose) {
                std::cout << "Loss = " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
            }
            
            metrics.final_loss = epoch_loss;
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        metrics.total_time = chrono::duration<double>(end_time - start_time).count();
        
        // Calculate derived metrics
        metrics.avg_feedforward_time = total_feedforward_time / total_batches;
        metrics.avg_backprop_time = total_backprop_time / total_batches;
        metrics.throughput_samples_per_sec = (config.num_samples * config.num_epochs) / metrics.total_time;
        
        // Calculate accuracy on test set
        vector<vector<float>> test_predictions;
        for (int i = 0; i < min(200, config.num_samples); i++) {
            auto prediction = network.FeedForward(inputs[i]);
            test_predictions.push_back(prediction);
        }
        
        vector<vector<float>> test_targets(test_predictions.size());
        for (size_t i = 0; i < test_predictions.size(); i++) {
            test_targets[i] = targets[i];
        }
        
        metrics.accuracy = calculateAccuracy(test_predictions, test_targets);
        
        // Estimate memory usage (simplified)
        size_t weights_memory = 0;
        weights_memory += config.input_size * config.hidden1_size * sizeof(float);
        weights_memory += config.hidden1_size * config.hidden2_size * sizeof(float);
        weights_memory += config.hidden2_size * config.output_size * sizeof(float);
        weights_memory += (config.hidden1_size + config.hidden2_size + config.output_size) * sizeof(float); // biases
        
        size_t activation_memory = config.batch_size * (config.input_size + config.hidden1_size + config.hidden2_size + config.output_size) * sizeof(float);
        
        metrics.memory_usage_bytes = weights_memory + activation_memory;
        
        return metrics;
    }

public:
    BenchmarkComparison(const BenchmarkConfig& cfg = {}) : config(cfg) {}
    
    void setConfig(const BenchmarkConfig& cfg) {
        config = cfg;
    }
    
    pair<PerformanceMetrics, PerformanceMetrics> runComparison() {
        std::cout << "=== SYCL Neural Network CPU vs GPU Benchmark ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Network: " << config.input_size << "-" << config.hidden1_size 
             << "-" << config.hidden2_size << "-" << config.output_size << std::endl;
        std::cout << "  Epochs: " << config.num_epochs << std::endl;
        std::cout << "  Batch Size: " << config.batch_size << std::endl;
        std::cout << "  Samples: " << config.num_samples << std::endl;
        std::cout << "  Learning Rate: " << config.learning_rate << std::endl;
        std::cout << "  Dataset: " << config.dataset_name << std::endl;
        std::cout << std::endl;
        
        // Create networks with same seed for reproducible comparison
        NeuralNetwork_CPU cpu_network(config.input_size, config.hidden1_size, 
                                     config.hidden2_size, config.output_size, config.random_seed);
        NeuralNetwork gpu_network(config.input_size, config.hidden1_size, 
                                config.hidden2_size, config.output_size, config.random_seed);
        
        // Benchmark CPU version
        auto cpu_metrics = benchmarkNetwork(cpu_network, "CPU");
        
        // Benchmark GPU version  
        auto gpu_metrics = benchmarkNetwork(gpu_network, "GPU");
        
        return make_pair(cpu_metrics, gpu_metrics);
    }
    
    void printComparison(const PerformanceMetrics& cpu_metrics, 
                        const PerformanceMetrics& gpu_metrics) {
        std::cout << "\n=== PERFORMANCE COMPARISON RESULTS ===" << std::endl;
        std::cout << left << std::setw(25) << "Metric" << std::setw(15) << "CPU" << std::setw(15) << "GPU" << std::setw(15) << "Ratio (GPU/CPU)" << std::endl;
        std::cout << string(70, '-') << std::endl;
        
        // Device Information
        std::cout << std::setw(25) << "Device Name:" << std::setw(15) << cpu_metrics.device_name.substr(0, 14) 
             << std::setw(15) << gpu_metrics.device_name.substr(0, 14) << std::setw(15) << "-" << std::endl;
        std::cout << std::setw(25) << "Compute Units:" << std::setw(15) << cpu_metrics.max_compute_units 
             << std::setw(15) << gpu_metrics.max_compute_units 
             << std::setw(15) << std::fixed << std::setprecision(2) << (double)gpu_metrics.max_compute_units / cpu_metrics.max_compute_units << std::endl;
        
        // Performance Metrics
        std::cout << std::setw(25) << "Total Time (s):" << std::setw(15) << std::fixed << std::setprecision(3) << cpu_metrics.total_time 
             << std::setw(15) << gpu_metrics.total_time 
             << std::setw(15) << (cpu_metrics.total_time / gpu_metrics.total_time) << "x" << std::endl;
        
        std::cout << std::setw(25) << "Throughput (samp/s):" << std::setw(15) << std::fixed << std::setprecision(1) << cpu_metrics.throughput_samples_per_sec 
             << std::setw(15) << gpu_metrics.throughput_samples_per_sec 
             << std::setw(15) << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) << "x" << std::endl;
        
        if (cpu_metrics.avg_feedforward_time > 0) {
            std::cout << std::setw(25) << "Avg FeedForward (ms):" << std::setw(15) << std::fixed << std::setprecision(3) << cpu_metrics.avg_feedforward_time * 1000 
                 << std::setw(15) << gpu_metrics.avg_feedforward_time * 1000 
                 << std::setw(15) << (cpu_metrics.avg_feedforward_time / gpu_metrics.avg_feedforward_time) << "x" << std::endl;
        }
        
        std::cout << std::setw(25) << "Avg BackProp (ms):" << std::setw(15) << std::fixed << std::setprecision(3) << cpu_metrics.avg_backprop_time * 1000 
             << std::setw(15) << gpu_metrics.avg_backprop_time * 1000 
             << std::setw(15) << (cpu_metrics.avg_backprop_time / gpu_metrics.avg_backprop_time) << "x" << std::endl;
        
        // Memory Usage
        std::cout << std::setw(25) << "Memory Usage (MB):" << std::setw(15) << std::fixed << std::setprecision(1) << cpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) 
             << std::setw(15) << gpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) 
             << std::setw(15) << (double)gpu_metrics.memory_usage_bytes / cpu_metrics.memory_usage_bytes << "x" << std::endl;
        
        // Quality Metrics
        std::cout << std::setw(25) << "Final Accuracy (%):" << std::setw(15) << std::fixed << std::setprecision(1) << cpu_metrics.accuracy * 100 
             << std::setw(15) << gpu_metrics.accuracy * 100 
             << std::setw(15) << (gpu_metrics.accuracy / cpu_metrics.accuracy) << "x" << std::endl;
        
        std::cout << std::setw(25) << "Final Loss:" << std::setw(15) << std::fixed << std::setprecision(4) << cpu_metrics.final_loss 
             << std::setw(15) << gpu_metrics.final_loss 
             << std::setw(15) << (cpu_metrics.final_loss / gpu_metrics.final_loss) << "x" << std::endl;
        
        std::cout << std::endl;
        
        // Summary
        std::cout << "=== SUMMARY ===" << std::endl;
        if (gpu_metrics.throughput_samples_per_sec > cpu_metrics.throughput_samples_per_sec) {
            std::cout << "GPU is " << std::fixed << std::setprecision(2) 
                 << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) 
                 << "x faster than CPU in throughput." << std::endl;
        } else {
            std::cout << "CPU is " << std::fixed << std::setprecision(2) 
                 << (cpu_metrics.throughput_samples_per_sec / gpu_metrics.throughput_samples_per_sec) 
                 << "x faster than GPU in throughput." << std::endl;
        }
        
        if (gpu_metrics.total_time < cpu_metrics.total_time) {
            std::cout << "GPU completed training " << std::fixed << std::setprecision(2) 
                 << (cpu_metrics.total_time / gpu_metrics.total_time) 
                 << "x faster than CPU." << std::endl;
        } else {
            std::cout << "CPU completed training " << std::fixed << std::setprecision(2) 
                 << (gpu_metrics.total_time / cpu_metrics.total_time) 
                 << "x faster than GPU." << std::endl;
        }
    }
    
    void saveResultsToCSV(const PerformanceMetrics& cpu_metrics, 
                         const PerformanceMetrics& gpu_metrics, 
                         const string& filename = "benchmark_results.csv") {
        ofstream file(filename);
        
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        // CSV Header
        file << "Device_Type,Device_Name,Total_Time_s,Throughput_samples_per_s,Avg_FeedForward_ms,Avg_BackProp_ms,Memory_MB,Accuracy_percent,Final_Loss,Max_Compute_Units,Max_Work_Group_Size" << std::endl;
        
        // CPU Results
        file << "CPU," << cpu_metrics.device_name << "," 
             << cpu_metrics.total_time << "," 
             << cpu_metrics.throughput_samples_per_sec << ","
             << cpu_metrics.avg_feedforward_time * 1000 << ","
             << cpu_metrics.avg_backprop_time * 1000 << ","
             << cpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) << ","
             << cpu_metrics.accuracy * 100 << ","
             << cpu_metrics.final_loss << ","
             << cpu_metrics.max_compute_units << ","
             << cpu_metrics.max_work_group_size << std::endl;
        
        // GPU Results
        file << "GPU," << gpu_metrics.device_name << "," 
             << gpu_metrics.total_time << "," 
             << gpu_metrics.throughput_samples_per_sec << ","
             << gpu_metrics.avg_feedforward_time * 1000 << ","
             << gpu_metrics.avg_backprop_time * 1000 << ","
             << gpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) << ","
             << gpu_metrics.accuracy * 100 << ","
             << gpu_metrics.final_loss << ","
             << gpu_metrics.max_compute_units << ","
             << gpu_metrics.max_work_group_size << std::endl;
        
        file.close();
        std::cout << "Results saved to " << filename << std::endl;
    }
    
    // Run multiple benchmark configurations for comprehensive analysis
    void runComprehensiveAnalysis() {
        std::cout << "=== COMPREHENSIVE BENCHMARK ANALYSIS ===" << std::endl;
        
        vector<BenchmarkConfig> configs = {};
        
        // Small network - Full dataset
        BenchmarkConfig small_config;
        small_config.input_size = 784; small_config.hidden1_size = 64; small_config.hidden2_size = 32;
        small_config.output_size = 10; small_config.num_epochs = 5; small_config.batch_size = 16;
        small_config.num_samples = 60000; small_config.learning_rate = 0.01f; small_config.verbose = false;
        small_config.dataset_name = "MNIST_small_net"; small_config.use_fashion_mnist = false; small_config.random_seed = 42;
        configs.push_back(small_config);
        
        // Medium network - Full dataset
        BenchmarkConfig medium_config;
        medium_config.input_size = 784; medium_config.hidden1_size = 128; medium_config.hidden2_size = 64;
        medium_config.output_size = 10; medium_config.num_epochs = 10; medium_config.batch_size = 32;
        medium_config.num_samples = 60000; medium_config.learning_rate = 0.01f; medium_config.verbose = false;
        medium_config.dataset_name = "MNIST_medium_net"; medium_config.use_fashion_mnist = false; medium_config.random_seed = 42;
        configs.push_back(medium_config);
        
        // Large network - Full dataset
        BenchmarkConfig large_config;
        large_config.input_size = 784; large_config.hidden1_size = 256; large_config.hidden2_size = 128;
        large_config.output_size = 10; large_config.num_epochs = 10; large_config.batch_size = 64;
        large_config.num_samples = 60000; large_config.learning_rate = 0.01f; large_config.verbose = false;
        large_config.dataset_name = "MNIST_large_net"; large_config.use_fashion_mnist = false; large_config.random_seed = 42;
        configs.push_back(large_config);
        
        // Different batch sizes - Full dataset
        BenchmarkConfig batch8_config;
        batch8_config.input_size = 784; batch8_config.hidden1_size = 128; batch8_config.hidden2_size = 64;
        batch8_config.output_size = 10; batch8_config.num_epochs = 5; batch8_config.batch_size = 8;
        batch8_config.num_samples = 60000; batch8_config.learning_rate = 0.01f; batch8_config.verbose = false;
        batch8_config.dataset_name = "MNIST_batch_8"; batch8_config.use_fashion_mnist = false; batch8_config.random_seed = 42;
        configs.push_back(batch8_config);
        
        BenchmarkConfig batch128_config;
        batch128_config.input_size = 784; batch128_config.hidden1_size = 128; batch128_config.hidden2_size = 64;
        batch128_config.output_size = 10; batch128_config.num_epochs = 5; batch128_config.batch_size = 128;
        batch128_config.num_samples = 60000; batch128_config.learning_rate = 0.01f; batch128_config.verbose = false;
        batch128_config.dataset_name = "MNIST_batch_128"; batch128_config.use_fashion_mnist = false; batch128_config.random_seed = 42;
        configs.push_back(batch128_config);
        
        // Fashion-MNIST comparison - Full dataset
        BenchmarkConfig fashion_config;
        fashion_config.input_size = 784; fashion_config.hidden1_size = 128; fashion_config.hidden2_size = 64;
        fashion_config.output_size = 10; fashion_config.num_epochs = 10; fashion_config.batch_size = 32;
        fashion_config.num_samples = 60000; fashion_config.learning_rate = 0.01f; fashion_config.verbose = false;
        fashion_config.dataset_name = "Fashion_MNIST_comparison"; fashion_config.use_fashion_mnist = true; fashion_config.random_seed = 42;
        configs.push_back(fashion_config);
        
        for (const auto& cfg : configs) {
            std::cout << "\n--- Testing Configuration: " << cfg.dataset_name << " ---" << std::endl;
            setConfig(cfg);
            auto [cpu_metrics, gpu_metrics] = runComparison();
            printComparison(cpu_metrics, gpu_metrics);
            saveResultsToCSV(cpu_metrics, gpu_metrics, "results_" + cfg.dataset_name + ".csv");
        }
    }
}; 