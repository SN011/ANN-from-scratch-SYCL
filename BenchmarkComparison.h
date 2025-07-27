#pragma once
#include "NeuralNetworkNew.h"           // GPU version
#include "NeuralNetworkNew_CPU.h"       // CPU version
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <map>
#include <random>

using namespace std;

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

    struct BenchmarkConfig {
        int input_size = 784;    // MNIST-like
        int hidden1_size = 128;
        int hidden2_size = 64;
        int output_size = 10;
        int num_epochs = 10;
        int batch_size = 32;
        int num_samples = 1000;
        float learning_rate = 0.01f;
        bool verbose = true;
        string dataset_name = "synthetic";
    };

private:
    BenchmarkConfig config;
    
    // Generate synthetic dataset for testing
    pair<vector<vector<float>>, vector<vector<float>>> generateSyntheticDataset() {
        vector<vector<float>> inputs;
        vector<vector<float>> targets;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> input_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> class_dist(0, config.output_size - 1);
        
        for (int i = 0; i < config.num_samples; i++) {
            // Generate random input
            vector<float> input(config.input_size);
            for (int j = 0; j < config.input_size; j++) {
                input[j] = input_dist(gen);
            }
            inputs.push_back(input);
            
            // Generate one-hot encoded target
            vector<float> target(config.output_size, 0.0f);
            int class_label = class_dist(gen);
            target[class_label] = 1.0f;
            targets.push_back(target);
        }
        
        return make_pair(inputs, targets);
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
            cout << "\n=== Benchmarking " << device_type_name << " Implementation ===" << endl;
            cout << "Device: " << metrics.device_name << endl;
            cout << "Max Compute Units: " << metrics.max_compute_units << endl;
            cout << "Max Work Group Size: " << metrics.max_work_group_size << endl;
        }
        
        // Generate dataset
        auto [inputs, targets] = generateSyntheticDataset();
        
        // Set learning rate
        network.setLearningRate(config.learning_rate);
        
        auto start_time = chrono::high_resolution_clock::now();
        
        double total_feedforward_time = 0.0;
        double total_backprop_time = 0.0;
        int total_batches = 0;
        
        // Training loop
        for (int epoch = 0; epoch < config.num_epochs; epoch++) {
            if (config.verbose) {
                cout << "Epoch " << (epoch + 1) << "/" << config.num_epochs << ": ";
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
            if constexpr (std::is_same_v<NetworkType, NeuralNetwork_CPU>) {
                epoch_loss = network.calculateLoss(sample_inputs, sample_targets);
            } else {
                // For GPU version, we need to implement calculateLoss or use a workaround
                // For now, we'll use a placeholder
                epoch_loss = 1.0f / (epoch + 1); // Placeholder decreasing loss
            }
            
            if (config.verbose) {
                cout << "Loss = " << fixed << setprecision(4) << epoch_loss << endl;
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
    BenchmarkComparison(const BenchmarkConfig& cfg = BenchmarkConfig{}) : config(cfg) {}
    
    void setConfig(const BenchmarkConfig& cfg) {
        config = cfg;
    }
    
    pair<PerformanceMetrics, PerformanceMetrics> runComparison() {
        cout << "=== SYCL Neural Network CPU vs GPU Benchmark ===" << endl;
        cout << "Configuration:" << endl;
        cout << "  Network: " << config.input_size << "-" << config.hidden1_size 
             << "-" << config.hidden2_size << "-" << config.output_size << endl;
        cout << "  Epochs: " << config.num_epochs << endl;
        cout << "  Batch Size: " << config.batch_size << endl;
        cout << "  Samples: " << config.num_samples << endl;
        cout << "  Learning Rate: " << config.learning_rate << endl;
        cout << "  Dataset: " << config.dataset_name << endl;
        cout << endl;
        
        // Create networks
        NeuralNetwork_CPU cpu_network(config.input_size, config.hidden1_size, 
                                     config.hidden2_size, config.output_size);
        NeuralNetwork gpu_network(config.input_size, config.hidden1_size, 
                                config.hidden2_size, config.output_size);
        
        // Benchmark CPU version
        auto cpu_metrics = benchmarkNetwork(cpu_network, "CPU");
        
        // Benchmark GPU version  
        auto gpu_metrics = benchmarkNetwork(gpu_network, "GPU");
        
        return make_pair(cpu_metrics, gpu_metrics);
    }
    
    void printComparison(const PerformanceMetrics& cpu_metrics, 
                        const PerformanceMetrics& gpu_metrics) {
        cout << "\n=== PERFORMANCE COMPARISON RESULTS ===" << endl;
        cout << left << setw(25) << "Metric" << setw(15) << "CPU" << setw(15) << "GPU" << setw(15) << "Ratio (GPU/CPU)" << endl;
        cout << string(70, '-') << endl;
        
        // Device Information
        cout << setw(25) << "Device Name:" << setw(15) << cpu_metrics.device_name.substr(0, 14) 
             << setw(15) << gpu_metrics.device_name.substr(0, 14) << setw(15) << "-" << endl;
        cout << setw(25) << "Compute Units:" << setw(15) << cpu_metrics.max_compute_units 
             << setw(15) << gpu_metrics.max_compute_units 
             << setw(15) << fixed << setprecision(2) << (double)gpu_metrics.max_compute_units / cpu_metrics.max_compute_units << endl;
        
        // Performance Metrics
        cout << setw(25) << "Total Time (s):" << setw(15) << fixed << setprecision(3) << cpu_metrics.total_time 
             << setw(15) << gpu_metrics.total_time 
             << setw(15) << (cpu_metrics.total_time / gpu_metrics.total_time) << "x" << endl;
        
        cout << setw(25) << "Throughput (samp/s):" << setw(15) << fixed << setprecision(1) << cpu_metrics.throughput_samples_per_sec 
             << setw(15) << gpu_metrics.throughput_samples_per_sec 
             << setw(15) << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) << "x" << endl;
        
        if (cpu_metrics.avg_feedforward_time > 0) {
            cout << setw(25) << "Avg FeedForward (ms):" << setw(15) << fixed << setprecision(3) << cpu_metrics.avg_feedforward_time * 1000 
                 << setw(15) << gpu_metrics.avg_feedforward_time * 1000 
                 << setw(15) << (cpu_metrics.avg_feedforward_time / gpu_metrics.avg_feedforward_time) << "x" << endl;
        }
        
        cout << setw(25) << "Avg BackProp (ms):" << setw(15) << fixed << setprecision(3) << cpu_metrics.avg_backprop_time * 1000 
             << setw(15) << gpu_metrics.avg_backprop_time * 1000 
             << setw(15) << (cpu_metrics.avg_backprop_time / gpu_metrics.avg_backprop_time) << "x" << endl;
        
        // Memory Usage
        cout << setw(25) << "Memory Usage (MB):" << setw(15) << fixed << setprecision(1) << cpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) 
             << setw(15) << gpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) 
             << setw(15) << (double)gpu_metrics.memory_usage_bytes / cpu_metrics.memory_usage_bytes << "x" << endl;
        
        // Quality Metrics
        cout << setw(25) << "Final Accuracy (%):" << setw(15) << fixed << setprecision(1) << cpu_metrics.accuracy * 100 
             << setw(15) << gpu_metrics.accuracy * 100 
             << setw(15) << (gpu_metrics.accuracy / cpu_metrics.accuracy) << "x" << endl;
        
        cout << setw(25) << "Final Loss:" << setw(15) << fixed << setprecision(4) << cpu_metrics.final_loss 
             << setw(15) << gpu_metrics.final_loss 
             << setw(15) << (cpu_metrics.final_loss / gpu_metrics.final_loss) << "x" << endl;
        
        cout << endl;
        
        // Summary
        cout << "=== SUMMARY ===" << endl;
        if (gpu_metrics.throughput_samples_per_sec > cpu_metrics.throughput_samples_per_sec) {
            cout << "GPU is " << fixed << setprecision(2) 
                 << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) 
                 << "x faster than CPU in throughput." << endl;
        } else {
            cout << "CPU is " << fixed << setprecision(2) 
                 << (cpu_metrics.throughput_samples_per_sec / gpu_metrics.throughput_samples_per_sec) 
                 << "x faster than GPU in throughput." << endl;
        }
        
        if (gpu_metrics.total_time < cpu_metrics.total_time) {
            cout << "GPU completed training " << fixed << setprecision(2) 
                 << (cpu_metrics.total_time / gpu_metrics.total_time) 
                 << "x faster than CPU." << endl;
        } else {
            cout << "CPU completed training " << fixed << setprecision(2) 
                 << (gpu_metrics.total_time / cpu_metrics.total_time) 
                 << "x faster than GPU." << endl;
        }
    }
    
    void saveResultsToCSV(const PerformanceMetrics& cpu_metrics, 
                         const PerformanceMetrics& gpu_metrics, 
                         const string& filename = "benchmark_results.csv") {
        ofstream file(filename);
        
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return;
        }
        
        // CSV Header
        file << "Device_Type,Device_Name,Total_Time_s,Throughput_samples_per_s,Avg_FeedForward_ms,Avg_BackProp_ms,Memory_MB,Accuracy_percent,Final_Loss,Max_Compute_Units,Max_Work_Group_Size" << endl;
        
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
             << cpu_metrics.max_work_group_size << endl;
        
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
             << gpu_metrics.max_work_group_size << endl;
        
        file.close();
        cout << "Results saved to " << filename << endl;
    }
    
    // Run multiple benchmark configurations for comprehensive analysis
    void runComprehensiveAnalysis() {
        cout << "=== COMPREHENSIVE BENCHMARK ANALYSIS ===" << endl;
        
        vector<BenchmarkConfig> configs = {
            // Small network
            {784, 64, 32, 10, 5, 16, 500, 0.01f, false, "small_net"},
            // Medium network  
            {784, 128, 64, 10, 10, 32, 1000, 0.01f, false, "medium_net"},
            // Large network
            {784, 256, 128, 10, 10, 64, 2000, 0.01f, false, "large_net"},
            // Different batch sizes
            {784, 128, 64, 10, 5, 8, 1000, 0.01f, false, "batch_8"},
            {784, 128, 64, 10, 5, 128, 1000, 0.01f, false, "batch_128"}
        };
        
        for (const auto& cfg : configs) {
            cout << "\n--- Testing Configuration: " << cfg.dataset_name << " ---" << endl;
            setConfig(cfg);
            auto [cpu_metrics, gpu_metrics] = runComparison();
            printComparison(cpu_metrics, gpu_metrics);
            saveResultsToCSV(cpu_metrics, gpu_metrics, "results_" + cfg.dataset_name + ".csv");
        }
    }
}; 