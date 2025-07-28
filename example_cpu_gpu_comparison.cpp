#include "BenchmarkComparison.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

/**
 * SYCL Neural Network CPU vs GPU Benchmark Example - REAL MNIST DATA
 * 
 * This example demonstrates how to use the CPU-optimized SYCL neural network
 * implementation as a baseline for comparing against GPU performance using
 * the actual MNIST dataset (28x28 grayscale handwritten digits).
 * 
 * Key Features:
 * - Uses REAL MNIST dataset (not synthetic data!)
 * - CPU-optimized SYCL implementation using buffers/accessors instead of USM
 * - GPU implementation using USM for device memory management
 * - Comprehensive performance metrics including throughput, latency, and accuracy
 * - Multiple benchmark configurations for thorough analysis
 * - CSV output for further analysis and paper writing
 */

void runBasicComparison() {
    std::cout << "=== BASIC CPU vs GPU COMPARISON ===" << std::endl;
    
    // Configure benchmark parameters
    BenchmarkConfig config;
    config.input_size = 784;      // MNIST-like input size
    config.hidden1_size = 128;    // First hidden layer
    config.hidden2_size = 64;     // Second hidden layer  
    config.output_size = 10;      // Classification classes
    config.num_epochs = 10;       // Training epochs
    config.batch_size = 32;       // Batch size for training
    config.num_samples = 60000;   // Full MNIST training dataset
    config.learning_rate = 0.01f; // Learning rate
    config.verbose = true;        // Print detailed progress
    config.dataset_name = "MNIST_basic_comparison";
    config.random_seed = 42;      // Fixed seed for reproducibility
    
    // Create benchmark instance
    BenchmarkComparison benchmark(config);
    
    // Run the comparison
    auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
    
    // Print detailed comparison
    benchmark.printComparison(cpu_metrics, gpu_metrics);
    
    // Save results to CSV for further analysis
    benchmark.saveResultsToCSV(cpu_metrics, gpu_metrics, "basic_comparison.csv");
    
    std::cout << "\n=== ANALYSIS NOTES ===" << std::endl;
    std::cout << "1. Using REAL MNIST dataset (28x28 grayscale images, 10 classes)" << std::endl;
    std::cout << "2. CPU implementation uses buffer/accessor model optimized for host memory" << std::endl;
    std::cout << "3. GPU implementation uses USM for device memory management" << std::endl;
    std::cout << "4. Performance differences depend on problem size and hardware capabilities" << std::endl;
    std::cout << "5. CPU may outperform GPU for small problems due to lower overhead" << std::endl;
    std::cout << "6. GPU typically excels with larger batch sizes and network dimensions" << std::endl;
}

void runNetworkSizeAnalysis() {
    std::cout << "\n=== NETWORK SIZE ANALYSIS ===" << std::endl;
    std::cout << "Testing different network architectures to find CPU vs GPU crossover points" << std::endl;
    
    struct NetworkConfig {
        std::string name;
        int hidden1;
        int hidden2;
        std::string description;
    };
    
    std::vector<NetworkConfig> networks = {
        {"tiny", 32, 16, "Tiny network - CPU should excel"},
        {"small", 64, 32, "Small network - competitive performance"},
        {"medium", 128, 64, "Medium network - GPU starts to win"},
        {"large", 256, 128, "Large network - GPU should dominate"},
        {"xlarge", 512, 256, "Extra large - GPU advantage clear"}
    };
    
    for (const auto& net : networks) {
        std::cout << "\n--- Testing " << net.name << " network: " << net.description << " ---" << std::endl;
        
        BenchmarkConfig config;
        config.input_size = 784;
        config.hidden1_size = net.hidden1;
        config.hidden2_size = net.hidden2;
        config.output_size = 10;
        config.num_epochs = 5;
        config.batch_size = 32;
        config.num_samples = 60000;  // Full MNIST dataset
        config.learning_rate = 0.01f;
        config.verbose = false;  // Reduce verbosity for multiple runs
        config.dataset_name = net.name + "_network";
        config.random_seed = 42;     // Fixed seed for reproducibility
        
        BenchmarkComparison benchmark(config);
        auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
        
        // Quick summary
        double speedup = gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec;
        std::cout << "Network: " << net.name 
                  << " | GPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
        
        if (speedup > 1.0) {
            std::cout << " (GPU faster)";
        } else {
            std::cout << " (CPU faster)";
        }
        std::cout << std::endl;
        
        // Save detailed results
        benchmark.saveResultsToCSV(cpu_metrics, gpu_metrics, 
                                   "network_size_" + net.name + ".csv");
    }
}

void runBatchSizeAnalysis() {
    std::cout << "\n=== BATCH SIZE ANALYSIS ===" << std::endl;
    std::cout << "Testing different batch sizes to understand parallelization efficiency" << std::endl;
    
    std::vector<int> batch_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048};
    
    for (int batch_size : batch_sizes) {
        std::cout << "\n--- Testing batch size: " << batch_size << " ---" << std::endl;
        
        BenchmarkConfig config;
        config.input_size = 784;
        config.hidden1_size = 128;
        config.hidden2_size = 64;
        config.output_size = 10;
        config.num_epochs = 3;
        config.batch_size = batch_size;
        config.num_samples = 60000;  // Full MNIST dataset
        config.learning_rate = 0.01f;
        config.verbose = false;
        config.dataset_name = "batch_" + std::to_string(batch_size);
        config.random_seed = 42;     // Fixed seed for reproducibility
        
        BenchmarkComparison benchmark(config);
        auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
        
        // Quick summary focusing on throughput
        std::cout << "Batch " << std::setw(3) << batch_size 
                  << " | CPU: " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << cpu_metrics.throughput_samples_per_sec << " samp/s"
                  << " | GPU: " << std::setw(8) << gpu_metrics.throughput_samples_per_sec << " samp/s"
                  << " | Ratio: " << std::setw(6) << std::setprecision(2) 
                  << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) << "x"
                  << std::endl;
        
        benchmark.saveResultsToCSV(cpu_metrics, gpu_metrics, 
                                   "batch_analysis_" + std::to_string(batch_size) + ".csv");
    }
}

void runMemoryUsageAnalysis() {
    std::cout << "\n=== MEMORY USAGE ANALYSIS ===" << std::endl;
    std::cout << "Analyzing memory efficiency of CPU vs GPU implementations" << std::endl;
    
    struct TestCase {
        std::string name;
        int samples;
        int batch_size;
    };
    
    std::vector<TestCase> cases = {
        {"small_dataset", 500, 16},
        {"medium_dataset", 2000, 32},
        {"large_dataset", 5000, 64},
        {"xlarge_dataset", 10000, 128}
    };
    
    for (const auto& test_case : cases) {
        std::cout << "\n--- Testing " << test_case.name << " ---" << std::endl;
        
        BenchmarkConfig config;
        config.input_size = 784;
        config.hidden1_size = 128;
        config.hidden2_size = 64;
        config.output_size = 10;
        config.num_epochs = 3;
        config.batch_size = test_case.batch_size;
        config.num_samples = test_case.samples;
        config.learning_rate = 0.01f;
        config.verbose = false;
        config.dataset_name = test_case.name;
        config.random_seed = 42;     // Fixed seed for reproducibility
        
        BenchmarkComparison benchmark(config);
        auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
        
        std::cout << "Dataset: " << test_case.name << std::endl;
        std::cout << "  CPU Memory: " << std::fixed << std::setprecision(1) 
                  << cpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  GPU Memory: " << gpu_metrics.memory_usage_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Memory Ratio (GPU/CPU): " << std::setprecision(2) 
                  << (double)gpu_metrics.memory_usage_bytes / cpu_metrics.memory_usage_bytes << "x" << std::endl;
        std::cout << "  Performance Ratio (GPU/CPU): " 
                  << (gpu_metrics.throughput_samples_per_sec / cpu_metrics.throughput_samples_per_sec) << "x" << std::endl;
        
        benchmark.saveResultsToCSV(cpu_metrics, gpu_metrics, 
                                   "memory_analysis_" + test_case.name + ".csv");
    }
}

void runComprehensiveAnalysis() {
    std::cout << "\n=== COMPREHENSIVE ANALYSIS FOR RESEARCH PAPER ===" << std::endl;
    std::cout << "Running full suite of benchmarks for academic publication" << std::endl;
    
    BenchmarkComparison benchmark;
    benchmark.runComprehensiveAnalysis();
    
    std::cout << "\n=== RESEARCH INSIGHTS ===" << std::endl;
    std::cout << "1. **CPU Implementation Strategy**:" << std::endl;
    std::cout << "   - Uses SYCL buffers/accessors for better CPU cache utilization" << std::endl;
    std::cout << "   - Optimized work group sizes (64) for CPU architecture" << std::endl;
    std::cout << "   - Host-side computation for small operations (< 10K elements)" << std::endl;
    std::cout << "   - Blocked transpose operations for cache efficiency" << std::endl;
    
    std::cout << "\n2. **GPU Implementation Strategy**:" << std::endl;
    std::cout << "   - Uses USM device allocations for minimal host-device transfers" << std::endl;
    std::cout << "   - oneMKL GEMM for optimized matrix multiplications" << std::endl;
    std::cout << "   - Larger work group sizes optimized for GPU architecture" << std::endl;
    std::cout << "   - SYCL kernels for element-wise operations" << std::endl;
    
    std::cout << "\n3. **Performance Characteristics**:" << std::endl;
    std::cout << "   - CPU excels for small networks and batch sizes" << std::endl;
    std::cout << "   - GPU advantages emerge with larger parallelism" << std::endl;
    std::cout << "   - Memory bandwidth is key limiting factor" << std::endl;
    std::cout << "   - SYCL provides good portability with reasonable performance" << std::endl;
    
    std::cout << "\n4. **Recommended Usage**:" << std::endl;
    std::cout << "   - Use CPU implementation for prototyping and small-scale experiments" << std::endl;
    std::cout << "   - Use GPU implementation for production and large-scale training" << std::endl;
    std::cout << "   - Consider hybrid approaches for different training phases" << std::endl;
}

void demonstrateDeviceInformation() {
    std::cout << "\n=== DEVICE INFORMATION ===" << std::endl;
    std::cout << "Gathering detailed information about available compute devices" << std::endl;
    
    try {
        // CPU Network
        NeuralNetwork_CPU cpu_network(784, 128, 64, 10);
        std::cout << "\nCPU Implementation:" << std::endl;
        cpu_network.printDeviceInfo();
        
        // GPU Network  
        NeuralNetwork gpu_network(784, 128, 64, 10);
        std::cout << "\nGPU Implementation:" << std::endl;
        gpu_network.printDeviceInfo();
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing networks: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "SYCL Neural Network CPU vs GPU Baseline Comparison - MNIST Dataset" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "This program demonstrates CPU-optimized SYCL neural network" << std::endl;
    std::cout << "implementations using REAL MNIST data for baseline comparisons." << std::endl;
    
    try {
        // Show available devices
        demonstrateDeviceInformation();
        
        // Run basic comparison
        runBasicComparison();
        
        // Run detailed analyses
        runNetworkSizeAnalysis();
        runBatchSizeAnalysis();
        runMemoryUsageAnalysis();
        
        // Run comprehensive analysis for research
        runComprehensiveAnalysis();
        
        std::cout << "\n=== BENCHMARK COMPLETE ===" << std::endl;
        std::cout << "All results have been saved to CSV files for further analysis." << std::endl;
        std::cout << "Use these results as baselines for your research paper." << std::endl;
        
        std::cout << "\n=== FILES GENERATED ===" << std::endl;
        std::cout << "- basic_comparison.csv: Basic CPU vs GPU metrics" << std::endl;
        std::cout << "- network_size_*.csv: Network architecture analysis" << std::endl;
        std::cout << "- batch_analysis_*.csv: Batch size performance study" << std::endl;
        std::cout << "- memory_analysis_*.csv: Memory usage comparison" << std::endl;
        std::cout << "- results_*.csv: Comprehensive benchmark suite" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 