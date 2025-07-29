#include "BenchmarkComparison.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdint>
#include "ImageNew.h"

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
    config.num_samples = 10000;   // Full MNIST training dataset
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

// Run the benchmark using the Fashion-MNIST dataset
void runFashionComparison() {
    std::cout << "\n=== BASIC CPU vs GPU COMPARISON (Fashion-MNIST) ===" << std::endl;
    
    BenchmarkConfig config;
    config.input_size = 784;
    config.hidden1_size = 128;
    config.hidden2_size = 64;
    config.output_size = 10;
    config.num_epochs = 10;
    config.batch_size = 32;
    config.num_samples = 10000;
    config.learning_rate = 0.01f;
    config.verbose = true;
    config.dataset_name = "Fashion_MNIST_basic_comparison";
    config.use_fashion_mnist = true;  // key difference
    config.random_seed = 42;

    BenchmarkComparison benchmark(config);
    auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
    benchmark.printComparison(cpu_metrics, gpu_metrics);
}

// Basic comparison for CIFAR-10
void runCIFARComparison() {
    std::cout << "\n=== BASIC CPU vs GPU COMPARISON (CIFAR-10) ===" << std::endl;

    BenchmarkConfig config;
    config.input_size = 3072;  // 32x32x3
    config.hidden1_size = 512;
    config.hidden2_size = 256;
    config.output_size = 10;
    config.num_epochs = 10;
    config.batch_size = 64;
    config.num_samples = 10000;
    config.learning_rate = 0.01f;
    config.verbose = true;
    config.dataset_name = "CIFAR10_basic_comparison";
    config.use_cifar10 = true;
    config.random_seed = 42;

    BenchmarkComparison benchmark(config);
    auto [cpu_metrics, gpu_metrics] = benchmark.runComparison();
    benchmark.printComparison(cpu_metrics, gpu_metrics);
}

// Network size analysis helper for arbitrary dataset
void runNetworkSizeAnalysisGeneric(const std::string& tag, int input_size, bool use_fashion, bool use_cifar) {
    std::cout << "\n=== NETWORK SIZE ANALYSIS (" << tag << ") ===" << std::endl;

    struct NetCfg { std::string name; int h1; int h2; std::string desc; };
    std::vector<NetCfg> nets = {
        {"tiny", 32, 16, "Tiny network - CPU should excel"},
        {"small", 64, 32, "Small network - competitive performance"},
        {"medium", 128, 64, "Medium network - GPU starts to win"},
        {"large", 256, 128, "Large network - GPU should dominate"},
        {"xlarge", 512, 256, "Extra large - GPU advantage clear"}
    };

    for (const auto& n : nets) {
        std::cout << "\n--- " << n.name << " : " << n.desc << " ---" << std::endl;

        BenchmarkConfig cfg;
        cfg.input_size = input_size;
        cfg.hidden1_size = n.h1;
        cfg.hidden2_size = n.h2;
        cfg.output_size = 10;
        cfg.num_epochs = 5;
        cfg.batch_size = 32;
        cfg.num_samples = 10000;
        cfg.learning_rate = 0.01f;
        cfg.verbose = false;
        cfg.dataset_name = tag + "_" + n.name;
        cfg.use_fashion_mnist = use_fashion;
        cfg.use_cifar10      = use_cifar;
        cfg.random_seed = 42;

        BenchmarkComparison bench(cfg);
        auto [cpu_metrics, gpu_metrics] = bench.runComparison();
        bench.printComparison(cpu_metrics, gpu_metrics);
    }
}

// Batch size analysis generic
void runBatchSizeAnalysisGeneric(const std::string& tag, int input_size, bool use_fashion, bool use_cifar) {
    std::cout << "\n=== BATCH SIZE ANALYSIS (" << tag << ") ===" << std::endl;

    std::vector<int> batch_sizes = {16, 32, 64, 128, 256, 512, 1024};
    for (int bs : batch_sizes) {
        std::cout << "\n--- Batch size: " << bs << " ---" << std::endl;
        BenchmarkConfig cfg;
        cfg.input_size = input_size;
        cfg.hidden1_size = 128;
        cfg.hidden2_size = 64;
        cfg.output_size = 10;
        cfg.num_epochs = 3;
        cfg.batch_size = bs;
        cfg.num_samples = 10000;
        cfg.learning_rate = 0.01f;
        cfg.verbose = false;
        cfg.dataset_name = tag + "_batch_" + std::to_string(bs);
        cfg.use_fashion_mnist = use_fashion;
        cfg.use_cifar10       = use_cifar;
        cfg.random_seed = 42;

        BenchmarkComparison bench(cfg);
        auto [cpu_m, gpu_m] = bench.runComparison();
        bench.printComparison(cpu_m, gpu_m);
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
        
        // --- MNIST ---
        runBasicComparison();
        runNetworkSizeAnalysisGeneric("MNIST", 784, false, false);
        runBatchSizeAnalysisGeneric("MNIST", 784, false, false);

        // --- Fashion-MNIST ---
        runFashionComparison();
        runNetworkSizeAnalysisGeneric("FashionMNIST", 784, true, false);
        runBatchSizeAnalysisGeneric("FashionMNIST", 784, true, false);

        // --- CIFAR-10 ---
        runCIFARComparison();
        runNetworkSizeAnalysisGeneric("CIFAR10", 3072, false, true);
        runBatchSizeAnalysisGeneric("CIFAR10", 3072, false, true);

        std::cout << "\n=== BENCHMARK COMPLETE ===" << std::endl;
        std::cout << "Redirect this output to a TXT file if desired." << std::endl;
        
    } catch (const std::exception& e) {
        
        std::cerr << "Error during benchmark execution: " << e.what() << std::endl;
        std::rethrow_exception(std::current_exception());
        return 1;
    }
    
    return 0;
} 