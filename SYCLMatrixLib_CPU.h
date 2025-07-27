#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp> // For oneMKL's GEMM function
#include <limits> // Required for std::numeric_limits

using namespace std;
using namespace sycl;

class Matrix_CPU {
public:
    int rows;
    int cols;
    std::vector<float> host_data; // CPU-optimized: Keep data on host for better cache performance
    sycl::queue* q_ptr; // Pointer to the SYCL queue

    Matrix_CPU() : rows(0), cols(0), q_ptr(nullptr) {}

    Matrix_CPU(int r, int c, sycl::queue& q) : rows(r), cols(c), q_ptr(&q) {
        host_data.resize(rows * cols, 0.0f);
    }

    // Copy constructor 
    Matrix_CPU(const Matrix_CPU& other) : rows(other.rows), cols(other.cols), q_ptr(other.q_ptr) {
        host_data = other.host_data;
    }

    // Move constructor
    Matrix_CPU(Matrix_CPU&& other) noexcept 
        : rows(other.rows), cols(other.cols), host_data(std::move(other.host_data)), q_ptr(other.q_ptr) {
        other.rows = 0;
        other.cols = 0;
        other.q_ptr = nullptr;
    }

    // Copy assignment operator
    Matrix_CPU& operator=(const Matrix_CPU& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            q_ptr = other.q_ptr;
            host_data = other.host_data;
        }
        return *this;
    }

    // Move assignment operator
    Matrix_CPU& operator=(Matrix_CPU&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            host_data = std::move(other.host_data);
            q_ptr = other.q_ptr;

            other.rows = 0;
            other.cols = 0;
            other.q_ptr = nullptr;
        }
        return *this;
    }

    ~Matrix_CPU() = default; // No need for explicit cleanup with std::vector

    void RandInit(int fan_in = 0, int fan_out = 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit;
        if (fan_in > 0 && fan_out > 0) {
            limit = std::sqrt(6.0f / (fan_in + fan_out));
        }
        else {
            limit = 1.0f; // Default range if no fan_in/fan_out is provided
        }
        std::uniform_real_distribution<float> dist(-limit, limit);
        
        for (int i = 0; i < rows * cols; i++) {
            host_data[i] = dist(gen);
        }
    }

    void printMatrix() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << host_data[i * cols + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    void multiplyScalar(float n) {
        int total_size = rows * cols;
        
        // CPU-optimized: Use buffer with read_write access for in-place operations
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                // CPU-optimized work group size
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] *= n;
                        }
                    });
            });
        } // Buffer destructor ensures data is copied back to host_data
    }

    void addScalar(float n) {
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] += n;
                        }
                    });
            });
        }
    }

    void negate() {
        multiplyScalar(-1.0f);
    }

    Matrix_CPU Negate() const {
        Matrix_CPU result(rows, cols, *q_ptr);
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> in_buf(host_data.data(), sycl::range<1>(total_size));
            sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            out_acc[gid] = -in_acc[gid];
                        }
                    });
            });
        }
        
        return result;
    }

    void add(const Matrix_CPU& other) {
        if (rows == other.rows && cols == other.cols) {
            int total_size = rows * cols;
            
            {
                sycl::buffer<float, 1> this_buf(host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> other_buf(other.host_data.data(), sycl::range<1>(total_size));
                
                q_ptr->submit([&](sycl::handler& h) {
                    auto this_acc = this_buf.get_access<sycl::access::mode::read_write>(h);
                    auto other_acc = other_buf.get_access<sycl::access::mode::read>(h);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                this_acc[gid] += other_acc[gid];
                            }
                        });
                });
            }
        }
        // Handle bias addition (column vector to matrix)
        else if (rows == other.rows && other.cols == 1 && cols > 1) {
            int total_size = rows * cols;
            
            {
                sycl::buffer<float, 1> this_buf(host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> bias_buf(other.host_data.data(), sycl::range<1>(other.rows));
                
                q_ptr->submit([&](sycl::handler& h) {
                    auto this_acc = this_buf.get_access<sycl::access::mode::read_write>(h);
                    auto bias_acc = bias_buf.get_access<sycl::access::mode::read>(h);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                int row = gid / cols;
                                this_acc[gid] += bias_acc[row];
                            }
                        });
                });
            }
        }
        else {
            std::cout << "Dims of matrices not compatible for add(). No changes made.\n";
        }
    }

    static Matrix_CPU add(const Matrix_CPU& m1, const Matrix_CPU& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix_CPU result(m1.rows, m1.cols, q);
            int total_size = m1.rows * m1.cols;
            
            {
                sycl::buffer<float, 1> m1_buf(m1.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> m2_buf(m2.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> result_buf(result.host_data.data(), sycl::range<1>(total_size));
                
                q.submit([&](sycl::handler& h) {
                    auto m1_acc = m1_buf.get_access<sycl::access::mode::read>(h);
                    auto m2_acc = m2_buf.get_access<sycl::access::mode::read>(h);
                    auto result_acc = result_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                result_acc[gid] = m1_acc[gid] + m2_acc[gid];
                            }
                        });
                });
            }
            
            return result;
        }
        return Matrix_CPU(); // Return empty matrix if dimensions don't match
    }

    Matrix_CPU Add(const Matrix_CPU& other) const {
        return add(*this, other, *q_ptr);
    }

    static Matrix_CPU subtract(const Matrix_CPU& m1, const Matrix_CPU& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix_CPU result(m1.rows, m1.cols, q);
            int total_size = m1.rows * m1.cols;
            
            {
                sycl::buffer<float, 1> m1_buf(m1.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> m2_buf(m2.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> result_buf(result.host_data.data(), sycl::range<1>(total_size));
                
                q.submit([&](sycl::handler& h) {
                    auto m1_acc = m1_buf.get_access<sycl::access::mode::read>(h);
                    auto m2_acc = m2_buf.get_access<sycl::access::mode::read>(h);
                    auto result_acc = result_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                result_acc[gid] = m1_acc[gid] - m2_acc[gid];
                            }
                        });
                });
            }
            
            return result;
        }
        return Matrix_CPU(); // Return empty matrix if dimensions don't match
    }

    void multiply(const Matrix_CPU& other) {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix_CPU result(rows, other.cols, *q_ptr);
        
        // Use oneMKL GEMM with buffers for CPU optimization
        try {
            {
                sycl::buffer<float, 1> a_buf(host_data.data(), sycl::range<1>(rows * cols));
                sycl::buffer<float, 1> b_buf(other.host_data.data(), sycl::range<1>(other.rows * other.cols));
                sycl::buffer<float, 1> c_buf(result.host_data.data(), sycl::range<1>(result.rows * result.cols));
                
                oneapi::mkl::blas::row_major::gemm(
                    *q_ptr,
                    oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans,
                    rows,                    // m
                    other.cols,              // n
                    cols,                    // k
                    1.0f,                    // alpha
                    a_buf,                   // A buffer
                    cols,                    // lda
                    b_buf,                   // B buffer
                    other.cols,              // ldb
                    0.0f,                    // beta
                    c_buf,                   // C buffer
                    result.cols              // ldc
                );
                
                q_ptr->wait(); // Ensure computation completes
            } // Buffers go out of scope, data is copied back
            
            // Copy result back to this matrix
            host_data = std::move(result.host_data);
            cols = other.cols; // Update dimensions
            
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during CPU GEMM: " << e.what() << std::endl;
        }
    }

    static Matrix_CPU multiply(const Matrix_CPU& m1, const Matrix_CPU& m2, sycl::queue& q) {
        if (m1.cols != m2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix_CPU result(m1.rows, m2.cols, q);
        
        try {
            {
                sycl::buffer<float, 1> a_buf(m1.host_data.data(), sycl::range<1>(m1.rows * m1.cols));
                sycl::buffer<float, 1> b_buf(m2.host_data.data(), sycl::range<1>(m2.rows * m2.cols));
                sycl::buffer<float, 1> c_buf(result.host_data.data(), sycl::range<1>(result.rows * result.cols));
                
                oneapi::mkl::blas::row_major::gemm(
                    q,
                    oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans,
                    m1.rows,
                    m2.cols,
                    m1.cols,
                    1.0f,
                    a_buf,
                    m1.cols,
                    b_buf,
                    m2.cols,
                    0.0f,
                    c_buf,
                    result.cols
                );
                
                q.wait();
            }
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during static CPU GEMM: " << e.what() << std::endl;
        }

        return result;
    }

    void elementWiseMult(const Matrix_CPU& other) {
        if (rows == other.rows && cols == other.cols) {
            int total_size = rows * cols;
            
            {
                sycl::buffer<float, 1> this_buf(host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> other_buf(other.host_data.data(), sycl::range<1>(total_size));
                
                q_ptr->submit([&](sycl::handler& h) {
                    auto this_acc = this_buf.get_access<sycl::access::mode::read_write>(h);
                    auto other_acc = other_buf.get_access<sycl::access::mode::read>(h);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                this_acc[gid] *= other_acc[gid];
                            }
                        });
                });
            }
        }
        else {
            std::cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
        }
    }

    Matrix_CPU ElementWiseMult(const Matrix_CPU& other) const {
        if (rows == other.rows && cols == other.cols) {
            Matrix_CPU result(rows, cols, *q_ptr);
            int total_size = rows * cols;
            
            {
                sycl::buffer<float, 1> this_buf(host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> other_buf(other.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> result_buf(result.host_data.data(), sycl::range<1>(total_size));
                
                q_ptr->submit([&](sycl::handler& h) {
                    auto this_acc = this_buf.get_access<sycl::access::mode::read>(h);
                    auto other_acc = other_buf.get_access<sycl::access::mode::read>(h);
                    auto result_acc = result_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                result_acc[gid] = this_acc[gid] * other_acc[gid];
                            }
                        });
                });
            }
            
            return result;
        }
        return Matrix_CPU(); // Return empty matrix if dimensions don't match
    }

    static Matrix_CPU ElementWiseMult(const Matrix_CPU& m1, const Matrix_CPU& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix_CPU result(m1.rows, m1.cols, q);
            int total_size = m1.rows * m1.cols;
            
            {
                sycl::buffer<float, 1> m1_buf(m1.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> m2_buf(m2.host_data.data(), sycl::range<1>(total_size));
                sycl::buffer<float, 1> result_buf(result.host_data.data(), sycl::range<1>(total_size));
                
                q.submit([&](sycl::handler& h) {
                    auto m1_acc = m1_buf.get_access<sycl::access::mode::read>(h);
                    auto m2_acc = m2_buf.get_access<sycl::access::mode::read>(h);
                    auto result_acc = result_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                    
                    auto local_range = std::min(64, total_size);
                    auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                    
                    h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                        [=](sycl::nd_item<1> item) {
                            auto gid = item.get_global_id(0);
                            if (gid < total_size) {
                                result_acc[gid] = m1_acc[gid] * m2_acc[gid];
                            }
                        });
                });
            }
            
            return result;
        }
        return Matrix_CPU(); // Return empty matrix if dimensions don't match
    }

    void transpose() {
        std::vector<float> temp_data(rows * cols);
        
        // Copy current data
        temp_data = host_data;
        
        // Swap dimensions
        int original_rows = rows;
        rows = cols;
        cols = original_rows;
        
        // Resize host_data to accommodate transposed matrix
        host_data.resize(rows * cols);
        
        // Perform transpose on host (more efficient for CPU)
        for (int i = 0; i < original_rows; i++) {
            for (int j = 0; j < cols; j++) {
                host_data[j * original_rows + i] = temp_data[i * cols + j];
            }
        }
    }

    Matrix_CPU Transpose() const {
        Matrix_CPU result(cols, rows, *q_ptr);
        
        // CPU-optimized transpose using blocked approach for better cache performance
        const int block_size = 32; // Cache-friendly block size
        
        for (int ii = 0; ii < rows; ii += block_size) {
            for (int jj = 0; jj < cols; jj += block_size) {
                int i_end = std::min(ii + block_size, rows);
                int j_end = std::min(jj + block_size, cols);
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        result.host_data[j * rows + i] = host_data[i * cols + j];
                    }
                }
            }
        }
        
        return result;
    }

    // Static activation functions
    static float Sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float dSigmoid(float x) {
        return x * (1.0f - x);
    }

    static float ReLU(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    static float dReLU(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    void applyReLU() {
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] = ReLU(acc[gid]);
                        }
                    });
            });
        }
    }

    void applyReLUDerivative() {
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] = dReLU(acc[gid]);
                        }
                    });
            });
        }
    }

    static Matrix_CPU ApplyReLU(const Matrix_CPU& m, sycl::queue& q) {
        Matrix_CPU result(m.rows, m.cols, q);
        int total_size = m.rows * m.cols;
        
        {
            sycl::buffer<float, 1> in_buf(m.host_data.data(), sycl::range<1>(total_size));
            sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(total_size));
            
            q.submit([&](sycl::handler& h) {
                auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            out_acc[gid] = ReLU(in_acc[gid]);
                        }
                    });
            });
        }
        
        return result;
    }

    static Matrix_CPU ApplyReLUDerivative(const Matrix_CPU& m, sycl::queue& q) {
        Matrix_CPU result(m.rows, m.cols, q);
        int total_size = m.rows * m.cols;
        
        {
            sycl::buffer<float, 1> in_buf(m.host_data.data(), sycl::range<1>(total_size));
            sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(total_size));
            
            q.submit([&](sycl::handler& h) {
                auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            out_acc[gid] = dReLU(in_acc[gid]);
                        }
                    });
            });
        }
        
        return result;
    }

    void applySigmoid() {
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] = Sigmoid(acc[gid]);
                        }
                    });
            });
        }
    }

    void applySigmoidDerivative() {
        int total_size = rows * cols;
        
        {
            sycl::buffer<float, 1> buf(host_data.data(), sycl::range<1>(total_size));
            
            q_ptr->submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            acc[gid] = dSigmoid(acc[gid]);
                        }
                    });
            });
        }
    }

    static Matrix_CPU ApplySigmoid(const Matrix_CPU& m, sycl::queue& q) {
        Matrix_CPU result(m.rows, m.cols, q);
        int total_size = m.rows * m.cols;
        
        {
            sycl::buffer<float, 1> in_buf(m.host_data.data(), sycl::range<1>(total_size));
            sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(total_size));
            
            q.submit([&](sycl::handler& h) {
                auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            out_acc[gid] = Sigmoid(in_acc[gid]);
                        }
                    });
            });
        }
        
        return result;
    }

    static Matrix_CPU ApplySigmoidDerivative(const Matrix_CPU& m, sycl::queue& q) {
        Matrix_CPU result(m.rows, m.cols, q);
        int total_size = m.rows * m.cols;
        
        {
            sycl::buffer<float, 1> in_buf(m.host_data.data(), sycl::range<1>(total_size));
            sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(total_size));
            
            q.submit([&](sycl::handler& h) {
                auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                
                auto local_range = std::min(64, total_size);
                auto global_range = ((total_size + local_range - 1) / local_range) * local_range;
                
                h.parallel_for(sycl::nd_range<1>(global_range, local_range), 
                    [=](sycl::nd_item<1> item) {
                        auto gid = item.get_global_id(0);
                        if (gid < total_size) {
                            out_acc[gid] = dSigmoid(in_acc[gid]);
                        }
                    });
            });
        }
        
        return result;
    }

    Matrix_CPU sumAlongAxis(int axis) const {
        if (axis == 1) { // Sum along columns (reduce each row to single value)
            Matrix_CPU result(rows, 1, *q_ptr);
            
            // CPU-optimized: Use host computation for reductions when data is small
            if (rows * cols < 10000) {
                // Host computation for small matrices
                for (int i = 0; i < rows; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < cols; j++) {
                        sum += host_data[i * cols + j];
                    }
                    result.host_data[i] = sum;
                }
            } else {
                // SYCL computation for larger matrices
                {
                    sycl::buffer<float, 1> in_buf(host_data.data(), sycl::range<1>(rows * cols));
                    sycl::buffer<float, 1> out_buf(result.host_data.data(), sycl::range<1>(rows));
                    
                    q_ptr->submit([&](sycl::handler& h) {
                        auto in_acc = in_buf.get_access<sycl::access::mode::read>(h);
                        auto out_acc = out_buf.get_access<sycl::access::mode::write>(h, sycl::no_init);
                        
                        h.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> row_id) {
                            float sum = 0.0f;
                            for (int j = 0; j < cols; j++) {
                                sum += in_acc[row_id * cols + j];
                            }
                            out_acc[row_id] = sum;
                        });
                    });
                }
            }
            
            return result;
        }
        else {
            throw std::invalid_argument("Only axis 1 is implemented.");
        }
    }

    static Matrix_CPU fromArr(const vector<float>& arr, sycl::queue& q) {
        Matrix_CPU result(static_cast<int>(arr.size()), 1, q);
        result.host_data = arr;
        return result;
    }

    vector<float> toArr() const {
        return host_data;
    }

    static Matrix_CPU Softmax(const Matrix_CPU& m, sycl::queue& q) {
        Matrix_CPU result(m.rows, m.cols, q);
        
        // CPU-optimized softmax with better numerical stability
        for (int col = 0; col < m.cols; col++) {
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int row = 0; row < m.rows; row++) {
                max_val = std::max(max_val, m.host_data[row * m.cols + col]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int row = 0; row < m.rows; row++) {
                float exp_val = std::exp(m.host_data[row * m.cols + col] - max_val);
                result.host_data[row * m.cols + col] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (int row = 0; row < m.rows; row++) {
                result.host_data[row * m.cols + col] /= sum;
            }
        }
        
        return result;
    }

    static float crossEntropy(const Matrix_CPU& outputs, const Matrix_CPU& targets, sycl::queue& q) {
        if (outputs.rows != targets.rows || outputs.cols != targets.cols) {
            throw std::invalid_argument("Outputs and targets matrices must have same dimensions for cross-entropy.");
        }

        // CPU-optimized cross-entropy computation
        float total_loss = 0.0f;
        int total_elements = outputs.rows * outputs.cols;
        
        for (int i = 0; i < total_elements; i++) {
            total_loss += -targets.host_data[i] * std::log(outputs.host_data[i] + 1e-9f);
        }

        return total_loss / outputs.cols; // Average over batch size
    }
}; 