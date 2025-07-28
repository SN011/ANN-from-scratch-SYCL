#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>
#include <map>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp> // For oneMKL's GEMM function
#include <limits> // Required for std::numeric_limits

using namespace std;
using namespace sycl;

class Matrix {
public:
    enum class ElementWiseOp {
        ADD_CONST,
        MUL_CONST,
        NEGATE
    };

    int rows;
    int cols;
    float* data; // Changed from vector<float> to float*
    sycl::queue* q_ptr; // Pointer to the SYCL queue

private:
    // Static ones vector for GEMV operations - indexed by size
    static std::map<std::pair<sycl::queue*, int>, float*> ones_cache;
    
    // USM Memory pool - keyed by (queue, rows, cols) and stores vector of free pointers
    static std::map<std::tuple<sycl::queue*, int, int>, std::vector<float*>> memory_pool;

    // Helper function to get or create ones vector
    float* getOnesVector(int size) {
        auto key = std::make_pair(q_ptr, size);
        auto it = ones_cache.find(key);
        if (it != ones_cache.end()) {
            return it->second;
        }
        
        // Create new ones vector
        float* ones = sycl::malloc_device<float>(size, *q_ptr);
        if (!ones) {
            throw std::runtime_error("Failed to allocate USM for ones vector.");
        }
        
        // Initialize to all ones
        q_ptr->submit([&](handler& h) {
            h.parallel_for(range<1>(size), [=](id<1> i) {
                ones[i] = 1.0f;
            });
        }).wait();
        
        ones_cache[key] = ones;
        return ones;
    }
    
    // Helper functions for memory pool management
    float* getPooledMemory(int rows, int cols) {
        auto key = std::make_tuple(q_ptr, rows, cols);
        auto it = memory_pool.find(key);
        
        if (it != memory_pool.end() && !it->second.empty()) {
            float* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        
        // No available memory in pool, allocate new
        float* ptr = sycl::malloc_device<float>(rows * cols, *q_ptr);
        return ptr;
    }
    
    void returnPooledMemory(float* ptr, int rows, int cols) {
        if (!ptr) return;
        
        auto key = std::make_tuple(q_ptr, rows, cols);
        memory_pool[key].push_back(ptr);
    }

public:
    Matrix() : rows(0), cols(0), data(nullptr), q_ptr(nullptr) {}

private:
    // Fused element-wise operation kernel
    template<ElementWiseOp OP>
    void fused_elementwise(float value = 0.0f) {
        int r = rows;
        int c = cols;
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                if constexpr (OP == ElementWiseOp::ADD_CONST) {
                    ptr[i] += value;
                } else if constexpr (OP == ElementWiseOp::MUL_CONST) {
                    ptr[i] *= value;
                } else if constexpr (OP == ElementWiseOp::NEGATE) {
                    ptr[i] *= -1.0f;
                }
            });
        });
    }

public:
    Matrix(int r, int c, sycl::queue& q) : rows(r), cols(c), q_ptr(&q) {
        data = getPooledMemory(rows, cols);
        if (!data) {
            throw std::runtime_error("Failed to allocate USM for Matrix data.");
        }
    }

    // Copy constructor for Matrix (deep copy for USM)
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), q_ptr(other.q_ptr) {
        if (q_ptr) {
            data = const_cast<Matrix*>(&other)->getPooledMemory(rows, cols);
            if (!data) {
                throw std::runtime_error("Failed to allocate USM for Matrix data in copy constructor.");
            }
            q_ptr->memcpy(data, other.data, rows * cols * sizeof(float)).wait();
        }
        else {
            data = nullptr; // Should not happen if q_ptr is always set
        }
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data), q_ptr(other.q_ptr) {
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
        other.q_ptr = nullptr;
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (data && q_ptr) {
                returnPooledMemory(data, rows, cols);
            }
            rows = other.rows;
            cols = other.cols;
            q_ptr = other.q_ptr;
            if (q_ptr) {
                data = getPooledMemory(rows, cols);
                if (!data) {
                    throw std::runtime_error("Failed to allocate USM for Matrix data in copy assignment.");
                }
                q_ptr->memcpy(data, other.data, rows * cols * sizeof(float)).wait();
            }
            else {
                data = nullptr;
            }
        }
        return *this;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (data && q_ptr) {
                returnPooledMemory(data, rows, cols);
            }
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            q_ptr = other.q_ptr;

            other.rows = 0;
            other.cols = 0;
            other.data = nullptr;
            other.q_ptr = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        if (data && q_ptr) {
            returnPooledMemory(data, rows, cols);
            data = nullptr;
            q_ptr = nullptr;
        }
    }

    void RandInit(int fan_in = 0, int fan_out = 0, unsigned int seed = 0) {
        float limit;
        if (fan_in > 0 && fan_out > 0) {
            limit = std::sqrt(6.0f / (fan_in + fan_out));
        }
        else {
            limit = 1.0f; // Default range if no fan_in/fan_out is provided
        }
        
        // Use shared USM for temporary storage
        float* temp_data = sycl::malloc_shared<float>(rows * cols, *q_ptr);
        if (!temp_data) {
            throw std::runtime_error("Failed to allocate shared USM for RandInit.");
        }
        
        unsigned int base_seed = (seed == 0) ? std::random_device{}() : seed;
        int total_elements = rows * cols;
        
        q_ptr->submit([&](handler& h) {
            h.parallel_for(range<1>(total_elements), [=](id<1> i) {
                // XOR-SHIFT random number generator
                unsigned int local_seed = base_seed + i[0] + 1; // Ensure non-zero seed
                
                // XOR-SHIFT algorithm
                local_seed ^= local_seed << 13;
                local_seed ^= local_seed >> 17;
                local_seed ^= local_seed << 5;
                
                // Convert to float in range [0, 1)
                float rand_01 = (local_seed & 0x7FFFFFFF) / float(0x7FFFFFFF);
                
                // Scale to desired range [-limit, limit]
                temp_data[i] = (rand_01 * 2.0f - 1.0f) * limit;
            });
        }).wait();
        
        // Copy from shared USM to device USM
        q_ptr->memcpy(data, temp_data, rows * cols * sizeof(float)).wait();
        
        sycl::free(temp_data, *q_ptr);
    }

    void printMatrix() const {
        std::vector<float> host_data(rows * cols);
        q_ptr->memcpy(host_data.data(), data, rows * cols * sizeof(float)).wait();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << host_data[i * cols + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    void multiplyScalar(float n) {
        fused_elementwise<ElementWiseOp::MUL_CONST>(n);
    }

    void addScalar(float n) {
        fused_elementwise<ElementWiseOp::ADD_CONST>(n);
    }

    void negate() {
        fused_elementwise<ElementWiseOp::NEGATE>();
    }

    Matrix Negate() {
        Matrix m(rows, cols, *q_ptr); // Pass queue to new Matrix constructor
        int r = rows;
        int c = cols;
        const float* inPtr = data;
        float* outPtr = m.data;


        q_ptr->submit([&](handler& h) {
            auto inAcc = inPtr;
            auto outAcc = outPtr;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                outAcc[i] = inAcc[i] * -1.0f;
                });
            });

        return m;
    }

    void add(const Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            int r = rows;
            int c = cols;
            float* ptr = data;
            const float* optr = other.data;

            q_ptr->submit([&](handler& h) {
                auto a = ptr;
                auto b = optr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    a[i] += b[i];
                    });
                });
        }
        // ONLY CHECKING IF OTHER IS A COL VEC (BIAS MATRIX)
        else if (rows == other.rows && other.cols == 1 && cols > 1) {
            int r = rows;
            int c = cols;
            float* ptr = data;
            const float* optr = other.data;

            q_ptr->submit([&](handler& h) {
                auto a = ptr;
                auto b = optr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    int row = i[0] / c;
                    a[i] += b[row];
                    });
                });
        }
        else {
            std::cout << "Dims of matrices not compatible for add(). No changes made.\n";
        }
    }

    static Matrix add(const Matrix& m1, const Matrix& m2, sycl::queue& q, bool in_place = false) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix result = in_place ? Matrix() : Matrix(m1.rows, m1.cols, q);
            int r = m1.rows;
            int c = m1.cols;

            const float* aPtr = m1.data;
            const float* bPtr = m2.data;
            float* resPtr = in_place ? const_cast<float*>(m1.data) : result.data;

            q.submit([&](handler& h) {
                auto a = aPtr;
                auto b = bPtr;
                auto out = resPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    out[i] = a[i] + b[i];
                    });
                });
            
            if (in_place) {
                // Return a copy of m1 for consistency with API
                result = Matrix(m1.rows, m1.cols, q);
                q.memcpy(result.data, m1.data, r * c * sizeof(float));
            }
            return result;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    Matrix Add(const Matrix& other, bool in_place = false) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output = in_place ? Matrix() : Matrix(rows, cols, *q_ptr);
            int r = rows;
            int c = cols;

            const float* aPtr = data;
            const float* bPtr = other.data;
            float* outPtr = in_place ? data : output.data;

            q_ptr->submit([&](handler& h) {
                auto accA = aPtr;
                auto accB = bPtr;
                auto accC = outPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    accC[i] = accA[i] + accB[i];
                    });
                });
            
            if (in_place) {
                // Return a copy of this for consistency with API
                output = Matrix(rows, cols, *q_ptr);
                q_ptr->memcpy(output.data, data, r * c * sizeof(float));
            }
            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    static Matrix subtract(const Matrix& m1, const Matrix& m2, sycl::queue& q, bool in_place = false) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix result = in_place ? Matrix() : Matrix(m1.rows, m1.cols, q);
            int r = m1.rows;
            int c = m1.cols;

            const float* aPtr = m1.data;
            const float* bPtr = m2.data;
            float* resPtr = in_place ? const_cast<float*>(m1.data) : result.data;

            q.submit([&](handler& h) {
                auto a = aPtr;
                auto b = bPtr;
                auto out = resPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    out[i] = a[i] - b[i];
                    });
                });
            
            if (in_place) {
                // Return a copy of m1 for consistency with API
                result = Matrix(m1.rows, m1.cols, q);
                q.memcpy(result.data, m1.data, r * c * sizeof(float));
            }
            return result;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    void multiply(const Matrix& other) {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix output(rows, other.cols, *q_ptr); // Pass queue to new Matrix constructor

        // Using oneMKL's gemm function for matrix multiplication
        // C = alpha * A * B + beta * C
        // Here, alpha = 1.0f, beta = 0.0f, C is initialized to zeros (implicitly by Matrix constructor)
        try {
            oneapi::mkl::blas::row_major::gemm(
                *q_ptr,
                oneapi::mkl::transpose::nontrans, // transa
                oneapi::mkl::transpose::nontrans, // transb
                rows,                             // m (rows of A and C)
                other.cols,                       // n (cols of B and C)
                cols,                             // k (cols of A and rows of B)
                1.0f,                             // alpha
                data,                             // A data (USM pointer)
                cols,                             // lda (leading dimension of A - number of columns if row-major)
                other.data,                       // B data (USM pointer)
                other.cols,                       // ldb (leading dimension of B - number of columns if row-major)
                0.0f,                             // beta
                output.data,                      // C data (USM pointer)
                output.cols                       // ldc (leading dimension of C - number of columns if row-major)
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during GEMM: " << e.what() << std::endl;
        }

        // No need to move data, just copy values to current object's data and update dimensions
        // This assumes 'output' is a temporary and its USM will be freed by its destructor
        // It is more efficient to swap the data pointers if 'output' is truly meant to replace 'this' data
        // However, for simplicity and to match previous behavior, a memcpy is used for now.
        q_ptr->memcpy(data, output.data, rows * other.cols * sizeof(float));
        // Rows and cols already match output, so no change needed
    }

    static Matrix multiply(const Matrix& m1, const Matrix& m2, sycl::queue& q, 
                          const std::vector<sycl::event>& deps = {}) {
        if (m1.cols != m2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(m1.rows, m2.cols, q); // Pass queue to new Matrix constructor

        // Using oneMKL's gemm function for matrix multiplication
        try {
            oneapi::mkl::blas::row_major::gemm(
                q,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans,
                m1.rows,
                m2.cols,
                m1.cols,
                1.0f,
                m1.data, // USM pointer
                m1.cols,
                m2.data, // USM pointer
                m2.cols,
                0.0f,
                result.data, // USM pointer
                result.cols,
                deps  // Event dependencies
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during static GEMM: " << e.what() << std::endl;
        }

        return result;
    }

    // New multiply function with transpose flags to avoid explicit transpose operations
    static Matrix multiply_with_transpose(const Matrix& m1, const Matrix& m2, sycl::queue& q,
                                          bool transpose_m1 = false, bool transpose_m2 = false,
                                          const std::vector<sycl::event>& deps = {}) {
        int m1_effective_rows = transpose_m1 ? m1.cols : m1.rows;
        int m1_effective_cols = transpose_m1 ? m1.rows : m1.cols;
        int m2_effective_rows = transpose_m2 ? m2.cols : m2.rows;
        int m2_effective_cols = transpose_m2 ? m2.rows : m2.cols;
        
        if (m1_effective_cols != m2_effective_rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication after transpose.");
        }

        Matrix result(m1_effective_rows, m2_effective_cols, q);

        // Using oneMKL's gemm function with transpose flags
        try {
            oneapi::mkl::blas::row_major::gemm(
                q,
                transpose_m1 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                transpose_m2 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                m1_effective_rows,
                m2_effective_cols,
                m1_effective_cols,
                1.0f,
                m1.data, // USM pointer
                m1.cols, // Leading dimension of original matrix
                m2.data, // USM pointer
                m2.cols, // Leading dimension of original matrix
                0.0f,
                result.data, // USM pointer
                result.cols,
                deps  // Event dependencies
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during transpose GEMM: " << e.what() << std::endl;
        }

        return result;
    }

    void elementWiseMult(const Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            int r = rows;
            int c = cols;
            float* ptrA = data;
            const float* ptrB = other.data;

            // buffer<float, 1> buf_a(ptrA, range<1>(r * c)); 
            // buffer<float, 1> buf_b(ptrB, range<1>(r * c)); 

            q_ptr->submit([&](handler& h) {
                auto a = ptrA;
                auto b = ptrB;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    a[i] *= b[i];
                    });
                });
        }
        else {
            std::cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
        }
    }

    Matrix ElementWiseMult(const Matrix& other, bool in_place = false) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output = in_place ? Matrix() : Matrix(rows, cols, *q_ptr);
            int r = rows;
            int c = cols;

            const float* ptrA = data;
            const float* ptrB = other.data;
            float* ptrC = in_place ? data : output.data;

            q_ptr->submit([&](handler& h) {
                auto a = ptrA;
                auto b = ptrB;
                auto cacc = ptrC;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    cacc[i] = a[i] * b[i];
                    });
                });

            if (in_place) {
                // Return a copy of this for consistency with API
                output = Matrix(rows, cols, *q_ptr);
                q_ptr->memcpy(output.data, data, r * c * sizeof(float));
            }
            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    static Matrix ElementWiseMult(const Matrix& m1, const Matrix& m2, sycl::queue& q, bool in_place = false) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix output = in_place ? Matrix() : Matrix(m1.rows, m1.cols, q);
            int r = m1.rows;
            int c = m1.cols;

            const float* ptrA = m1.data;
            const float* ptrB = m2.data;
            float* ptrC = in_place ? const_cast<float*>(m1.data) : output.data;

            q.submit([&](handler& h) {
                auto a = ptrA;
                auto b = ptrB;
                auto cacc = ptrC;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    cacc[i] = a[i] * b[i];
                    });
                });
            
            if (in_place) {
                // Return a copy of m1 for consistency with API
                output = Matrix(m1.rows, m1.cols, q);
                q.memcpy(output.data, m1.data, r * c * sizeof(float));
            }
            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    // DELETED: transpose() function to avoid expensive host-device copies
    // Use oneMKL GEMM with transpose flags instead: oneapi::mkl::transpose::trans

    // DELETED: Transpose() function to avoid expensive host-device copies
    // Use oneMKL GEMM with transpose flags instead: oneapi::mkl::transpose::trans
    Matrix Transpose() const {
        // This function should not be used - use GEMM with transpose flags instead
        throw std::runtime_error("Transpose() function has been deleted for performance. Use GEMM with transpose flags.");
    }

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
        int r = rows;
        int c = cols;
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] = ReLU(ptr[i]);
                });
            });
    }

    void applyReLUDerivative() {
        int r = rows;
        int c = cols;
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] = dReLU(ptr[i]);
                });
            });
    }

    static Matrix ApplyReLU(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols, q);
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = output.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = ReLU(a[i]);
                });
            });
        return output;
    }

    static Matrix ApplyReLUDerivative(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols, q);
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = output.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = dReLU(a[i]);
                });
            });
        return output;
    }

    void applySigmoid() {
        int r = rows;
        int c = cols;
        // float* ptr = data.data(); 

        // buffer<float, 1> buf(ptr, range<1>(r * c)); 
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] = Sigmoid(ptr[i]);
                });
            });
    }

    void applySigmoidDerivative() {
        int r = rows;
        int c = cols;
        // float* ptr = data.data(); 

        // buffer<float, 1> buf(ptr, range<1>(r * c)); 
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] = dSigmoid(ptr[i]);
                });
            });
    }

    static Matrix ApplySigmoid(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols, q); // Pass queue to new Matrix constructor
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = output.data;

        // buffer<float, 1> buf_a(ptrA, range<1>(r * c)); 
        // buffer<float, 1> buf_b(ptrOut, range<1>(r * c)); 

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = Sigmoid(a[i]);
                });
            });
        return output;
    }

    static Matrix ApplySigmoidDerivative(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols, q); // Pass queue to new Matrix constructor
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = output.data;

        // buffer<float, 1> buf_a(ptrA, range<1>(r * c)); 
        // buffer<float, 1> buf_b(ptrOut, range<1>(r * c)); 

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = dSigmoid(a[i]);
                });
            });
        return output;
    }

    Matrix sumAlongAxis(int axis) const {
        if (axis == 1) {
            Matrix result(rows, 1, *q_ptr);
            float* ones = const_cast<Matrix*>(this)->getOnesVector(cols);
            
            // Use GEMV: result = A * ones_vector (where A is this matrix)
            // result = 1.0 * data * ones + 0.0 * result
            try {
                oneapi::mkl::blas::row_major::gemv(
                    *q_ptr,
                    oneapi::mkl::transpose::nontrans,  // trans = nontrans
                    rows,           // m (rows of A)
                    cols,           // n (cols of A)
                    1.0f,           // alpha
                    data,           // A data (USM pointer)
                    cols,           // lda (leading dimension of A)
                    ones,           // x vector (ones vector)
                    1,              // incx (increment for x)
                    0.0f,           // beta
                    result.data,    // y vector (result)
                    1               // incy (increment for y)
                );
            }
            catch (sycl::exception const& e) {
                std::cerr << "SYCL exception caught during GEMV: " << e.what() << std::endl;
            }
            
            return result;
        }
        else {
            throw std::invalid_argument("Only axis 1 is implemented.");
        }
    }

    static Matrix fromArr(const vector<float>& arr, sycl::queue& q) {
        Matrix output((int)arr.size(), 1, q); // Pass queue to new Matrix constructor
        q.memcpy(output.data, arr.data(), arr.size() * sizeof(float)).wait();
        return output;
    }

    vector<float> toArr() const {
        vector<float> host_data(rows * cols);
        q_ptr->memcpy(host_data.data(), data, rows * cols * sizeof(float)).wait();
        return host_data;
    }

    static Matrix Softmax(const Matrix& m, sycl::queue& q) {
        Matrix out(m.rows, m.cols, q);
        int R = m.rows, C = m.cols;
        const float* A = m.data;
        float* B = out.data;

        // Use work-groups with local memory for efficient row reduction
        const int local_size = 256; // Workgroup size
        
        q.submit([&](sycl::handler& h) {
            // Local memory for reduction operations
            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> 
                local_max(sycl::range<1>(local_size), h);
            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> 
                local_sum(sycl::range<1>(local_size), h);
            
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((C + local_size - 1) / local_size * local_size), 
                                           sycl::range<1>(local_size)),
                          [=](sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                int lid = item.get_local_id(0);
                int group_id = item.get_group(0);
                
                if (gid >= C) return; // Check bounds
                
                // Phase 1: Find max for this column
                float max_val = -std::numeric_limits<float>::infinity();
                for (int r = 0; r < R; r++) {
                    max_val = sycl::max(max_val, A[r * C + gid]);
                }
                local_max[lid] = max_val;
                item.barrier(sycl::access::fence_space::local_space);
                
                // Reduce max within workgroup (though each thread handles one column)
                // This step is mainly for future extensibility
                
                // Phase 2: Compute sum of exp(x - max)
                float sum_exp = 0.0f;
                for (int r = 0; r < R; r++) {
                    sum_exp += sycl::exp(A[r * C + gid] - max_val);
                }
                local_sum[lid] = sum_exp;
                item.barrier(sycl::access::fence_space::local_space);
                
                // Phase 3: Write normalized values
                for (int r = 0; r < R; r++) {
                    B[r * C + gid] = sycl::exp(A[r * C + gid] - max_val) / sum_exp;
                }
            });
        });
        return out;
    }

    static float crossEntropy(const Matrix& outputs, const Matrix& targets, sycl::queue& q) {
        if (outputs.rows != targets.rows || outputs.cols != targets.cols) {
            throw std::invalid_argument("Outputs and targets matrices must have same dimensions for cross-entropy.");
        }

        int totalElements = outputs.rows * outputs.cols;
        const float* outPtr = outputs.data;
        const float* targetPtr = targets.data;

        // Create a buffer for reduction results (sum of logs)
        float* sum_log_device = sycl::malloc_shared<float>(1, q);
        q.memset(sum_log_device, 0, sizeof(float)).wait(); // Initialize to 0

        auto red = sycl::reduction(sum_log_device, sycl::plus<float>());

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(totalElements), red,
                [=](sycl::id<1> idx, auto& sum) {
                    sum += -targetPtr[idx] *
                        sycl::log(outPtr[idx] + 1e-9f);
                });
            }).wait();
        float loss = *sum_log_device;
        sycl::free(sum_log_device, q);

        return loss / outputs.cols; // Average over the batch size (number of columns)
    }
};

// Static member definitions
std::map<std::pair<sycl::queue*, int>, float*> Matrix::ones_cache;
std::map<std::tuple<sycl::queue*, int, int>, std::vector<float*>> Matrix::memory_pool;

// template <>
// struct is_device_copyable<Matrix> : std::true_type {}; 

