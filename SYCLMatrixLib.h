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

class Matrix {
public:
    int rows;
    int cols;
    float* data; // Changed from vector<float> to float*
    sycl::queue* q_ptr; // Pointer to the SYCL queue

    Matrix() : rows(0), cols(0), data(nullptr), q_ptr(nullptr) {}

    Matrix(int r, int c, sycl::queue& q) : rows(r), cols(c), q_ptr(&q) {
        data = sycl::malloc_device<float>(rows * cols, q);
        if (!data) {
            throw std::runtime_error("Failed to allocate USM for Matrix data.");
        }
    }

    // Copy constructor for Matrix (deep copy for USM)
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), q_ptr(other.q_ptr) {
        if (q_ptr) {
            data = sycl::malloc_device<float>(rows * cols, *q_ptr);
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
                sycl::free(data, *q_ptr);
            }
            rows = other.rows;
            cols = other.cols;
            q_ptr = other.q_ptr;
            if (q_ptr) {
                data = sycl::malloc_device<float>(rows * cols, *q_ptr);
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
                sycl::free(data, *q_ptr);
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
            sycl::free(data, *q_ptr);
            data = nullptr;
            q_ptr = nullptr;
        }
    }

    void RandInit(int fan_in = 0, int fan_out = 0) {
        std::random_device rd;
        float limit;
        if (fan_in > 0 && fan_out > 0) {
            limit = std::sqrt(6.0f / (fan_in + fan_out));
        }
        else {
            limit = 1.0f; // Default range if no fan_in/fan_out is provided
        }
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> host_data(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            host_data[i] = dist(rd);
        }
        q_ptr->memcpy(data, host_data.data(), rows * cols * sizeof(float)).wait();
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
        int r = rows;
        int c = cols;
        // float* ptr = data.data(); 

        // buffer<float, 1> buf(ptr, range<1>(r * c)); 
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] *= n;
                });
            });
    }

    void addScalar(float n) {
        int r = rows;
        int c = cols;
        // float* ptr = data.data(); 

        // buffer<float, 1> buf(ptr, range<1>(r * c)); 
        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] += n;
                });
            });
    }

    void negate() {
        int r = rows;
        int c = cols;


        q_ptr->submit([&](handler& h) {
            auto ptr = data;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                ptr[i] *= -1;
                });
            });
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

    static Matrix add(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix result(m1.rows, m1.cols, q); // Pass queue to new Matrix constructor
            int r = m1.rows;
            int c = m1.cols;

            const float* aPtr = m1.data;
            const float* bPtr = m2.data;
            float* resPtr = result.data;


            q.submit([&](handler& h) {
                auto a = aPtr;
                auto b = bPtr;
                auto out = resPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    out[i] = a[i] + b[i];
                    });
                });
            return result;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    Matrix Add(const Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols, *q_ptr); // Pass queue to new Matrix constructor
            int r = rows;
            int c = cols;

            const float* aPtr = data;
            const float* bPtr = other.data;
            float* outPtr = output.data;

            q_ptr->submit([&](handler& h) {
                auto accA = aPtr;
                auto accB = bPtr;
                auto accC = outPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    accC[i] = accA[i] + accB[i];
                    });
                });
            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    static Matrix subtract(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix result(m1.rows, m1.cols, q); // Pass queue to new Matrix constructor
            int r = m1.rows;
            int c = m1.cols;

            const float* aPtr = m1.data;
            const float* bPtr = m2.data;
            float* resPtr = result.data;

            // buffer<float, 1> buf_a(aPtr, range<1>(r * c)); 
            // buffer<float, 1> buf_b(bPtr, range<1>(r * c)); 
            // buffer<float, 1> buf_c(resPtr, range<1>(r * c)); 

            q.submit([&](handler& h) {
                auto a = aPtr;
                auto b = bPtr;
                auto out = resPtr;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    out[i] = a[i] - b[i];
                    });
                });
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

    static Matrix multiply(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
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
                result.cols
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during static GEMM: " << e.what() << std::endl;
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

    Matrix ElementWiseMult(const Matrix& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols, *q_ptr); // Pass queue to new Matrix constructor
            int r = rows;
            int c = cols;

            const float* ptrA = data;
            const float* ptrB = other.data;
            float* ptrC = output.data;


            q_ptr->submit([&](handler& h) {
                auto a = ptrA;
                auto b = ptrB;
                auto cacc = ptrC;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    cacc[i] = a[i] * b[i];
                    });
                });

            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    static Matrix ElementWiseMult(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix output(m1.rows, m1.cols, q); // Pass queue to new Matrix constructor
            int r = m1.rows;
            int c = m1.cols;

            const float* ptrA = m1.data;
            const float* ptrB = m2.data;
            float* ptrC = output.data;


            q.submit([&](handler& h) {
                auto a = ptrA;
                auto b = ptrB;
                auto cacc = ptrC;
                h.parallel_for(range<1>(r * c), [=](id<1> i) {
                    cacc[i] = a[i] * b[i];
                    });
                });
            return output;
        }
        return Matrix(); // Needs a queue or default constructor with nullptr
    }

    // Transpose operations are inherently host-side and involve data movement.
    // For efficiency with USM, one might prefer to re-think operations that require transpose
    // or use oneMKL functions with transpose flags if available. 
    void transpose() {
        std::vector<float> host_data(rows * cols);
        q_ptr->memcpy(host_data.data(), data, rows * cols * sizeof(float)).wait();

        // Free old device memory if this object already held data
        // This needs careful management if 'this' matrix is being reused/reassigned.
        // For now, assuming it's safe to free and reallocate, or that the swap handles it.
        sycl::free(data, *q_ptr);

        int originalRows = rows;
        rows = cols;
        cols = originalRows;

        data = sycl::malloc_device<float>(rows * cols, *q_ptr);
        if (!data) {
            throw std::runtime_error("Failed to allocate USM for transposed Matrix data.");
        }

        std::vector<float> output_host_data(rows * cols);
        for (int i = 0; i < originalRows; i++) {
            for (int j = 0; j < cols; j++) { // Loop over new cols (original rows)
                output_host_data[j * originalRows + i] = host_data[i * cols + j];
            }
        }
        q_ptr->memcpy(data, output_host_data.data(), rows * cols * sizeof(float)).wait();
    }

    Matrix Transpose() const {
        Matrix output(cols, rows, *q_ptr); // Pass queue to new Matrix constructor

        std::vector<float> host_data(rows * cols);
        q_ptr->memcpy(host_data.data(), data, rows * cols * sizeof(float)).wait();

        std::vector<float> output_host_data(cols * rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output_host_data[j * rows + i] = host_data[i * cols + j];
            }
        }
        q_ptr->memcpy(output.data, output_host_data.data(), cols * rows * sizeof(float)).wait();

        return output;
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
            // Pass queue to new Matrix constructor
            // For sumAlongAxis, it's simpler to copy to host, compute, and copy back to USM
            // For larger matrices, a SYCL kernel for reduction would be more efficient
            std::vector<float> host_data(rows * cols);
            q_ptr->memcpy(host_data.data(), data, rows * cols * sizeof(float)).wait();

            std::vector<float> result_host_data(rows);
            for (int i = 0; i < rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += host_data[i * cols + j];
                }
                result_host_data[i] = sum;
            }
            q_ptr->memcpy(result.data, result_host_data.data(), rows * sizeof(float)).wait();

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

        // one work‑group per column
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(C, R), [=](sycl::id<2> id) {
                int col = id[0], row = id[1];

                // 1. find per‑column max (naïve reduction in registers)
                float maxv = -1e30f;
                for (int r_idx = 0; r_idx < R; ++r_idx)
                    maxv = sycl::max(maxv, A[r_idx * C + col]);

                // 2. compute exp and running sum
                float sum = 0.0f;
                for (int r_idx = 0; r_idx < R; ++r_idx)
                    sum += sycl::exp(A[r_idx * C + col] - maxv);

                // 3. write normalized value
                B[row * C + col] = sycl::exp(A[row * C + col] - maxv) / sum;
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

// template <>
// struct is_device_copyable<Matrix> : std::true_type {}; 

