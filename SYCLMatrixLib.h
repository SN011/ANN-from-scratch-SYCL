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
    int rows;
    int cols;
    float* data; // Changed from vector<float> to float*
    sycl::queue* q_ptr; // Pointer to the SYCL queue

    // ---------------------------------------------------------------------
    // Shared scratch buffers for Softmax to avoid frequent device allocations
    // These live for the duration of the program and resize only when needed
    // ---------------------------------------------------------------------
    static inline float* softmax_tmp_max  = nullptr;
    static inline float* softmax_tmp_sum  = nullptr;
    static inline std::size_t softmax_capacity = 0;


public:
    Matrix() : rows(0), cols(0), data(nullptr), q_ptr(nullptr) {}



public:
    Matrix(int r, int c, sycl::queue& q) : rows(r), cols(c), q_ptr(&q) {
        data = sycl::malloc_shared<float>(rows * cols, q);
        if (!data) {
            throw std::runtime_error("Failed to allocate USM for Matrix data.");
        }
    }

    // Copy constructor for Matrix (deep copy for USM)
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), q_ptr(other.q_ptr) {
        if (q_ptr) {
            data = sycl::malloc_shared<float>(rows * cols, *q_ptr);
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
                // Ensure all operations using this buffer are complete before freeing
                q_ptr->wait();
                sycl::free(data, *q_ptr);
            }
            rows = other.rows;
            cols = other.cols;
            q_ptr = other.q_ptr;
            if (q_ptr) {
                data = sycl::malloc_shared<float>(rows * cols, *q_ptr);
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
                // Ensure all operations using this buffer are complete before freeing
                q_ptr->wait();
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
            try {
                q_ptr->wait_and_throw();
            } catch (const sycl::exception& e) {
                std::cerr << "SYCL async error during Matrix destruction: " << e.what() << std::endl;
                // Swallow exception to avoid abort inside destructor
            }
            try {
                sycl::free(data, *q_ptr);
            } catch (const sycl::exception& e) {
                // This can happen if the queue/device is already in a bad state
                std::cerr << "SYCL error during free in Matrix destructor: " << e.what() << std::endl;
            }
            data = nullptr;
            q_ptr = nullptr;
        }
    }

    // Resize matrix dimensions without reallocating (for reusing scratch matrices)
    void resizeInPlace(int new_rows, int new_cols) {
        if (new_rows <= 0 || new_cols <= 0) {
            throw std::runtime_error("Cannot resize matrix to zero or negative dimensions.");
        }
        if (new_rows * new_cols > rows * cols) {
            throw std::runtime_error("Cannot resize matrix to larger size than originally allocated.");
        }
        rows = new_rows;
        cols = new_cols;
    }

    void RandInit(int fan_in = 0, int fan_out = 0, unsigned int seed = 0) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
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
            host_data[i] = dist(gen);
        }
        q_ptr->memcpy(data, host_data.data(), rows * cols * sizeof(float)).wait();
        // Ensure all kernels that might have been enqueued for this matrix finish
        q_ptr->wait_and_throw();
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

    static void subtract(const Matrix& m1, const Matrix& m2, Matrix& dest, sycl::queue& q) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }
        if (dest.rows != m1.rows || dest.cols != m1.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match result dimensions.");
        }

        int r = m1.rows;
        int c = m1.cols;
        const float* aPtr = m1.data;
        const float* bPtr = m2.data;
        float* resPtr = dest.data;

        q.submit([&](handler& h) {
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                resPtr[i] = aPtr[i] - bPtr[i];
                });
            });
    }

    void multiply(const Matrix& other) {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create a temporary matrix to hold the result
        Matrix result(rows, other.cols, *q_ptr);

        // Using oneMKL's row-major gemm function for matrix multiplication
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
                cols,                             // lda (leading dimension of A)
                other.data,                       // B data (USM pointer)
                other.cols,                       // ldb (leading dimension of B)
                0.0f,                             // beta
                result.data,                      // C data (USM pointer)
                result.cols                       // ldc (leading dimension of C)
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during GEMM: " << e.what() << std::endl;
        }

        // Replace this matrix's data with the result (efficient swap)
        if (data && q_ptr) {
            sycl::free(data, *q_ptr);
        }
        data = result.data;
        cols = result.cols;
        // Clear result's data pointer so it won't be freed by result's destructor
        result.data = nullptr;
    }

    static Matrix multiply(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.cols != m2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(m1.rows, m2.cols, q); // Pass queue to new Matrix constructor

        // Using oneMKL's row-major gemm function for matrix multiplication
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
                m1.cols, // lda - leading dimension (cols for row-major)
                m2.data, // USM pointer
                m2.cols, // ldb - leading dimension (cols for row-major)
                0.0f,
                result.data, // USM pointer
                result.cols // ldc - leading dimension (cols for row-major)
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during static GEMM: " << e.what() << std::endl;
        }

        return result;
    }

    // Version that writes result to existing destination matrix (avoids allocation)
    static void multiply(const Matrix& m1, const Matrix& m2, Matrix& dest, sycl::queue& q) {
        // Assert that no matrix has zero dimensions
        assert(m1.rows > 0 && m1.cols > 0 && m2.rows > 0 && m2.cols > 0 && dest.rows > 0 && dest.cols > 0 &&
               "Zero-dimension matrix passed to GEMM");
        
        if (m1.cols != m2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        
        if (dest.rows != m1.rows || dest.cols != m2.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match result dimensions.");
        }

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
                m1.cols, // lda - leading dimension (cols for row-major)
                m2.data, // USM pointer
                m2.cols, // ldb - leading dimension (cols for row-major)
                0.0f,
                dest.data, // USM pointer
                dest.cols // ldc - leading dimension (cols for row-major)
            );
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught during static GEMM to dest: " << e.what() << std::endl;
        }
    }

    // helper – single compilation unit
    static Matrix mult_T(const Matrix& A,
        const Matrix& B,
        bool At, bool Bt,
        sycl::queue& q)
    {
        auto trans_A = At ? oneapi::mkl::transpose::trans
            : oneapi::mkl::transpose::nontrans;
        auto trans_B = Bt ? oneapi::mkl::transpose::trans
            : oneapi::mkl::transpose::nontrans;

        int M = At ? A.cols : A.rows;   // rows  of op(A)
        int N = Bt ? B.rows : B.cols;   // cols  of op(B)
        int K = At ? A.rows : A.cols;   // inner dimension

        if ((At ? A.rows : A.cols) != (Bt ? B.cols : B.rows))
            throw std::invalid_argument("mult_T: dimension mismatch.");

        Matrix C(M, N, q);

        //   In row‑major the leading dimension is ALWAYS “original cols”.
        std::int64_t lda = A.cols;
        std::int64_t ldb = B.cols;
        std::int64_t ldc = C.cols;      // = N

        oneapi::mkl::blas::row_major::gemm(
            q, trans_A, trans_B,
            M, N, K,
            1.0f,
            A.data, lda,
            B.data, ldb,
            0.0f,
            C.data, ldc);

        return C;
    }

    static void mult_T(const Matrix& A,
        const Matrix& B,
        bool At, bool Bt,
        Matrix& C,
        sycl::queue& q)
    {
        auto trans_A = At ? oneapi::mkl::transpose::trans
            : oneapi::mkl::transpose::nontrans;
        auto trans_B = Bt ? oneapi::mkl::transpose::trans
            : oneapi::mkl::transpose::nontrans;

        int M = At ? A.cols : A.rows;   // rows  of op(A)
        int N = Bt ? B.rows : B.cols;   // cols  of op(B)
        int K = At ? A.rows : A.cols;   // inner dimension

        if (K != (Bt ? B.cols : B.rows))
            throw std::invalid_argument("mult_T: dimension mismatch.");
        
        if (C.rows != M || C.cols != N)
            throw std::invalid_argument("mult_T: destination C has incorrect dimensions.");


        // In row‑major the leading dimension is ALWAYS “original cols”.
        std::int64_t lda = A.cols;
        std::int64_t ldb = B.cols;
        std::int64_t ldc = C.cols;

        oneapi::mkl::blas::row_major::gemm(
            q, trans_A, trans_B,
            M, N, K,
            1.0f,
            A.data, lda,
            B.data, ldb,
            0.0f,
            C.data, ldc);
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

    static void ElementWiseMult(const Matrix& m1, const Matrix& m2, Matrix& dest, sycl::queue& q) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw std::invalid_argument("Matrix dimensions must be equal for element-wise multiplication.");
        }
        if (dest.rows != m1.rows || dest.cols != m1.cols) {
            throw std::invalid_argument("Destination matrix has incorrect dimensions.");
        }
        
        int r = m1.rows;
        int c = m1.cols;
        const float* ptrA = m1.data;
        const float* ptrB = m2.data;
        float* ptrC = dest.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrB;
            auto cacc = ptrC;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                cacc[i] = a[i] * b[i];
            });
        });
    }

    Matrix Transpose() const {
        Matrix output(cols, rows, *q_ptr);
        
        const int inputRows = rows;
        const int inputCols = cols;
        const float* inputPtr = data;
        float* outputPtr = output.data;

        // Use 2D parallel_for for efficient memory access patterns
        q_ptr->submit([&](handler& h) {
            h.parallel_for(range<2>(inputRows, inputCols), [=](id<2> idx) {
                int i = idx[0]; // row in input
                int j = idx[1]; // col in input
                // Transpose: input[i][j] -> output[j][i]
                outputPtr[j * inputRows + i] = inputPtr[i * inputCols + j];
            });
        });

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

    static void ApplyReLU(const Matrix& m, Matrix& dest, sycl::queue& q) {
        if (m.rows != dest.rows || m.cols != dest.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match source for ApplyReLU.");
        }
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = dest.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = ReLU(a[i]);
            });
        });
    }

    static void ApplyReLUDerivative(const Matrix& m, Matrix& dest, sycl::queue& q) {
        if (m.rows != dest.rows || m.cols != dest.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match source for ApplyReLUDerivative.");
        }
        int r = m.rows;
        int c = m.cols;

        const float* ptrA = m.data;
        float* ptrOut = dest.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = dReLU(a[i]);
            });
        });
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

    static void ApplySigmoid(const Matrix& m, Matrix& dest, sycl::queue& q) {
        if (m.rows != dest.rows || m.cols != dest.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match source for ApplySigmoid.");
        }
        int r = m.rows;
        int c = m.cols;
        const float* ptrA = m.data;
        float* ptrOut = dest.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = Sigmoid(a[i]);
                });
            });
    }

    static void ApplySigmoidDerivative(const Matrix& m, Matrix& dest, sycl::queue& q) {
        if (m.rows != dest.rows || m.cols != dest.cols) {
            throw std::invalid_argument("Destination matrix dimensions do not match source for ApplySigmoidDerivative.");
        }
        int r = m.rows;
        int c = m.cols;
        const float* ptrA = m.data;
        float* ptrOut = dest.data;

        q.submit([&](handler& h) {
            auto a = ptrA;
            auto b = ptrOut;
            h.parallel_for(range<1>(r * c), [=](id<1> i) {
                b[i] = dSigmoid(a[i]);
                });
            });
    }

    Matrix sumAlongAxis(int axis) const {
        if (axis == 1) {
            Matrix res(rows, 1, *q_ptr);
            
            // Bounds check to prevent undefined behavior
            if (cols == 0 || rows == 0) {
                return res; // Return zero-initialized matrix
            }
            
            int N = cols;
            auto ptr = data;
            auto out = res.data;
            q_ptr->submit([&](handler& h) {
                h.parallel_for(range<1>(rows), [=](id<1> i) {
                    float s = 0;
                    for (int j = 0; j < N; ++j) s += ptr[i[0] * N + j];
                    out[i] = s;
                });
            });
            return res;
        }
        else {
            throw std::invalid_argument("Only axis 1 is implemented.");
        }
    }

    void sumAlongAxis(int axis, Matrix& dest) const {
        if (axis != 1) {
            throw std::invalid_argument("Only axis 1 is implemented.");
        }
        if (dest.rows != this->rows || dest.cols != 1) {
            throw std::invalid_argument("Destination matrix for sumAlongAxis must have 1 column and same number of rows as source.");
        }

        int N = cols;
        auto ptr = data;
        auto out = dest.data;
        q_ptr->submit([&](handler& h) {
            h.parallel_for(range<1>(rows), [=](id<1> i) {
                float s = 0;
                for (int j = 0; j < N; ++j) s += ptr[i[0] * N + j];
                out[i] = s;
                });
            });
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
        Softmax(m, out, q);
        return out;
    }

    static void Softmax(const Matrix& m, Matrix& out, sycl::queue& q) {
        if (m.rows != out.rows || m.cols != out.cols) {
            throw std::invalid_argument("Input and output matrices must have same dimensions for Softmax.");
        }
        int R = m.rows, C = m.cols;
        const float* A = m.data;
        float* B = out.data;

        // Re-use shared scratch buffers to avoid excessive device allocations
        if (static_cast<std::size_t>(C) > softmax_capacity) {
            // Ensure any previous kernels using the buffers are complete
            q.wait_and_throw();
            if (softmax_tmp_max)  sycl::free(softmax_tmp_max, q);
            if (softmax_tmp_sum)  sycl::free(softmax_tmp_sum, q);

            softmax_tmp_max  = sycl::malloc_shared<float>(C, q);
            softmax_tmp_sum  = sycl::malloc_shared<float>(C, q);
            softmax_capacity = static_cast<std::size_t>(C);
        }

        float* max_vals = softmax_tmp_max;
        float* sums     = softmax_tmp_sum;

        // Kernel 1: Find max value and sum of exps for each column
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
                int col = id[0];

                // 1. Find per-column max
                float maxv = -std::numeric_limits<float>::infinity();
                for (int r_idx = 0; r_idx < R; ++r_idx) {
                    maxv = sycl::max(maxv, A[r_idx * C + col]);
                }
                max_vals[col] = maxv;

                // 2. Compute sum of exps
                float sum = 0.0f;
                for (int r_idx = 0; r_idx < R; ++r_idx) {
                    sum += sycl::exp(A[r_idx * C + col] - maxv);
                }
                sums[col] = sum;
            });
        }).wait(); // Wait for max/sum to be computed before normalization

        // Kernel 2: Normalize
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(R, C), [=](sycl::id<2> id) {
                int row = id[0];
                int col = id[1];
                float maxv = max_vals[col];
                float sum = sums[col];
                
                // 3. Write normalized value
                if (sum > 0) { // Avoid division by zero
                    B[row * C + col] = sycl::exp(A[row * C + col] - maxv) / sum;
                } else {
                    B[row * C + col] = 0.0f;
                }
            });
        });

        // NOTE: We do not free the shared scratch buffers here; they are reused.
        // They will be freed automatically at program termination, or can be
        // manually resized in future calls when a larger capacity is needed.
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

