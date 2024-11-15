#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>
#include <CL/sycl.hpp>


using namespace std;
using namespace cl::sycl;

class Matrix {
public:
    int rows;
    int cols;
    vector<float> data;
    
    Matrix() : cols(0), rows(0), data({}) {}

    Matrix(int rows, int cols) : cols(cols), rows(rows), data(rows*cols) {}

    void RandInit() {
        std::random_device rd;
        std::uniform_real_distribution<float> dist(-1, 1);
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dist(rd);
        }
    }

    void printMatrix() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[i * cols + j] << "\t";
            }
            cout << "\n";
        }
    }

    void multiplyScalar(float n, sycl::queue& q) {
        
        buffer<float, 1> buf(data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto acc = buf.get_access<access::mode::read_write>(h);
            h.parallel_for(buf.get_range(), [=](id<1> i) {
                acc[i] *= n;
                });
            }).wait();
    }

    void addScalar(float n, sycl::queue& q) {
        
        buffer<float, 1> buf(data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto acc = buf.get_access<access::mode::read_write>(h);
            h.parallel_for(buf.get_range(), [=](id<1> i) {
                acc[i] += n;
                });
            }).wait();
    }

    void negate(sycl::queue& q) {
        
        buffer<float, 1> buf(data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto acc = buf.get_access<access::mode::read_write>(h);
            h.parallel_for(buf.get_range(), [=](id<1> i) {
                acc[i] *= -1;
                });
            }).wait();
    }

    Matrix Negate(sycl::queue& q) {
        Matrix m(rows, cols);
        
        buffer<float, 1> buf_in(data.data(), range<1>(rows * cols));
        buffer<float, 1> buf_out(m.data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto in = buf_in.get_access<access::mode::read>(h);
            auto out = buf_out.get_access<access::mode::write>(h);
            h.parallel_for(buf_in.get_range(), [=](id<1> i) {
                out[i] = in[i] * -1;
                });
            }).wait();

        return m;
    }

    void add(const Matrix& other, sycl::queue& q) {
        if (rows == other.rows && cols == other.cols) {
            
            buffer<float, 1> buf_a(data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_b(other.data.data(), range<1>(rows * cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read_write>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    a[i] += b[i];
                    });
                }).wait();
        }
        else {
            cout << "Dims of matrices must be equal to add both of them. Current object remains UNCHANGED.\n";
        }
    }

    static Matrix add(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        Matrix result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            
            buffer<float, 1> buf_a(m1.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_b(m2.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_c(result.data.data(), range<1>(m1.rows * m1.cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                auto c = buf_c.get_access<access::mode::write>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    c[i] = a[i] + b[i];
                    });
                }).wait();

            return result;
        }
        return Matrix();
    }

    Matrix Add(const Matrix& other, sycl::queue& q) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols);
            
            buffer<float, 1> buf_a(data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_b(other.data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_c(output.data.data(), range<1>(rows * cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                auto c = buf_c.get_access<access::mode::write>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    c[i] = a[i] + b[i];
                    });
                }).wait();

            return output;
        }
        return Matrix();
    }

    static Matrix subtract(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        Matrix result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            
            buffer<float, 1> buf_a(m1.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_b(m2.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_c(result.data.data(), range<1>(m1.rows * m1.cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                auto c = buf_c.get_access<access::mode::write>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    c[i] = a[i] - b[i];
                    });
                }).wait();

            return result;
        }
        return Matrix();
    }

    
    void multiply(const Matrix& other, sycl::queue& q) {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        const size_t TILE_SIZE = 8; // Tile size for optimization

        // Create output matrix
        Matrix output(rows, other.cols);

        // Create padded data vectors to avoid out-of-bounds access
        int paddedRows = (rows % TILE_SIZE == 0) ? rows : rows + (TILE_SIZE - (rows % TILE_SIZE));
        int paddedCols = (cols % TILE_SIZE == 0) ? cols : cols + (TILE_SIZE - (cols % TILE_SIZE));
        int paddedOtherCols = (other.cols % TILE_SIZE == 0) ? other.cols : other.cols + (TILE_SIZE - (other.cols % TILE_SIZE));

        std::vector<float> paddedA(paddedRows * paddedCols, 0.0f);
        std::vector<float> paddedB(paddedCols * paddedOtherCols, 0.0f);
        std::vector<float> paddedC(paddedRows * paddedOtherCols, 0.0f);

        // Copy original data into padded vectors
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                paddedA[i * paddedCols + j] = data[i * cols + j];
            }
        }

        for (int i = 0; i < other.rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                paddedB[i * paddedOtherCols + j] = other.data[i * other.cols + j];
            }
        }

        buffer<float, 2> bufferA(paddedA.data(), range<2>(paddedRows, paddedCols));
        buffer<float, 2> bufferB(paddedB.data(), range<2>(paddedCols, paddedOtherCols));
        buffer<float, 2> bufferC(paddedC.data(), range<2>(paddedRows, paddedOtherCols));

        // Set the global range to be padded to the nearest multiple of TILE_SIZE
        size_t globalRows = ((paddedRows + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        size_t globalCols = ((paddedOtherCols + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

        q.submit([&](handler& cgh) {
            accessor a(bufferA, cgh, read_only);
            accessor b(bufferB, cgh, read_only);
            accessor c(bufferC, cgh, read_write);

            accessor<float, 1, access::mode::read_write, access::target::local> localA(range<1>(TILE_SIZE * TILE_SIZE), cgh);
            accessor<float, 1, access::mode::read_write, access::target::local> localB(range<1>(TILE_SIZE * TILE_SIZE), cgh);

            cgh.parallel_for(nd_range<2>(range<2>(globalRows, globalCols), range<2>(TILE_SIZE, TILE_SIZE)), [=](nd_item<2> item) {
                int globalRow = item.get_global_id(0);
                int globalCol = item.get_global_id(1);
                int localRow = item.get_local_id(0);
                int localCol = item.get_local_id(1);

                float sum = 0.0f;

                for (int k = 0; k < paddedCols; k += TILE_SIZE) {
                    if (globalRow < paddedRows && (k + localCol) < paddedCols)
                        localA[localRow * TILE_SIZE + localCol] = a[globalRow][k + localCol];
                    else
                        localA[localRow * TILE_SIZE + localCol] = 0.0f;

                    if (globalCol < paddedOtherCols && (k + localRow) < paddedCols)
                        localB[localRow * TILE_SIZE + localCol] = b[k + localRow][globalCol];
                    else
                        localB[localRow * TILE_SIZE + localCol] = 0.0f;

                    item.barrier(access::fence_space::local_space);

                    for (int j = 0; j < TILE_SIZE; ++j) {
                        sum += localA[localRow * TILE_SIZE + j] * localB[j * TILE_SIZE + localCol];
                    }
                    item.barrier(access::fence_space::local_space);
                }

                if (globalRow < paddedRows && globalCol < paddedOtherCols) {
                    c[globalRow][globalCol] = sum;
                }
                });
            }).wait();

        // Copy result back to the original output matrix (non-padded)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                output.data[i * other.cols + j] = paddedC[i * paddedOtherCols + j];
            }
        }

        // Update current matrix with the result
        data = std::move(output.data);
        rows = output.rows;
        cols = output.cols;
    }


    static Matrix multiply(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.cols != m2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        const size_t TILE_SIZE = 16; // Tile size for optimization

        // Create output matrix
        Matrix result(m1.rows, m2.cols);

        // Create padded data vectors to avoid out-of-bounds access
        int paddedRows = (m1.rows % TILE_SIZE == 0) ? m1.rows : m1.rows + (TILE_SIZE - (m1.rows % TILE_SIZE));
        int paddedCols = (m1.cols % TILE_SIZE == 0) ? m1.cols : m1.cols + (TILE_SIZE - (m1.cols % TILE_SIZE));
        int paddedOtherCols = (m2.cols % TILE_SIZE == 0) ? m2.cols : m2.cols + (TILE_SIZE - (m2.cols % TILE_SIZE));

        std::vector<float> paddedA(paddedRows * paddedCols, 0.0f);
        std::vector<float> paddedB(paddedCols * paddedOtherCols, 0.0f);
        std::vector<float> paddedC(paddedRows * paddedOtherCols, 0.0f);

        // Copy original data into padded vectors
        for (int i = 0; i < m1.rows; ++i) {
            for (int j = 0; j < m1.cols; ++j) {
                paddedA[i * paddedCols + j] = m1.data[i * m1.cols + j];
            }
        }

        for (int i = 0; i < m2.rows; ++i) {
            for (int j = 0; j < m2.cols; ++j) {
                paddedB[i * paddedOtherCols + j] = m2.data[i * m2.cols + j];
            }
        }

        buffer<float, 2> bufferA(paddedA.data(), range<2>(paddedRows, paddedCols));
        buffer<float, 2> bufferB(paddedB.data(), range<2>(paddedCols, paddedOtherCols));
        buffer<float, 2> bufferC(paddedC.data(), range<2>(paddedRows, paddedOtherCols));

        // Set the global range to be padded to the nearest multiple of TILE_SIZE
        size_t globalRows = ((paddedRows + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        size_t globalCols = ((paddedOtherCols + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

        q.submit([&](handler& cgh) {
            accessor a(bufferA, cgh, read_only);
            accessor b(bufferB, cgh, read_only);
            accessor c(bufferC, cgh, write_only, noinit);

            accessor<float, 1, access::mode::read_write, access::target::local> localA(TILE_SIZE * TILE_SIZE, cgh);
            accessor<float, 1, access::mode::read_write, access::target::local> localB(TILE_SIZE * TILE_SIZE, cgh);

            cgh.parallel_for(nd_range<2>(range<2>(globalRows, globalCols), range<2>(TILE_SIZE, TILE_SIZE)), [=](nd_item<2> item) {
                int globalRow = item.get_global_id(0);
                int globalCol = item.get_global_id(1);
                int localRow = item.get_local_id(0);
                int localCol = item.get_local_id(1);

                float sum = 0.0f;

                for (int k = 0; k < paddedCols; k += TILE_SIZE) {
                    if (globalRow < paddedRows && (k + localCol) < paddedCols)
                        localA[localRow * TILE_SIZE + localCol] = a[globalRow][k + localCol];
                    else
                        localA[localRow * TILE_SIZE + localCol] = 0.0f;

                    if (globalCol < paddedOtherCols && (k + localRow) < paddedCols)
                        localB[localRow * TILE_SIZE + localCol] = b[k + localRow][globalCol];
                    else
                        localB[localRow * TILE_SIZE + localCol] = 0.0f;

                    item.barrier(access::fence_space::local_space);

                    for (int j = 0; j < TILE_SIZE; ++j) {
                        sum += localA[localRow * TILE_SIZE + j] * localB[j * TILE_SIZE + localCol];
                    }

                    item.barrier(access::fence_space::local_space);
                }

                if (globalRow < paddedRows && globalCol < paddedOtherCols) {
                    c[globalRow][globalCol] = sum;
                }
                });
            }).wait();

        // Copy result back to the original output matrix (non-padded)
        for (int i = 0; i < m1.rows; ++i) {
            for (int j = 0; j < m2.cols; ++j) {
                result.data[i * m2.cols + j] = paddedC[i * paddedOtherCols + j];
            }
        }

        return result;
    }


    void elementWiseMult(const Matrix& other, sycl::queue& q) {
        if (rows == other.rows && cols == other.cols) {
            
            buffer<float, 1> buf_a(data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_b(other.data.data(), range<1>(rows * cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read_write>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    a[i] *= b[i];
                    });
                }).wait();
        }
        else {
            cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
        }
    }

    Matrix ElementWiseMult(const Matrix& other, sycl::queue& q) {
        if (rows == other.rows && cols == other.cols) {
            Matrix output(rows, cols);
            
            buffer<float, 1> buf_a(data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_b(other.data.data(), range<1>(rows * cols));
            buffer<float, 1> buf_c(output.data.data(), range<1>(rows * cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                auto c = buf_c.get_access<access::mode::write>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    c[i] = a[i] * b[i];
                    });
                }).wait();

            return output;
        }
        return Matrix();
    }

    static Matrix ElementWiseMult(const Matrix& m1, const Matrix& m2, sycl::queue& q) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix output(m1.rows, m1.cols);
            
            buffer<float, 1> buf_a(m1.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_b(m2.data.data(), range<1>(m1.rows * m1.cols));
            buffer<float, 1> buf_c(output.data.data(), range<1>(m1.rows * m1.cols));

            q.submit([&](handler& h) {
                auto a = buf_a.get_access<access::mode::read>(h);
                auto b = buf_b.get_access<access::mode::read>(h);
                auto c = buf_c.get_access<access::mode::write>(h);
                h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                    c[i] = a[i] * b[i];
                    });
                }).wait();

            return output;
        }
        return Matrix();
    }

    void transpose() {
        Matrix output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        *this = output;
    }

    Matrix Transpose() const {
        Matrix output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        return output;
    }

    static float Sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

    static float dSigmoid(float x) {
        return x * (1 - x);
    }

    void applySigmoid(sycl::queue& q) {
        
        buffer<float, 1> buf(data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto acc = buf.get_access<access::mode::read_write>(h);
            h.parallel_for(buf.get_range(), [=](id<1> i) {
                acc[i] = Sigmoid(acc[i]);
                });
            }).wait();
    }

    void applySigmoidDerivative(sycl::queue& q) {
        
        buffer<float, 1> buf(data.data(), range<1>(rows * cols));

        q.submit([&](handler& h) {
            auto acc = buf.get_access<access::mode::read_write>(h);
            h.parallel_for(buf.get_range(), [=](id<1> i) {
                acc[i] = dSigmoid(acc[i]);
                });
            }).wait();
    }

    static Matrix ApplySigmoid(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols);
        
        buffer<float, 1> buf_a(m.data.data(), range<1>(m.rows * m.cols));
        buffer<float, 1> buf_b(output.data.data(), range<1>(m.rows * m.cols));

        q.submit([&](handler& h) {
            auto a = buf_a.get_access<access::mode::read>(h);
            auto b = buf_b.get_access<access::mode::write>(h);
            h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                b[i] = Sigmoid(a[i]);
                });
            }).wait();

        return output;
    }

    static Matrix ApplySigmoidDerivative(const Matrix& m, sycl::queue& q) {
        Matrix output(m.rows, m.cols);
        
        buffer<float, 1> buf_a(m.data.data(), range<1>(m.rows * m.cols));
        buffer<float, 1> buf_b(output.data.data(), range<1>(m.rows * m.cols));

        q.submit([&](handler& h) {
            auto a = buf_a.get_access<access::mode::read>(h);
            auto b = buf_b.get_access<access::mode::write>(h);
            h.parallel_for(buf_a.get_range(), [=](id<1> i) {
                b[i] = dSigmoid(a[i]);
                });
            }).wait();

        return output;
    }

    Matrix sumAlongAxis(int axis) const {
        if (axis == 1) {
            // Sum along each row (axis = 1), which means sum across columns for each row
            Matrix result(rows, 1);
            for (int i = 0; i < rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += data[i * cols + j];
                }
                result.data[i] = sum;
            }
            return result;
        }
        else {
            throw std::invalid_argument("Only axis 1 (sum along rows) is implemented.");
        }
    }


    static Matrix fromArr(const vector<float>& arr) {
        Matrix output(arr.size(), 1);
        for (int i = 0; i < output.rows; i++) {
            output.data[i] = arr[i];
        }
        return output;
    }

    vector<float> toArr() const {
        return data;
    }
};

template<>
struct is_device_copyable<Matrix> : std::true_type {};
