#pragma once
#include "SYCLMatrixLib.h"
#include "im2col.hpp"
#include <vector>

class ConvLayer {
private:
    int C, H, W, K;          // in‑channels, height, width, out‑channels
    int kH = 3, kW = 3;
    sycl::queue* q_ptr;

    Matrix W_;               // [K, C*kH*kW]
    Matrix b_;               // [K,1]
    Matrix colBuf_;          // [C*kH*kW, batch*Hout*Wout]
    
    // For backward pass
    float* input_cache;      // Cache input for backward pass
    int cached_batch_size;

public:
    ConvLayer(int inC, int inH, int inW, int outC, sycl::queue& q)
      : C(inC), H(inH), W(inW), K(outC), q_ptr(&q),
        W_(outC, inC*kH*kW, q),
        b_(outC, 1, q),
        colBuf_(0, 0, q),
        input_cache(nullptr),
        cached_batch_size(0)
    {
        // He initialization for ReLU
        W_.RandInit(inC*kH*kW, outC);
        // Biases initialized to 0 (already done by Matrix constructor)
    }

    ~ConvLayer() {
        if (input_cache && q_ptr) {
            sycl::free(input_cache, *q_ptr);
        }
    }

    // Forward pass
    Matrix forward(const float* x, int N) {
        int Hout = H - kH + 1, Wout = W - kW + 1;
        int cols = N * Hout * Wout;
        
        // Resize colBuf if needed
        if (colBuf_.rows != C*kH*kW || colBuf_.cols != cols) {
            colBuf_ = Matrix(C*kH*kW, cols, *q_ptr);
        }

        // Cache input for backward pass
        cached_batch_size = N;
        if (!input_cache) {
            input_cache = sycl::malloc_device<float>(N * C * H * W, *q_ptr);
        } else if (cached_batch_size > N) {
            // Reallocate if needed
            sycl::free(input_cache, *q_ptr);
            input_cache = sycl::malloc_device<float>(N * C * H * W, *q_ptr);
        }
        q_ptr->memcpy(input_cache, x, N * C * H * W * sizeof(float));

        // im2col transformation
        im2col<3,3>(x, N, C, H, W, colBuf_.data, *q_ptr);
        
        // Matrix multiplication: W * colBuf
        Matrix out = Matrix::multiply(W_, colBuf_, *q_ptr);   // [K, cols]
        
        // Add bias (broadcast)
        out.add(b_);
        
        return out;  // Shape: [K, N*Hout*Wout]
    }

    // Backward pass
    Matrix backward(const Matrix& dY, float learning_rate) {
        int Hout = H - kH + 1, Wout = W - kW + 1;
        int cols = cached_batch_size * Hout * Wout;

        // 1. Compute gradients w.r.t. weights: dW = dY * colBuf.T
        Matrix dW = Matrix::multiply(dY, colBuf_.Transpose(), *q_ptr);
        dW.multiplyScalar(-learning_rate);
        W_.add(dW);

        // 2. Compute gradients w.r.t. bias: db = sum(dY, axis=1)
        Matrix db = dY.sumAlongAxis(1);
        db.multiplyScalar(-learning_rate);
        b_.add(db);

        // 3. Compute gradients w.r.t. input: dX = W.T * dY
        Matrix dColBuf = Matrix::multiply(W_.Transpose(), dY, *q_ptr);

        // 4. col2im to get gradients w.r.t. input
        float* dX = sycl::malloc_device<float>(cached_batch_size * C * H * W, *q_ptr);
        col2im<3,3>(dColBuf.data, cached_batch_size, C, H, W, dX, *q_ptr);

        // Create Matrix wrapper for dX
        Matrix dX_matrix(cached_batch_size * C * H * W, 1, *q_ptr);
        q_ptr->memcpy(dX_matrix.data, dX, cached_batch_size * C * H * W * sizeof(float));
        sycl::free(dX, *q_ptr);

        return dX_matrix;
    }

    int getOutputChannels() const { return K; }
    int getOutputHeight() const { return H - kH + 1; }
    int getOutputWidth() const { return W - kW + 1; }
}; 