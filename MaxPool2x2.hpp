#pragma once
#include "SYCLMatrixLib.h"
#include <vector>

class MaxPool2x2 {
private:
    int C, H, W;
    sycl::queue* q_ptr;
    int* idxMask_device;     // Device memory for storing arg-max indices
    int cached_batch_size;

public:
    MaxPool2x2(int c, int h, int w, sycl::queue& q) 
        : C(c), H(h), W(w), q_ptr(&q), idxMask_device(nullptr), cached_batch_size(0) {}

    ~MaxPool2x2() {
        if (idxMask_device && q_ptr) {
            sycl::free(idxMask_device, *q_ptr);
        }
    }

    Matrix forward(const Matrix& x, int N) {
        int Hout = H / 2, Wout = W / 2;
        Matrix y(C * Hout * Wout, N, *q_ptr);
        
        // Allocate or reallocate mask if needed
        cached_batch_size = N;
        if (!idxMask_device) {
            idxMask_device = sycl::malloc_device<int>(C * Hout * Wout * N, *q_ptr);
        } else if (cached_batch_size > N) {
            sycl::free(idxMask_device, *q_ptr);
            idxMask_device = sycl::malloc_device<int>(C * Hout * Wout * N, *q_ptr);
        }

        const float* x_data = x.data;
        float* y_data = y.data;
        int* mask_data = idxMask_device;

        q_ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(N, C * Hout * Wout), [=](sycl::id<2> id) {
                int n = id[0];
                int idx = id[1];

                int c = idx / (Hout * Wout);
                int hw = idx % (Hout * Wout);
                int y0 = (hw / Wout) * 2;
                int x0 = (hw % Wout) * 2;

                float maxv = -1e9f; 
                int arg = 0;
                
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int in_y = y0 + dy, in_x = x0 + dx;
                        // Input format: [C*H*W, N] so index is [c*H*W + in_y*W + in_x][n]
                        size_t flat_spatial = c * H * W + in_y * W + in_x;
                        size_t input_idx = flat_spatial * N + n;
                        float v = x_data[input_idx];
                        if (v > maxv) {
                            maxv = v; 
                            arg = flat_spatial;
                        }
                    }
                }
                
                // Output format: [C*Hout*Wout, N]
                size_t output_idx = idx * N + n;
                y_data[output_idx] = maxv;
                mask_data[output_idx] = arg;
            });
        });
        
        return y;
    }

    Matrix backward(const Matrix& dY, int N) {
        int Hout = H / 2, Wout = W / 2;
        Matrix dX(C * H * W, N, *q_ptr);
        
        // Zero out gradients
        q_ptr->memset(dX.data, 0, C * H * W * N * sizeof(float));

        const float* dY_data = dY.data;
        float* dX_data = dX.data;
        int* mask_data = idxMask_device;

        q_ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(N, C * Hout * Wout), [=](sycl::id<2> id) {
                int n = id[0];
                int idx = id[1];

                // Get the stored argmax index
                size_t output_idx = idx * N + n;
                int arg_max_spatial = mask_data[output_idx];
                float grad = dY_data[output_idx];

                // Scatter gradient back to the argmax position
                size_t input_idx = arg_max_spatial * N + n;
                
                // Use atomic add to handle potential race conditions
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> 
                    atomic_dx(dX_data[input_idx]);
                atomic_dx += grad;
            });
        });

        return dX;
    }

    int getOutputChannels() const { return C; }
    int getOutputHeight() const { return H / 2; }
    int getOutputWidth() const { return W / 2; }
}; 