#pragma once
#include <sycl/sycl.hpp>

// N  : batch size
// C  : in‑channels
// H,W: spatial dims
// kH,kW, pad, stride = 1 for simplicity
// out = [C*kH*kW, N*Hout*Wout]
template<int kH = 3, int kW = 3>
void im2col(const float* in,            // USM pointer (N,C,H,W)
            int N, int C, int H, int W,
            float* out,                 // USM pointer (cols)
            sycl::queue& q)
{
    const int Hout = H - kH + 1;        // no padding, stride=1
    const int Wout = W - kW + 1;
    const int Cols = N * Hout * Wout;
    const int Rows = C * kH * kW;

    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<2>(Rows, Cols), [=](sycl::id<2> id){
            int r = id[0];                  // which weight element
            int c = id[1];                  // which output position

            int kw =  r % kW;               // unravel r
            int kh = (r / kW)        % kH;
            int ch = (r / kW / kH);

            int n  =  c % N;
            int x  = (c / N) % Wout;
            int y  = (c / N / Wout);

            int in_x = x + kw;
            int in_y = y + kh;

            size_t in_idx  = ((((size_t)n*C + ch)*H + in_y)*W + in_x);
            size_t out_idx = r*Cols + c;
            out[out_idx]   = in[in_idx];
        });
    });
}

// col2im for backward pass (reverse of im2col)
template<int kH = 3, int kW = 3>
void col2im(const float* in,            // USM pointer (cols)
            int N, int C, int H, int W,
            float* out,                 // USM pointer (N,C,H,W)
            sycl::queue& q)
{
    const int Hout = H - kH + 1;        // no padding, stride=1
    const int Wout = W - kW + 1;
    const int Cols = N * Hout * Wout;
    const int Rows = C * kH * kW;

    // First zero out the output
    q.memset(out, 0, N * C * H * W * sizeof(float));

    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<2>(Rows, Cols), [=](sycl::id<2> id){
            int r = id[0];                  // which weight element
            int c = id[1];                  // which output position

            int kw =  r % kW;               // unravel r
            int kh = (r / kW)        % kH;
            int ch = (r / kW / kH);

            int n  =  c % N;
            int x  = (c / N) % Wout;
            int y  = (c / N / Wout);

            int in_x = x + kw;
            int in_y = y + kh;

            size_t out_idx = ((((size_t)n*C + ch)*H + in_y)*W + in_x);
            size_t in_idx  = r*Cols + c;
            
            // Atomic add to handle overlapping regions
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_out(out[out_idx]);
            atomic_out += in[in_idx];
        });
    });
} 