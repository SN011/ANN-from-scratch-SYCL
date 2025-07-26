#pragma once
#include <sycl/sycl.hpp>

//
// 16×16 tiled transpose that operates in GPU.
// Works for any M×N tensor stored in row-major USM.
//
inline sycl::event gpu_transpose(sycl::queue& q,
                          const float* src,
                          float*       dst,
                          int          rows,
                          int          cols)
{
    constexpr size_t BLK = 16;
    sycl::range<2> g{ (cols + BLK - 1) / BLK * BLK,
                      (rows + BLK - 1) / BLK * BLK };
    sycl::range<2> l{ BLK, BLK };

    return q.submit([&](sycl::handler& h)
    {
        h.parallel_for(
            sycl::nd_range<2>(g, l),
            [=](sycl::nd_item<2> it)
            [[sycl::reqd_sub_group_size(16)]]
        {
            int r = it.get_global_id(1);
            int c = it.get_global_id(0);
            if (r < rows && c < cols)
                dst[c * rows + r] = src[r * cols + c];
        });
    });
} 
