cmake_minimum_required(VERSION 3.20)
project(ann_sycl LANGUAGES CXX)

set(CMAKE_CXX_COMPILER dpcpp)
set(CMAKE_CXX_STANDARD 20)

# put your include dirs here:
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(ann_app
  ImageNew.cpp
  MNIST_experiment.cpp
)

# Compile‑time flags
target_compile_options(ann_app PRIVATE
  -fno-sycl-rdc
  -fsycl-device-code-split=per_kernel
)

# Link‑time flags
target_link_libraries(ann_app PRIVATE
  "-fno-sycl-device-lib=all"
  "-fsycl-device-lib=libm-fp32"
  "-fsycl-max-parallel-link-jobs=4"
  mkl_sycl_dll
  mkl_intel_ilp64
  mkl_tbb_thread_dll
  mkl_core_dll
  OpenCL
)

# Tell CMake where oneMKL lives:
link_directories("$ENV{ONEAPI_ROOT}/mkl/2024.2/lib")
