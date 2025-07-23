#include "SYCLMatrixLib.h"
#include "1DMatrixLib.h"
#include <omp.h>
#include <iostream>
#include "NeuralNetwork1DMat.h"
#include <cassert>

int N = 2048;
void testMatrixMultiply(sycl::queue& q) {
    // int N = 1024;
    Matrix A(N, N);
    Matrix B(N, N);
    A.RandInit();
    B.RandInit();

    double start = omp_get_wtime();
    A.multiply(B, q);
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Matrix Multiply (" << N << "x" << N << ") took " << duration << " seconds" << std::endl;
}

void testMatrix1DMultiply() {
    // int N = 256;
    Matrix1D A(N, N);
    Matrix1D B(N, N);
    A.RandInit();
    B.RandInit();

    double start = omp_get_wtime();
    A.multiply(B);
    double duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Multiply (" << N << "x" << N << ") took " << duration << " seconds" << std::endl;
}

void testScalarOperations(sycl::queue& q) {
    Matrix A(128, 128);
    A.RandInit();

    double start = omp_get_wtime();
    A.multiplyScalar(2.0f, q);
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Scalar Multiplication took " << duration << " seconds" << std::endl;

    start = omp_get_wtime();
    A.addScalar(5.0f, q);
    duration = omp_get_wtime() - start;
    std::cout << "SYCL Scalar Addition took " << duration << " seconds" << std::endl;

    Matrix1D M(128, 128);
    M.RandInit();

    start = omp_get_wtime();
    M.multiplyScalar(2.0f);
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Scalar Multiplication took " << duration << " seconds" << std::endl;

    start = omp_get_wtime();
    M.addScalar(5.0f);
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Scalar Addition took " << duration << " seconds" << std::endl;
}

void testElementWiseMultiplication(sycl::queue& q) {
    // int N = 128;
    Matrix A(N, N);
    Matrix B(N, N);
    A.RandInit();
    B.RandInit();

    double start = omp_get_wtime();
    A.elementWiseMult(B, q);
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Element-wise Multiplication took " << duration << " seconds" << std::endl;

    Matrix1D M(N, N);
    Matrix1D n(N, N);
    M.RandInit();
    n.RandInit();

    start = omp_get_wtime();
    M.elementWiseMult(n);
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Element-wise Multiplication took " << duration << " seconds" << std::endl;
}

void testAddition(sycl::queue& q) {
    // int N = 256;
    Matrix A(N, N);
    Matrix B(N, N);
    A.RandInit();
    B.RandInit();

    double start = omp_get_wtime();
    A.add(B, q);
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Matrix Addition took " << duration << " seconds" << std::endl;

    Matrix1D M(N, N);
    Matrix1D n(N, N);
    M.RandInit();
    n.RandInit();

    start = omp_get_wtime();
    M.add(n);
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Addition took " << duration << " seconds" << std::endl;
}

void testTranspose(sycl::queue& q) {
    // int N = 128;
    Matrix A(N, N);
    A.RandInit();

    double start = omp_get_wtime();
    A.transpose();
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Matrix Transpose took " << duration << " seconds" << std::endl;

    Matrix1D M(N, N);
    M.RandInit();

    start = omp_get_wtime();
    M.transpose();
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Transpose took " << duration << " seconds" << std::endl;
}

void testSigmoidOperations(sycl::queue& q) {
    // int N = 128;
    Matrix A(N, N);
    A.RandInit();

    double start = omp_get_wtime();
    A.applySigmoid(q);
    double duration = omp_get_wtime() - start;
    std::cout << "SYCL Sigmoid Application took " << duration << " seconds" << std::endl;

    start = omp_get_wtime();
    A.applySigmoidDerivative(q);
    duration = omp_get_wtime() - start;
    std::cout << "SYCL Sigmoid Derivative Application took " << duration << " seconds" << std::endl;

    Matrix1D M(N, N);
    M.RandInit();

    start = omp_get_wtime();
    M.applySigmoid();
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Sigmoid Application took " << duration << " seconds" << std::endl;

    start = omp_get_wtime();
    M.applySigmoidDerivative();
    duration = omp_get_wtime() - start;
    std::cout << "Matrix1D Sigmoid Derivative Application took " << duration << " seconds" << std::endl;
}

int main() {
    sycl::queue q{ gpu_selector{} };

    std::cout << "Testing Matrix Multiplication..." << std::endl;
    testMatrixMultiply(q);
    testMatrix1DMultiply();

    std::cout << "\nTesting Scalar Operations..." << std::endl;
    testScalarOperations(q);

    std::cout << "\nTesting Element-wise Multiplication..." << std::endl;
    testElementWiseMultiplication(q);

    std::cout << "\nTesting Matrix Addition..." << std::endl;
    testAddition(q);

    std::cout << "\nTesting Transpose..." << std::endl;
    testTranspose(q);

    std::cout << "\nTesting Sigmoid Operations..." << std::endl;
    testSigmoidOperations(q);

    return 0;
}
