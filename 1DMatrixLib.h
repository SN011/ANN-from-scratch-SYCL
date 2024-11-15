#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>

using namespace std;

class Matrix1D {
public:
    int rows;
    int cols;
    vector<float> data;

    Matrix1D() : cols(0), rows(0), data({}) {}

    Matrix1D(int rows, int cols) : cols(cols), rows(rows), data(rows* cols) {}

    void RandInit() {
        std::random_device rd;
        std::uniform_real_distribution<float> dist(-1, 1);
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dist(rd);
        }
    }

    void printMatrix1D() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[i * cols + j] << "\t";
            }
            cout << "\n";
        }
    }

    // Multiplies current object by a scalar
    void multiplyScalar(float n) {
        for (int i = 0; i < rows * cols; i++) {
            data[i] *= n;
        }
    }

    // Adds scalar value to current object
    void addScalar(float n) {
        for (int i = 0; i < rows * cols; i++) {
            data[i] += n;
        }
    }

    // Makes all values in current Matrix1D object negative
    void negate() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] *= -1;
        }
    }

    // This negates the values in the current object and loads them into another Matrix1D which is returned
    Matrix1D Negate() {
        Matrix1D m(rows, cols);
        for (int i = 0; i < rows * cols; i++) {
            m.data[i] = data[i] * -1;
        }
        return m;
    }

    // Current object is manipulated -> other Matrix1D's values are added to the current object only if condition is met
    // otherwise the message will be printed
    void add(Matrix1D& other) {
        if (rows == other.rows && cols == other.cols) {
            for (int i = 0; i < rows * cols; i++) {
                data[i] += other.data[i];
            }
        }
        else {
            cout << "Dims of matrices must be equal to add both of them. Current object remains UNCHANGED.\n";
        }
    }

    // Static method to represent 'add' method for the whole Matrix1D class; synthesizes two Matrix1D objects and returns result
    // if condition is not met, Matrix1D() is returned
    static Matrix1D add(Matrix1D& m1, Matrix1D& m2) {
        Matrix1D result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                result.data[i] = m1.data[i] + m2.data[i];
            }
            return result;
        }
        return Matrix1D();
    }

    // Instance method Add which does same thing as void add method but returns a new Matrix1D object
    Matrix1D Add(Matrix1D& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix1D output(rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                output.data[i] = data[i] + other.data[i];
            }
            return output;
        }
        return Matrix1D();
    }

    // Method that subtracts 2 matrices and returns result. It is static in order to represent subtraction method for the whole Matrix1D class
    static Matrix1D subtract(Matrix1D m1, Matrix1D m2) {
        Matrix1D result(m1.rows, m1.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                result.data[i] = m1.data[i] - m2.data[i];
            }
            return result;
        }
        return Matrix1D();
    }

    // Instance method that multiplies other Matrix1D's values with current Matrix1D's values and sets dimensions accordingly using setter methods
    // if condition not met a message is printed
    void multiply(Matrix1D& other) {
        if (cols != other.rows) {
            cout << "Cols of first Matrix1D should equal rows of second Matrix1D to multiply both of them. Current object remains UNCHANGED.\n";
            return;
        }

        Matrix1D output(rows, other.cols);
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                float sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i * cols + k] * other.data[k * other.cols + j];
                }
                output.data[i * output.cols + j] = sum;
            }
        }

        *this = output;
    }

    // Static method for the whole class for multiplication; multiplies two matrices and returns result
    // if condition not met Matrix1D() is returned
    static Matrix1D multiply(Matrix1D m1, Matrix1D m2) {
        if (m1.cols != m2.rows) {
            cout << "Cols of first Matrix1D should equal rows of second Matrix1D to multiply both of them.\n";
            return Matrix1D();
        }

        Matrix1D output(m1.rows, m2.cols);
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                float sum = 0;
                for (int k = 0; k < m1.cols; k++) {
                    sum += m1.data[i * m1.cols + k] * m2.data[k * m2.cols + j];
                }
                output.data[i * output.cols + j] = sum;
            }
        }

        return output;
    }

    // HADAMARD PRODUCT - ELEMENT WISE Matrix1D MULTIPLICATION
    // Multiplies val in 1st Matrix1D to corresponding val in second Matrix1D
    void elementWiseMult(Matrix1D& other) {
        if (rows == other.rows && cols == other.cols) {
            for (int i = 0; i < rows * cols; i++) {
                data[i] *= other.data[i];
            }
        }
        else {
            cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
        }
    }

    // Instance method that returns a Matrix1D after doing element-wise multiplication
    // Returns Matrix1D() if condition not met
    Matrix1D ElementWiseMult(Matrix1D& other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix1D output(rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                output.data[i] = data[i] * other.data[i];
            }
            return output;
        }
        return Matrix1D();
    }

    // Static method for element-wise multiplication, for the whole Matrix1D class
    // Returns Matrix1D() if condition not met
    static Matrix1D ElementWiseMult(Matrix1D& m1, Matrix1D& m2) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            Matrix1D output(m1.rows, m1.cols);
            for (int i = 0; i < m1.rows * m1.cols; i++) {
                output.data[i] = m1.data[i] * m2.data[i];
            }
            return output;
        }
        return Matrix1D();
    }

    // Transposes current Matrix1D object and uses setter methods to set dimensions of the modified current object accordingly
    void transpose() {
        Matrix1D output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        *this = output;
    }

    // Instance method for transpose that returns resultant, transposed Matrix1D
    Matrix1D Transpose() {
        Matrix1D output(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output.data[j * rows + i] = data[i * cols + j];
            }
        }
        return output;
    }

    // Neural Network functions
    static float Sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

    static float dSigmoid(float x) {
        return x * (1 - x);
    }

    void applySigmoid() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] = Sigmoid(data[i]);
        }
    }

    void applySigmoidDerivative() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dSigmoid(data[i]);
        }
    }

    static Matrix1D ApplySigmoid(Matrix1D& m) {
        Matrix1D output(m.rows, m.cols);
        for (int i = 0; i < m.rows * m.cols; i++) {
            output.data[i] = Sigmoid(m.data[i]);
        }
        return output;
    }

    static Matrix1D ApplySigmoidDerivative(Matrix1D& m) {
        Matrix1D output(m.rows, m.cols);
        for (int i = 0; i < m.rows * m.cols; i++) {
            output.data[i] = dSigmoid(m.data[i]);
        }
        return output;
    }

    // Functions to convert from 1d array to Matrix1D and from Matrix1D to 1d array
    // Returns (1 col. Matrix1D) column vector of 'arr.length' no of rows and 1 column
    static Matrix1D fromArr(vector<float>& arr) {
        Matrix1D output(arr.size(), 1);
        for (int i = 0; i < output.rows; i++) {
            output.data[i] = arr[i];
        }
        return output;
    }

    // Load Matrix1D elements into array of size [rows*cols]
    vector<float> toArr() {
        return data;
    }
};

