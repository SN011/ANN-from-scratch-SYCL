#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <random>
using namespace std;
class Matrix
{
public:
	int rows;
	int cols;
	vector<vector<float>> data;

	Matrix()
		:cols(0), rows(0), data({})
	{

	};

	Matrix(int rows, int cols)
		:cols(cols), rows(rows), data({})
	{
		data.resize(rows, vector<float>(cols));
	};

	void RandInit() {
		std::random_device rd;
		std::uniform_real_distribution<float> dist(-1,1);
		//std::normal_distribution<float> dist2(-1,1);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				data[i][j] = dist(rd);
			}
		}
	}

	void printMatrix() {
		for (size_t i = 0; i < this->data.size(); i++)
		{
			for (size_t j = 0; j < this->data[0].size(); j++)
			{
				cout << this->data[i][j] << "\t";
			}
			cout << "\n";
		}
	}
	//Multiplies current object by a scalar
	//Multiplies current object by a scalar
	void multiplyScalar(float n) {
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				this->data[i][j] *= n;
			}
		}
	}

	//Adds scalar value to current object
	void addScalar(float n) {
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				this->data[i][j] += n;
			}
		}
	}

	//Makes all values in current Matrix object negative
	void negate() {
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				this->data[i][j] *= (-1);
			}
		}
	}

	//This negates the values in the current object and loads them into another matrix which is returned
	Matrix Negate() {
		Matrix m(this->rows, this->cols);
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				m.data[i][j] = this->data[i][j] * (-1);
			}
		}
		return m;
	}

	//Current object is manipulated --> other matrix's values are added to the current object only if condition is met
	//otherwise the message will be printed
	void add(Matrix& other) {
		if (this->rows == other.rows && this->cols == other.cols) {
			for (int i = 0; i < this->rows; i++) {
				for (int j = 0; j < this->cols; j++) {
					this->data[i][j] += other.data[i][j];
				}
			}
		}
		else cout << "Dims of matrices must be equal to add both of them. Current object remains UNCHANGED.\n";
	}

	//Static method to represent 'add' method for the whole Matrix class; synthesizes two matrix objects and returns result
	//if condition is not met, Matrix() is returned
	static Matrix add(Matrix& m1, Matrix& m2) {
		Matrix result(m1.rows, m1.cols);
		if (m1.rows == m2.rows && m1.cols == m2.cols) {
			for (int i = 0; i < m1.rows; i++) {
				for (int j = 0; j < m1.cols; j++) {
					result.data[i][j] = m1.data[i][j] + m2.data[i][j];
				}
			}
			return result;
		}
		return Matrix();
	}

	//Instance method Add which does same thing as void add method but returns a new matrix object
	Matrix Add(Matrix& other) {

		if (this->rows == other.rows && this->cols == other.cols) {
			Matrix output(this->rows, this->cols);
			for (int i = 0; i < this->rows; i++) {
				for (int j = 0; j < this->cols; j++) {
					output.data[i][j] = this->data[i][j] + other.data[i][j];
				}
			}
			return output;
		}
		return Matrix();
	}

	//Method that subtracts 2 matrices and returns result. It is static in order to represent subtraction method for the whole Matrix class
	static Matrix subtract(Matrix m1, Matrix m2) {
		Matrix result(m1.rows, m1.cols);
		if (m1.rows == m2.rows && m1.cols == m2.cols) {
			for (int i = 0; i < m1.rows; i++) {
				for (int j = 0; j < m1.cols; j++) {
					result.data[i][j] = m1.data[i][j] - m2.data[i][j];
				}
			}
			return result;
		}
		return Matrix();
	}

	//Instance method that mutiplies other matrix's values with current matrix's values and sets dimensions accordingly using setter methods
	//if condition not met a message is printed
	void multiply(Matrix& other) {
		Matrix output(this->rows, other.cols);
		if (this->cols == other.rows) {
			for (int i = 0; i < output.rows; i++) {
				for (int j = 0; j < output.cols; j++) {
					float sum = 0;
					for (int k = 0; k < other.rows; k++) {
						sum += this->data[i][k] * other.data[k][j];
					}
					output.data[i][j] = sum;
				}
			}
			this->data = (output.data);
			this->rows = (output.rows);
			this->cols = (output.cols);
		}
		else cout << "Cols of first matrix should equal rows of second matrix to multiply both of them. Current object remains UNCHANGED.\n";
	}
	//Static method for the whole class for multiplication; multiplies two matrices and returns result
   //if condition not met Matrix() is returned 
	static Matrix multiply(Matrix m1, Matrix m2) {
		if (m1.cols == m2.rows) {
			Matrix output(m1.rows, m2.cols);
			for (int i = 0; i < output.rows; i++) {
				for (int j = 0; j < output.cols; j++) {
					float sum = 0;
					for (int k = 0; k < m1.cols; k++) {
						sum += m1.data[i][k] * m2.data[k][j];
					}
					output.data[i][j] = sum;
				}
			}
			return output;
		}
		return Matrix();
	}

	//HADAMARD PRODUCT - ELEMENT WISE MATRIX MULTIPLICATION
	//Multplies val in 1st matrix to corresponding val in second matrix
	void elementWiseMult(Matrix& other) {
		if (this->rows == other.rows && this->cols == other.cols) {
			Matrix output(rows, cols);
			for (int i = 0; i < output.rows; i++) {
				for (int j = 0; j < output.cols; j++) {
					this->data[i][j] *= other.data[i][j];
				}
			}
		}
		else cout << "Dims of matrices must be equal to perform element wise multiplication. Current object remains UNCHANGED.\n";
	}

	//Instance method that returns a matrix after doing elem-wise mult.
	//Returns Matrix() if condition not met
	Matrix ElementWiseMult(Matrix& other) {
		if (this->rows == other.rows && this->cols == other.cols) {
			Matrix output(rows, cols);
			for (int i = 0; i < output.rows; i++) {
				for (int j = 0; j < output.cols; j++) {
					output.data[i][j] = this->data[i][j] * other.data[i][j];
				}
			}
			return output;
		}
		return Matrix();
	}

	//Static method for elem-wise mult., for whole Matrix class
	//Returns Matrix() if condition not met
	static Matrix ElementWiseMult(Matrix& m1, Matrix& m2) {
		if (m1.rows == m2.rows && m1.cols == m2.cols) {
			Matrix output(m1.rows, m1.cols);
			for (int i = 0; i < output.rows; i++) {
				for (int j = 0; j < output.cols; j++) {
					output.data[i][j] = m1.data[i][j] * m2.data[i][j];
				}
			}
			return output;
		}
		return Matrix();
	}

	//Transposes current matrix object and uses setter methods to set dims of the modified current object accordingly
	void transpose() {
		Matrix output(this->cols, this->rows);
		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {
				output.data[i][j] = this->data[j][i];
			}
		}
		this->data = (output.data);
		this->rows = (output.rows);
		this->cols = (output.cols);
	}

	//Instance method for transpose that returns resultant, transposed matrix
	Matrix Transpose() {
		Matrix output(this->cols, this->rows);
		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {
				output.data[i][j] = this->data[j][i];
			}
		}
		return output;
	}
	//Neural Network functions------------------------Sigmoid and dSigmoid are private methods because there is 
   //absolutely no need to access them outside this class
	static float Sigmoid(float x) {
		return 1.0f / (1 + exp(-x));
		//return tanh(x);
	}
	static float dSigmoid(float x) {
		return x * (1 - x);
		//return 1 - (x * x);
	}

	void applySigmoid() {
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				this->data[i][j] = Sigmoid(this->data[i][j]);
			}
		}
	}

	void applySigmoidDerivative() {
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				this->data[i][j] = dSigmoid(this->data[i][j]);
			}
		}
	}

	static Matrix ApplySigmoid(Matrix& m) {
		Matrix output(m.rows, m.cols);
		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {
				output.data[i][j] = Sigmoid(m.data[i][j]);
			}
		}
		return output;
	}

	static Matrix ApplySigmoidDerivative(Matrix& m) {
		Matrix output(m.rows, m.cols);
		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {
				output.data[i][j] = dSigmoid(m.data[i][j]);
			}
		}
		return output;
	}
	//-----------------------------------------------------

	//Functions to convert from 1d arr to matrix and from Matrix to 1d arr:
	//Returns (1 col. Matrix) column vector of 'arr.length' no of rows and 1 column
	static Matrix fromArr(vector<float>& arr) {
		Matrix output(arr.size(), 1);
		for (int i = 0; i < output.rows; i++) {
			output.data[i][0] = arr[i];
		}
		return output;
	}

	//Load Matrix elements into arr of size [rows*cols]
	vector<float> toArr() {
		vector<float> output(this->rows * this->cols);
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				output[i] = this->data[i][j];
			}
		}
		return output;
	}


};
