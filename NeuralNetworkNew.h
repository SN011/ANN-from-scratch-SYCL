#pragma once
#include "SYCLMatrixLib.h"
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace sycl;
class NeuralNetwork {
private:
    float learningRate = 0.1f;

    vector<int> nn_topology;
    vector<Matrix> weightMatrices;
    vector<Matrix> biasMatrices;
    sycl::queue q{ sycl::gpu_selector{}, sycl::property_list{ sycl::property::queue::in_order {} } };

public:

    NeuralNetwork(int in, int hid1, int hid2, int out) {
        //Add the params to the topology array list --> list will be [in, hid, out]
        //Topology of an NN pertains to its architecture of layers and the number of nodes in those layers
        nn_topology.push_back(in);
        nn_topology.push_back(hid1);
        nn_topology.push_back(hid2);
        nn_topology.push_back(out);

        /*** Modified for USM **/
        for (size_t i = 0; i < nn_topology.size() - 1; i++) {
            Matrix wts(nn_topology[i + 1], nn_topology[i], q);
            // Xavier initialization
            wts.RandInit(nn_topology[i], nn_topology[i + 1]);
            weightMatrices.push_back(wts);

            Matrix b(nn_topology[i + 1], 1, q);
            // Biases initialized to 0
            biasMatrices.push_back(b);
        }
    }

    void setLearningRate(float lr) {
        this->learningRate = lr;
    }

    /*
       FeedForward Algorithm:
       Precondition: targetInputs array must have length that is equal to number of input nodes (i.e. nn_topology[0])
       [Steps]
       1.Convert an array of inputs into a one column matrix (also called column vector)
       2.Send the matrix product of the inputs and weights (between input->hidden layer) to the next layer (i.e. hidden layer)
       3.Apply activation function to each element in the matrix that resulted from prior operation

       Repeat previous steps on the next layer (hidden->output)

       Return values in the last layer (i.e. output layer) converted from a matrix to an array
    */

    vector<float> FeedForward(vector<float> targetInputs) {
        Matrix inputs = Matrix::fromArr(targetInputs, q);
        //inputL->hidden1L
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();
        //hidden1L -> hidden2L
        Matrix hidden2 = Matrix::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();
        //hiddenL->outputL
        Matrix outputs = Matrix::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix::Softmax(outputs, q);

        return outputs.toArr();
    }

    void printDeviceInfo() {
        std::cout << "SYCL Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    }

    sycl::queue& getQueue() { return q; }

    /*
       Precondition: targetOutputs array must have length that is equal to number of output nodes (i.e. nn_topology.get(2))
       Backpropagation Algorithm:

       [Steps] 0: Perform the same steps as in FeedForward

       1. Convert targets from array into matrix
       2. Calculate errors between targets and outputs via Matrix subtraction --> subtract outputs (prediction vals) from targets (actual vals)
       3. Take the sigmoid derivative s(x) * (1-s(x)) of the outputs matrix (where s(x) is sigmoid function --> 1.0/(1.0+e^-x) )
          and save this in a matrix called gradients
       4. Multiply gradients matrix ELEMENT-WISE (not matrix multiplication by rows and columns, but just multiplying the elemtns straightforwardly)
       5. Multiply gradients matrix by LEARNING RATE (0.1 by default)
       6. Find the delta weights of the weights between output and hidden layers (out->hid) by multiplying THE TRANSPOSE of the hidden layer
          (because we are traversing the nn backward)
       7. Add the delta weights to the weights between output->hidden layers (to tune them up or down for the nn to make a better prediction)
          and add the gradients matrix to the bias matrix for out->hidden bias (biasMatrices[1])
       CALCULATE HIDDEN ERRORS
       8. Multiply TRANSPOSE of weightMatrices[1] or the weights that were modified in step 7, to the output_errors matrix
       9. REPEAT steps 3-7 on the next layers (hid->input layer)

    */

    void BackPropagate(vector<float> targetInputs, vector<float> targetOutputs) {
        //FeedForward ---------------------------------------------------------   
        Matrix inputs = Matrix::fromArr(targetInputs, q);
        //inputL->hidden1L
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();
        //hidden1L -> hidden2L
        Matrix hidden2 = Matrix::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();
        //hiddenL->outputL
        Matrix outputs = Matrix::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix::Softmax(outputs, q);
        // --------------------------------------------------------------------   

        //Target array --> converted to matrix
        Matrix targets = Matrix::fromArr(targetOutputs, q);

        //Output errors = targets-outputs (matrix subtraction) - For Softmax + Cross-Entropy, this is simply (outputs - targets)
        Matrix output_errors = Matrix::subtract(outputs, targets, q);

        //Calculate the gradients which will be used to calculate the values by which to tune the weights up or down to get a better prediction from NN
        // With Softmax and Cross-Entropy, the gradient for the output layer is simply output_errors
        Matrix gradients = output_errors; // Removed ApplySigmoidDerivative and elementWiseMult
        gradients.multiplyScalar(-learningRate);

        //Multiply by the prev layer matrices transposed because this is essentially a backward traversal of the NN
        Matrix weight_h2o_deltas = Matrix::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));
        //-----------------------------------------------------------------------------------------------------------------------------------------------

        Matrix hidden2_errors = Matrix::multiply(weightMatrices[2].Transpose(), output_errors, q);

        //Calculate the gradients which will be used to calculate the values by which to tune the weights up or down to get a better prediction from NN
        Matrix hidden2_gradient = Matrix::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-learningRate);

        //Multiply by the prev layer matrices transposed because this is essentially a backward traversal of the NN
        Matrix weight_h1h2_deltas = Matrix::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        //-----------------------------------------------------------------------------------------------------------------------------------------------
        //Find the hidden errors and do the same thing as above
        Matrix hidden_errors = Matrix::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        // Calculate the gradients for hidden layer 1
        Matrix hidden_gradient = Matrix::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-learningRate);

        Matrix weight_ih1_deltas = Matrix::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));
    }

    void BackPropagateBatch(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return;

        // Combine inputs into a batch matrix
        Matrix inputs(nn_topology[0], batchSize, q);
        // Copy data from host vector of vectors to USM. This might be a bottleneck.
        // For true USM efficiency, inputs should ideally be directly managed in USM from the start.
        // For now, copying to USM via a host intermediate for simplicity.
        std::vector<float> host_inputs_flat(nn_topology[0] * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                host_inputs_flat[j * batchSize + i] = batchInputs[i][j];
            }
        }
        q.memcpy(inputs.data, host_inputs_flat.data(), nn_topology[0] * batchSize * sizeof(float)); // Removed .wait()

        // FeedForward Step
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        Matrix hidden2 = Matrix::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();

        Matrix outputs = Matrix::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix::Softmax(outputs, q);

        // Combine targets into a batch matrix
        Matrix targets(nn_topology.back(), batchSize, q);
        // Copy data from host vector of vectors to USM.
        std::vector<float> host_targets_flat(nn_topology.back() * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                host_targets_flat[j * batchSize + i] = batchOutputs[i][j];
            }
        }
        q.memcpy(targets.data, host_targets_flat.data(), nn_topology.back() * batchSize * sizeof(float)); // Removed .wait()

        // Output errors = targets - outputs (matrix subtraction)
        Matrix output_errors = Matrix::subtract(outputs, targets, q);
        output_errors.multiplyScalar(1.0f / batchSize);   //  ← NEW

        // Calculate the gradients for output layer
        Matrix gradients = output_errors; // Removed ApplySigmoidDerivative and elementWiseMult
        float lr = learningRate; // NOT lr / batchSize
        gradients.multiplyScalar(-lr);

        // Update weights and biases for hidden2 -> output layer
        Matrix weight_h2o_deltas = Matrix::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));

        // Hidden layer 2 errors
        Matrix hidden2_errors = Matrix::multiply(weightMatrices[2].Transpose(), output_errors, q);

        // Calculate the gradients for hidden layer 2
        Matrix hidden2_gradient = Matrix::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-lr);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix weight_h1h2_deltas = Matrix::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        // Hidden layer 1 errors
        Matrix hidden_errors = Matrix::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        // Calculate the gradients for hidden layer 1
        Matrix hidden_gradient = Matrix::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-lr);

        // Update weights and biases for input -> hidden1 layer
        Matrix weight_ih1_deltas = Matrix::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));
    }

    Matrix BackPropagateBatchAndReturnOutputs(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return Matrix();

        // Combine inputs into a batch matrix
        Matrix inputs(nn_topology[0], batchSize, q);
        std::vector<float> host_inputs_flat(nn_topology[0] * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                host_inputs_flat[j * batchSize + i] = batchInputs[i][j];
            }
        }
        q.memcpy(inputs.data, host_inputs_flat.data(), nn_topology[0] * batchSize * sizeof(float));

        // FeedForward Step
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs, q);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        Matrix hidden2 = Matrix::multiply(weightMatrices[1], hidden, q);
        hidden2.add(biasMatrices[1]);
        hidden2.applySigmoid();

        Matrix outputs = Matrix::multiply(weightMatrices[2], hidden2, q);
        outputs.add(biasMatrices[2]);
        outputs = Matrix::Softmax(outputs, q);

        // Combine targets into a batch matrix
        Matrix targets(nn_topology.back(), batchSize, q);
        std::vector<float> host_targets_flat(nn_topology.back() * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                host_targets_flat[j * batchSize + i] = batchOutputs[i][j];
            }
        }
        q.memcpy(targets.data, host_targets_flat.data(), nn_topology.back() * batchSize * sizeof(float));

        // Output errors = targets - outputs (matrix subtraction)
        Matrix output_errors = Matrix::subtract(outputs, targets, q);
        output_errors.multiplyScalar(1.0f / batchSize);

        // Calculate the gradients for output layer
        Matrix gradients = output_errors;
        float lr = learningRate;
        gradients.multiplyScalar(-lr);

        // Update weights and biases for hidden2 -> output layer
        Matrix weight_h2o_deltas = Matrix::multiply(gradients, hidden2.Transpose(), q);
        weightMatrices[2].add(weight_h2o_deltas);
        biasMatrices[2].add(gradients.sumAlongAxis(1));

        // Hidden layer 2 errors
        Matrix hidden2_errors = Matrix::multiply(weightMatrices[2].Transpose(), output_errors, q);

        // Calculate the gradients for hidden layer 2
        Matrix hidden2_gradient = Matrix::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-lr);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix weight_h1h2_deltas = Matrix::multiply(hidden2_gradient, hidden.Transpose(), q);
        weightMatrices[1].add(weight_h1h2_deltas);
        biasMatrices[1].add(hidden2_gradient.sumAlongAxis(1));

        // Hidden layer 1 errors
        Matrix hidden_errors = Matrix::multiply(weightMatrices[1].Transpose(), hidden2_errors, q);

        // Calculate the gradients for hidden layer 1
        Matrix hidden_gradient = Matrix::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-lr);

        // Update weights and biases for input -> hidden1 layer
        Matrix weight_ih1_deltas = Matrix::multiply(hidden_gradient, inputs.Transpose(), q);
        weightMatrices[0].add(weight_ih1_deltas);
        biasMatrices[0].add(hidden_gradient.sumAlongAxis(1));
        /*std::cout << "Inside BackPropagateBatchAndReturnOutputs: outputs dimensions: " << outputs.rows << "x" << outputs.cols << std::endl;
        std::cout << "Inside BackPropagateBatchAndReturnOutputs: targets dimensions: " << targets.rows << "x" << targets.cols << std::endl;
        */
        return outputs; // Return the outputs matrix
    }

};
