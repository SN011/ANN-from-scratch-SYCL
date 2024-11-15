#pragma once
#include "MatrixLib.h"
#include <vector>
#include <cstdlib>
#include <cmath>
using namespace std;
class NeuralNet {
private:
    float learningRate = 0.4f;

    vector<int> nn_topology;
    vector<Matrix> weightMatrices;
    vector<Matrix> biasMatrices;
public:

    NeuralNet(int in, int hid, int out) {
        //Add the params to the topology array list --> list will be [in, hid, out]
        //Topology of an NN pertains to its architecture of layers and the number of nodes in those layers
        nn_topology.push_back(in);
        nn_topology.push_back(hid);
        nn_topology.push_back(out);

        /***
           dimensions notation: [rows * cols]
           weightMatrix 0 dimensions --> hid * in
           weightMatrix 1 dimensions --> out * hid [pattern is: next layer * prev layer]
           biasMatrix 0 dimensions --> hid * 1
           biasMatrix 1 dimensions --> out * 1
           Initialize weights and biases to random decimals such that 0 <= x < 1
        ***/

        for (size_t i = 0; i < nn_topology.size() - 1; i++) {
            Matrix wts(nn_topology[i + 1], nn_topology[i]);
            wts.RandInit();
            weightMatrices.push_back(wts);

            Matrix b(nn_topology[i + 1], 1);
            b.RandInit();
            biasMatrices.push_back(b);
        }
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
        Matrix inputs = Matrix::fromArr(targetInputs);
        //inputL->hiddenL
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs);
        hidden.add(biasMatrices[0]);
        hidden.applySigmoid();

        //hiddenL->outputL
        Matrix outputs = Matrix::multiply(weightMatrices[1], hidden);
        outputs.add(biasMatrices[1]);
        outputs.applySigmoid();

        return outputs.toArr();
    }

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
        Matrix inputs = Matrix::fromArr(targetInputs);
        Matrix hidden = Matrix::multiply(weightMatrices[0], inputs);
        hidden.add(biasMatrices[0]);

        hidden.applySigmoid();

        Matrix outputs = Matrix::multiply(weightMatrices[1], hidden);
        outputs.add(biasMatrices[1]);
        outputs.applySigmoid();
        // --------------------------------------------------------------------   

        //Target array --> converted to matrix
        Matrix targets = Matrix::fromArr(targetOutputs);

        //Output errors = targets-outputs (matrix subtraction)
        Matrix output_errors = Matrix::subtract(targets, outputs);

        //Calculate the gradients which will be used to calculate the values by which to tune the weights up or down to get a better prediction from NN
        Matrix gradients = Matrix::ApplySigmoidDerivative(outputs);
        gradients.elementWiseMult(output_errors);
        gradients.multiplyScalar(learningRate);

        //Multiply by the prev layer matrices transposed because this is essentially a backward traversal of the NN
        Matrix weight_ho_deltas = Matrix::multiply(gradients, hidden.Transpose());
        weightMatrices[1].add(weight_ho_deltas);
        biasMatrices[1].add(gradients);

        //Find the hidden errors and do the same thing as above
        Matrix hidden_errors = Matrix::multiply(weightMatrices[1].Transpose(), output_errors);

        Matrix hidden_gradient = Matrix::ApplySigmoidDerivative(hidden);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(learningRate);

        Matrix weight_ih_deltas = Matrix::multiply(hidden_gradient, inputs.Transpose());
        weightMatrices[0].add(weight_ih_deltas);
        biasMatrices[0].add(hidden_gradient);
    }

    void setLearningRate(float f) {
        this->learningRate = f;
    }
};
