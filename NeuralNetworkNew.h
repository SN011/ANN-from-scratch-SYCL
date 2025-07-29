#pragma once
#include "SYCLMatrixLib.h"
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace sycl;
class NeuralNetwork {
private:
    // IMPORTANT: Queue must be declared first to ensure it's destroyed last
    // This prevents access violations when Matrix destructors try to free USM memory
    sycl::async_handler ah = [](auto l) { 
        for (auto& e : l) try { 
            std::rethrow_exception(e); 
        } catch (const sycl::exception& ex) { 
            std::cerr << "ASYNC " << ex.what() << '\n'; 
        } 
    };
    sycl::queue q{ sycl::gpu_selector{}, ah };
    
    float learningRate = 0.1f;
    vector<int> nn_topology;
    vector<Matrix> weightMatrices;
    vector<Matrix> biasMatrices;
    
    // Reusable scratch matrices to avoid allocation overhead
    Matrix scratch_inputs, scratch_targets;
    Matrix scratch_hidden, scratch_hidden2, scratch_outputs;
    Matrix scratch_grad_out, scratch_grad_h2, scratch_grad_h1;
    Matrix scratch_errors_out, scratch_errors_h2, scratch_errors_h1;

    // Scratch matrices for weight deltas
    Matrix scratch_w_h2o_deltas, scratch_w_h1h2_deltas, scratch_w_ih1_deltas;

    // Scratch matrices for bias gradient sums
    Matrix scratch_bias_grad_sum_out, scratch_bias_grad_sum_h2, scratch_bias_grad_sum_h1;

    // Ensure scratch matrices have capacity for the current batch size.
    void ensureBatchCapacity(int batch) {
        auto resize = [&](Matrix& M, int r, int c) {
            if (M.data == nullptr) {
                M = Matrix(r, c, q);
                return;
            }

            std::size_t required = static_cast<std::size_t>(r) * c;
            std::size_t current  = static_cast<std::size_t>(M.rows) * M.cols;

            if (required > current) {
                // Need a larger allocation: ensure all prior work done then free
                q.wait_and_throw();
                sycl::free(M.data, q);
                M.data = sycl::malloc_shared<float>(required, q);
            }
            // Update logical dimensions (no realloc if capacity sufficient)
            M.rows = r;
            M.cols = c;
        };

        resize(scratch_inputs , nn_topology[0] , batch);
        resize(scratch_targets, nn_topology.back(), batch);
        resize(scratch_hidden , nn_topology[1] , batch);
        resize(scratch_hidden2, nn_topology[2] , batch);
        resize(scratch_outputs, nn_topology.back(), batch);

        resize(scratch_grad_out , nn_topology.back(), batch);
        resize(scratch_grad_h2  , nn_topology[2]   , batch);
        resize(scratch_grad_h1  , nn_topology[1]   , batch);
        resize(scratch_errors_out, nn_topology.back(), batch);
        resize(scratch_errors_h2 , nn_topology[2]   , batch);
        resize(scratch_errors_h1 , nn_topology[1]   , batch);
    }

public:
    ~NeuralNetwork() {
        // Ensure all queued operations have completed before resources are released
        try {
            q.wait_and_throw();
        } catch (const sycl::exception& e) {
            std::cerr << "NeuralNetwork destructor caught SYCL exception: " << e.what() << std::endl;
        }
    }

    NeuralNetwork(int in, int hid1, int hid2, int out, unsigned int seed = 42) {
        //Add the params to the topology array list --> list will be [in, hid, out]
        //Topology of an NN pertains to its architecture of layers and the number of nodes in those layers
        nn_topology.push_back(in);
        nn_topology.push_back(hid1);
        nn_topology.push_back(hid2);
        nn_topology.push_back(out);

        /*** Modified for USM **/
        // Reserve to avoid reallocations that would move USM pointers
        weightMatrices.reserve(nn_topology.size() - 1);
        biasMatrices.reserve(nn_topology.size() - 1);

        for (size_t i = 0; i + 1 < nn_topology.size(); ++i) {
            // Construct weights directly inside the vector to avoid an extra copy and destructor
            weightMatrices.emplace_back(nn_topology[i + 1], nn_topology[i], q);
            auto& W = weightMatrices.back();
            // Xavier initialization with fixed seed for reproducibility
            W.RandInit(nn_topology[i], nn_topology[i + 1], seed + static_cast<unsigned int>(i));

            // Bias matrix (already zero-initialised by default)
            biasMatrices.emplace_back(nn_topology[i + 1], 1, q);
        }

        // Scratch matrices start empty; they will grow on first batch
        scratch_inputs = Matrix();
        scratch_targets = Matrix();
        scratch_hidden = Matrix();
        scratch_hidden2 = Matrix();
        scratch_outputs = Matrix();

        scratch_grad_out = Matrix();
        scratch_grad_h2 = Matrix();
        scratch_grad_h1 = Matrix();
        scratch_errors_out = Matrix();
        scratch_errors_h2 = Matrix();
        scratch_errors_h1 = Matrix();

        // Scratch matrices for weight deltas
        scratch_w_h2o_deltas = Matrix(weightMatrices[2].rows, weightMatrices[2].cols, q);
        scratch_w_h1h2_deltas = Matrix(weightMatrices[1].rows, weightMatrices[1].cols, q);
        scratch_w_ih1_deltas = Matrix(weightMatrices[0].rows, weightMatrices[0].cols, q);

        // Scratch matrices for bias gradient sums
        scratch_bias_grad_sum_out = Matrix(nn_topology.back(), 1, q);
        scratch_bias_grad_sum_h2 = Matrix(nn_topology[2], 1, q);
        scratch_bias_grad_sum_h1 = Matrix(nn_topology[1], 1, q);
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

    vector<float> FeedForward(const vector<float>& targetInputs) {
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
        // In-place softmax to avoid discarding the result
        Matrix::Softmax(outputs, outputs, q);

        return outputs.toArr();
    }

    void printDeviceInfo() {
        std::cout << "SYCL Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        auto d = q.get_device();
        std::cout << "global_mem_size = "
            << d.get_info<sycl::info::device::global_mem_size>() / 1024 / 1024
            << " MB\n";
        std::cout << "USM allocation size limit  = "
            << d.get_info<sycl::info::device::usm_device_allocations>() << "\n";
    }

    sycl::queue& getQueue() { return q; }

    // Loss-only calculation method (for monitoring/evaluation)
    float calculateLoss(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchTargets) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return 0.0f;

        // Combine inputs into a batch matrix
        Matrix inputs(nn_topology[0], batchSize, q);
        std::vector<float> host_inputs_flat(nn_topology[0] * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                host_inputs_flat[j * batchSize + i] = batchInputs[i][j];
            }
        }
        q.memcpy(inputs.data, host_inputs_flat.data(), nn_topology[0] * batchSize * sizeof(float));

        // Forward pass to get outputs
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
                host_targets_flat[j * batchSize + i] = batchTargets[i][j];
            }
        }
        q.memcpy(targets.data, host_targets_flat.data(), nn_topology.back() * batchSize * sizeof(float));

        // Calculate cross-entropy loss
        float loss = Matrix::crossEntropy(outputs, targets, q);
        
        q.wait(); // Ensure computation completes
        
        return loss;
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

    void BackPropagate(const vector<float>& targetInputs, const vector<float>& targetOutputs) {
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
        // Apply softmax in-place to obtain probabilities
        Matrix::Softmax(outputs, outputs, q);
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
        Matrix weight_h2o_deltas = Matrix::mult_T(gradients, hidden2, false, true, q);
        weightMatrices[2].add(weight_h2o_deltas);
        
        // --- Bias Update ---
        Matrix bias_grad_sum_out(gradients.rows, 1, q);
        gradients.sumAlongAxis(1, bias_grad_sum_out);
        //-----------------------------------------------------------------------------------------------------------------------------------------------

        Matrix hidden2_errors = Matrix::mult_T(weightMatrices[2], output_errors, true, false, q);

        //Calculate the gradients which will be used to calculate the values by which to tune the weights up or down to get a better prediction from NN
        Matrix hidden2_gradient = Matrix::ApplySigmoidDerivative(hidden2, q);
        hidden2_gradient.elementWiseMult(hidden2_errors);
        hidden2_gradient.multiplyScalar(-learningRate);

        //Multiply by the prev layer matrices transposed because this is essentially a backward traversal of the NN
        Matrix weight_h1h2_deltas = Matrix::mult_T(hidden2_gradient, hidden, false, true, q);
        weightMatrices[1].add(weight_h1h2_deltas);
        
        // --- Bias Update ---
        Matrix bias_grad_sum_h2(hidden2_gradient.rows, 1, q);
        hidden2_gradient.sumAlongAxis(1, bias_grad_sum_h2);

        //-----------------------------------------------------------------------------------------------------------------------------------------------
        //Find the hidden errors and do the same thing as above
        Matrix hidden_errors = Matrix::mult_T(weightMatrices[1], hidden2_errors, true, false, q);

        // Calculate the gradients for hidden layer 1
        Matrix hidden_gradient = Matrix::ApplySigmoidDerivative(hidden, q);
        hidden_gradient.elementWiseMult(hidden_errors);
        hidden_gradient.multiplyScalar(-learningRate);

        Matrix weight_ih1_deltas = Matrix::mult_T(hidden_gradient, inputs, false, true, q);
        weightMatrices[0].add(weight_ih1_deltas);
        
        // --- Bias Update ---
        Matrix bias_grad_sum_h1(hidden_gradient.rows, 1, q);
        hidden_gradient.sumAlongAxis(1, bias_grad_sum_h1);
    }

    float BackPropagateBatch(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs, bool return_loss = false) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return 0.0f;
        
        if (batchSize > 2048) {
            throw std::runtime_error("Batch size exceeds maximum supported size of 2048");
        }

        // Ensure scratch matrices have sufficient capacity for this mini-batch
        ensureBatchCapacity(batchSize);

        // Copy input data to scratch matrix
        std::vector<float> host_inputs_flat(nn_topology[0] * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                host_inputs_flat[j * batchSize + i] = batchInputs[i][j];
            }
        }
        q.memcpy(scratch_inputs.data, host_inputs_flat.data(), nn_topology[0] * batchSize * sizeof(float));

        // FeedForward Step using scratch matrices
        Matrix::multiply(weightMatrices[0], scratch_inputs, scratch_hidden, q);
        scratch_hidden.add(biasMatrices[0]);
        scratch_hidden.applySigmoid();

        Matrix::multiply(weightMatrices[1], scratch_hidden, scratch_hidden2, q);
        scratch_hidden2.add(biasMatrices[1]);
        scratch_hidden2.applySigmoid();

        Matrix::multiply(weightMatrices[2], scratch_hidden2, scratch_outputs, q);
        scratch_outputs.add(biasMatrices[2]);
        // In-place softmax on scratch_outputs
        Matrix::Softmax(scratch_outputs, scratch_outputs, q);

        // Copy target data to scratch matrix
        std::vector<float> host_targets_flat(nn_topology.back() * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                host_targets_flat[j * batchSize + i] = batchOutputs[i][j];
            }
        }
        q.memcpy(scratch_targets.data, host_targets_flat.data(), nn_topology.back() * batchSize * sizeof(float));

        // Backpropagation using scratch matrices
        // Compute output errors into dedicated scratch matrix
        Matrix::subtract(scratch_outputs, scratch_targets, scratch_errors_out, q);

        // Calculate the gradients for output layer (inline batch size division)
        // scratch_grad_out will be a copy of scratch_errors_out, but scaled
        q.memcpy(scratch_grad_out.data, scratch_errors_out.data, scratch_errors_out.rows * scratch_errors_out.cols * sizeof(float)).wait();
        float lr = learningRate / batchSize;
        scratch_grad_out.multiplyScalar(-lr);

        // Update weights and biases for hidden2 -> output layer
        Matrix::mult_T(scratch_grad_out, scratch_hidden2, false, true, scratch_w_h2o_deltas, q);
        weightMatrices[2].add(scratch_w_h2o_deltas);
        scratch_grad_out.sumAlongAxis(1, scratch_bias_grad_sum_out);
        biasMatrices[2].add(scratch_bias_grad_sum_out);

        // Hidden layer 2 errors
        Matrix::mult_T(weightMatrices[2], scratch_errors_out, true, false, scratch_errors_h2, q);

        // Calculate the gradients for hidden layer 2
        Matrix::ApplySigmoidDerivative(scratch_hidden2, scratch_grad_h2, q);
        scratch_grad_h2.elementWiseMult(scratch_errors_h2);
        scratch_grad_h2.multiplyScalar(-lr);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix::mult_T(scratch_grad_h2, scratch_hidden, false, true, scratch_w_h1h2_deltas, q);
        weightMatrices[1].add(scratch_w_h1h2_deltas);
        scratch_grad_h2.sumAlongAxis(1, scratch_bias_grad_sum_h2);
        biasMatrices[1].add(scratch_bias_grad_sum_h2);

        // Hidden layer 1 errors
        Matrix::mult_T(weightMatrices[1], scratch_errors_h2, true, false, scratch_errors_h1, q);

        // Calculate the gradients for hidden layer 1
        Matrix::ApplySigmoidDerivative(scratch_hidden, scratch_grad_h1, q);
        scratch_grad_h1.elementWiseMult(scratch_errors_h1);
        scratch_grad_h1.multiplyScalar(-lr);

        // Update weights and biases for input -> hidden1 layer
        Matrix::mult_T(scratch_grad_h1, scratch_inputs, false, true, scratch_w_ih1_deltas, q);
        weightMatrices[0].add(scratch_w_ih1_deltas);
        scratch_grad_h1.sumAlongAxis(1, scratch_bias_grad_sum_h1);
        biasMatrices[0].add(scratch_bias_grad_sum_h1);
        
        // Calculate and return loss if requested
        if (return_loss) {
            float loss = Matrix::crossEntropy(scratch_outputs, scratch_targets, q);
            q.wait(); // Ensure computation completes
            return loss;
        }
        
        return 0.0f;
    }

    Matrix BackPropagateBatchAndReturnOutputs(const vector<vector<float>>& batchInputs, const vector<vector<float>>& batchOutputs) {
        int batchSize = batchInputs.size();
        if (batchSize == 0) return Matrix();
        
        if (batchSize > 2048) {
            throw std::runtime_error("Batch size exceeds maximum supported size of 2048");
        }

        // Ensure scratch matrices have sufficient capacity for this mini-batch
        ensureBatchCapacity(batchSize);

        // Copy input data to scratch matrix
        std::vector<float> host_inputs_flat(nn_topology[0] * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology[0]; j++) {
                host_inputs_flat[j * batchSize + i] = batchInputs[i][j];
            }
        }
        q.memcpy(scratch_inputs.data, host_inputs_flat.data(), nn_topology[0] * batchSize * sizeof(float));

        // FeedForward Step using scratch matrices
        Matrix::multiply(weightMatrices[0], scratch_inputs, scratch_hidden, q);
        scratch_hidden.add(biasMatrices[0]);
        scratch_hidden.applySigmoid();

        Matrix::multiply(weightMatrices[1], scratch_hidden, scratch_hidden2, q);
        scratch_hidden2.add(biasMatrices[1]);
        scratch_hidden2.applySigmoid();

        Matrix::multiply(weightMatrices[2], scratch_hidden2, scratch_outputs, q);
        scratch_outputs.add(biasMatrices[2]);
        // In-place softmax on scratch_outputs
        Matrix::Softmax(scratch_outputs, scratch_outputs, q);

        // Copy target data to scratch matrix
        std::vector<float> host_targets_flat(nn_topology.back() * batchSize);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nn_topology.back(); j++) {
                host_targets_flat[j * batchSize + i] = batchOutputs[i][j];
            }
        }
        q.memcpy(scratch_targets.data, host_targets_flat.data(), nn_topology.back() * batchSize * sizeof(float));

        // Backpropagation using scratch matrices
        // Compute output errors into dedicated scratch matrix
        Matrix::subtract(scratch_outputs, scratch_targets, scratch_errors_out, q);

        // Calculate the gradients for output layer (inline batch size division)
        q.memcpy(scratch_grad_out.data, scratch_errors_out.data, scratch_errors_out.rows * scratch_errors_out.cols * sizeof(float)).wait();
        float lr = learningRate / batchSize;
        scratch_grad_out.multiplyScalar(-lr);

        // Update weights and biases for hidden2 -> output layer
        Matrix::mult_T(scratch_grad_out, scratch_hidden2, false, true, scratch_w_h2o_deltas, q);
        weightMatrices[2].add(scratch_w_h2o_deltas);
        scratch_grad_out.sumAlongAxis(1, scratch_bias_grad_sum_out);
        biasMatrices[2].add(scratch_bias_grad_sum_out);

        // Hidden layer 2 errors
        Matrix::mult_T(weightMatrices[2], scratch_errors_out, true, false, scratch_errors_h2, q);

        // Calculate the gradients for hidden layer 2
        Matrix::ApplySigmoidDerivative(scratch_hidden2 , scratch_grad_h2, q);
        scratch_grad_h2.elementWiseMult(scratch_errors_h2);
        scratch_grad_h2.multiplyScalar(-lr);

        // Update weights and biases for hidden1 -> hidden2 layer
        Matrix::mult_T(scratch_grad_h2, scratch_hidden, false, true , scratch_w_h1h2_deltas, q);
        weightMatrices[1].add(scratch_w_h1h2_deltas);
        scratch_grad_h2.sumAlongAxis(1, scratch_bias_grad_sum_h2);
        biasMatrices[1].add(scratch_bias_grad_sum_h2);

        // Hidden layer 1 errors
        Matrix::mult_T(weightMatrices[1], scratch_errors_h2, true, false, scratch_errors_h1, q);

        // Calculate the gradients for hidden layer 1
        Matrix::ApplySigmoidDerivative(scratch_hidden, scratch_grad_h1, q);
        scratch_grad_h1.elementWiseMult(scratch_errors_h1);
        scratch_grad_h1.multiplyScalar(-lr);

        // Update weights and biases for input -> hidden1 layer
        Matrix::mult_T(scratch_grad_h1, scratch_inputs, false, true, scratch_w_ih1_deltas, q);
        weightMatrices[0].add(scratch_w_ih1_deltas);
        scratch_grad_h1.sumAlongAxis(1, scratch_bias_grad_sum_h1);
        biasMatrices[0].add(scratch_bias_grad_sum_h1);
        
        // Return a copy of the outputs matrix (since scratch_outputs will be reused)
        return Matrix(scratch_outputs);
    }

};
