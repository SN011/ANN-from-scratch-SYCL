#include "NeuralNetworkNew.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <chrono>
using namespace std;

int generateRandNum() {
	static std::random_device rd;
	static std::uniform_int_distribution<int> dist(0, 3);
	return dist(rd);
}

std::vector<std::vector<float>> targetInputs = {
		{0.0f, 0.0f},
		{1.0f, 1.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f}
};

vector<vector<float>> targetOutputs = {
 {0.0},
 {0.0},
 {1.0},
 {1.0}
};
NeuralNetwork nn(2, 64, 2, 1);
NeuralNet nn2(2,64,1);
int main() {

	std::cout.precision(4);
	std::cout << std::fixed;
	int epoch = 10000;
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "Training started...\n";
	for (int i = 0; i < epoch; i++) {
		int index = generateRandNum();
		nn.BackPropagate(targetInputs[index],targetOutputs[index]);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	std::cout << "Training completed... (" << dur / (1e+9) << " seconds)\n";

	start = std::chrono::high_resolution_clock::now();
	std::cout << "Training started (2)...\n";
	for (int i = 0; i < epoch; i++) {
		int index = generateRandNum();
		nn2.BackPropagate(targetInputs[index], targetOutputs[index]);
	}
	end = std::chrono::high_resolution_clock::now();
	dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	std::cout << "Training completed... (" << dur / (1e+9) << " seconds)\n";

	int ctr = 0;
	for (std::vector<float> input : targetInputs) {
		nn.FeedForward(input);
		std::vector<float> preds = nn.FeedForward(input);
		std::cout << input[0] << ", " << input[1] << " --> " << preds[0] << " (" << round(preds[0]) << ") ";
		if (abs(preds[0] - targetOutputs[ctr][0]) <= 0.1) {
			printf("Correct!(%f)\n", abs(preds[0] - targetOutputs[ctr][0]));
		}
		else {
			printf("Wrong\n");
		}
		ctr++;
	}

	ctr = 0;
	for (std::vector<float> input : targetInputs) {
		nn.FeedForward(input);
		std::vector<float> preds = nn2.FeedForward(input);
		std::cout << input[0] << ", " << input[1] << " --> " << preds[0] << " (" << round(preds[0]) << ") ";
		if (abs(preds[0] - targetOutputs[ctr][0]) <= 0.1) {
			printf("Correct!(%f)\n", abs(preds[0] - targetOutputs[ctr][0]));
		}
		else {
			printf("Wrong\n");
		}
		ctr++;
	}
}