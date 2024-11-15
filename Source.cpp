#include <iostream>
#include <fstream>
#include <vector>
#include "ImageNew.h"
#include "NeuralNetwork.h"
#include <random>
#include <algorithm>
#include <omp.h>
using namespace std;

std::vector<unsigned int> getRGBFromInt(unsigned int x) {
	unsigned int red = (x & 0xff0000) >> 16;
	unsigned int green = (x & 0x00ff00) >> 8;
	unsigned int blue = (x & 0x0000ff);
	return { red, green, blue };
}

std::vector<ImageData> cats_training = {};
std::vector<ImageData> rainbows_training = {};
std::vector<ImageData> trains_training = {};
std::vector<ImageData> cats_testing = {};
std::vector<ImageData> rainbows_testing  = {};
std::vector<ImageData> trains_testing = {};
std::vector<ImageData> trainingSet = {};
std::vector<ImageData> testingSet = {};


NeuralNet nn(784, 32, 3);

std::vector<unsigned char> data1;
std::vector<unsigned char> data2;
std::vector<unsigned char> data3;
std::vector<Image> images = {};
const int total = 5000;
void loadMore() {
	std::ifstream infile1("C:\\Users\\pc-user\\downloads\\full_numpy_bitmap_cat.npy", ios::binary);
	std::ifstream infile2("C:\\Users\\pc-user\\downloads\\full_numpy_bitmap_rainbow.npy", ios::binary);
	std::ifstream infile3("C:\\Users\\pc-user\\downloads\\full_numpy_bitmap_train.npy", ios::binary);
	std::ofstream binary1("C:\\Users\\pc-user\\desktop\\cats10000.bin", ios::binary);
	std::ofstream binary2("C:\\Users\\pc-user\\desktop\\rainbows10000.bin", ios::binary);
	std::ofstream binary3("C:\\Users\\pc-user\\desktop\\trains10000.bin", ios::binary);
	std::vector<unsigned char>data01((std::istreambuf_iterator<char>(infile1)),
		std::istreambuf_iterator<char>());
	std::vector<unsigned char>data02((std::istreambuf_iterator<char>(infile2)),
		std::istreambuf_iterator<char>());
	std::vector<unsigned char>data03((std::istreambuf_iterator<char>(infile3)),
		std::istreambuf_iterator<char>());
	infile1.close();
	infile2.close();
	infile3.close();
	for (int n = 0; n < total; n++)
	{
		int start = 80 + n * 784;
		for (int i = 0; i < 784; i++)
		{
			int index = i + start;
			unsigned char val1 = static_cast<const char>(data01[index]);
			unsigned char val2 = static_cast<const char>(data02[index]);
			unsigned char val3 = static_cast<const char>(data03[index]);
			binary1.write(reinterpret_cast<char*>(&val1), sizeof(unsigned char));
			binary2.write(reinterpret_cast<char*>(&val2), sizeof(unsigned char));
			binary3.write(reinterpret_cast<char*>(&val3), sizeof(unsigned char));
		}
	}
	
	binary1.close();
	binary2.close();
	binary3.close();
}



void prepareData() {
	//C:\\Users\\pc-user\\downloads\\full_numpy_bitmap_cat.npy
	//std::ofstream binary("C:\\Users\\pc-user\\desktop\\trains1000.bin", ios::binary);
	//loadMore();
	std::ifstream infile1("C:\\Users\\pc-user\\desktop\\cats10000.bin", ios::binary);
	std::ifstream infile2("C:\\Users\\pc-user\\desktop\\rainbows10000.bin", ios::binary);
	std::ifstream infile3("C:\\Users\\pc-user\\desktop\\trains10000.bin", ios::binary);
	
	data1 = std::vector<unsigned char>((std::istreambuf_iterator<char>(infile1)),
		std::istreambuf_iterator<char>());
	data2 = std::vector<unsigned char>((std::istreambuf_iterator<char>(infile2)),
		std::istreambuf_iterator<char>());
	data3 = std::vector<unsigned char>((std::istreambuf_iterator<char>(infile3)),
		std::istreambuf_iterator<char>());
	
	infile1.close();
	infile2.close();
	infile3.close();
	std::cout << data1.size() << '\n';
	
	
	std::cout << total << '\n';

	for (int i = 0; i < total; i++)
	{
		int offset = i * 784;
		vector<float> subvector1 = { data1.begin() + offset, data1.begin() + offset + 784 };
		if (i < (int)(0.8* total))cats_training.push_back(ImageData(subvector1,0));
		else cats_testing.push_back(ImageData(subvector1, 0));

		vector<float> subvector2 = { data2.begin() + offset, data2.begin() + offset + 784 };
		if (i < (int)(0.8 * total))rainbows_training.push_back(ImageData(subvector2, 1));
		else rainbows_testing.push_back(ImageData(subvector2, 1));

		vector<float> subvector3 = { data3.begin() + offset, data3.begin() + offset + 784 };
		if (i < (int)(0.8 * total))trains_training.push_back(ImageData(subvector3, 2));
		else trains_testing.push_back(ImageData(subvector3, 2));
	}


	trainingSet.insert(trainingSet.end(), cats_training.begin(),cats_training.end());
	trainingSet.insert(trainingSet.end(), rainbows_training.begin(), rainbows_training.end());
	trainingSet.insert(trainingSet.end(), trains_training.begin(), trains_training.end());
	
	for (auto& set : trainingSet) {
		std::for_each(set.vec.begin(), set.vec.end(), [](float& n) {n = n / 255.0f; });
	}

	testingSet.insert(testingSet.end(), cats_testing.begin(), cats_testing.end());
	testingSet.insert(testingSet.end(), rainbows_testing.begin(), rainbows_testing.end());
	testingSet.insert(testingSet.end(), trains_testing.begin(), trains_testing.end());

	cats_training.clear();
	cats_testing.clear();
	rainbows_training.clear();
	rainbows_testing.clear();
	trains_training.clear();
	trains_testing.clear();

	for (auto& set2 : testingSet) {
		std::for_each(set2.vec.begin(), set2.vec.end(), [](float& n) {n = n / 255.0f; });
	}
	std::shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(0));
	std::shuffle(testingSet.begin(), testingSet.end(), std::default_random_engine(0));
}

void test() {
	int correct = 0, ctr = 0;
	for (int i = 0; i < testingSet.size(); i++)
	{
		auto inputs = testingSet[i].vec;
		auto label = testingSet[i].label;
		auto guess = nn.FeedForward(inputs);

		float maxElement = *std::max_element(guess.begin(), guess.end());
		int maxIndex = max_element(guess.begin(), guess.end()) - guess.begin();
		std::string guessStr, realStr;
		switch (maxIndex)
		{
		case 0:guessStr = "cat"; break;
		case 1:guessStr = "rainbow"; break;
		case 2:guessStr = "train"; break;
		
		default:
			break;
		}

		switch (label)
		{
		case 0:realStr = "cat"; break;
		case 1:realStr = "rainbow"; break;
		case 2:realStr = "train"; break;
		default:
			break;
		}

		std::cout << "Guess: " << guessStr << " (Real: " << realStr << ")\n";
		if (maxIndex == label) {
			correct++;
		}
	}
	std::cout << " ====================== % Correct = " << 100 * (float)correct / testingSet.size() << " ======================\n";
}
void train() {
	for (int i = 0; i < trainingSet.size(); i++)
	{
		std::vector<float> inputs = trainingSet[i].vec;
		int label = trainingSet[i].label;
		std::vector<float> targets(3, 0.0f);
		targets[label] = 1.0f;
		nn.BackPropagate(inputs, targets);
	}
}

void testUserImage() {
	std::ifstream imageFile("C:\\users\\pc-user\\desktop\\imagedata.npy", ios::binary);
	//cout << "File to be processed: " << path << endl;
	std::vector<unsigned char> imgData((std::istreambuf_iterator<char>(imageFile)),
		std::istreambuf_iterator<char>());
	imageFile.close();
	int L=0;
	//cout << "Is your drawing a cat (0), rainbow (1), or train (2) ?\n";
	//cin >> L;
	vector<float> subvector = { imgData.begin(), imgData.end() };
	ImageData set(subvector, L);
	std::for_each(set.vec.begin(), set.vec.end(), [](float& n) {n = n / 255.0f; });

	auto inputs = set.vec;
	auto label = set.label;
	auto guess = nn.FeedForward(inputs);

	float maxElement = *std::max_element(guess.begin(), guess.end());
	int maxIndex = max_element(guess.begin(), guess.end()) - guess.begin();
	std::string guessStr, realStr;
	switch (maxIndex)
	{
	case 0:
	{
		guessStr = "cat"; break;
	}
	case 1:
	{
		guessStr = "rainbow"; break;
	}
	case 2:
	{
		guessStr = "train"; break;
	}
	default:
		break;
	}

	switch (label)
	{
	case 0:
	{
		realStr = "cat"; break;
	}
	case 1:
	{
		realStr = "rainbow"; break;
	}
	case 2:
	{
		realStr = "train"; break;
	}
	default:
		break;
	}

	std::cout << "Guess: " << guessStr << " \n";
	std::cout << "(" << guess[0] << ", " << guess[1] << ", " << guess[2] << ")\n";
	

}

int main() {
	nn.setLearningRate(0.05f);
	std::srand(std::time(0));
	double start = omp_get_wtime();
	prepareData();
	std::cout << "Preparing data took " << omp_get_wtime() - start << " seconds\n";
	for (int j = 0; j < 8; j++)
	{
		std::shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(0));
		
		start = omp_get_wtime();
		train();
		std::cout << "Epoch " << j + 1 << " Took " << omp_get_wtime() - start << " seconds\n";
	}
	test();
	int cont = -1;
	while (cont != 1) {
		// This python script will format the image that is drawn in paint (in specific file)
		system("D:\\DEV\\NeuralNet\\PythonApplication1\\PythonApplication1\\PythonApplication1.py");
		testUserImage();

		cout << "To test another drawing, press 0 (continue) or 1 (break)\n";
		cin >> cont;
		if(cont==0)system("D:\\DEV\\NeuralNet\\PythonApplication1\\PythonApplication1\\PythonApplication1.py");
	}
	
	/*std::ifstream infile("C:\\Users\\PC-User\\Desktop\\consonants.bin", ios::binary);
	std::vector<unsigned char>data((std::istreambuf_iterator<char>(infile)),
		std::istreambuf_iterator<char>());
	for (int n = 0; n < out; n++)
	{
		Image img(28, 28);
		int start = n * 784;
		for (int i = 0; i < 784; i++)
		{
			int index = i + start;
			unsigned char val = static_cast<const char>(data[index]);
			auto vec = getRGBFromInt(static_cast<unsigned int>(val));
			img.setColor(Color(vec[0],vec[1],vec[2]), i);
		}
		images.push_back(img);
	}

	for (int i = 0; i < images.size(); i++)
	{
		std::string fileStr = "C:\\Users\\pc-user\\desktop\\test" + std::to_string(i) + ".bmp";
		images[i].Export(fileStr);
	}*/
	
	return 0;
}