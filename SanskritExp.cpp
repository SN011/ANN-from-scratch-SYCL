#include <iostream>
#include <fstream>
#include <vector>
#include "ImageNew.h"
#include "NeuralNetworkNew.h"
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

std::vector<ImageData> dataset = {};
std::vector<ImageData> trainingSet = {};
std::vector<ImageData> testingSet = {};

std::vector<std::string> fileVec{
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_01_ka.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_02_kha.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_03_ga.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_04_gha.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_05_kna.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_06_cha.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_07_chha.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_08_ja.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_09_jha.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_10_yna.npy",
		"C:\\Users\\PC-User\\Desktop\\SanskritDataNpyFiles\\character_11_taamatar.npy"
};
int globalOutputNodesCount = fileVec.size();

NeuralNetwork nn(1024, 64, 32, globalOutputNodesCount);

std::vector<Image> images = {};
const int total = 2000;
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
		int start = 80 + n * 1024;
		for (int i = 0; i < 1024; i++)
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
	
	int label = 0;
	for (int i = 0; i < fileVec.size(); i++) {
		std::ifstream file(fileVec[i], ios::binary);
		std::vector<unsigned char> data = std::vector<unsigned char>((std::istreambuf_iterator<char>(file)),
			std::istreambuf_iterator<char>());

		file.close();

		for (int i = 0; i < total; i++)
		{
			int offset = 80+i * 1024;
			vector<float> subvector = { data.begin() + offset, data.begin() + offset + 1024 };
			dataset.push_back(ImageData(subvector, label));
		}
		label++;
	}

	std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine(0));

	for (int i = 0; i < dataset.size(); i++) {
		if (i < (int)(0.9 * dataset.size()))trainingSet.push_back(dataset.at(i));
		else testingSet.push_back(dataset.at(i));
	}

	for (auto& set : trainingSet) {
		std::for_each(set.vec.begin(), set.vec.end(), [](float& n) {n = n / 255.0f; });
	}

	for (auto& set2 : testingSet) {
		std::for_each(set2.vec.begin(), set2.vec.end(), [](float& n) {n = n / 255.0f; });
	}
	//std::shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(0));
	//std::shuffle(testingSet.begin(), testingSet.end(), std::default_random_engine(0));
}

void test() {
	int correct = 0, ctr = 0;
	for (int i = 0; i < testingSet.size(); i++)
	{
		vector<float> inputs = testingSet[i].vec;
		int label = testingSet[i].label;
		vector<float> guess = nn.FeedForward(inputs);

		float maxElement = *std::max_element(guess.begin(), guess.end());
		int maxIndex = max_element(guess.begin(), guess.end()) - guess.begin();
		std::string guessStr, realStr;
		switch (maxIndex)
		{
		case 0:
		{
			guessStr = "ka"; break;
		}
		case 1:
		{
			guessStr = "kha"; break;
		}
		case 2:
		{
			guessStr = "ga"; break;
		}
		case 3:guessStr = "gha"; break;
		case 4:guessStr = "nga"; break;
		case 5:guessStr = "ca"; break;
		case 6:guessStr = "chha"; break;
		case 7:guessStr = "ja"; break;
		case 8:guessStr = "jhha"; break;
		case 9:guessStr = "nya"; break;
		case 10:guessStr = "Ta"; break;
		default:
			break;
		}

		switch (label)
		{
		case 0:
		{
			realStr = "ka"; break;
		}
		case 1:
		{
			realStr = "kha"; break;
		}
		case 2:
		{
			realStr = "ga"; break;
		}
		case 3: realStr = "gha"; break;
		case 4:realStr = "nga"; break;
		case 5:realStr = "ca"; break;
		case 6:realStr = "chha"; break;
		case 7:realStr = "ja"; break;
		case 8:realStr = "jhha"; break;
		case 9:realStr = "nya"; break;
		case 10:realStr = "Ta"; break;
		default:
			break;
		}

		
		if (maxIndex == label) {
			correct++;
		}
		else
		{
			std::cout << "Guess: " << guessStr << " (Real: " << realStr << ")\n";
		}
	}
	std::cout << " ====================== % Correct = " << 100 * (float)correct / testingSet.size() << " ======================\n";
}
void train() {
	for (int i = 0; i < trainingSet.size(); i++)
	{
		std::vector<float> inputs = trainingSet[i].vec;
		int label = trainingSet[i].label;
		std::vector<float> targets(globalOutputNodesCount, 0.0f);
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
	vector<float> subvector = { imgData.begin(), imgData.end() };
	ImageData set(subvector, -1);
	std::for_each(set.vec.begin(), set.vec.end(), [](float& n) {n = n / 255.0f; });

	vector<float> inputs = set.vec;
	int label = set.label;
	vector<float> guess = nn.FeedForward(inputs);

	float maxElement = *std::max_element(guess.begin(), guess.end());
	int maxIndex = max_element(guess.begin(), guess.end()) - guess.begin();
	std::string guessStr, realStr;
	switch (maxIndex)
	{
	case 0:
	{
		guessStr = "ka"; break;
	}
	case 1:
	{
		guessStr = "kha"; break;
	}
	case 2:
	{
		guessStr = "ga"; break;
	}
	case 3:guessStr = "gha"; break;
	case 4:guessStr = "nga"; break;
	case 5:guessStr = "ca"; break;
	case 6:guessStr = "chha"; break;
	case 7:guessStr = "ja"; break;
	case 8:guessStr = "jhha"; break;
	case 9:guessStr = "nya"; break;
	case 10:guessStr = "Ta"; break;
	default:
		break;
	}

	switch (label)
	{
	case 0:
	{
		realStr = "ka"; break;
	}
	case 1:
	{
		realStr = "kha"; break;
	}
	case 2:
	{
		realStr = "ga"; break;
	}
	case 3 : realStr = "gha"; break;
	case 4:realStr = "nga"; break;
	case 5:realStr = "ca"; break;
	case 6:realStr = "chha"; break;
	case 7:realStr = "ja"; break;
	case 8:realStr = "jhha"; break;
	case 9:realStr = "nya"; break;
	case 10:realStr = "Ta"; break;
	default:
		break;
	}

	std::cout << "Guess: " << guessStr << " \n";
	std::cout << "(";
	for (int i = 0; i < guess.size(); i++)
	{
		std::cout << guess[i] << " ";
	}
	std::cout << ")\n";


}

int main() {
	
	nn.setLearningRate(0.01f);
	std::srand(std::time(0));
	double start = omp_get_wtime();
	prepareData();
	std::cout << "Preparing data took " << omp_get_wtime() - start << " seconds\n";
	for (int j = 0; j < 10; j++)
	{
		std::shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(0));

		start = omp_get_wtime();
		train();
		std::cout << "Epoch " << j + 1 << " Took " << omp_get_wtime() - start << " seconds\n";
	}
	test();
	std::cout << "Epoch Took " << omp_get_wtime() - start << " seconds\n";
	int cont = -1;
	while (cont != 1) {
		system("D:\\DEV\\NeuralNet\\PythonApplication1\\PythonApplication1\\PythonApplication1.py");
		testUserImage();

		cout << "To test another drawing, press 0 (continue) or 1 (break)\n";
		cin >> cont;
		if (cont == 0)system("D:\\DEV\\NeuralNet\\PythonApplication1\\PythonApplication1\\PythonApplication1.py");
	}

	/*std::ifstream infile("C:\\Users\\PC-User\\Desktop\\consonants.bin", ios::binary);
	std::vector<unsigned char>data((std::istreambuf_iterator<char>(infile)),
		std::istreambuf_iterator<char>());
	for (int n = 0; n < out; n++)
	{
		Image img(28, 28);
		int start = n * 1024;
		for (int i = 0; i < 1024; i++)
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