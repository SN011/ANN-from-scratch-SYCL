#include <iostream>
#include <fstream>
#include <string>
#include "ImageNew.h"
Color::Color()
{
	r = g = b = 0;
}

Color::Color(float r, float g, float b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}

Color::~Color()
{
}

Image::Image(int width, int height)
{
	m_width = width;
	m_height = height;

	m_colors = std::vector<Color>((size_t)width * height);
}

Image::~Image()
{
}

void Image::Read(const char* path)
{
	std::ifstream f;
	f.open(path, std::ios::in | std::ios::binary);
	if (!f.is_open()) {
		std::cout << "Error opening file\n";
		return;
	}

	const int paddingAmt = ((4 - (m_width * 3) % 4) % 4);

	const int fileHeaderSize = 14;
	const int infoHeaderSize = 40;

	//const int fileSize = fileHeaderSize + infoHeaderSize + m_width * m_height * 3 + paddingAmt * m_height;
	unsigned char fileHeader[fileHeaderSize];
	f.read(reinterpret_cast<char*>(fileHeader), fileHeaderSize);
	unsigned char infoHeader[infoHeaderSize];
	f.read(reinterpret_cast<char*>(infoHeader), infoHeaderSize);

	int fileSize = fileHeader[2] + (fileHeader[3] << 8) +
		(fileHeader[4] << 16) + (fileHeader[5] << 24);

	m_width = infoHeader[4] + (infoHeader[5] << 8) + (infoHeader[6] << 16) + (infoHeader[7] << 24);
	m_height = infoHeader[8] + (infoHeader[9] << 8) + (infoHeader[10] << 16) + (infoHeader[11] << 24);
	m_width = m_width - (m_width % 4);
	m_height = m_height - (m_height % 4);
	m_colors.resize(m_width * m_height);

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			unsigned char color[3];
			f.read(reinterpret_cast<char*>(color), 3);
			m_colors[y * m_width + x].r = (static_cast<float>(color[2]) / 255.0f);
			m_colors[y * m_width + x].g = (static_cast<float>(color[1]) / 255.0f);
			m_colors[y * m_width + x].b = (static_cast<float>(color[0]) / 255.0f);
		}
		f.ignore(paddingAmt);
	}
	f.close();
	printf("File Read!\n");
}

Color Image::getColor(int x, int y) const
{
	return m_colors[y * (size_t)m_width + x];
}

void Image::setColor(const Color& color, int x, int y)
{
	m_colors[y * (size_t)m_width + x].r = color.r;
	m_colors[y * (size_t)m_width + x].g = color.g;
	m_colors[y * (size_t)m_width + x].b = color.b;
}

void Image::setColor(const Color& color, int i)
{
	m_colors[i].r = color.r;
	m_colors[i].g = color.g;
	m_colors[i].b = color.b;
}


void Image::Export(std::string path) const
{
	std::ofstream f;
	f.open(path, std::ios::out | std::ios::binary);
	if (!f.is_open()) {
		std::cout << "Error opening file\n";
		return;
	}

	unsigned char bmpPad[3] = { 0,0,0 };
	const int paddingAmt = ((4 - (m_width * 3) % 4) % 4);



	const int fileHeaderSize = 14;
	const int infoHeaderSize = 40;

	const int fileSize = fileHeaderSize + infoHeaderSize + m_width * m_height * 3 + paddingAmt * m_height;

	//File Type
	unsigned char fileHeader[fileHeaderSize];
	fileHeader[0] = 'B';
	fileHeader[1] = 'M';

	//File size
	fileHeader[2] = fileSize;
	fileHeader[3] = fileSize >> 8;
	fileHeader[4] = fileSize >> 16;
	fileHeader[5] = fileSize >> 24;

	//Reserved 1 (Not used)
	fileHeader[6] = 0;
	fileHeader[7] = 0;
	//Reserved 2 (Not used)
	fileHeader[8] = 0;
	fileHeader[9] = 0;

	//Pixel data offset
	fileHeader[10] = fileHeaderSize + infoHeaderSize;
	fileHeader[11] = 0;
	fileHeader[12] = 0;
	fileHeader[13] = 0;

	unsigned char infoHeader[infoHeaderSize];
	//Header size
	infoHeader[0] = infoHeaderSize;
	infoHeader[1] = 0;
	infoHeader[2] = 0;
	infoHeader[3] = 0;

	//Image width
	infoHeader[4] = m_width;
	infoHeader[5] = m_width >> 8;
	infoHeader[6] = m_width >> 16;
	infoHeader[7] = m_width >> 24;

	//Image height
	infoHeader[8] = m_height;
	infoHeader[9] = m_height >> 8;
	infoHeader[10] = m_height >> 16;
	infoHeader[11] = m_height >> 24;

	//Planes
	infoHeader[12] = 1;
	infoHeader[13] = 0;

	//Bits per pixel
	infoHeader[14] = 24;
	infoHeader[15] = 0;

	//Compression (No compression - comp.)
	infoHeader[16] = 0;
	infoHeader[17] = 0;
	infoHeader[18] = 0;
	infoHeader[19] = 0;

	//Image size (no comp.)
	infoHeader[20] = 0;
	infoHeader[21] = 0;
	infoHeader[22] = 0;
	infoHeader[23] = 0;

	//X pixels per meter (not specified)
	infoHeader[24] = 0;
	infoHeader[25] = 0;
	infoHeader[26] = 0;
	infoHeader[27] = 0;

	//Y pixels per meter (not specified)
	infoHeader[28] = 0;
	infoHeader[29] = 0;
	infoHeader[30] = 0;
	infoHeader[31] = 0;

	//Total colors (color palette not used)
	infoHeader[32] = 0;
	infoHeader[33] = 0;
	infoHeader[34] = 0;
	infoHeader[35] = 0;

	//Important colors (generally ignored)
	infoHeader[36] = 0;
	infoHeader[37] = 0;
	infoHeader[38] = 0;
	infoHeader[39] = 0;

	f.write(reinterpret_cast<char*>(fileHeader), fileHeaderSize);
	f.write(reinterpret_cast<char*>(infoHeader), infoHeaderSize);

	for (int y = m_height - 1; y >= 0; y--)
	{
		for (int x = m_width - 1; x >= 0; x--)
		{
			unsigned char r = static_cast<unsigned char>(getColor(x, y).r * 255.0f);
			unsigned char g = static_cast<unsigned char>(getColor(x, y).g * 255.0f);
			unsigned char b = static_cast<unsigned char>(getColor(x, y).b * 255.0f);

			unsigned char color[] = { b,g,r }; //in bitmap, start with b,g -- not rgb order
			f.write(reinterpret_cast<char*>(color), 3);
		}
		f.write(reinterpret_cast<char*>(bmpPad), paddingAmt);
	}

	f.close();
	std::cout << "File created!!!\n";
}

ImageData::ImageData(std::vector<float> vec, int label)
{
	this->vec = vec;
	this->label = label;
}
