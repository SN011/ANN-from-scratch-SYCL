#pragma once

#include <vector>
#include <string>

struct Color {
	float r, g, b;
	Color();
	Color(float r, float g, float b);
	~Color();
};

struct ImageData {
	std::vector<float> vec;
	int label;
	ImageData(std::vector<float> vec, int label);
};

class Image {
public:
	Image(int width, int height);
	~Image();
	void Read(const char* path);
	void CustomRead(const char* path);
	void writeHeaders(std::fstream& f);
	Color getColor(int x, int y) const;
	Color getColor(int x, int y, int width) const;
	void setColor(const Color& color, int x, int y);

	void setColor(const Color& color, int i);
	int getWidth() {
		return m_width;
	};
	int getHeight() {
		return m_height;
	}

	void Export(std::string path) const;
	void ExportSection(std::string path, int row, int col);
	void ExportSectionToBinary(std::string path, int row, int col);
	void CustomExport(const char* path);
private:
	int m_width;
	int m_height;
	std::vector<Color> m_colors;
};

