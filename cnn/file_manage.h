#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "Matrix.h"
using namespace std;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(string img_name, int NumberOfImages, int DataOfAnImage, vector<Matrix>& arr)
{
	arr.resize(NumberOfImages, Matrix(28, 28));

	ifstream file(img_name, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		char inputstring[1000];
		for (int i = 0; i < NumberOfImages; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					arr[i](r, c) = (double)temp / 255 * (1.175 + 0.1) - 0.1;
				}
			}
		}
	}
}

void ReadMNISTLabel(string label_name, int NumberOfImages, Vector& arr)
{
	arr.resize(NumberOfImages);
	ifstream file(label_name);
	for (int i = 0; i < NumberOfImages + 8; ++i)
	{
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));

		if (i > 7)
			arr[i - 8] = (int)temp;
	}
}

void write_matrix(ofstream& out, const Matrix& mat)
{
	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
			out << mat(i, j) << ',';
		out << endl;
	}
}

void write_vector(ofstream& out, const Vector& vec)
{
	for (int i = 0; i < vec.size(); i++)
		out << vec[i] << ',';
	out << endl;
}