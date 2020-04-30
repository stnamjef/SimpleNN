#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "Vector.h"
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

void ReadMNIST(int NumberOfImages, int DataOfAnImage, Vector<Vector<double>>& arr)
{
	arr.resize(NumberOfImages);
	for (int i = 0; i < NumberOfImages; i++)
		arr[i] = Vector<double>(DataOfAnImage);

	ifstream file("train-images.idx3-ubyte", ios::binary);
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
					arr[i][(n_rows * r) + c] = (double)temp / 255;
				}
			}
		}
	}
}

void ReadMNISTLabel(int NumberOfImages, Vector<int>& arr)
{
	arr.resize(NumberOfImages);
	ifstream file("train-labels.idx1-ubyte");
	for (int i = 0; i < NumberOfImages + 8; ++i)
	{
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));

		if (i > 7)
			arr[i - 8] = (int)temp;
	}
}

//void min_max_in_column(const vector<vector<double>>& vec, vector<vector<double>>& min_max)
//{
//	min_max.resize(2, vector<double>{});
//	for (int j = 0; j < vec[0].size(); j++)
//	{
//		double min = vec[0][j];
//		double max = vec[0][j];
//		for (int i = 0; i < vec.size(); i++)
//		{
//			if (vec[i][j] < min)
//				min = vec[i][j];
//			if (vec[i][j] > max)
//				max = vec[i][j];
//		}
//		min_max[0].push_back(min);
//		min_max[1].push_back(max);
//	}
//}
//
//void min_max_normalization(vector<vector<double>>& X)
//{
//	vector<vector<double>> min_max;
//	min_max_in_column(X, min_max);
//
//	for (int j = 0; j < X[0].size(); j++)
//		for (int i = 0; i < X.size(); i++)
//			X[i][j] = (X[i][j] - min_max[0][j]) / (min_max[1][j] - min_max[0][j]);
//}