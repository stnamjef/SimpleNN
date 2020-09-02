#pragma once
#include <fstream>
#include <string>
using namespace std;

namespace simple_nn
{
	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}

	void ReadMNIST(string img_name, int NumberOfImages, int DataOfAnImage, float* arr)
	{
		ifstream file(img_name, ios::binary);
		if (file.is_open()) {
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

			int im_size = n_rows * n_cols;
			for (int n = 0; n < NumberOfImages; n++) {
				for (int i = 0; i < n_rows; i++) {
					for (int j = 0; j < n_cols; j++) {
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));

						int idx = j + n_cols * (i + n_rows * n);
						arr[idx] = (float)temp / 255 * (1.175F + 0.1F) - 0.1F;
						//arr[idx] = float(temp);
					}
				}
			}
		}
	}

	void ReadMNISTLabel(string label_name, int NumberOfImages, int* arr)
	{
		ifstream file(label_name);
		for (int i = 0; i < NumberOfImages + 8; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				arr[i - 8] = (int)temp;
			}
		}
	}
}