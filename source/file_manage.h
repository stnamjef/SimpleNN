#pragma once
#include "common.h"

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

	MatXf read_mnist(string path, int n_imgs, bool train = true)
	{
		MatXf img;

		ifstream fin(path, ios::binary);
		if (fin.is_open()) {
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;

			fin.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			fin.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			fin.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			fin.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);

			img.resize(n_imgs, n_rows * n_cols);

			float m = 0.1306604762738431f;
			float s = 0.3081078038564622f;

			if (!train) {
				m = 0.13251460584233699f;
				s = 0.3104802479305348f;
			}

			for (int n = 0; n < n_imgs; n++) {
				for (int i = 0; i < n_rows; i++) {
					for (int j = 0; j < n_cols; j++) {
						unsigned char temp = 0;
						fin.read((char*)&temp, sizeof(temp));
						img(n, j + n_cols * i) = (temp / 255.f - m) / s;
					}
				}
			}
		}

		return img;
	}

	VecXi read_mnist_label(string path, int n_imgs)
	{
		VecXi label(n_imgs);

		ifstream fin(path);
		for (int i = 0; i < n_imgs + 8; ++i) {
			unsigned char temp = 0;
			fin.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				label[i - 8] = (int)temp;
			}
		}
		return label;
	}
}