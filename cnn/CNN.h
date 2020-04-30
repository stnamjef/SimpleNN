#pragma once
#include <iostream>
#include <vector>
#include "Vector.h"
#include "Layer.h"

class CNN
{
private:
	vector<Layer*> network;
	vec<vec<double>> bias_weight;
	vec<vec<vec<double>>> weight;
public:
	CNN() {}
	void add(Layer* layer);
	void fit(const vec<vec<double>>& X, const vec<int>& Y, double l_rate = 0.001, int n_epoch = 100, int batch = 30);
	void print_network();
};

void CNN::add(Layer* layer) { network.push_back(layer); }


void init_weight(vec<vec<double>>& bias_weight, vec<vec<vec<double>>>& weight);

void xavier(vec<vec<double>>& bias_weight, vec<vec<vec<double>>>& weight);

double train_network(vector<Layer*>& network, vec<vec<double>>& bias_weight, vec<vec<vec<double>>>& weight,
	const vec<vec<double>>& X, const vec<int>& Y, double l_rate, int batch);

void CNN::print_network()
{
	for (auto layer : network)
	{
		layer->print_layer();
		cout << endl;
	}
}

vec<int> conv2d(const vec<int>& img, const vec<int>& kernel, int bias, int P, int S = 1)
{
	int img_size = (int)sqrt(img.size());
	int kernel_size = (int)sqrt(kernel.size());

	int out_size = (int)floor((img_size + 2 * P - kernel_size) / S) + 1;
	vec<int> out(out_size * out_size, 0);

	for (int i = 0; i < out_size; i++)
		for (int j = 0; j < out_size; j++)
			for (int x = 0; x < kernel_size; x++)
				for (int y = 0; y < kernel_size; y++)
				{
					int ii = i + x - P;
					int jj = j + y - P;

					if (i != 0)
						ii += (S - 1);
					if (j != 0)
						jj += (S - 1);

					if (ii >= 0 && ii < img_size && jj >= 0 && jj < img_size)
						out[i * out_size + j] += img[ii * img_size + jj] * kernel[x * kernel_size + y];
				}

	return out;
}

vec<int> pool2d(const vec<int>& img, int kernel, int S = 1)
{
	int img_size = (int)sqrt(img.size());

	int out_size = (int)floor((img_size - kernel) / S) + 1;
	vec<int> out(out_size * out_size, 0);

	for (int i = 0; i < out_size; i++)
		for (int j = 0; j < out_size; j++)
		{
			vec<int> temp(kernel * kernel);
			for (int x = 0; x < kernel; x++)
				for (int y = 0; y < kernel; y++)
				{
					int ii = i + x;
					int jj = j + y;

					if (i != 0)
						ii += (S - 1);
					if (j != 0)
						jj += (S - 1);

					temp[x * kernel + y] = img[ii * img_size + jj];
				}
			out[i * out_size + j] = *max_element(temp.begin(), temp.end());
		}
	return out;
}