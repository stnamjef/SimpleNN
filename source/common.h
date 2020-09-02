#pragma once
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <openblas/cblas.h>
using namespace std;
using namespace chrono;

namespace simple_nn
{
	template<class T>
	void allocate_memory(T*& p, int size)
	{
		p = new T[size];
		std::for_each(p, p + size, [](T& elem) { elem = 0; });
	}

	template<class T>
	void delete_memory(T*& p)
	{
		delete[] p;
	}

	void set_zero(float* p, int size)
	{
		std::for_each(p, p + size, [](float& elem) { elem = 0.0F; });
	}

	void set_one(float* p, int size)
	{
		std::for_each(p, p + size, [](float& elem) { elem = 1.0F; });
	}

	template<class T>
	void print(const T* p, int batch, int channels, int height, int width)
	{
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < height; i++) {
					for (int j = 0; j < width; j++) {
						cout << fixed << setprecision(4) << setw(10);
						cout << p[j + width * (i + height * (c + channels * n))];
					}
					cout << '\n';
				}
				cout << '\n';
			}
			cout << '\n';
		}
	}

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

	void init_weight(float* W, int size, int n_in, int n_out, string init_opt)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (init_opt == "normal") {
			float var = std::sqrt(2.0F / (n_in + n_out));
			normal_distribution<float> dist(0, var);
			std::for_each(W, W + size, [&](float& elem) { elem = dist(e); });
		}
		else {
			float r = 1 / std::sqrt((float)n_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W, W + size, [&](float& elem) { elem = dist(e); });
		}
	}
}