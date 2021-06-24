#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <assert.h>
#include <Eigen/Dense>
#include "im2col.h"
#include "col2im.h"
using namespace std;
using namespace chrono;
using namespace Eigen;

namespace simple_nn
{

	typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatXf;
	typedef Matrix<float, Dynamic, 1> VecXf;
	typedef Matrix<float, 1, Dynamic> RowVecXf;
	typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatXi;
	typedef Matrix<int, Dynamic, 1> VecXi;

	void write_file(const MatXf& data, int channels, string fname)
	{
		ofstream fout(fname, std::ios::app);
		if (channels != 0) {
			int n_row = (int)data.rows() / channels; // batch size
			int n_col = (int)data.cols() * channels; // feature size * channels
			for (int i = 0; i < n_row; i++) {
				const float* begin = data.data() + n_col * i;
				for (int j = 0; j < n_col; j++) {
					fout << *begin;
					if (j == n_col - 1) {
						fout << endl;
					}
					else {
						fout << ',';
					}
					begin++;
				}
			}
		}
		else {
			for (int i = 0; i < data.rows(); i++) {
				for (int j = 0; j < data.cols(); j++) {
					fout << data(i, j);
					if (j == data.cols() - 1) {
						fout << endl;
					}
					else {
						fout << ',';
					}
				}
			}
		}
		fout.close();
	}

	enum class Init
	{
		LecunNormal,
		LecunUniform,
		XavierNormal,
		XavierUniform,
		KaimingNormal,
		KaimingUniform,
		Normal,
		Uniform
	};

	void init_weight(MatXf& W, int fan_in, int fan_out, Init option)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(4);

		if (option == Init::LecunNormal) {
			float s = std::sqrt(1.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::LecunUniform) {
			float r = std::sqrt(1.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::XavierNormal) {
			float s = std::sqrt(2.f / (fan_in + fan_out));
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::XavierUniform) {
			float r = std::sqrt(6.f / (fan_in + fan_out));
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::KaimingNormal) {
			float s = std::sqrt(2.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::KaimingUniform) {
			float r = std::sqrt(6.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == Init::Normal) {
			normal_distribution<float> dist(0.f, 0.1f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else {
			uniform_real_distribution<float> dist(-0.01f, 0.01f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
	}

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}
}