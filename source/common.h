#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include "matrix.h"
using namespace std;

namespace simple_nn
{
	enum class LayerType
	{
		DENSE,
		CONV2D,
		POOL2D,
		ACTIVATION,
		BATCHNORM,
		OUTPUT
	};

	enum class Init
	{
		NORMAL,
		UNIFORM
	};

	enum class Activate
	{
		TANH,
		RELU,
		SOFTMAX
	};

	enum class Pool
	{
		MAX,
		AVG
	};

	enum class Loss
	{
		NONE,
		MSE,
		CROSS_ENTROPY
	};


	//------------------------------- function declaration -------------------------------


	int calc_outsize(int in_size, int filt_size, int stride, int pad);

	vector<vector<Vector>> flatten(const vector<vector<Matrix>>& input, int batch_size);

	vector<vector<Matrix>> unflatten(const vector<vector<Vector>>& input, int channels, int img_size);

	Vector as_vector(int label, int size);

	Matrix pow2d(const Matrix& mat);

	Matrix sqrt2d(const Matrix& mat);

	int max_idx(const Vector& vec);

	double calc_error(const Vector& expected, const Vector& predicted);


	//------------------------------- function definition -------------------------------

	int calc_outsize(int in_size, int filt_size, int stride, int pad)
	{
		return (int)floor((in_size + 2 * pad - filt_size) / stride) + 1;
	}

	vector<vector<Vector>> flatten(const vector<vector<Matrix>>& input, int batch_size)
	{
		int n_channels = (int)input.at(0).size(), img_size = (int)input.at(0).at(0).size();
		vector<vector<Vector>> out(batch_size, vector<Vector>(1, Vector(n_channels * img_size)));

		for (int batch = 0; batch < batch_size; batch++)
		{
			int k = 0;
			for (const auto& img : input[batch])
				for (int i = 0; i < img.rows(); i++)
					for (int j = 0; j < img.cols(); j++)
						out[batch][0][k++] = img(i, j);
		}

		return out;
	}

	vector<vector<Matrix>> unflatten(const vector<vector<Vector>>& input, int channels, int img_size)
	{
		vector<vector<Matrix>> out(input.size(), vector<Matrix>(channels, Matrix(img_size, img_size)));

		for (unsigned int batch = 0; batch < out.size(); batch++)
			for (int ch = 0; ch < channels; ch++)
				for (int i = 0; i < img_size; i++)
					for (int j = 0; j < img_size; j++)
					{
						int idx = i * img_size + j + (ch * out[batch][ch].size());
						out[batch][ch](i, j) = input[batch][0][idx];
					}

		return out;
	}

	Vector as_vector(int label, int size)
	{
		Vector out(size);
		out.setZero();
		out[label]++;
		return out;
	}

	Matrix pow2d(const Matrix& mat)
	{
		Vector out(mat.rows(), mat.cols());
		std::transform(mat.begin(), mat.end(), out.begin(), [](const double& elem) { return std::pow(elem, 2); });
		return out;
	}

	Matrix sqrt2d(const Matrix& mat)
	{
		Matrix out(mat.rows(), mat.cols());
		std::transform(mat.begin(), mat.end(), out.begin(), [](const double& elem) { return std::sqrt(elem); });
		return out;
	}

	int max_idx(const Vector& vec)
	{
		return (int)std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
	}

	double calc_error(const Vector& expected, const Vector& predicted)
	{
		int error = 0;
		double loss = 0.0;
		for (int i = 0; i < expected.size(); i++)
			if (expected[i] != predicted[i])
				error++;
		return error / (double)expected.size();
	}
}