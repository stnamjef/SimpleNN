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

	void init_weight(vector<Matrix>& Ws, Init opt, int n_in, int n_out);

	void init_weight(Matrix& W, Init opt, int n_in, int n_out);

	void init_delta_weight(vector<Matrix>& dWs);

	Matrix conv2d(const Matrix& img, const Matrix& filt, int pad);

	Matrix pool2d(const Matrix& img, int filt, int stride, Pool opt);

	Matrix delta_img(const Matrix& img, const Matrix& delta, int filt, int stride, Pool opt);

	Matrix tanh(const Matrix& sum);

	Matrix tanh_prime(const Matrix& sum);

	Matrix relu(const Matrix& sum);

	Matrix relu_prime(const Matrix& sum);

	double sum_exp(const Vector& sum, double max);

	Vector softmax(const Vector& sum);

	Matrix activate(const Matrix& sum, Activate opt);

	Matrix activate_prime(const Matrix& sum, Activate opt);

	vector<vector<Vector>> flatten(const vector<vector<Matrix>>& input, int batch_size);

	vector<vector<Matrix>> unflatten(const vector<vector<Vector>>& input, int channels, int img_size);

	Matrix rotate_180(const Matrix& filt);

	Vector as_vector(int label, int size);

	Vector pow2(const Vector& vec);

	int max_idx(const Vector& vec);

	double calc_error(const Vector& expected, const Vector& predicted);


	//------------------------------- function definition -------------------------------

	int calc_outsize(int in_size, int filt_size, int stride, int pad)
	{
		return (int)floor((in_size + 2 * pad - filt_size) / stride) + 1;
	}

	void init_weight(vector<Matrix>& Ws, Init opt, int n_in, int n_out)
		// seed 값 수정
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (opt == Init::NORMAL)
		{
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);

			for (unsigned int n = 0; n < Ws.size(); n++)
				for (int i = 0; i < Ws[n].rows(); i++)
					for (int j = 0; j < Ws[n].cols(); j++)
						Ws[n](i, j) = dist(e);
		}
		else
		{
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);

			for (unsigned int n = 0; n < Ws.size(); n++)
				for (int i = 0; i < Ws[n].rows(); i++)
					for (int j = 0; j < Ws[n].cols(); j++)
						Ws[n](i, j) = dist(e);
		}
	}

	void init_weight(Matrix& W, Init opt, int n_in, int n_out)
		// seed 값 수정
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (opt == Init::NORMAL)
		{
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);

			for (int i = 0; i < W.rows(); i++)
				for (int j = 0; j < W.cols(); j++)
					W(i, j) = dist(e);
		}
		else
		{
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);

			for (int i = 0; i < W.rows(); i++)
				for (int j = 0; j < W.cols(); j++)
					W(i, j) = dist(e);
		}
	}

	void init_delta_weight(vector<Matrix>& dWs)
	{
		for (unsigned int i = 0; i < dWs.size(); i++)
			dWs[i].setZero();
	}

	Matrix conv2d(const Matrix& img, const Matrix& filt, int pad)
	{
		int out = calc_outsize(img.rows(), filt.rows(), 1, pad);
		Matrix output(out, out);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				output(i, j) = 0;
				for (int x = 0; x < filt.rows(); x++)
					for (int y = 0; y < filt.cols(); y++)
					{
						int ii = i + x - pad;
						int jj = j + y - pad;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
							output(i, j) += img(ii, jj) * filt(x, y);
					}
			}
		return output;
	}

	Matrix pool2d(const Matrix& img, int filt, int stride, Pool opt)
	{
		int out = calc_outsize(img.rows(), filt, stride, 0);
		Matrix output(out, out);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				vector<double> temp((__int64)filt * filt);
				for (int x = 0; x < filt; x++)
					for (int y = 0; y < filt; y++)
					{
						int ii = i + x + (stride - 1) * i;
						int jj = j + y + (stride - 1) * j;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
							temp[x * filt + y] = img(ii, jj);
					}
				if (opt == Pool::MAX)
					output(i, j) = *std::max_element(temp.begin(), temp.end());
				else
					output(i, j) = std::accumulate(temp.begin(), temp.end(), 0.0) / (double)temp.size();
			}
		return output;
	}

	Matrix delta_img(const Matrix& img, const Matrix& delta, int filt, int stride, Pool opt)
	{
		Matrix output(img.rows(), img.cols());
		output.setZero();

		int out = calc_outsize(img.rows(), filt, stride, 0);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				vector<double> temp;
				for (int x = 0; x < filt; x++)
					for (int y = 0; y < filt; y++)
					{
						int ii = i + x + (stride - 1) * i;
						int jj = j + y + (stride - 1) * j;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
						{
							if (opt == Pool::MAX)
								temp.push_back(img(ii, jj));
							else
								output(ii, jj) = delta(i, j) / pow((double)filt, 2);
						}
					}
				if (opt == Pool::MAX)
				{
					int max = (int)std::distance(temp.begin(), std::max_element(temp.begin(), temp.end()));
					int ii = i + (max / filt) + (stride - 1) * i;
					int jj = j + (max % filt) + (stride - 1) * j;
					output(ii, jj) = delta(i, j);
				}
			}
		return output;
	}

	Matrix tanh(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			return 2 / (1 + std::exp(-elem)) - 1;
		});
		return out;
	}

	Matrix tanh_prime(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		std::transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			double tanh = 2 / (1 + std::exp(-elem)) - 1;
			return 0.5 * (1 - pow(tanh, 2));
		});
		return out;
	}

	Matrix relu(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		std::transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			return std::max(0.0, elem);
		});
		return out;
	}

	Matrix relu_prime(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		std::transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			return (elem < 0) ? 0 : 1;
		});
		return out;
	}

	double sum_exp(const Vector& sum, double max)
	{
		double out = std::accumulate(sum.begin(), sum.end(), 0.0, [&](const double& sum, const double& elem) {
			return sum + exp(elem + max);
		});
		return out;
	}

	Vector softmax(const Vector& sum)
	{
		if (sum.rows() > 1 && sum.cols() > 1)
		{
			cout << "softmax(const Vector&): Not a matrix function." << endl;
			exit(100);
		}
		Vector out(sum.size());
		double max = sum.max();
		double exps = sum_exp(sum, max);
		std::transform(sum.begin(), sum.end(), out.begin(), [&](const double& elem) {
			return exp(elem + max) / exps;
		});
		return out;
	}

	Matrix activate(const Matrix& sum, Activate opt)
	{
		if (opt == Activate::TANH)
			return tanh(sum);
		else if (opt == Activate::RELU)
			return relu(sum);
		else
			return softmax(sum);
	}

	Matrix activate_prime(const Matrix& sum, Activate opt)
	{
		if (opt == Activate::TANH)
			return tanh_prime(sum);
		else if (opt == Activate::RELU)
			return relu_prime(sum);
		else
		{
			cout << "activate_prime(const Matrix&, Activate): Invalid argument." << endl;
			exit(100);
		}
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

	Matrix rotate_180(const Matrix& filt)
	{
		int r = (int)filt.rows(), c = (int)filt.cols();
		Matrix out(r, c);
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				out(i, j) = filt(r - i - 1, c - j - 1);
		return out;
	}

	Vector as_vector(int label, int size)
	{
		Vector out(size);
		out.setZero();
		out[label]++;
		return out;
	}

	Vector pow2(const Vector& vec)
	{
		Vector out(vec.size());
		std::transform(vec.begin(), vec.end(), out.begin(), [](const double& elem) { return std::pow(elem, 2); });
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