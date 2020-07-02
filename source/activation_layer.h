#pragma once
#include "layer.h"

namespace simple_nn
{
	class Activation : public Layer
	{
	private:
		int batch_size;
		int channels;
		Activate opt;
	public:
		Activation(int channels, Activate opt) :
			Layer(LayerType::ACTIVATION),
			batch_size(0),
			opt(opt),
			channels(channels) {}

		~Activation() {}

		void set_batch(int batch_size) override;

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override;

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override;
	};

	Matrix activate(const Matrix& sum, Activate opt);

	Matrix tanh(const Matrix& sum);

	Matrix relu(const Matrix& sum);

	Vector softmax(const Vector& sum);

	double sum_exp(const Vector& sum, double max);

	Matrix activate_prime(const Matrix& sum, Activate opt);

	Matrix tanh_prime(const Matrix& sum);

	Matrix relu_prime(const Matrix& sum);

	//------------------------------- function definition -------------------------------

	void Activation::set_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, vector<Matrix>(channels));
		delta.resize(batch_size, vector<Matrix>(channels));
	}

	void Activation::forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction)
	{
		for (int n = 0; n < batch_size; n++)
			for (int ch = 0; ch < channels; ch++)
				output[n][ch] = activate(prev_out[n][ch], opt);
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

	Matrix tanh(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			return 2 / (1 + std::exp(-elem)) - 1;
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

	double sum_exp(const Vector& sum, double max)
	{
		double out = std::accumulate(sum.begin(), sum.end(), 0.0, [&](const double& sum, const double& elem) {
			return sum + exp(elem + max);
		});
		return out;
	}

	vector<vector<Matrix>> Activation::backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst)
	{
		if (opt != Activate::SOFTMAX)
		{
			for (int n = 0; n < batch_size; n++)
				for (int ch = 0; ch < channels; ch++)
					delta[n][ch] *= activate_prime(prev_out[n][ch], opt);
		}
		return delta;
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

	Matrix tanh_prime(const Matrix& sum)
	{
		Matrix out(sum.rows(), sum.cols());
		std::transform(sum.begin(), sum.end(), out.begin(), [](const double& elem) {
			double tanh = 2 / (1 + std::exp(-elem)) - 1;
			return 0.5 * (1 - pow(tanh, 2));
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
}