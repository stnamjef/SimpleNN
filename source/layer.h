#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <limits>
#include "tensor.h"
#define DOUBLE_MIN std::numeric_limits<double>::min();
using namespace std;

namespace simple_nn
{
	class Layer
	{
	public:
		string type;
		Tensor output;
		Tensor delta;
	public:
		Layer(string type) : type(type) {}
		virtual void set_layer(int batch_size, const vector<int>& input_shape) = 0;
		virtual void reset_batch(int batch_size) = 0;
		virtual void forward_propagate(const Tensor& prev_out, bool isPrediction = false) = 0;
		virtual void backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst = false) = 0;
		virtual void update_weight(double l_rate, double lambda) { return; }
		virtual vector<int> input_shape() { return {}; }
		virtual vector<int> output_shape() = 0;
		virtual double calc_loss(const Vector& Y) { return 0.0; }
	};

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

	double sum_exp(const Matrix& mat, double max)
	{
		double out = std::accumulate(mat.begin(), mat.end(), 0.0,
			[&](const double& sum, const double& elem) { return sum + std::exp(elem + max); });
		return out;
	}
}