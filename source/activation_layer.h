#pragma once
#include "layer.h"

namespace simple_nn
{
	class Activation : public Layer
	{
	public:
		int batch_size;
		int channels;
		int in_h;
		int in_w;
		string activate_opt;
		bool is2d;
	public:
		Activation(string opt);
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size);
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
	private:
		void tanh(const Tensor& prev_out);
		void relu(const Tensor& prev_out);
		void softmax(const Tensor& prev_out);
		void calc_prev_delta_tanh(const Tensor& prev_out, Tensor& prev_delta) const;
		void calc_prev_delta_relu(const Tensor& prev_out, Tensor& prev_delta) const;
		void calc_prev_delta_softmax(Tensor& prev_delta) const;
	};

	Activation::Activation(string opt) :
		Layer("activation"),
		batch_size(0),
		channels(0),
		in_h(0),
		in_w(0),
		activate_opt(opt),
		is2d(false) {}

	void Activation::set_layer(int batch_size, const vector<int>& input_shape)
	{
		this->batch_size = batch_size;
		if (input_shape.size() == 3) {
			channels = input_shape[2];
			in_h = input_shape[0];
			in_w = input_shape[1];
			is2d = true;
			output.resize(batch_size, channels, in_h, in_w);
			delta.resize(batch_size, channels, in_h, in_w);
		}
		else {
			channels = 1;
			in_h = input_shape[0];
			in_w = 1;
			is2d = false;
			output.resize(batch_size, channels, in_h);
			delta.resize(batch_size, channels, in_h);
		}
	}

	void Activation::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		if (is2d) {
			output.resize(batch_size, channels, in_h, in_w);
		}
		else {
			output.resize(batch_size, channels, in_h);
		}
	}

	void Activation::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		if (activate_opt == "tanh") {
			tanh(prev_out);
		}
		else if (activate_opt == "relu") {
			relu(prev_out);
		}
		else {
			softmax(prev_out);
		}
	}

	void Activation::tanh(const Tensor& prev_out)
	{
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < channels; c++) {
				std::transform(prev_out[n][c].begin(), prev_out[n][c].end(), output[n][c].begin(),
					[](const double& elem) {
					return 2 / (1 + std::exp(-elem)) - 1;
				});
			}
		}
	}

	void Activation::relu(const Tensor& prev_out)
	{
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < channels; c++) {
				std::transform(prev_out[n][c].begin(), prev_out[n][c].end(), output[n][c].begin(),
					[](const double& elem) {
					return std::max(0.0, elem);
				});
			}
		}
	}

	void Activation::softmax(const Tensor& prev_out)
	{
		if (in_h > 1 && in_w > 1) {
			cout << "softmax(const Vector&): Not a matrix function." << endl;
			exit(100);
		}
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < channels; c++) {
				double max = prev_out[n][c].max();
				double sum_exp_ = sum_exp(prev_out[n][c], max);
				std::transform(prev_out[n][c].begin(), prev_out[n][c].end(), output[n][c].begin(),
					[&](const double& elem) {
					return std::exp(elem + max) / sum_exp_;
				});
			}
		}
	}

	void Activation::backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst)
	{
		if (activate_opt == "tanh") {
			calc_prev_delta_tanh(prev_out, prev_delta);
		}
		else if (activate_opt == "relu") {
			calc_prev_delta_relu(prev_out, prev_delta);
		}
		else {
			calc_prev_delta_softmax(prev_delta);
		}
	}

	void Activation::calc_prev_delta_tanh(const Tensor& prev_out, Tensor& prev_delta) const
	{
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < channels; c++) {
				std::transform(prev_out[n][c].begin(), prev_out[n][c].end(), delta[n][c].begin(), prev_delta[n][c].begin(),
					[](const double& elem1, const double& elem2) {
					double tanh = 2 / (1 + std::exp(-elem1)) - 1;
					return elem2 * 0.5 * (1 - tanh * tanh);
				});
			}
		}
	}

	void Activation::calc_prev_delta_relu(const Tensor& prev_out, Tensor& prev_delta) const
	{
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < channels; c++) {
				std::transform(prev_out[n][c].begin(), prev_out[n][c].end(), delta[n][c].begin(), prev_delta[n][c].begin(),
					[](const double& elem1, const double& elem2) {
					return elem1 < 0 ? 0 : elem2;
				});
			}
		}
	}

	void Activation::calc_prev_delta_softmax(Tensor& prev_delta) const { prev_delta = delta; }

	vector<int> Activation::output_shape()
	{
		if (is2d) {
			return { in_h, in_w, channels };
		}
		else {
			return { in_h };
		}
	}
}