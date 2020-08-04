#pragma once
#include "layer.h"

namespace simple_nn
{
	class Output : public Layer
	{
	private:
		int batch_size;
		int n_node;
		string loss_opt;
	public:
		Output(int n_node, string opt);
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size) override;
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& Y, Tensor& prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
		double calc_loss(const Vector& Y) override;
	private:
		Vector as_vector(double label);
	};

	Output::Output(int n_node, string opt) :
		Layer("output"),
		batch_size(0),
		n_node(n_node),
		loss_opt(opt) {}

	void Output::set_layer(int batch_size, const vector<int>& input_shape)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, 1, n_node);
	}

	void Output::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, 1, n_node);
	}

	void Output::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		output = prev_out;
	}

	void Output::backward_propagate(const Tensor& Y, Tensor& prev_delta, bool isFirst)
	{
		for (int n = 0; n < batch_size; n++) {
			prev_delta[n][0] = output[n][0] - as_vector(Y[n][0](0)); // -(Y - Out);
		}
	}

	vector<int>Output::output_shape() { return {}; }

	double Output::calc_loss(const Vector& Y)
	{
		double loss = 0.0;
		for (int n = 0; n < batch_size; n++) {
			if (loss_opt == "mse") {
				loss += 0.5 * (output[n][0] - as_vector(Y(n))).pow(2).sum();
			}
			else {
				double max = output[n][0].max();
				double softmax = std::exp(max + max) / sum_exp(output[n][0], max);
				loss += -std::log(softmax);
			}
		}
		return loss / batch_size;
	}

	Vector Output::as_vector(double label)
	{
		Vector output(n_node, 1);
		output.setZero();
		output((int)label) = 1;
		return output;
	}
}