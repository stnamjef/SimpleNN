#pragma once
#include "layer.h"

namespace simple_nn
{
	class Activation : public Layer
	{
	private:
		int batch;
		int channels;
		int in_h;
		int in_w;
		int out_block_size;
		string activ_opt;
		bool is2d;
	public:
		Activation(string activ_opt);
		~Activation();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval = false) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	private:
		void tanh(const float* prev_out);
		void relu(const float* prev_out);
		void softmax(const float* prev_out);
		void calc_prev_delta_tanh(const float* prev_out, float* prev_delta) const;
		void calc_prev_delta_relu(const float* prev_out, float* prev_delta) const;
		void calc_prev_delta_softmax(float* prev_delta) const;
	};

	Activation::Activation(string activ_opt) :
		Layer(ACTIVATION),
		batch(0),
		channels(0),
		in_h(0),
		in_w(0),
		out_block_size(0),
		activ_opt(activ_opt),
		is2d(false) {}

	Activation::~Activation()
	{
		delete_memory(output);
		delete_memory(delta);
	}

	void Activation::set_layer(int batch, const vector<int>& input_shape)
	{
		this->batch = batch;
		if (input_shape.size() == 3) {
			channels = input_shape[2];
			in_h = input_shape[0];
			in_w = input_shape[1];
			out_block_size = batch * channels * in_h * in_w;
			is2d = true;
		}
		else {
			channels = 1;
			in_h = input_shape[0];
			in_w = 1;
			is2d = false;
			out_block_size = batch * in_h;
		}
		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
	}

	void Activation::forward_propagate(const float* prev_out, bool isEval)
	{
		if (activ_opt == "tanh") {
			tanh(prev_out);
		}
		else if (activ_opt == "relu") {
			relu(prev_out);
		}
		else if (activ_opt == "softmax") {
			if (in_h > 1 && in_w > 1) {
				throw (logic_error("Activation::forward_propagate(const float*, bool): Not a matrix function."));
			}
			softmax(prev_out);
		}
		else {
			throw(logic_error("Activation::forward_propagate(const float*, bool): Invalid activation option."));
		}
	}

	void Activation::tanh(const float* prev_out)
	{
		std::transform(prev_out, prev_out + out_block_size, output,
			[](const float& elem) { return 2 / (1 + std::exp(-2.0F * elem)) - 1; });
	}

	void Activation::relu(const float* prev_out)
	{
		std::transform(prev_out, prev_out + out_block_size, output,
			[](const float& elem) { return std::max(0.0F, elem); });
	}

	float sum_exp(const float* p, int size, float max)
	{
		float out = std::accumulate(p, p + size, 0.0F,
			[&](const float& sum, const float& elem) { return sum + std::exp(elem - max); });
		return out;
	}

	void Activation::softmax(const float* prev_out)
	{
		int im_size = in_h * in_w;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int offset = im_size * (c + channels * n);
				const float* begin = prev_out + offset;
				float max = *std::max_element(begin, begin + im_size);
				float sum = sum_exp(begin, im_size, max);
				std::transform(begin, begin + im_size, output + offset,
					[&](const float& elem) {
					return std::exp(elem - max) / sum;
				});
			}
		}
	}

	void Activation::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		if (activ_opt == "tanh") {
			calc_prev_delta_tanh(prev_out, prev_delta);
		}
		else if (activ_opt == "relu") {
			calc_prev_delta_relu(prev_out, prev_delta);
		}
		else if (activ_opt == "softmax") {
			if (in_h > 1 && in_w > 1) {
				throw (logic_error("Activation::forward_propagate(const float*, bool): Not a matrix function."));
			}
			calc_prev_delta_softmax(prev_delta);
		}
		else {
			throw(logic_error("Activation::forward_propagate(const float*, bool): Invalid activation option."));
		}
	}

	void Activation::calc_prev_delta_tanh(const float* prev_out, float* prev_delta) const
	{
		std::transform(prev_out, prev_out + out_block_size, delta, prev_delta,
			[](const float& elem1, const float& elem2) {
			float tanh = 2 / (1 + std::exp(-elem1)) - 1;
			return elem2 * (1 - tanh * tanh);
		});
	}

	void Activation::calc_prev_delta_relu(const float* prev_out, float* prev_delta) const
	{
		std::transform(prev_out, prev_out + out_block_size, delta, prev_delta,
			[](const float& elem1, const float& elem2) {
			return (elem1 < 0) ? 0 : elem2;
		});
	}

	void Activation::calc_prev_delta_softmax(float* prev_delta) const
	{
		// 복사를 해야할까?
		std::copy(delta, delta + out_block_size, prev_delta);
		//prev_delta = delta;
	}

	vector<int> Activation::output_shape()
	{
		if (is2d) {
			return { in_h, in_w, channels };
		}
		else {
			return { in_h };
		}
	}

	int Activation::get_out_block_size() { return out_block_size; }
}