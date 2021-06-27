#pragma once
#include "layer.h"

namespace simple_nn
{
	class Flatten : public Layer
	{
	private:
		int batch;
		int channels;
		int height;
		int width;
		int out_block_size;
	public:
		Flatten();
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

	Flatten::Flatten() : Layer(LayerType::FLATTEN) {}

	void Flatten::set_layer(const vector<int>& input_shape)
	{
		assert(input_shape.size() == 4 && "Flatten::set_layer(const vector<int>&): Must be followed by 2d layer.");
		batch = input_shape[0];
		channels = input_shape[1];
		height = input_shape[2];
		width = input_shape[3];
		out_block_size = batch * channels * height * width;

		output.resize(batch, channels * height * width);
		delta.resize(batch, channels * height * width);
	}

	void Flatten::forward(const MatXf& prev_out, bool is_training)
	{
		std::copy(prev_out.data(), prev_out.data() + out_block_size, output.data());
	}

	void Flatten::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		std::copy(delta.data(), delta.data() + out_block_size, prev_delta.data());
	}

	void Flatten::zero_grad() { delta.setZero(); }

	vector<int> Flatten::output_shape() { return { batch, channels, height, width }; }
}