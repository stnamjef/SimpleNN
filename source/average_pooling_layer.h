#pragma once
#include "layer.h"

namespace simple_nn
{
	class AvgPool2d : public Layer
	{
	public:
		int batch;
		int channels;
		int in_h;
		int in_w;
		int ker_h;
		int ker_w;
		int out_h;
		int out_w;
		int stride;
		int out_block_size;
	public:
		AvgPool2d(int kernel_size, int stride);
		~AvgPool2d();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval = false) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	};

	AvgPool2d::AvgPool2d(int kernel_size, int stride) :
		Layer(AVGPOOL2D),
		batch(0),
		channels(0),
		in_h(0),
		in_w(0),
		ker_h(kernel_size),
		ker_w(kernel_size),
		out_h(0),
		out_w(0),
		stride(stride),
		out_block_size(0) {}

	AvgPool2d::~AvgPool2d()
	{
		delete_memory(output);
		delete_memory(delta);
	}

	void AvgPool2d::set_layer(int batch, const vector<int>& input_shape)
	{
		this->batch = batch;
		channels = input_shape[2];
		in_h = input_shape[0];
		in_w = input_shape[1];
		out_h = calc_outsize(in_h, ker_h, stride, 0);
		out_w = calc_outsize(in_w, ker_w, stride, 0);
		out_block_size = batch * channels * out_h * out_w;
		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
	}

	void AvgPool2d::forward_propagate(const float* prev_out, bool isEval)
	{
		float denominator = float(ker_h * ker_w);
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < out_h; i++) {
					for (int j = 0; j < out_w; j++) {
						int out_idx = j + out_w * (i + out_h * (c + channels * n));
						for (int x = 0; x < ker_h; x++) {
							for (int y = 0; y < ker_w; y++) {
								int im_i = i * stride + x;
								int im_j = j * stride + y;
								int in_idx = im_j + in_w * (im_i + in_h * (c + channels * n));
								if (im_i >= 0 && im_i < in_h &&
									im_j >= 0 && im_j < in_w) {
									output[out_idx] += prev_out[in_idx];
								}
							}
						}
						output[out_idx] /= denominator;
					}
				}
			}
		}
	}

	void AvgPool2d::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		float denominator = float(ker_h * ker_w);
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < out_h; i++) {
					for (int j = 0; j < out_w; j++) {
						int cur_idx = j + out_w * (i + out_h * (c + channels * n));
						for (int x = 0; x < ker_h; x++) {
							for (int y = 0; y < ker_w; y++) {
								int im_i = i * stride + x;
								int im_j = j * stride + y;
								int prev_idx = im_j + in_w * (im_i + in_h * (c + channels * n));
								if (im_i >= 0 && im_i < in_h &&
									im_j >= 0 && im_j < in_w) {
									prev_delta[prev_idx] = delta[cur_idx] / denominator;
								}
							}
						}
					}
				}
			}
		}
	}

	vector<int> AvgPool2d::output_shape() { return { out_h, out_w, channels }; }

	int AvgPool2d::get_out_block_size() { return out_block_size; }
}