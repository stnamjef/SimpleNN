#pragma once
#include "layer.h"
#include <limits>
#define FLOAT_MIN std::numeric_limits<float>::min();

namespace simple_nn
{
	class MaxPool2d : public Layer
	{
	private:
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
		int* indices;
	public:
		MaxPool2d(int kernel_size, int stride);
		~MaxPool2d();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval = false) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	};

	MaxPool2d::MaxPool2d(int kernel_size, int stride) :
		Layer(MAXPOOL2D),
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

	MaxPool2d::~MaxPool2d()
	{
		delete_memory(output);
		delete_memory(delta);
		delete_memory(indices);
	}

	void MaxPool2d::set_layer(int batch, const vector<int>& input_shape)
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
		allocate_memory(indices, out_block_size);
	}

	void MaxPool2d::forward_propagate(const float* prev_out, bool isEval)
	{
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < out_h; i++) {
					for (int j = 0; j < out_w; j++) {
						int out_idx = j + out_w * (i + out_h * (c + channels * n));
						float max = FLOAT_MIN;
						int max_idx = -1;
						for (int x = 0; x < ker_h; x++) {
							for (int y = 0; y < ker_w; y++) {
								int im_i = i * stride + x;
								int im_j = j * stride + y;
								int in_idx = im_j + in_w * (im_i + in_h * (c + channels * n));
								int valid = (im_i >= 0 && im_i < in_h && im_j >= 0 && im_j < in_w);
								float val = (valid != 0) ? prev_out[in_idx] : FLOAT_MIN;
								max_idx = (val > max) ? in_idx : max_idx;
								max = (val > max) ? val : max;
							}
						}
						output[out_idx] = max;
						indices[out_idx] = max_idx;
					}
				}
			}
		}
	}

	void MaxPool2d::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		for (int i = 0; i < out_block_size; i++) {
			int idx = indices[i];
			prev_delta[idx] += delta[i];
		}
	}

	vector<int> MaxPool2d::output_shape() { return { out_h, out_w, channels }; }

	int MaxPool2d::get_out_block_size() { return out_block_size; }
}