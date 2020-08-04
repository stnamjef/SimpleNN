#pragma once
#include "layer.h"

namespace simple_nn
{
	class Pool2D : public Layer
	{
	private:
		int batch_size;
		int in_channels;
		int out_channels;
		int in_h;
		int in_w;
		int kernel_h;
		int kernel_w;
		int out_h;
		int out_w;
		int stride;
		string pool_opt;
	public:
		Pool2D(const vector<int>& kernel_size, int stride, string pool_opt = "max");
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size) override;
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst) override;
		vector<int> output_shape() override;
	};

	Pool2D::Pool2D(const vector<int>& kernel_size,
		int stride,
		string pool_opt) :
		Layer("pool2d"),
		batch_size(0),
		in_channels(0),
		out_channels(0),
		in_h(0),
		in_w(0),
		kernel_h(kernel_size[0]),
		kernel_w(kernel_size[1]),
		out_h(0),
		out_w(0),
		stride(stride),
		pool_opt(pool_opt) {}

	void Pool2D::set_layer(int batch_size, const vector<int>& input_shape)
	{
		this->batch_size = batch_size;
		in_channels = input_shape[2];
		out_channels = input_shape[2];
		in_h = input_shape[0];
		in_w = input_shape[1];
		out_h = calc_outsize(in_h, kernel_h, stride, 0);
		out_w = calc_outsize(in_w, kernel_w, stride, 0);

		output.resize(batch_size, out_channels, out_h, out_w);
		delta.resize(batch_size, out_channels, out_h, out_w);
	}

	void Pool2D::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, out_channels, out_h, out_w);
	}

	void Pool2D::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		if (pool_opt == "max") {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < in_channels; c++) {
					for (int i = 0; i < out_h; i++) {
						for (int j = 0; j < out_w; j++) {
							Vector temp(kernel_h * kernel_w);
							for (int x = 0; x < kernel_h; x++) {
								for (int y = 0; y < kernel_w; y++) {
									int ii = i + x + (stride - 1) * i;
									int jj = j + y + (stride - 1) * j;
									if (ii >= 0 && ii < in_h &&
										jj >= 0 && jj < in_w) {
										temp(x * kernel_w + y) = prev_out[n][c](ii, jj);
									}
								}
							}
							output[n][c](i, j) = temp.max();
						}
					}
				}
			}
		}
		else {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < in_channels; c++) {
					for (int i = 0; i < out_h; i++) {
						for (int j = 0; j < out_w; j++) {
							double sum = 0.0;
							for (int x = 0; x < kernel_h; x++) {
								for (int y = 0; y < kernel_w; y++) {
									int ii = i + x + (stride - 1) * i;
									int jj = j + y + (stride - 1) * j;
									if (ii >= 0 && ii < in_h &&
										jj >= 0 && jj < in_w) {
										sum += prev_out[n][c](ii, jj);
									}
								}
							}
							output[n][c](i, j) = sum / (kernel_h * kernel_w);
						}
					}
				}
			}
		}
	}

	void Pool2D::backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst)
	{
		if (pool_opt == "MAX") {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < in_channels; c++) {
					prev_delta[n][c].setZero();
					for (int i = 0; i < out_h; i++) {
						for (int j = 0; j < out_w; j++) {
							double max = DOUBLE_MIN;
							int max_i = -1;
							int max_j = -1;
							for (int x = 0; x < kernel_h; x++) {
								for (int y = 0; y < kernel_w; y++) {
									int ii = i + x + (stride - 1) * i;
									int jj = j + y + (stride - 1) * j;
									if (ii >= 0 && ii < in_h &&
										jj >= 0 && jj < in_w &&
										max < prev_out[n][c](ii, jj)) {
										max = prev_out[n][c](ii, jj);
										max_i = ii;
										max_j = jj;
									}
								}
							}
							prev_delta[n][c](max_i, max_j) = delta[n][c](i, j);
						}
					}
				}
			}
		}
		else {
			double denominator = (double)kernel_h * kernel_w;
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < in_channels; c++) {
					for (int i = 0; i < out_h; i++) {
						for (int j = 0; j < out_w; j++) {
							for (int x = 0; x < kernel_h; x++) {
								for (int y = 0; y < kernel_w; y++) {
									int ii = i + x + (stride - 1) * i;
									int jj = j + y + (stride - 1) * j;
									if (ii >= 0 && ii < in_h &&
										jj >= 0 && jj < in_w) {
										prev_delta[n][c](ii, jj) = delta[n][c](i, j) / denominator;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	vector<int> Pool2D::output_shape() { return { out_h, out_w, out_channels }; }
}