#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2D : public Layer
	{
	private:
		int batch_size;
		int in_channels;
		int out_channels;	// also # of kernels
		int in_h;
		int in_w;
		int kernel_h;
		int kernel_w;
		int out_h;
		int out_w;
		int pad;
		string init_opt;
		vector<vector<int>> indices;
		Tensor kernel;
		Tensor dkernel;
		Vector bias;
		Vector dbias;
	public:
		Conv2D(int out_channels,
			const vector<int>& kernel_size,
			int pad,
			string init_opt,
			const vector<int>& input_shape,
			const vector<vector<int>>& indices = {});
		Conv2D(int out_channels,
			const vector<int>& kernel_size,
			int pad,
			string init_opt,
			const vector<vector<int>>& indices = {});
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size) override;
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst) override;
		void update_weight(double l_rate, double lambda) override;
		vector<int> input_shape() override;
		vector<int> output_shape() override;
	private:
		void init_kernel(int n_in, int n_out);
		void init_dkernel();
	};

	Conv2D::Conv2D(int out_channels,
		const vector<int>& kernel_size,
		int pad,
		string init_opt,
		const vector<int>& input_shape,
		const vector<vector<int>>& indices) :
		Layer("conv2d"),
		batch_size(0),
		in_channels(input_shape[2]),
		out_channels(out_channels),
		in_h(input_shape[0]),
		in_w(input_shape[1]),
		kernel_h(kernel_size[0]),
		kernel_w(kernel_size[1]),
		out_h(calc_outsize(in_h, kernel_h, 1, pad)),
		out_w(calc_outsize(in_w, kernel_w, 1, pad)),
		pad(pad),
		init_opt(init_opt),
		indices(indices) {}

	Conv2D::Conv2D(int out_channels,
		const vector<int>& kernel_size,
		int pad,
		string init_opt,
		const vector<vector<int>>& indices) :
		Layer("conv2d"),
		batch_size(0),
		in_channels(0),
		out_channels(out_channels),
		in_h(0),
		in_w(0),
		kernel_h(kernel_size[0]),
		kernel_w(kernel_size[1]),
		out_h(0),
		out_w(0),
		pad(pad),
		init_opt(init_opt),
		indices(indices) {}

	void Conv2D::set_layer(int batch_size, const vector<int>& input_shape)
	{
		this->batch_size = batch_size;
		in_h = input_shape[0];
		in_w = input_shape[1];
		in_channels = input_shape[2];
		out_h = calc_outsize(in_h, kernel_h, 1, pad);
		out_w = calc_outsize(in_w, kernel_w, 1, pad);

		output.resize(batch_size, out_channels, out_h, out_w);
		delta.resize(batch_size, out_channels, out_h, out_w);
		kernel.resize(out_channels, in_channels, kernel_h, kernel_w);
		dkernel.resize(out_channels, in_channels, kernel_h, kernel_w);
		bias.resize(out_channels);
		dbias.resize(out_channels);

		init_kernel(in_h * in_w * in_channels, out_h * out_w);
		bias.setZero();
		init_dkernel();
		dbias.setZero();
		if (this->indices.size() == 0) {
			this->indices.resize(in_channels, vector<int>(out_channels, 1));
		}
	}

	void Conv2D::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, out_channels, out_h, out_w);
	}

	void Conv2D::init_kernel(int n_in, int n_out)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (init_opt == "normal")
		{
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);
			for (int n = 0; n < out_channels; n++) {
				for (int c = 0; c < in_channels; c++) {
					for (int i = 0; i < kernel_h; i++) {
						for (int j = 0; j < kernel_w; j++) {
							kernel[n][c](i, j) = dist(e);
						}
					}
				}
			}
		}
		else
		{
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);
			for (int n = 0; n < out_channels; n++) {
				for (int c = 0; c < in_channels; c++) {
					for (int i = 0; i < kernel_h; i++) {
						for (int j = 0; j < kernel_w; j++) {
							kernel[n][c](i, j) = dist(e);
						}
					}
				}
			}
		}
	}

	void Conv2D::init_dkernel()
	{
		for (int n = 0; n < out_channels; n++) {
			for (int c = 0; c < in_channels; c++) {
				dkernel[n][c].setZero();
			}
		}
	}

	void Conv2D::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		for (int n = 0; n < batch_size; n++) {
			for (int k = 0; k < out_channels; k++) {
				output[n][k].setZero();
				for (int c = 0; c < in_channels; c++) {
					if (indices[c][k] != 0) {
						for (int i = 0; i < out_h; i++) {
							for (int j = 0; j < out_w; j++) {
								for (int x = 0; x < kernel_h; x++) {
									for (int y = 0; y < kernel_w; y++) {
										int ii = i + x - pad;
										int jj = j + y - pad;
										if (ii >= 0 && ii < in_h &&
											jj >= 0 && jj < in_w) {
											output[n][k](i, j) += prev_out[n][c](ii, jj) * kernel[k][c](x, y);
										}
									}
								}
							}
						}
					}
				}
				output[n][k] += bias(k);
			}
		}
	}

	void Conv2D::backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst)
	{
		// calc delta w.r.t the kernel & the bias of this layer
		for (int n = 0; n < batch_size; n++) {
			for (int k = 0; k < out_channels; k++) {
				for (int c = 0; c < in_channels; c++) {
					if (indices[c][k] != 0) {
						for (int i = 0; i < kernel_h; i++) {
							for (int j = 0; j < kernel_w; j++) {
								for (int x = 0; x < out_h; x++) {
									for (int y = 0; y < out_w; y++) {
										int ii = i + x - pad;
										int jj = j + y - pad;
										if (ii >= 0 && ii < in_h &&
											jj >= 0 && jj < in_w) {
											dkernel[k][c](i, j) += prev_out[n][c](ii, jj) * delta[n][k](x, y);
										}
									}
								}
							}
						}
					}
				}
				dbias(k) += delta[n][k].sum();
			}
		}
		if (!isFirst) {
			// calc delta w.r.t the output of the previous layer
			int pad = kernel_h - 1;
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < in_channels; c++) {
					prev_delta[n][c].setZero();
					for (int k = 0; k < out_channels; k++) {
						if (indices[c][k] != 0) {
							for (int i = 0; i < in_h; i++) {
								for (int j = 0; j < in_w; j++) {
									for (int x = 0; x < kernel_h; x++) {
										for (int y = 0; y < kernel_w; y++) {
											int ii = i + x - pad;
											int jj = j + y - pad;
											if (ii >= 0 && ii < out_h &&
												jj >= 0 && jj < out_w) {
												int xx = kernel_h - x - 1;
												int yy = kernel_w - y - 1;
												prev_delta[n][c](i, j) += delta[n][k](ii, jj) * kernel[k][c](xx, yy);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	void Conv2D::update_weight(double l_rate, double lambda)
	{
		for (int k = 0; k < out_channels; k++) {
			for (int c = 0; c < in_channels; c++) {
				if (indices[c][k] != 0) {
					kernel[k][c] = (1 - (2 * l_rate * lambda) / batch_size) * kernel[k][c] - (l_rate / batch_size) * dkernel[k][c];
					dkernel[k][c].setZero();
				}
			}
		}
		bias = (1 - (2 * l_rate * lambda) / batch_size) * bias - (l_rate / batch_size) * dbias;
		dbias.setZero();
	}

	vector<int> Conv2D::input_shape()
	{
		if (in_h == 0 || in_w == 0 || in_channels == 0) {
			cout << "Conv2D::input_shape(): Input shape is empty." << endl;
			exit(100);
		}
		return { in_h, in_w, in_channels };
	}

	vector<int> Conv2D::output_shape() { return { out_h, out_w, out_channels }; }
}