#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2d : public Layer
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
		string option;
		MatXf dkernel;
		VecXf dbias;
		MatXf im_col;
	public:
		MatXf kernel;
		VecXf bias;
		Conv2d(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

	Conv2d::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		pad(padding),
		option(option) {}

	void Conv2d::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		output.resize(batch * oc, ohw);
		delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
		bias.resize(oc);
		dbias.resize(oc);
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
	}

	void Conv2d::forward(const MatXf& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}

	void Conv2d::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!is_first) {
			for (int n = 0; n < batch; n++) {
				float* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

	void Conv2d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;

		if (t1 != 1) {
			kernel *= t1;
			bias *= t1;
		}

		kernel -= t2 * dkernel;
		bias -= t2 * dbias;
	}

	void Conv2d::zero_grad()
	{
		delta.setZero();
		dkernel.setZero();
		dbias.setZero();
	}

	vector<int> Conv2d::output_shape() { return { batch, oc, oh, ow }; }
}