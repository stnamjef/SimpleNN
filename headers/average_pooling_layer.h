#pragma once
#include "layer.h"

namespace simple_nn
{
	class AvgPool2d : public Layer
	{
	private:
		int batch;
		int ch;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int stride;
		// MatXf im_col;
	public:
		AvgPool2d(int kernel_size, int stride);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

	AvgPool2d::AvgPool2d(int kernel_size, int stride) :
		Layer(LayerType::AVGPOOL2D),
		batch(0),
		ch(0),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		stride(stride) {}

	void AvgPool2d::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ch = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, stride, 0);
		ow = calc_outsize(iw, kw, stride, 0);
		ohw = oh * ow;

		output.resize(batch * ch, ohw);
		delta.resize(batch * ch, ohw);
		// im_col.resize(kh * kw, ohw);
	}

	void AvgPool2d::forward(const MatXf& prev_out, bool is_training)
	{
		output.setZero();
		float* out = output.data();
		const float* pout = prev_out.data();
		float denominator = (float)(kh * kw);
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				for (int i = 0; i < oh; i++) {
					for (int j = 0; j < ow; j++) {
						int out_idx = j + ow * (i + oh * (c + ch * n));
						for (int y = 0; y < kh; y++) {
							for (int x = 0; x < kw; x++) {
								int ii = i * stride + y;
								int jj = j * stride + x;
								int in_idx = jj + iw * (ii + ih * (c + ch * n));
								if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
									out[out_idx] += pout[in_idx];
								}
							}
						}
						out[out_idx] /= denominator;
					}
				}
			}
		}

		/*for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				const float* im = prev_out.data() + ihw * (c + channels * n);
				im2col(im, 1, ih, iw, kh, stride, 0, im_col.data());
				output.row(c + channels * n) = im_col.colwise().mean();
			}
		}*/
	}

	void AvgPool2d::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		float* pd = prev_delta.data();
		const float* d = delta.data();
		float denominator = (float)(kh * kw);
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				for (int i = 0; i < oh; i++) {
					for (int j = 0; j < ow; j++) {
						int cur_idx = j + ow * (i + oh * (c + ch * n));
						for (int y = 0; y < kh; y++) {
							for (int x = 0; x < kw; x++) {
								int ii = y + stride * i;
								int jj = x + stride * j;
								int prev_idx = jj + iw * (ii + ih * (c + ch * n));
								if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
									pd[prev_idx] = d[cur_idx] / denominator;
								}
							}
						}
					}
				}
			}
		}
	}

	void AvgPool2d::zero_grad() { delta.setZero(); }

	vector<int> AvgPool2d::output_shape() { return { batch, ch, oh, ow }; }
}