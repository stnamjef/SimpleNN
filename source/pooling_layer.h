#pragma once
#include "layer.h"

namespace simple_nn
{
	class Pool2D : public Layer
	{
	private:
		int in_channels;
		int out_channels;
		int in_size;
		int filt_size;
		int out_size;
		int stride;
		Pool pool_opt;
	public:
		Pool2D(const vector<int>& input_size,
			const vector<int>& filter_size,
			int stride,
			Pool pool_opt = Pool::MAX) :
			Layer(LayerType::POOL2D),
			in_channels(input_size[2]),
			out_channels(filter_size[2]),
			in_size(input_size[0]),
			filt_size(filter_size[0]),
			out_size(calc_outsize(in_size, filt_size, stride, 0)),
			stride(stride),
			pool_opt(pool_opt)
		{
			if (in_channels != out_channels)
			{
				cout << "Pool2D::Pool2D(): Invalid channel size." << endl;
				exit(100);
			}
		}

		~Pool2D() {}

		void set_batch(int batch_size, int n_batch) override
		{
			output.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
			delta.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
		}

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override
		{
			//#pragma omp parallel for
			for (int batch = 0; batch < prev_out.size(); batch++)
				for (int ch = 0; ch < out_channels; ch++)
					output[batch][ch] = pool2d(prev_out[batch][ch], filt_size, stride, pool_opt);
		}

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override
		{
			int batch_size = (int)prev_out.size();
			vector<vector<Matrix>> prev_delta(batch_size, vector<Matrix>(in_channels));

			//#pragma omp parallel for
			for (int batch = 0; batch < batch_size; batch++)
				for (int ch = 0; ch < in_channels; ch++)
					prev_delta[batch][ch] = delta_img(prev_out[batch][ch], delta[batch][ch], filt_size, stride, pool_opt);

			return prev_delta;
		}
	};
}