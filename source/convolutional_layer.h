#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2D : public Layer
	{
	private:
		int in_channels;
		int out_channels;
		int in_size;
		int filt_size;
		int out_size;
		int pad;
		vector<vector<int>> indices;
		vector<Matrix> Ws;
		vector<Matrix> dWs;
		Vector b;
		Vector db;
	public:
		Conv2D(const vector<int>& input_size,
			const vector<int>& filter_size,
			int pad,
			Init opt = Init::NORMAL,
			const vector<vector<int>>& indices = {},
			const vector<Matrix>& Ws_trained = {},
			const Vector b_trained = {}) :
			Layer(LayerType::CONV2D),
			in_channels(input_size[2]),
			out_channels(filter_size[2]),
			in_size(input_size[0]),
			filt_size(filter_size[0]),
			out_size(calc_outsize(in_size, filt_size, 1, pad)),
			pad(pad),
			indices(indices)
		{
			if (this->indices.size() == 0)
				this->indices.resize(in_channels, vector<int>(out_channels, 1));

			Ws.resize(out_channels, Matrix(filt_size, filt_size));
			dWs.resize(out_channels, Matrix(filt_size, filt_size));
			b.resize(out_channels);
			db.resize(out_channels);

			if (Ws_trained.size() != 0)
			{
				Ws = Ws_trained;
				b = b_trained;
			}
			else
			{
				// in_size * in_size * channels로도 해볼 것
				init_weight(Ws, opt, in_size * in_size, out_size * out_size);
				b.setZero();
			}

			init_delta_weight(dWs);
			db.setZero();
		}

		~Conv2D() {}

		void set_batch(int batch_size, int n_batch) override
		{
			output.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
			delta.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
		}

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override
		{
			for (int batch = 0; batch < prev_out.size(); batch++)
				for (int out_ch = 0; out_ch < out_channels; out_ch++)
				{
					output[batch][out_ch].setZero();
					for (int in_ch = 0; in_ch < in_channels; in_ch++)
					{
						if (indices[in_ch][out_ch] != 0)
							output[batch][out_ch] += conv2d(prev_out[batch][in_ch], Ws[out_ch], pad);
					}
					output[batch][out_ch] += b[out_ch];
				}
		}

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override
		{
			int batch_size = (int)prev_out.size();
			int r = prev_out.at(0).at(0).rows(), c = prev_out.at(0).at(0).cols();
			vector<vector<Matrix>> prev_delta(batch_size, vector<Matrix>(in_channels, Matrix(r, c)));

			for (int batch = 0; batch < batch_size; batch++)
			{
				// calc delta w.r.t weight & bias of this layer
				for (int out_ch = 0; out_ch < out_channels; out_ch++)
				{
					for (int in_ch = 0; in_ch < in_channels; in_ch++)
						if (indices[in_ch][out_ch] != 0)
							dWs[out_ch] += conv2d(prev_out[batch][in_ch], delta[batch][out_ch], pad);
					db[out_ch] += delta[batch][out_ch].sum();
				}

				if (!isFirst)
				{
					// calc delta w.r.t weighted sum of the previous layer
					for (int in_ch = 0; in_ch < in_channels; in_ch++)
					{
						prev_delta[batch][in_ch].setZero();
						for (int out_ch = 0; out_ch < out_channels; out_ch++)
							if (indices[in_ch][out_ch] != 0)
								prev_delta[batch][in_ch] += conv2d(delta[batch][out_ch], rotate_180(Ws[out_ch]), filt_size - 1);
					}
				}
			}

			return prev_delta;
		}

		void update_weight(double l_rate, double lambda, int batch_size) override
		{
			for (int ch = 0; ch < out_channels; ch++)
			{
				Ws[ch] = (1 - (2 * l_rate * lambda) / batch_size) * Ws[ch] - (l_rate / batch_size) * dWs[ch];
				dWs[ch].setZero();
			}
			b = (1 - (2 * l_rate * lambda) / batch_size) * b - (l_rate / batch_size) * db;
			db.setZero();
		}
	};
}