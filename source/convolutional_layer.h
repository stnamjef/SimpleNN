#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2D : public Layer
	{
	private:
		int batch_size;
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
			const Vector b_trained = {});

		~Conv2D() {}

		void set_batch(int batch_size) override;

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override;

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override;

		void update_weight(double l_rate, double lambda) override;
	};

	void init_weight(vector<Matrix>& Ws, Init opt, int n_in, int n_out);

	void init_delta_weight(vector<Matrix>& dWs);

	Matrix conv2d(const Matrix& img, const Matrix& filt, int pad);

	Matrix rotate_180(const Matrix& filt);

	//------------------------------- function definition -------------------------------

	Conv2D::Conv2D(const vector<int>& input_size,
				   const vector<int>& filter_size,
				   int pad,
				   Init opt,
				   const vector<vector<int>>& indices,
				   const vector<Matrix>& Ws_trained,
				   const Vector b_trained) :
				   Layer(LayerType::CONV2D),
				   batch_size(0),
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
			init_weight(Ws, opt, in_size * in_size * in_channels, out_size * out_size);
			b.setZero();
			init_delta_weight(dWs);
			db.setZero();
		}
	}

	void init_weight(vector<Matrix>& Ws, Init opt, int n_in, int n_out)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (opt == Init::NORMAL)
		{
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);

			for (int n = 0; n < Ws.size(); n++)
				for (int i = 0; i < Ws[n].rows(); i++)
					for (int j = 0; j < Ws[n].cols(); j++)
						Ws[n](i, j) = dist(e);
		}
		else
		{
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);

			for (int n = 0; n < Ws.size(); n++)
				for (int i = 0; i < Ws[n].rows(); i++)
					for (int j = 0; j < Ws[n].cols(); j++)
						Ws[n](i, j) = dist(e);
		}
	}

	void init_delta_weight(vector<Matrix>& dWs)
	{
		for (int i = 0; i < dWs.size(); i++)
			dWs[i].setZero();
	}

	void Conv2D::set_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
		delta.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
	}

	void Conv2D::forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction)
	{
		for (int n = 0; n < batch_size; n++)
			for (int out_ch = 0; out_ch < out_channels; out_ch++)
			{
				output[n][out_ch].setZero();
				for (int in_ch = 0; in_ch < in_channels; in_ch++)
				{
					if (indices[in_ch][out_ch] != 0)
						output[n][out_ch] += conv2d(prev_out[n][in_ch], Ws[out_ch], pad);
				}
				output[n][out_ch] += b[out_ch];
			}
	}

	Matrix conv2d(const Matrix& img, const Matrix& filt, int pad)
	{
		int out = calc_outsize(img.rows(), filt.rows(), 1, pad);
		Matrix output(out, out);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				output(i, j) = 0;
				for (int x = 0; x < filt.rows(); x++)
					for (int y = 0; y < filt.cols(); y++)
					{
						int ii = i + x - pad;
						int jj = j + y - pad;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
							output(i, j) += img(ii, jj) * filt(x, y);
					}
			}
		return output;
	}

	vector<vector<Matrix>> Conv2D::backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst)
	{
		int r = prev_out.at(0).at(0).rows(), c = prev_out.at(0).at(0).cols();
		vector<vector<Matrix>> prev_delta(batch_size, vector<Matrix>(in_channels, Matrix(r, c)));

		for (int n = 0; n < batch_size; n++)
		{
			// calc delta w.r.t weight & bias of this layer
			for (int out_ch = 0; out_ch < out_channels; out_ch++)
			{
				for (int in_ch = 0; in_ch < in_channels; in_ch++)
					if (indices[in_ch][out_ch] != 0)
						dWs[out_ch] += conv2d(prev_out[n][in_ch], delta[n][out_ch], pad);
				db[out_ch] += delta[n][out_ch].sum();
			}

			if (!isFirst)
			{
				// calc delta w.r.t weighted sum of the previous layer
				for (int in_ch = 0; in_ch < in_channels; in_ch++)
				{
					prev_delta[n][in_ch].setZero();
					for (int out_ch = 0; out_ch < out_channels; out_ch++)
						if (indices[in_ch][out_ch] != 0)
							prev_delta[n][in_ch] += conv2d(delta[n][out_ch], rotate_180(Ws[out_ch]), filt_size - 1);
				}
			}
		}

		return prev_delta;
	}

	Matrix rotate_180(const Matrix& filt)
	{
		int r = (int)filt.rows(), c = (int)filt.cols();
		Matrix out(r, c);
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				out(i, j) = filt(r - i - 1, c - j - 1);
		return out;
	}

	void Conv2D::update_weight(double l_rate, double lambda)
	{
		for (int ch = 0; ch < out_channels; ch++)
		{
			Ws[ch] = (1 - (2 * l_rate * lambda) / batch_size) * Ws[ch] - (l_rate / batch_size) * dWs[ch];
			dWs[ch].setZero();
		}
		b = (1 - (2 * l_rate * lambda) / batch_size) * b - (l_rate / batch_size) * db;
		db.setZero();
	}
}