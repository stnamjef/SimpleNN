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
		int in_size;
		int filt_size;
		int out_size;
		int stride;
		Pool pool_opt;
	public:
		Pool2D(const vector<int>& input_size,
			const vector<int>& filter_size,
			int stride,
			Pool pool_opt = Pool::MAX);

		~Pool2D() {}

		void set_batch(int batch_size) override;

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override;

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override;
	};

	Matrix pool2d(const Matrix& img, int filt, int stride, Pool opt);

	Matrix delta_img(const Matrix& img, const Matrix& delta, int filt, int stride, Pool opt);

	//------------------------------- function definition -------------------------------

	Pool2D::Pool2D(const vector<int>& input_size,
				   const vector<int>& filter_size,
				   int stride,
				   Pool pool_opt) :
				   Layer(LayerType::POOL2D),
				   batch_size(0),
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

	void Pool2D::set_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
		delta.resize(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
	}

	void Pool2D::forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction)
	{
		for (int n = 0; n < batch_size; n++)
			for (int ch = 0; ch < out_channels; ch++)
				output[n][ch] = pool2d(prev_out[n][ch], filt_size, stride, pool_opt);
	}

	Matrix pool2d(const Matrix& img, int filt, int stride, Pool opt)
	{
		int out = calc_outsize(img.rows(), filt, stride, 0);
		Matrix output(out, out);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				vector<double> temp((__int64)filt * filt);
				for (int x = 0; x < filt; x++)
					for (int y = 0; y < filt; y++)
					{
						int ii = i + x + (stride - 1) * i;
						int jj = j + y + (stride - 1) * j;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
							temp[x * filt + y] = img(ii, jj);
					}
				if (opt == Pool::MAX)
					output(i, j) = *std::max_element(temp.begin(), temp.end());
				else
					output(i, j) = std::accumulate(temp.begin(), temp.end(), 0.0) / (double)temp.size();
			}
		return output;
	}

	vector<vector<Matrix>> Pool2D::backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst)
	{
		vector<vector<Matrix>> prev_delta(batch_size, vector<Matrix>(in_channels));

		for (int n = 0; n < batch_size; n++)
			for (int ch = 0; ch < in_channels; ch++)
				prev_delta[n][ch] = delta_img(prev_out[n][ch], delta[n][ch], filt_size, stride, pool_opt);

		return prev_delta;
	}

	Matrix delta_img(const Matrix& img, const Matrix& delta, int filt, int stride, Pool opt)
	{
		Matrix output(img.rows(), img.cols());
		output.setZero();

		int out = calc_outsize(img.rows(), filt, stride, 0);
		for (int i = 0; i < out; i++)
			for (int j = 0; j < out; j++)
			{
				vector<double> temp;
				for (int x = 0; x < filt; x++)
					for (int y = 0; y < filt; y++)
					{
						int ii = i + x + (stride - 1) * i;
						int jj = j + y + (stride - 1) * j;

						if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
						{
							if (opt == Pool::MAX)
								temp.push_back(img(ii, jj));
							else
								output(ii, jj) = delta(i, j) / pow((double)filt, 2);
						}
					}
				if (opt == Pool::MAX)
				{
					int max = (int)std::distance(temp.begin(), std::max_element(temp.begin(), temp.end()));
					int ii = i + (max / filt) + (stride - 1) * i;
					int jj = j + (max % filt) + (stride - 1) * j;
					output(ii, jj) = delta(i, j);
				}
			}
		return output;
	}
}