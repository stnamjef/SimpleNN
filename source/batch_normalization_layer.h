#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm : public Layer
	{
	public:
		int n_batch;
		int count;
		int channels;
		int size;
		double epsilon;
		Vector move_mu;
		Vector move_var;
		Vector mu;
		Vector var;
		Vector g;
		Vector dg;
		Vector b;
		Vector db;
	public:
		BatchNorm(const vector<int>& input_size, double epsilon = 0.00001) :
			Layer(LayerType::BATCHNORM),
			n_batch(0),
			count(0),
			channels(input_size[2]),
			size(input_size[0]),
			epsilon(epsilon)
		{
			move_mu.resize(channels);
			move_var.resize(channels);
			mu.resize(channels);
			var.resize(channels);
			g.resize(channels);
			dg.resize(channels);
			b.resize(channels);
			db.resize(channels);

			move_mu.setZero();
			move_var.setZero();
			mu.setZero();
			var.setZero();
			std::for_each(g.begin(), g.end(), [](double& elem) { elem = 1; });
			dg.setZero();
			b.setZero();
			db.setZero();
		}

		~BatchNorm() {}

		void set_batch(int batch_size, int n_batch) override
		{
			this->n_batch = n_batch;
			if (channels == 1)
			{
				output.resize(batch_size, vector<Matrix>(channels, Vector(size)));
				delta.resize(batch_size, vector<Matrix>(channels, Vector(size)));
			}
			else
			{
				output.resize(batch_size, vector<Matrix>(channels, Matrix(size, size)));
				delta.resize(batch_size, vector<Matrix>(channels, Matrix(size, size)));
			}
		}

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override
		{
			int batch_size = (int)prev_out.size();

			if (!isPrediction)
			{
				// calculate batch mean & variance of each channels
				for (int ch = 0; ch < channels; ch++)
					for (int batch = 0; batch < batch_size; batch++)
					{
						double mean = prev_out[batch][ch].mean();
						mu[ch] += mean / batch_size;
						var[ch] += prev_out[batch][ch].var(mean) / batch_size;
					}

				// normalize input data
				for (int ch = 0; ch < channels; ch++)
					for (int batch = 0; batch < batch_size; batch++)
						for (int i = 0; i < output[batch][ch].rows(); i++)
							for (int j = 0; j < output[batch][ch].cols(); j++)
							{
								double xhat = (prev_out[batch][ch](i, j) - mu[ch]) / std::sqrt(var[ch] + epsilon);
								output[batch][ch](i, j) = g[ch] * xhat + b[ch];
							}

				// accumulate Mu and Var
				move_mu += (1 / (double)n_batch) * mu;
				move_var += (1 / (double)n_batch) * var;
				mu.setZero();
				var.setZero();
			}
			else
			{
				for (int ch = 0; ch < channels; ch++)
					for (int batch = 0; batch < batch_size; batch++)
						for (int i = 0; i < output[batch][ch].rows(); i++)
							for (int j = 0; j < output[batch][ch].cols(); j++)
							{
								double xhat = (prev_out[batch][ch](i, j) - move_mu[ch]) / std::sqrt(move_var[ch] + epsilon);
								output[batch][ch](i, j) = g[ch] * xhat + b[ch];
							}

				// 임시 코드 수정해야됨.
				count++;
				if (count == 70000)
				{
					count = 0;
					move_mu.setZero();
					move_var.setZero();
				}
			}
		}

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& empty, bool isFirst) override
		{
			int batch_size = (int)delta.size();
			double N = (channels == 1) ? batch_size * size : batch_size * size * size;

			vector<vector<Matrix>> dxhat(batch_size, vector<Matrix>(channels));
			for (int ch = 0; ch < channels; ch++)
				for (int n = 0; n < batch_size; n++)
					dxhat[n][ch] = delta[n][ch] * g[ch];

			vector<vector<Matrix>> prev_delta(batch_size, vector<Matrix>(channels));
			Vector dxhat_sum(channels), weighted_dxhat_sum(channels);
			for (int ch = 0; ch < channels; ch++)
			{
				dxhat_sum[ch] = 0.0, weighted_dxhat_sum[ch] = 0.0;
				for (int n = 0; n < batch_size; n++)
				{
					dxhat_sum[ch] += dxhat[n][ch].sum();
					weighted_dxhat_sum[ch] += (dxhat[n][ch].element_wise(output[n][ch])).sum();
				}

				double denominator = N * std::sqrt(2 + epsilon);
				for (int n = 0; n < batch_size; n++)
				{
					prev_delta[n][ch] = (1 / denominator) * ((N * dxhat[n][ch]) - dxhat_sum[ch] - (output[n][ch] * weighted_dxhat_sum[ch]));
					db[ch] += delta[n][ch].sum();
					dg[ch] += (output[n][ch].element_wise(delta[n][ch])).sum();
				}
			}

			return prev_delta;
		}

		void update_weight(double l_rate, double lambda, int batch_size) override
		{
			for (int ch = 0; ch < channels; ch++)
			{
				g[ch] = (1 - (2 * l_rate * lambda) / batch_size) * g[ch] - (l_rate / batch_size) * dg[ch];
				b[ch] = (1 - (2 * l_rate * lambda) / batch_size) * b[ch] - (l_rate / batch_size) * db[ch];
				dg[ch] = 0.0;
				db[ch] = 0.0;
			}
		}

	};
}