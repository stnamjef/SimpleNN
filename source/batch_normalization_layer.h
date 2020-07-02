#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm : public Layer
	{
	private:
		int batch_size;
		int channels;
		int size;		// if 2d then row size
		bool is2d;
		double epsilon;
		double momentum;
		vector<vector<Matrix>> xhat;
		Vector move_mu;
		Vector move_var;
		Vector mu;
		Vector var;
		Vector gamma;
		Vector dgamma;
		Vector beta;
		Vector dbeta;
	public:

		BatchNorm(const vector<int>& input_size, bool is2d, double epsilon = 0.00001, double momentum = 0.9);

		void set_batch(int batch_size) override;

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override;

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& empty, bool isFirst) override;

		void update_weight(double l_rate, double lambda) override;

	private:

		void calc_batch_mu(const vector<vector<Matrix>>& prev_out);

		void calc_batch_var(const vector<vector<Matrix>>& prev_out);

		void normalize_and_shift(const vector<vector<Matrix>>& prev_out, const Vector& mu, const Vector& var);

		vector<vector<Matrix>> calc_dxhat(const vector<vector<Matrix>>& delta);

		vector<vector<Matrix>> calc_dx_dgamma_dbeta(const vector<vector<Matrix>>& dxhat);
	};

	//---------------------------------------------- function definition ----------------------------------------------

	BatchNorm::BatchNorm(const vector<int>& input_size, bool is2d, double epsilon, double momentum) :
		Layer(LayerType::BATCHNORM),
		batch_size(0),
		channels(input_size[2]),
		size(input_size[0]),
		is2d(is2d),
		epsilon(epsilon),
		momentum(momentum)
	{
		if (is2d)
		{
			move_mu.resize(channels);
			move_var.resize(channels);
			mu.resize(channels);
			var.resize(channels);
			gamma.resize(channels);
			dgamma.resize(channels);
			beta.resize(channels);
			dbeta.resize(channels);
		}
		else
		{
			move_mu.resize(size);
			move_var.resize(size);
			mu.resize(size);
			var.resize(size);
			gamma.resize(size);
			dgamma.resize(size);
			beta.resize(size);
			dbeta.resize(size);
		}

		move_mu.setZero();
		move_var.setZero();
		std::for_each(gamma.begin(), gamma.end(), [](double& elem) { elem = 1; });
		dgamma.setZero();
		beta.setZero();
		dbeta.setZero();
	}

	void BatchNorm::set_batch(int batch_size)
	{
		this->batch_size = batch_size;
		if (is2d)
		{
			output.resize(batch_size, vector<Matrix>(channels, Matrix(size, size)));
			delta.resize(batch_size, vector<Matrix>(channels, Matrix(size, size)));
			xhat.resize(batch_size, vector<Matrix>(channels, Matrix(size, size)));
		}
		else
		{
			output.resize(batch_size, vector<Matrix>(channels, Vector(size)));
			delta.resize(batch_size, vector<Matrix>(channels, Vector(size)));
			xhat.resize(batch_size, vector<Matrix>(channels, Vector(size)));
		}
	}

	void BatchNorm::forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction)
	{
		if (!isPrediction)
		{
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, mu, var);

			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
		}
		else
		{
			normalize_and_shift(prev_out, move_mu, move_var);
		}
	}

	void BatchNorm::calc_batch_mu(const vector<vector<Matrix>>& prev_out)
	{
		mu.setZero();
		if (is2d)
		{
			for (int n = 0; n < batch_size; n++)
				for (int ch = 0; ch < channels; ch++)
					mu[ch] += prev_out[n][ch].mean() / (double)batch_size;
		}
		else
		{
			for (int n = 0; n < batch_size; n++)
				mu += prev_out[n][0];
			mu /= (double)batch_size;
		}
	}

	void BatchNorm::calc_batch_var(const vector<vector<Matrix>>& prev_out)
	{
		var.setZero();
		if (is2d)
		{
			for (int n = 0; n < batch_size; n++)
				for (int ch = 0; ch < channels; ch++)
					var[ch] += prev_out[n][ch].var(mu[ch]) / (double)batch_size;
		}
		else
		{
			for (int n = 0; n < batch_size; n++)
				var += pow2d(prev_out[n][0] - mu);
			var /= (double)batch_size;
		}
	}

	void BatchNorm::normalize_and_shift(const vector<vector<Matrix>>& prev_out, const Vector& mu, const Vector& var)
	{
		if (is2d)
		{
			for (int n = 0; n < prev_out.size(); n++)
				for (int ch = 0; ch < prev_out[n].size(); ch++)
				{
					xhat[n][ch] = (prev_out[n][ch] - mu[ch]) / std::sqrt(var[ch] + epsilon);
					output[n][ch] = gamma[ch] * xhat[n][ch] + beta[ch];
				}
		}
		else
		{
			for (int n = 0; n < prev_out.size(); n++)
			{
				xhat[n][0] = (prev_out[n][0] - mu) / sqrt2d(var + epsilon);
				output[n][0] = gamma.element_wise(xhat[n][0]) + beta;
			}
		}
	}

	vector<vector<Matrix>> BatchNorm::backward_propagate(const vector<vector<Matrix>>& empty, bool isFirst)
	{
		return calc_dx_dgamma_dbeta(calc_dxhat(delta));
	}

	vector<vector<Matrix>> BatchNorm::calc_dxhat(const vector<vector<Matrix>>& delta)
	{
		vector<vector<Matrix>> dxhat(batch_size, vector<Matrix>(channels));
		if (is2d)
		{
			for (int n = 0; n < delta.size(); n++)
				for (int ch = 0; ch < delta[n].size(); ch++)
					dxhat[n][ch] = delta[n][ch] * gamma[ch];
		}
		else
		{
			for (int n = 0; n < delta.size(); n++)
				dxhat[n][0] = delta[n][0].element_wise(gamma);
		}
		return dxhat;
	}

	vector<vector<Matrix>> BatchNorm::calc_dx_dgamma_dbeta(const vector<vector<Matrix>>& dxhat)
	{
		vector<vector<Matrix>> dx(batch_size, vector<Matrix>(channels));
		double m = (double)batch_size;
		if (is2d)
		{
			Vector dxhat_sum(channels), weighted_dxhat_sum(channels);
			for (int ch = 0; ch < channels; ch++)
			{
				dxhat_sum[ch] = 0, weighted_dxhat_sum[ch] = 0;
				for (int n = 0; n < batch_size; n++)
				{
					dxhat_sum[ch] += dxhat[n][ch].mean();
					weighted_dxhat_sum[ch] += (dxhat[n][ch].element_wise(xhat[n][ch])).mean();
				}

				double denominator = m * std::sqrt(var[ch] + epsilon);
				for (int n = 0; n < batch_size; n++)
				{
					dx[n][ch] = (1.0 / denominator) * ((m * dxhat[n][ch]) - dxhat_sum[ch] - (xhat[n][ch] * weighted_dxhat_sum[ch]));
					dgamma[ch] += (xhat[n][ch].element_wise(delta[n][ch])).sum();
					dbeta[ch] += delta[n][ch].sum();
				}
			}
		}
		else
		{
			int n_node = xhat[0][0].size();
			Vector dxhat_sum(n_node), weighted_dxhat_sum(n_node);
			dxhat_sum.setZero(), weighted_dxhat_sum.setZero();

			for (int n = 0; n < xhat.size(); n++)
			{
				dxhat_sum += dxhat[n][0];
				weighted_dxhat_sum += dxhat[n][0].element_wise(xhat[n][0]);
			}

			Vector denominator = m * sqrt2d(var + epsilon);
			for (int n = 0; n < xhat.size(); n++)
			{
				dx[n][0] = (1.0 / denominator).element_wise(((m * dxhat[n][0]) - dxhat_sum - (xhat[n][0].element_wise(weighted_dxhat_sum))));
				dgamma += xhat[n][0].element_wise(delta[n][0]);
				dbeta += delta[n][0];
			}
		}
		return dx;
	}

	void BatchNorm::update_weight(double l_rate, double lambda)
	{
		gamma += -(l_rate / batch_size) * dgamma;
		beta += -(l_rate / batch_size) * dbeta;
		dgamma.setZero();
		dbeta.setZero();
	}
}