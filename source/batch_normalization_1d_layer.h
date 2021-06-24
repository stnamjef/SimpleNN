#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm1d : public Layer
	{
	private:
		int batch;
		int n_feat;
		float eps;
		float momentum;
		MatXf xhat;
		MatXf dxhat;
		RowVecXf mu;
		RowVecXf var;
		RowVecXf dgamma;
		RowVecXf dbeta;
		RowVecXf sum1;
		RowVecXf sum2;
	public:
		RowVecXf move_mu;
		RowVecXf move_var;
		RowVecXf gamma;
		RowVecXf beta;
		BatchNorm1d(float eps = 0.00001f, float momentum = 0.9f);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	private:
		void calc_batch_mu(const MatXf& prev_out);
		void calc_batch_var(const MatXf& prev_out);
		void normalize_and_shift(const MatXf& prev_out, bool is_training);
	};

	BatchNorm1d::BatchNorm1d(float eps, float momentum) :
		Layer(LayerType::BATCHNORM1D),
		batch(0),
		n_feat(0),
		eps(eps),
		momentum(momentum) {}

	void BatchNorm1d::set_layer(const vector<int>& input_shape)
	{
		assert(input_shape.size() == 2 && "BatchNorm1d::set_layer(const vector<int>&): Must be followed by Linear layer.");

		batch = input_shape[0];
		n_feat = input_shape[1];

		output.resize(batch, n_feat);
		delta.resize(batch, n_feat);
		xhat.resize(batch, n_feat);
		dxhat.resize(batch, n_feat);
		move_mu.resize(n_feat);
		move_var.resize(n_feat);
		mu.resize(n_feat);
		var.resize(n_feat);
		gamma.resize(n_feat);
		dgamma.resize(n_feat);
		beta.resize(n_feat);
		dbeta.resize(n_feat);
		sum1.resize(n_feat);
		sum2.resize(n_feat);

		move_mu.setZero();
		move_var.setZero();
		gamma.setConstant(1.f);
		beta.setZero();
	}

	void BatchNorm1d::forward(const MatXf& prev_out, bool is_training)
	{
		if (is_training) {
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, is_training);
			// update moving mu and var
			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
		}
		else {
			normalize_and_shift(prev_out, is_training);
		}
	}

	void BatchNorm1d::calc_batch_mu(const MatXf& prev_out)
	{
		mu = prev_out.colwise().mean();
	}

	void BatchNorm1d::calc_batch_var(const MatXf& prev_out)
	{
		var.setZero();
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				float diff = prev_out(i, j) - mu[j];
				var[j] += diff * diff / batch; // (batch - 1)?
			}
		}
	}

	void BatchNorm1d::normalize_and_shift(const MatXf& prev_out, bool is_training)
	{
		const float* M = mu.data();
		const float* V = var.data();

		if (!is_training) {
			M = move_mu.data();
			V = move_var.data();
		}

		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				xhat(i, j) = (prev_out(i, j) - M[j]) / std::sqrt(V[j] + eps);
				output(i, j) = gamma[j] * xhat(i, j) + beta[j];
			}
		}
	}

	void BatchNorm1d::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		// calc dxhat
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				dxhat(i, j) = delta(i, j) * gamma[j];
			}
		}

		// calc Sum(dxhat), Sum(dxhat * xhat)
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				sum1[j] += dxhat(i, j);
				sum2[j] += dxhat(i, j) * xhat(i, j);
			}
		}

		// calc dx, dgamma, dbeta
		float m = (float)batch;
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				prev_delta(i, j) = (m * dxhat(i, j)) - sum1[j] - (xhat(i, j) * sum2[j]);
				prev_delta(i, j) /= m * std::sqrt(var[j] + eps);
				dgamma[j] += xhat(i, j) * delta(i, j);
				dbeta[j] += delta(i, j);
			}
		}
	}

	void BatchNorm1d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		if (t1 != 1) {
			gamma *= t1;
			beta *= t1;
		}
		gamma -= t2 * dgamma;
		beta -= t2 * dbeta;
	}

	void BatchNorm1d::zero_grad()
	{
		delta.setZero();
		dxhat.setZero();
		dgamma.setZero();
		dbeta.setZero();
		sum1.setZero();
		sum2.setZero();
	}

	vector<int> BatchNorm1d::output_shape() { return { batch, n_feat }; }
}