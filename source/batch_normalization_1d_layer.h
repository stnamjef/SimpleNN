#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm1d : public Layer
	{
	public :
		int batch;
		int n_node;
		int out_block_size;
		float eps;
		float momentum;
		float* xhat;
		float* move_mu;
		float* move_var;
		float* mu;
		float* var;
		float* gamma;
		float* dgamma;
		float* beta;
		float* dbeta;
	public:
		BatchNorm1d(float eps = 0.00001F, float momentum = 0.9F);
		~BatchNorm1d();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval = false) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		void update_weight(float lr, float decay) override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	private:
		void calc_batch_mu(const float* prev_out, float* mu);
		void calc_batch_var(const float* prev_out, const float* mu, float* var);
		void normalize_and_shift(const float* prev_out, const float* mu, const float* var);
		void update_move_mu_move_var();
	};

	BatchNorm1d::BatchNorm1d(float eps, float momentum) :
		Layer(BATCHNORM1D),
		batch(0),
		n_node(0),
		out_block_size(0),
		eps(eps),
		momentum(momentum) {}

	BatchNorm1d::~BatchNorm1d()
	{
		delete_memory(output);
		delete_memory(delta);
		delete_memory(xhat);
		delete_memory(move_mu);
		delete_memory(move_var);
		delete_memory(mu);
		delete_memory(var);
		delete_memory(gamma);
		delete_memory(dgamma);
		delete_memory(beta);
		delete_memory(dbeta);
	}

	void BatchNorm1d::set_layer(int batch, const vector<int>& input_shape)
	{
		if (input_shape.size() != 1) {
			throw logic_error("BatchNorm1d::set_layer(int, const vector<int>): Invalid input shape.");
		}

		this->batch = batch;
		n_node = input_shape[0];
		out_block_size = batch * n_node;

		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
		allocate_memory(xhat, out_block_size);
		allocate_memory(move_mu, n_node);
		allocate_memory(move_var, n_node);
		allocate_memory(mu, n_node);
		allocate_memory(var, n_node);
		allocate_memory(gamma, n_node);
		allocate_memory(dgamma, n_node);
		allocate_memory(beta, n_node);
		allocate_memory(dbeta, n_node);

		set_zero(move_mu, n_node);
		set_zero(move_var, n_node);
		set_one(gamma, n_node);
		set_zero(dgamma, n_node);
		set_zero(beta, n_node);
		set_zero(dbeta, n_node);
	}

	void BatchNorm1d::forward_propagate(const float* prev_out, bool isEval)
	{
		if (!isEval) {
			calc_batch_mu(prev_out, mu);
			calc_batch_var(prev_out, mu, var);
			normalize_and_shift(prev_out, mu, var);
			update_move_mu_move_var();
		}
		else {
			normalize_and_shift(prev_out, move_mu, move_var);
		}
	}

	void BatchNorm1d::calc_batch_mu(const float* prev_out, float* mu)
	{
		set_zero(mu, n_node);
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				mu[i] += prev_out[i + n_node * n] / batch;
			}
		}
	}

	void BatchNorm1d::calc_batch_var(const float* prev_out, const float* mu, float* var)
	{
		set_zero(var, n_node);
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				float diff = prev_out[i + n_node * n] - mu[i];
				var[i] += diff * diff / batch;
			}
		}
	}

	void BatchNorm1d::normalize_and_shift(const float* prev_out, const float* mu, const float* var)
	{
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				int idx = i + n_node * n;
				xhat[idx] = (prev_out[idx] - mu[i]) / std::sqrt(var[i] + eps);
				output[idx] = gamma[i] * xhat[idx] + beta[i];
			}
		}
	}

	void BatchNorm1d::update_move_mu_move_var()
	{
		for (int i = 0; i < n_node; i++) {
			move_mu[i] = move_mu[i] * momentum + mu[i] * (1 - momentum);
			move_var[i] = move_var[i] * momentum + var[i] * (1 - momentum);
		}
	}

	void BatchNorm1d::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		// calc dxhat
		float* dxhat = new float[out_block_size];
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				int idx = i + n_node * n;
				dxhat[idx] = delta[idx] * gamma[i];
			}
		}

		// calc dx, dgamma, dbeta
		float* sum1 = new float[n_node];
		float* sum2 = new float[n_node];

		set_zero(sum1, n_node);
		set_zero(sum2, n_node);
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				int idx = i + n_node * n;
				sum1[i] += dxhat[idx];
				sum2[i] += dxhat[idx] * xhat[idx];
			}
		}

		float m = (float)batch;
		for (int n = 0; n < batch; n++) {
			for (int i = 0; i < n_node; i++) {
				int idx = i + n_node * n;
				prev_delta[idx] = (m * dxhat[idx]) - sum1[i] - (xhat[idx] * sum2[i]);
				prev_delta[idx] /= m * std::sqrt(var[i] + eps);
				dgamma[i] += xhat[idx] * delta[idx];
				dbeta[i] += delta[idx];
			}
		}

		delete_memory(dxhat);
		delete_memory(sum1);
		delete_memory(sum2);
	}

	void BatchNorm1d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		for (int i = 0; i < n_node; i++) {
			gamma[i] = t1 * gamma[i] - t2 * dgamma[i];
			beta[i] = t1 * beta[i] - t2 * dbeta[i];
			dgamma[i] = 0.0F;
			dbeta[i] = 0.0F;
		}
	}

	vector<int> BatchNorm1d::output_shape() { return { n_node }; }

	int BatchNorm1d::get_out_block_size() { return out_block_size; }
}