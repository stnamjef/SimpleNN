#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm2d : public Layer
	{
	private:
		int batch;
		int channels;
		int in_h;
		int in_w;
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
		BatchNorm2d(float eps = 0.00001F, float momentum = 0.9F);
		~BatchNorm2d();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval = false) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		void update_weight(float lr, float decay) override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	private:
		void calc_batch_mu(const float* prev_out, float* mu);
		void calc_batch_var(const float* prev_out, const float* mu, float* vat);
		void normalize_and_shift(const float* prev_out, const float* mu, const float* var);
		void update_move_mu_move_var();
	};

	BatchNorm2d::BatchNorm2d(float eps, float momentum) :
		Layer(BATCHNORM2D),
		batch(0),
		channels(0),
		in_h(0),
		in_w(0),
		out_block_size(0),
		eps(eps),
		momentum(momentum) {}

	BatchNorm2d::~BatchNorm2d()
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

	void BatchNorm2d::set_layer(int batch, const vector<int>& input_shape)
	{
		if (input_shape.size() != 3) {
			throw logic_error("BatchNorm2d::set_layer(int, const vector<int>): Invalid input shape.");
		}

		this->batch = batch;
		channels = input_shape[2];
		in_h = input_shape[0];
		in_w = input_shape[1];
		out_block_size = batch * channels * in_h * in_w;

		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
		allocate_memory(xhat, out_block_size);
		allocate_memory(move_mu, channels);
		allocate_memory(move_var, channels);
		allocate_memory(mu, channels);
		allocate_memory(var, channels);
		allocate_memory(gamma, channels);
		allocate_memory(dgamma, channels);
		allocate_memory(beta, channels);
		allocate_memory(dbeta, channels);

		set_zero(move_mu, channels);
		set_zero(move_var, channels);
		set_one(gamma, channels);
		set_zero(dgamma, channels);
		set_zero(beta, channels);
		set_zero(dbeta, channels);
	}

	void BatchNorm2d::forward_propagate(const float* prev_out, bool isEval)
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

	void BatchNorm2d::calc_batch_mu(const float* prev_out, float* mu)
	{
		set_zero(mu, channels);
		int im_size = in_h * in_w;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int offset = im_size * (c + channels * n);
				const float* begin = prev_out + offset;
				mu[c] += std::accumulate(begin, begin + im_size, 0.0F) / im_size / batch;
			}
		}
	}

	void BatchNorm2d::calc_batch_var(const float* prev_out, const float* mu, float* vat)
	{
		set_zero(var, channels);
		int im_size = in_h * in_w;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int temp = c + channels * n;
				float m = mu[c];
				float v = 0.0F;
				for (int i = 0; i < in_h; i++) {
					for (int j = 0; j < in_w; j++) {
						int idx = j + in_w * (i + in_h * temp);
						float dif = prev_out[idx] - m;
						v += dif * dif;
					}
				}
				var[c] += v / im_size / batch;
			}
		}

		/*for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				float _mu = mu[c];
				int offset = im_size * (c + channels * n);
				const float* begin = prev_out + offset;
				var[c] += std::accumulate(begin, begin + im_size, 0.0F,
					[&](const float& sum, const float& elem) {
					return sum + (elem - _mu) * (elem - _mu);
				});
				var[c] /= denominator;
			}
		}*/
	}

	void BatchNorm2d::normalize_and_shift(const float* prev_out, const float* mu, const float* var)
	{
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int temp = c + channels * n;
				float m = mu[c];
				float s = std::sqrt(var[c] + eps);
				float g = gamma[c];
				float b = beta[c];
				for (int i = 0; i < in_h; i++) {
					for (int j = 0; j < in_w; j++) {
						int idx = j + in_w * (i + in_h * temp);
						xhat[idx] = (prev_out[idx] - m) / s;
						output[idx] = g * xhat[idx] + b;
					}
				}
			}
		}
	}

	void BatchNorm2d::update_move_mu_move_var()
	{
		for (int c = 0; c < channels; c++) {
			move_mu[c] = move_mu[c] * momentum + mu[c] * (1 - momentum);
			move_var[c] = move_var[c] * momentum + var[c] * (1 - momentum);
		}
	}

	void BatchNorm2d::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		// calc dxhat
		float* dxhat = new float[out_block_size];
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int temp = c + channels * n;
				float g = gamma[c];
				for (int i = 0; i < in_h; i++) {
					for (int j = 0; j < in_w; j++) {
						int idx = j + in_w * (i + in_h * temp);
						dxhat[idx] = delta[idx] * g;
					}
				}
			}
		}

		// calc dx, dgamma, dbeta
		float* sum1 = new float[channels];
		float* sum2 = new float[channels];

		set_zero(sum1, channels);
		set_zero(sum2, channels);

		int im_size = in_h * in_w;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int temp = c + channels * n;
				float s1 = 0.0F;
				float s2 = 0.0F;
				for (int i = 0; i < in_h; i++) {
					for (int j = 0; j < in_w; j++) {
						int idx = j + in_w * (i + in_h * temp);
						s1 += dxhat[idx];
						s2 += dxhat[idx] * xhat[idx];
					}
				}
				sum1[c] += s1 / im_size;
				sum2[c] += s2 / im_size;
			}
		}
		
		float m = (float)batch;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				int temp = c + channels * n;
				float s1 = sum1[c];
				float s2 = sum2[c];
				float dg = 0.0F;
				float db = 0.0F;
				float denominator = m * std::sqrt(var[c] + eps);
				for (int i = 0; i < in_h; i++) {
					for (int j = 0; j < in_w; j++) {
						int idx = j + in_w * (i + in_h * temp);
						prev_delta[idx] = (m * dxhat[idx]) - s1 - (xhat[idx] * s2);
						prev_delta[idx] /= denominator;
						dg += (xhat[idx] * delta[idx]);
						db += (delta[idx]);
					}
				}
				dgamma[c] += dg;
				dbeta[c] += db;
			}
		}

		delete_memory(dxhat);
		delete_memory(sum1);
		delete_memory(sum2);
	}

	void BatchNorm2d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		for (int c = 0; c < channels; c++) {
			gamma[c] = t1 * gamma[c] - t2 * dgamma[c];
			beta[c] = t1 * dbeta[c] - t2 * dbeta[c];
			dgamma[c] = 0.0F;
			dbeta[c] = 0.0F;
		}
	}

	vector<int> BatchNorm2d::output_shape() { return { in_h, in_w, channels }; }

	int BatchNorm2d::get_out_block_size() { return out_block_size; }
}