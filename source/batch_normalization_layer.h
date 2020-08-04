#pragma once
#include "layer.h"

namespace simple_nn
{
	class BatchNorm : public Layer
	{
	private:
		int batch_size;
		int channels;
		int in_h;
		int in_w;
		double epsilon;
		double momentum;
		bool is2d;
		Tensor xhat;
		Vector move_mu;
		Vector move_var;
		Vector mu;
		Vector var;
		Vector gamma;
		Vector dgamma;
		Vector beta;
		Vector dbeta;
	public:
		BatchNorm(double epsilon = 0.00001, double momentum = 0.9);
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size) override;
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& empty, Tensor& prev_delta, bool isFirst) override;
		void update_weight(double l_rate, double lambda) override;
		vector<int> output_shape() override;
	private:
		void calc_batch_mu(const Tensor& prev_out);
		void calc_batch_var(const Tensor& prev_out);
		void normalize_and_shift(const Tensor& prev_out, const Vector& mu, const Vector& var);
		/*void calc_dxhat(const Tensor& delta, Tensor& dxhat);
		void calc_dx_dgamma_dbeta(const Tensor& dxhat, Tensor& prev_delta);*/
	};

	BatchNorm::BatchNorm(double epsilon, double momentum) :
		Layer("batchnorm"),
		batch_size(0),
		channels(0),
		in_h(0),
		in_w(0),
		epsilon(epsilon),
		momentum(momentum),
		is2d(false) {}

	void BatchNorm::set_layer(int batch_size, const vector<int>& input_shape)
	{
		int size = 0;
		this->batch_size = batch_size;
		if (input_shape.size() == 3) {
			in_h = input_shape[0];
			in_w = input_shape[1];
			channels = input_shape[2];
			is2d = true;
			size = channels;
		}
		else {
			in_h = input_shape[0];
			in_w = 1;
			channels = 1;
			is2d = false;
			size = in_h;
		}
		// memory allocation
		output.resize(batch_size, channels, in_h, in_w);
		delta.resize(batch_size, channels, in_h, in_w);
		xhat.resize(batch_size, channels, in_h, in_w);
		move_mu.resize(size);
		move_var.resize(size);
		mu.resize(size);
		var.resize(size);
		gamma.resize(size);
		dgamma.resize(size);
		beta.resize(size);
		dbeta.resize(size);
		// initialization
		move_mu.setZero();
		move_var.setZero();
		std::for_each(gamma.begin(), gamma.end(), [](double& elem) { elem = 1; });
		dgamma.setZero();
		beta.setZero();
		dbeta.setZero();
	}

	void BatchNorm::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, channels, in_h, in_w);
		xhat.resize(batch_size, channels, in_h, in_w);
	}

	void BatchNorm::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		if (!isPrediction) {
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, mu, var);
			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
		}
		else {
			normalize_and_shift(prev_out, move_mu, move_var);
		}
	}

	void BatchNorm::calc_batch_mu(const Tensor& prev_out)
	{
		mu.setZero();
		if (is2d) {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < channels; c++) {
					mu(c) += prev_out[n][c].mean() / (double)batch_size;
				}
			}
		}
		else {
			for (int n = 0; n < batch_size; n++) {
				mu += prev_out[n][0];
			}
			mu /= (double)batch_size;
		}
	}

	void BatchNorm::calc_batch_var(const Tensor& prev_out)
	{
		var.setZero();
		if (is2d) {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < channels; c++) {
					var(c) += prev_out[n][c].var(mu(c)) / (double)batch_size;
				}
			}
		}
		else {
			for (int n = 0; n < batch_size; n++) {
				var += (prev_out[n][0] - mu).pow(2);
			}
			var /= (double)batch_size;
		}
	}

	void BatchNorm::normalize_and_shift(const Tensor& prev_out, const Vector& mu, const Vector& var)
	{
		if (is2d) {
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < channels; c++) {
					xhat[n][c] = (prev_out[n][c] - mu(c)) / std::sqrt(var(c) + epsilon);
					output[n][c] = gamma(c) * xhat[n][c] + beta(c);
				}
			}
		}
		else {
			for (int n = 0; n < batch_size; n++) {
				xhat[n][0] = (prev_out[n][0] - mu) / (var + epsilon).sqrt();
				output[n][0] = gamma.elem_wise_mult(xhat[n][0]) + beta;
			}
		}
	}

	void BatchNorm::backward_propagate(const Tensor& empty, Tensor& dx, bool isFirst)
	{
		double m = (double)batch_size;
		if (is2d) {
			// calc dxhat
			Tensor dxhat(batch_size, channels, in_h, in_w);
			for (int n = 0; n < batch_size; n++) {
				for (int c = 0; c < channels; c++) {
					dxhat[n][c] = delta[n][c] * gamma(c);
				}
			}
			
			// calc dx, dgamma, dbeta
			Vector sum1(channels), sum2(channels);
			for (int c = 0; c < channels; c++) {
				sum1(c) = 0;
				sum2(c) = 0;
				for (int n = 0; n < batch_size; n++) {
					sum1(c) += dxhat[n][c].mean();
					sum2(c) += (dxhat[n][c].elem_wise_mult(xhat[n][c])).mean();
				}

				double denominator = m * std::sqrt(var(c) + epsilon);
				for (int n = 0; n < batch_size; n++) {
					dx[n][c] = (m * dxhat[n][c]) - sum1(c) - (xhat[n][c] * sum2(c));
					dx[n][c] /= denominator;
					dgamma(c) += (xhat[n][c].elem_wise_mult(delta[n][c])).sum();
					dbeta(c) += delta[n][c].sum();
				}
			}
		}
		else {
			// calc dxhat
			Tensor dxhat(batch_size, channels, in_h, 1);
			for (int n = 0; n < batch_size; n++) {
				dxhat[n][0] = delta[n][0].elem_wise_mult(gamma);
			}

			// calc dx, dgamma, dbeta
			Vector sum1(in_h), sum2(in_h);
			sum1.setZero();
			sum2.setZero();
			for (int n = 0; n < batch_size; n++) {
				sum1 += dxhat[n][0];
				sum2 += dxhat[n][0].elem_wise_mult(xhat[n][0]);
			}

			Vector denominator = m * (var + epsilon).sqrt();
			for (int n = 0; n < batch_size; n++) {
				dx[n][0] = (m * dxhat[n][0]) - sum1 - (xhat[n][0].elem_wise_mult(sum2));
				dx[n][0] /= denominator;
				dgamma += xhat[n][0].elem_wise_mult(delta[n][0]);
				dbeta += delta[n][0];
			}
		}
	}

	void BatchNorm::update_weight(double l_rate, double lambda)
	{
		gamma += -(l_rate / batch_size) * dgamma;
		beta += -(l_rate / batch_size) * dbeta;
		dgamma.setZero();
		dbeta.setZero();
	}

	vector<int> BatchNorm::output_shape()
	{
		if (is2d) {
			return { in_h, in_w, channels };
		}
		else {
			return { in_h };
		}
	}
}