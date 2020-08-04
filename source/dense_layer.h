#pragma once
#include "layer.h"

namespace simple_nn
{
	class Dense : public Layer
	{
	private:
		int batch_size;
		int n_input;
		int n_node;
		string init_opt;
		Matrix W;
		Matrix dW;
		Vector b;
		Vector db;
	public:
		Dense(int n_node, string init_opt, int n_input = 0);
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void reset_batch(int batch_size) override;
		void forward_propagate(const Tensor& prev_out, bool isPrediction) override;
		void backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst) override;
		void update_weight(double l_rate, double lambda) override;
		vector<int> input_shape() override;
		vector<int> output_shape() override;
	private:
		void init_weight(int n_in, int n_out);
	};

	Dense::Dense(int n_node, string init_opt, int n_input) :
		Layer("dense"),
		batch_size(0),
		n_input(n_input),
		n_node(n_node),
		init_opt(init_opt) {}

	void Dense::set_layer(int batch_size, const vector<int>& input_shape)
	{
		this->batch_size = batch_size;
		if (input_shape.size() == 1) {
			n_input = input_shape[0];
		}
		else {
			cout << "Dense::set_layer(int ...): Add a flatten layer." << endl;
			exit(100);
		}
		output.resize(batch_size, 1, n_node);
		delta.resize(batch_size, 1, n_node);
		W.resize(n_node, n_input);
		dW.resize(n_node, n_input);
		b.resize(n_node);
		db.resize(n_node);
		init_weight(n_input, n_node);
		b.setZero();
		dW.setZero();
		db.setZero();
	}

	void Dense::reset_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, 1, n_node);
	}

	void Dense::init_weight(int n_in, int n_out)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (init_opt == "norml") {
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);
			for (int i = 0; i < n_node; i++) {
				for (int j = 0; j < n_input; j++) {
					W(i, j) = dist(e);
				}
			}
		}
		else {
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);
			for (int i = 0; i < n_node; i++) {
				for (int j = 0; j < n_input; j++) {
					W(i, j) = dist(e);
				}
			}
		}
	}

	void Dense::forward_propagate(const Tensor& prev_out, bool isPrediction)
	{
		// fast operation
		for (int n = 0; n < batch_size; n++) {
			for (int i = 0; i < n_node; i++) {
				output[n][0](i) = b(i);
				for (int k = 0; k < n_input; k++) {
					output[n][0](i) += W(i, k) * prev_out[n][0](k);
				}
			}
		}

		/*for (int n = 0; n < batch_size; n++) {
			output[n][0] = W * prev_out[n][0] + b;
		}*/
	}

	void Dense::backward_propagate(const Tensor& prev_out, Tensor& prev_delta, bool isFirst)
	{
		// fast operation
		for (int n = 0; n < batch_size; n++) {
			for (int i = 0; i < n_node; i++) {
				for (int j = 0; j < n_input; j++) {
					dW(i, j) += delta[n][0](i) * prev_out[n][0](j);
				}
				db(i) += delta[n][0](i);
			}
		}
		if (!isFirst) {
			for (int n = 0; n < batch_size; n++) {
				prev_delta[n][0].setZero();
				for (int i = 0; i < n_node; i++) {
					double temp = delta[n][0](i);
					for (int j = 0; j < n_input; j++) {
						prev_delta[n][0](j) += W(i, j) * temp;
					}
				}
			}
		}

		/*for (int n = 0; n < batch_size; n++) {
			dW += delta[n][0] * prev_out[n][0].transpose();
			db += delta[n][0];
		}
		if (!isFirst) {
			for (int n = 0; n < batch_size; n++) {
				prev_delta[n][0] = W.transpose() * delta[n][0];
			}
		}*/
	}

	void Dense::update_weight(double l_rate, double lambda)
	{
		W = (1 - (2 * l_rate * lambda) / batch_size) * W - (l_rate / batch_size) * dW;
		b = (1 - (2 * l_rate * lambda) / batch_size) * b - (l_rate / batch_size) * db;
		dW.setZero();
		db.setZero();
	}

	vector<int> Dense::input_shape()
	{
		if (n_input == 0) {
			cout << "Dense::input_shape(): Input shape is empty." << endl;
			exit(100);
		}
		return { n_input };
	}

	vector<int> Dense::output_shape() { return { n_node }; }
}