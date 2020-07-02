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
		Matrix W;
		Matrix dW;
		Vector b;
		Vector db;
	public:
		Dense(int n_input,
			int n_node,
			Init opt = Init::NORMAL,
			const Matrix& W_trained = {},
			const Matrix& b_trained = {});

		~Dense() {}

		void set_batch(int batch_size) override;

		void forward_propagate(const vector<vector<Vector>>& prev_out, bool isPrediction) override;

		vector<vector<Vector>> backward_propagate(const vector<vector<Vector>>& prev_out, bool isFirst) override;

		void update_weight(double l_rate, double lambda) override;
	};

	void init_weight(Matrix& W, Init opt, int n_in, int n_out);

	//------------------------------- function definition -------------------------------

	Dense::Dense(int n_input,
				 int n_node,
				 Init opt,
				 const Matrix& W_trained,
				 const Matrix& b_trained) :
				 Layer(LayerType::DENSE),
				 batch_size(0),
				 n_input(n_input),
				 n_node(n_node)
	{
		W.resize(n_node, n_input);
		dW.resize(n_node, n_input);
		b.resize(n_node);
		db.resize(n_node);

		if (W_trained.size() != 0)
		{
			W = W_trained;
			b = b_trained;
		}
		else
		{
			init_weight(W, opt, n_input, n_node);
			b.setZero();
			dW.setZero();
			db.setZero();
		}
	}

	void init_weight(Matrix& W, Init opt, int n_in, int n_out)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
		default_random_engine e(444);

		if (opt == Init::NORMAL)
		{
			double var = std::sqrt(2 / ((double)n_in + n_out));
			normal_distribution<double> dist(0, var);

			for (int i = 0; i < W.rows(); i++)
				for (int j = 0; j < W.cols(); j++)
					W(i, j) = dist(e);
		}
		else
		{
			double r = 1 / std::sqrt((double)n_in);
			uniform_real_distribution<double> dist(-r, r);

			for (int i = 0; i < W.rows(); i++)
				for (int j = 0; j < W.cols(); j++)
					W(i, j) = dist(e);
		}
	}

	void Dense::set_batch(int batch_size)
	{
		this->batch_size = batch_size;
		output.resize(batch_size, vector<Vector>(1, Vector(n_node)));
		delta.resize(batch_size, vector<Vector>(1, Vector(n_node)));
	}

	void Dense::forward_propagate(const vector<vector<Vector>>& prev_out, bool isPrediction)
	{
		for (int n = 0; n < batch_size; n++)
			output[n][0] = W * prev_out[n][0] + b;
	}

	vector<vector<Vector>> Dense::backward_propagate(const vector<vector<Vector>>& prev_out, bool isFirst)
	{
		// calc delta w.r.t weight & bias of this layer
		for (int n = 0; n < batch_size; n++)
		{
			dW += delta[n][0] * prev_out[n][0].transpose();
			db += delta[n][0];
		}

		vector<vector<Vector>> prev_delta(batch_size, vector<Vector>(1));
		if (!isFirst)
		{
			for (int n = 0; n < batch_size; n++)
				prev_delta[n][0] = W.transpose() * delta[n][0];
		}
		return prev_delta;
	}

	void Dense::update_weight(double l_rate, double lambda)
	{
		W = (1 - (2 * l_rate * lambda) / batch_size) * W - (l_rate / batch_size) * dW;
		b = (1 - (2 * l_rate * lambda) / batch_size) * b - (l_rate / batch_size) * db;

		dW.setZero();
		db.setZero();
	}
}