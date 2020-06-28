#pragma once
#include "layer.h"

namespace simple_nn
{
	class Dense : public Layer
	{
	private:
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
			const Matrix& b_trained = {}):
			Layer(LayerType::DENSE),
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
			}
			
			dW.setZero();
			db.setZero();
		}

		~Dense() {}

		void set_batch(int batch_size, int n_batch) override
		{
			output.resize(batch_size, vector<Vector>(1, Vector(n_node)));
			delta.resize(batch_size, vector<Vector>(1, Vector(n_node)));
		}

		void forward_propagate(const vector<vector<Vector>>& prev_out, bool isPrediction) override
		{
			//#pragma omp parallel for
			for (int batch = 0; batch < prev_out.size(); batch++)
				output[batch][0] = W * prev_out[batch][0] + b;
		}

		vector<vector<Vector>> backward_propagate(const vector<vector<Vector>>& prev_out, bool isFirst) override
		{
			int batch_size = (int)prev_out.size();

			// calc delta w.r.t weight & bias of this layer
			for (int batch = 0; batch < batch_size; batch++)
			{
				dW += delta[batch][0] * prev_out[batch][0].transpose();
				db += delta[batch][0];
			}

			vector<vector<Vector>> prev_delta(batch_size, vector<Vector>(1));
			if (!isFirst)
			{
				for (int batch = 0; batch < batch_size; batch++)
					prev_delta[batch][0] = W.transpose() * delta[batch][0];
			}
			return prev_delta;
		}

		void update_weight(double l_rate, double lambda, int batch_size) override
			// lambda is the regularization coeff of L2 norm
		{
			W = (1 - (2 * l_rate * lambda) / batch_size) * W - (l_rate / batch_size) * dW;
			b = (1 - (2 * l_rate * lambda) / batch_size) * b - (l_rate / batch_size) * db;

			dW.setZero();
			db.setZero();
		}
	};
}