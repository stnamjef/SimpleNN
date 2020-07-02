#pragma once
#include "layer.h"

namespace simple_nn
{
	class Output : public Layer
	{
	private:
		int batch_size;
		int n_node;
		Loss opt;
	public:
		Output(int n_node, Loss opt) :
			Layer(LayerType::OUTPUT),
			batch_size(0),
			n_node(n_node),
			opt(opt) {}

		~Output() {}

		void set_batch(int batch_size) override
		{
			this->batch_size = batch_size;
			output.resize(batch_size, vector<Vector>(1, Vector(n_node)));
			delta.resize(batch_size, vector<Vector>(1, Vector(n_node)));
		}

		void forward_propagate(const vector<vector<Vector>>& prev_out, bool isPrediction) override
		{
			for (int n = 0; n < batch_size; n++)
				output[n][0] = prev_out[n][0];
		}

		vector<vector<Vector>> backward_propagate(const vector<vector<Vector>>& Y, bool isFirst) override
		{
			for (int n = 0; n < Y[0][0].size(); n++)
				delta[n][0] = output[n][0] - as_vector((int)Y[0][0][n], 10); // -(Y - Out);
			return delta;
		}

		Loss get_loss_opt() override { return opt; }
	};
}