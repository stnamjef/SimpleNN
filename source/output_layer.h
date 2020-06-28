#pragma once
#include "layer.h"

namespace simple_nn
{
	class Output : public Layer
	{
	private:
		int n_node;
		Loss opt;
	public:
		Output(int n_node, Loss opt) :
			Layer(LayerType::OUTPUT),
			n_node(n_node),
			opt(opt) {}

		~Output() {}

		void set_batch(int batch_size, int n_batch) override
		{
			output.resize(batch_size, vector<Vector>(1, Vector(n_node)));
			delta.resize(batch_size, vector<Vector>(1, Vector(n_node)));
		}

		void forward_propagate(const vector<vector<Vector>>& prev_out, bool isPrediction) override
		{
			for (unsigned int batch = 0; batch < prev_out.size(); batch++)
				output[batch][0] = prev_out[batch][0];
		}

		vector<vector<Vector>> backward_propagate(const vector<vector<Vector>>& Y, bool isFirst) override
		{
			for (unsigned int batch = 0; batch < Y[0][0].size(); batch++)
				delta[batch][0] = output[batch][0] - as_vector((int)Y[0][0][batch], 10); // -(Y - Out);
			return delta;
		}

		Loss get_loss_opt() override { return opt; }
	};
}