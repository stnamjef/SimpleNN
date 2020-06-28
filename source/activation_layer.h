#pragma once
#include "layer.h"

namespace simple_nn
{
	class Activation : public Layer
	{
	private:
		int channels;
		Activate opt;
	public:
		Activation(int channels, Activate opt) :
			Layer(LayerType::ACTIVATION),
			opt(opt),
			channels(channels) {}

		~Activation() {}

		void set_batch(int batch_size, int n_batch) override
		{
			output.resize(batch_size, vector<Matrix>(channels));
			delta.resize(batch_size, vector<Matrix>(channels));
		}

		void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction) override
		{
			for (int batch = 0; batch < prev_out.size(); batch++)
				for (unsigned int ch = 0; ch < prev_out[batch].size(); ch++)
					output[batch][ch] = activate(prev_out[batch][ch], opt);
		}

		vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst) override
		{
			if (opt != Activate::SOFTMAX)
			{
				for (unsigned int batch = 0; batch < prev_out.size(); batch++)
					for (int ch = 0; ch < channels; ch++)
						delta[batch][ch] *= activate_prime(prev_out[batch][ch], opt);
			}
			return delta;
		}
	};
}