#pragma once
#include "common.h"

namespace simple_nn
{
	class SGD
	{
	private:
		int batch;
		int n_label;
		float _lr;
		float _decay;
		string loss_opt;
	public:
		SGD(float lr, float decay, string loss_opt);
		void set(int batch, int n_label);
		float loss_criterion(const float* prev_out, const int* labels, float* prev_delta);
		float error_criterion(const float* prev_out, const int* labels);
		float lr();
		float decay();
	};

	SGD::SGD(float lr, float decay, string loss_opt) :
		batch(0),
		n_label(0),
		_lr(lr),
		_decay(decay),
		loss_opt(loss_opt)
	{
		if (loss_opt != "MSE" && loss_opt != "cross entropy") {
			throw logic_error("SGD::SGD(float, float, string): Invalid loss option.");
		}
	}

	void SGD::set(int batch, int n_label)
	{
		this->batch = batch;
		this->n_label = n_label;
	}

	float SGD::loss_criterion(const float* prev_out, const int* labels, float* prev_delta)
		// prev_out is the output of the final layer
	{
		float loss = 0.0F;
		std::copy(prev_out, prev_out + batch * n_label, prev_delta);
		for (int i = 0; i < batch; i++) {
			float* begin = prev_delta + n_label * i;
			int label = labels[i];
			*(begin + label) -= 1;
			if (loss_opt == "MSE") {
				loss += 0.5F * std::accumulate(begin, begin + n_label, 0.0F,
					[](const float& sum, const float& elem) { return sum + elem * elem; });
			}
			else {
				const float* out_begin = prev_out + n_label * i;
				loss -= std::log(*(out_begin + label));
			}
		}
		return loss / batch;
	}

	float SGD::error_criterion(const float* output, const int* labels)
		// output is the predicted values
	{
		float error = 0.0F;
		for (int i = 0; i < batch; i++) {
			if (output[i] != labels[i]) {
				error++;
			}
		}
		return error / batch;
	}

	float SGD::lr() { return _lr; }

	float SGD::decay() { return _decay; }
}