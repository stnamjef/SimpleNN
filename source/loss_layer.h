#pragma once
#include "common.h"

namespace simple_nn
{
	class Loss
	{
	public:
		int batch;
		int n_label;
	public:
		Loss() : batch(0), n_label(0) {}

		void set_layer(const vector<int>& input_shape)
		{
			assert(input_shape.size() == 2 && "Loss::set(const vector<int>): Output layer must be linear.");
			batch = input_shape[0];
			n_label = input_shape[1];
		}

		virtual float calc_loss(const MatXf& prev_out, const VecXi& labels, MatXf& prev_delta) = 0;
	};

	class MSELoss : public Loss
	{
	public:
		MSELoss() : Loss() {}

		float calc_loss(const MatXf& prev_out, const VecXi& labels, MatXf& prev_delta) override
		{
			float loss_batch = 0.f, loss = 0.f;
			prev_delta = prev_out;
			for (int n = 0; n < batch; n++) {
				prev_delta(n, labels[n]) -= 1.f;
				for (int i = 0; i < n_label; i++) {
					loss = prev_delta(n, i);
					loss_batch += 0.5f * loss * loss;
				}
				// loss_batch += 0.5f * prev_delta.row(n).pow(2).sum();
			}
			return loss_batch / batch;
		}
	};

	class CrossEntropyLoss : public Loss
	{
	public:
		CrossEntropyLoss() : Loss() {}

		float calc_loss(const MatXf& prev_out, const VecXi& labels, MatXf& prev_delta)
		{
			float loss_batch = 0.f;
			prev_delta = prev_out;
			for (int n = 0; n < batch; n++) {
				int answer_idx = labels[n];
				prev_delta(n, answer_idx) -= 1.f;
				loss_batch -= std::log(prev_out(n, answer_idx));
			}
			return loss_batch / batch;
		}
	};
}