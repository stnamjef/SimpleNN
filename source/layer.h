#pragma once
#include "common.h"

namespace simple_nn
{
	class Layer
	{
	public:
		vector<vector<Matrix>> output;
		vector<vector<Matrix>> delta;
	public:
		LayerType type;
		Layer(LayerType type) : type(type) {}
		~Layer() {}
		virtual void set_batch(int batch_size, int n_batch = 0) = 0;
		virtual void forward_propagate(const vector<vector<Matrix>>& prev_out, bool isPrediction = false) = 0;
		virtual vector<vector<Matrix>> backward_propagate(const vector<vector<Matrix>>& prev_out, bool isFirst = false) = 0;
		virtual void update_weight(double l_rate, double lambda, int batch_size) { return; }
		virtual Loss get_loss_opt() { return Loss::MSE; }
	};
}