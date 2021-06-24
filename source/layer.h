#pragma once
#include "common.h"

namespace simple_nn
{
	enum class LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
		ACTIVATION,
		BATCHNORM1D,
		BATCHNORM2D,
		FLATTEN
	};

	class Layer
	{
	public:
		LayerType type;
		bool is_first;
		bool is_last;
		MatXf output;
		MatXf delta;
	public:
		Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
		virtual void set_layer(const vector<int>& input_shape) = 0;
		virtual void forward(const MatXf& prev_out, bool is_training = true) = 0;
		virtual void backward(const MatXf& prev_out, MatXf& prev_delta) = 0;
		virtual void update_weight(float lr, float decay) { return; }
		virtual void zero_grad() { return; }
		virtual vector<int> output_shape() = 0;
	};
}