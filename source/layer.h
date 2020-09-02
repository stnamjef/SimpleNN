#pragma once
#include "gemm.h"
#include "common.h"
using namespace std;

namespace simple_nn
{
	enum LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
		ACTIVATION,
		FLATTEN,
		BATCHNORM1D,
		BATCHNORM2D,
	};

	class Layer
	{
	public:
		LayerType type;
		float* output;
		float* delta;
	public:
		Layer(LayerType type) : type(type), output(nullptr), delta(nullptr) {}
		virtual void set_layer(int batch, const vector<int>& input_shape) = 0;
		virtual void forward_propagate(const float* prev_out, bool isEval = false) = 0;
		virtual void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) = 0;
		virtual void update_weight(float lr, float decay) { return; }
		virtual vector<int> input_shape() { return {}; }
		virtual vector<int> output_shape() = 0;
		virtual int get_out_block_size() = 0;
	};
}