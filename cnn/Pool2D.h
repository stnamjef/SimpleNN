#pragma once
#include "Layer.h"

class Pooling2D : public Layer
{
public:
	int stride;
	vec<int> dim_input;
	vec<int> dim_filter;
	vec<int> dim_fmap;

	Pooling2D(const vec<int>& dim_filter, int stride, const vec<int>& dim_input);
	~Pooling2D();
	void print_layer() override;
};

int out_size(int input_size, int filter_size, int stride)
{
	return (int)floor((input_size - filter_size) / stride) + 1;
}

Pooling2D::Pooling2D(const vec<int>& dim_filter, int stride, const vec<int>& dim_input) :
	Layer("Pooling2D", (int)pow(out_size(dim_input[0], dim_filter[0], stride), 2) * dim_filter[2]),
	stride(stride),
	dim_input(dim_input),
	dim_filter(dim_filter)
{
	int out = out_size(dim_input[0], dim_filter[0], stride);
	dim_fmap = { out, out, dim_filter[2] };
}

Pooling2D::~Pooling2D() {}

void Pooling2D::print_layer()
{
	cout << "[ " << Layer::type << " layer ] :" << endl;
	cout << "Number of node : " << n_node << endl;
	cout << "Dimension of input : ";
	for (int i = 0; i < 3; i++)
		cout << dim_input[i] << " ";
	cout << endl;
	cout << "Dimension of filter : ";
	for (int i = 0; i < 3; i++)
		cout << dim_filter[i] << " ";
	cout << endl;
	cout << "Dimension of feature map : ";
	for (int i = 0; i < 3; i++)
		cout << dim_fmap[i] << " ";
	cout << endl;
}