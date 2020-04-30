#pragma once
#include "Layer.h"

class Conv2D : public Layer
{
public:
	int padding;
	int stride;
	vec<int> dim_input;
	vec<int> dim_filter;
	vec<int> dim_fmap;

	Conv2D(const vec<int>& dim_filter, int padding, int stride, const vec<int>& dim_input);
	~Conv2D();
	void print_layer() override;
};

int out_size(int input_size, int filter_size, int padding, int stride)
{
	return (int)floor((input_size + 2 * padding - filter_size) / stride) + 1;
}

Conv2D::Conv2D(const vec<int>& dim_filter, int padding, int stride, const vec<int>& dim_input) :
	Layer("Conv2D", (int)pow(out_size(dim_input[0], dim_filter[0], padding, stride), 2) * dim_filter[2]),
	padding(padding),
	stride(stride),
	dim_input(dim_input),
	dim_filter(dim_filter)
{
	int out = out_size(dim_input[0], dim_filter[0], padding, stride);
	dim_fmap = { out, out, dim_filter[2] };
}

Conv2D::~Conv2D() {}

void Conv2D::print_layer()
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