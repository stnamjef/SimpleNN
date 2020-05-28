#pragma once
#include "common.h"

class Pool2D
{
public:
	int in_size;
	int filt_size;
	int out_size;
	int n_in;
	int n_out;
	int stride;
	int pool_opt;
	vector<Matrix> output;
	vector<Matrix> delta;
public:
	Pool2D(const vector<int>& input_size, const vector<int>& filter_size, int stride = 1, int pool_opt = 0);
	~Pool2D() {}
	void forward_propagate(const vector<Matrix>& prev_out);
	vector<Matrix> backward_propagate(const vector<Matrix>& prev_out);
};

Pool2D::Pool2D(const vector<int>& input_size, const vector<int>& filter_size, int stride, int pool_opt) :
	in_size(input_size[0]), filt_size(filter_size[0]),
	out_size(calc_outsize(in_size, filt_size, stride, 0)),
	n_in(input_size[2]), n_out(filter_size[2]), stride(stride), pool_opt(pool_opt)
{
	output.resize(n_out, Matrix(out_size, out_size));
	delta.resize(n_out, Matrix(out_size, out_size));
}

void Pool2D::forward_propagate(const vector<Matrix>& prev_out)
{
	for (int i = 0; i < output.size(); i++)
		output[i] = pool2d(prev_out[i], filt_size, stride, pool_opt);
}

vector<Matrix> Pool2D::backward_propagate(const vector<Matrix>& prev_out)
{
	vector<Matrix> prev_delta(prev_out.size());
	for (int i = 0; i < prev_delta.size(); i++)
	{
		if (pool_opt == 0)
			prev_delta[i] = delta_img_max(prev_out[i], delta[i], filt_size, stride);
		else
			prev_delta[i] = delta_img_avg(prev_out[0].rows(), delta[i], filt_size, stride);
	}
	return prev_delta;
}