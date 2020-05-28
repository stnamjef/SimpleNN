#pragma once
#include "common.h"

class Conv2D
{
public:
	int in_size;
	int filt_size;
	int out_size;
	int n_in;
	int n_out;
	int pad;
	vector<Matrix> sum;
	vector<Matrix> output;
	vector<Matrix> delta;
	vector<Matrix> Ws;
	vector<Matrix> dWs;
	Vector b;
	Vector db;
	vector<vector<int>> indices;
public:
	Conv2D(const vector<int>& input_size, const vector<int>& filter_size, int pad = 0, int init_opt = 0,
		const vector<vector<int>>& indices = {});
	~Conv2D() {}
	void forward_propagate(const Matrix& x);
	void forward_propagate(const vector<Matrix>& prev_out);
	vector<Matrix> backward_propagate(const vector<Matrix>& prev_out);
	void backward_propagate(const Matrix& prev_out);
	void update_weight(double l_rate);
};

Conv2D::Conv2D(const vector<int>& input_size, const vector<int>& filter_size, int pad, int init_opt,
	const vector<vector<int>>& indices) :
	in_size(input_size[0]), filt_size(filter_size[0]),
	out_size(calc_outsize(in_size, filt_size, 1, pad)),
	n_in(input_size[2]), n_out(filter_size[2]), pad(pad),
	indices(indices)
{
	sum.resize(n_out, Matrix(out_size, out_size));
	output.resize(n_out, Matrix(out_size, out_size));
	delta.resize(n_out, Matrix(out_size, out_size));
	Ws.resize(n_out, Matrix(filt_size, filt_size));
	dWs.resize(n_out, Matrix(filt_size, filt_size));
	b.resize(n_out);
	db.resize(n_out);

	if (init_opt == 0)
		init_filter_normal(Ws, b, filt_size * filt_size, out_size * out_size);
	else
		init_filter_uniform(Ws, b, in_size * in_size);
	for (int i = 0; i < n_out; i++)
		dWs[i].setZero();
	db.setZero();
}

void Conv2D::forward_propagate(const Matrix& x)
// if this is the first layer
{
	for (int i = 0; i < output.size(); i++)
	{
		sum[i] = conv2d(x, Ws[i], pad);
		sum[i] += b[i];
		output[i] = activate(sum[i]);
	}
}

void Conv2D::forward_propagate(const vector<Matrix>& prev_out)
{
	if (indices.size() == 0) // do not suffle input featuremap
	{
		for (int i = 0; i < output.size(); i++)
		{
			sum[i] = conv2d(prev_out[i], Ws[i], pad);
			sum[i] += b[i];
			output[i] = activate(sum[i]);
		}
	}
	else
	{
		for (int j = 0; j < indices[0].size(); j++)
		{
			sum[j].setZero();
			for (int i = 0; i < indices.size(); i++)
			{
				if (indices[i][j] != 0)
					sum[j] += conv2d(prev_out[i], Ws[j], pad);
			}
			sum[j] += b[j];
			output[j] = activate(sum[j]);
		}
	}
}

vector<Matrix> Conv2D::backward_propagate(const vector<Matrix>& prev_out)
{
	for (int i = 0; i < delta.size(); i++)
		delta[i].element_wise(activate_prime(sum[i]));

	vector<Matrix> prev_delta(prev_out.size(), Matrix(prev_out[0].rows(), prev_out[0].cols()));

	if (indices.size() == 0)
	{
		for (int i = 0; i < Ws.size(); i++)
		{
			dWs[i] -= conv2d(prev_out[i], delta[i], pad);
			db[i] -= delta[i].sum();
		}

		for (int i = 0; i < prev_delta.size(); i++)
			prev_delta[i] = conv2d(delta[i], rotate_180(Ws[i]), filt_size - 1);
	}
	else
	{	
		for (int j = 0; j < indices[0].size(); j++)
		{
			for (int i = 0; i < indices.size(); i++)
			{
				if (indices[i][j] != 0)
					dWs[j] -= conv2d(prev_out[i], delta[j], pad);
			}
			db[j] -= delta[j].sum();
		}

		for (int i = 0; i < indices.size(); i++)
		{
			prev_delta[i].setZero();
			for (int j = 0; j < indices[0].size(); j++)
				if (indices[i][j] != 0)
					prev_delta[i] += conv2d(delta[j], rotate_180(Ws[j]), filt_size - 1);
		}
	}

	return prev_delta;
}

void Conv2D::backward_propagate(const Matrix& prev_out)
// if this is the first layer
{
	for (int i = 0; i < delta.size(); i++)
		delta[i].element_wise(activate_prime(sum[i]));

	for (int i = 0; i < Ws.size(); i++)
	{
		dWs[i] -= conv2d(prev_out, delta[i], pad);
		db[i] -= delta[i].sum();
	}
}

void Conv2D::update_weight(double l_rate)
{
	for (int i = 0; i < Ws.size(); i++)
	{
		Ws[i] -= l_rate * dWs[i];
		dWs[i].setZero();
	}
	b -= l_rate * db;
	db.setZero();
}