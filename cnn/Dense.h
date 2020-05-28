#pragma once
#include "common.h"

class Dense
{
public:
	int n_node;
	int n_input;
	Vector sum;
	Vector output;
	Vector delta;
	Matrix W;
	Matrix dW;
	Vector b;
	Vector db;
public:
	Dense(int n_node, int n_input, int init_opt = 0);
	~Dense() {}
	void forward_propagate(const Vector& prev_out);
	Vector backward_propagate(double label, const Vector& prev_out);
	Vector backward_propagate(const Vector& prev_out);
	void update_weight(double l_rate);
	void print_layer();
};

Dense::Dense(int n_input, int n_node, int init_opt) :  n_node(n_node), n_input(n_input)
{
	sum.resize(n_node);
	output.resize(n_node);
	delta.resize(n_node);
	W.resize(n_node, n_input);
	dW.resize(n_node, n_input);
	b.resize(n_node);
	db.resize(n_node);

	if (init_opt == 0)
		init_weight_normal(W, b, n_input, n_node);
	else
		init_weight_uniform(W, b, n_input);

	dW.setZero();
	db.setZero();
}

void Dense::forward_propagate(const Vector& prev_out)
{
	sum = W * prev_out + b;
	output = activate(sum);
}

Vector Dense::backward_propagate(double label, const Vector& prev_out)
// if this is the last layer
{
	delta = as_vector(label, 10) - output;
	delta.element_wise(activate_prime(sum));
	dW -= delta * prev_out.transpose();
	db -= delta;

	return (W.transpose() * delta);
}

Vector Dense::backward_propagate(const Vector& prev_out)
{
	delta.element_wise(activate_prime(sum));
	dW -= delta * prev_out.transpose();
	db -= delta;

	return (W.transpose() * delta);
}

void Dense::update_weight(double l_rate)
{
	W -= l_rate * dW;
	b -= l_rate * db;
	dW.setZero();
	db.setZero();
}

void Dense::print_layer()
{

}