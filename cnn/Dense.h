#pragma once
#include "Layer.h"

class Dense : public Layer
{
public:
	Dense(int n_node) : Layer("Dense", n_node) {}
	~Dense() {}
	void print_layer() override;
};

void Dense::print_layer()
{
	cout << "[ " << Layer::type << " layer ] :" << endl;
	cout << "Number of node : " << n_node << endl;
}