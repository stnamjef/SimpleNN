#pragma once
#include <iostream>
#include <string>
#include "Vector.h"
using namespace std;

class Layer
{
public:
	string type;
	int n_node;
	vec<double> sum;
	vec<double> output;
	vec<double> delta;

	Layer(string type, int n_node);
	~Layer();

	virtual void print_layer() = 0;
};

Layer::Layer(string type, int n_node) : type(type), n_node(n_node)
{
	sum.resize(n_node);
	output.resize(n_node);
	delta.resize(n_node);
}

Layer::~Layer() {}