#pragma once
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include "Vector.h"
using namespace std;

struct Neuron
{
	double sum;
	double output;
	double delta;
	Neuron() : sum(0), output(0), delta(0) {}
};

class DNN
{
private:
	Vector<Vector<Neuron>> network;
	Vector<Vector<Vector<double>>> weights;
public:
	DNN(const Vector<int>& layer_sizes);
	void fit(const Vector<Vector<double>>& X, const Vector<int>& Y, double l_rate = 0.001, int n_epoch = 100, int batch = 30);
	Vector<int> predict(const Vector<Vector<double>>& X);
	void print_network();
};

void form_network(Vector<Vector<Neuron>>& network, Vector<Vector<Vector<double>>>& weights, const Vector<int>& layer_sizes);

void init_weights_normal(Vector<Vector<Vector<double>>>& weights);

void init_weights_lecun(Vector<Vector<Vector<double>>>& weights);

void init_weights_xavier(Vector<Vector<Vector<double>>>& weights);

double train_network(Vector<Vector<Neuron>>& network, Vector<Vector<Vector<double>>>& weights,
	const Vector<Vector<double>>& X, const Vector<int>& Y, double l_rate, int batch);

void init_delta_weights(Vector<Vector<Vector<double>>>& del_weights, const Vector<Vector<Vector<double>>>& weights);

void forward_propagate(Vector<Vector<Neuron>>& network, const Vector<Vector<Vector<double>>>& weights,
	Vector<double> input, Vector<double>& output);

double dot(const Vector<double>& u, const Vector<double>& v);

double sigmoid(double x);

void backward_propagate(Vector<Vector<Neuron>>& N, Vector<Vector<Vector<double>>>& Ws,
	Vector<Vector<Vector<double>>>& del_Ws, const Vector<double>& input, double expected);

double sigmoid_prime(double x);

void update_weights(Vector<Vector<Vector<double>>>& Ws, const Vector<Vector<Vector<double>>>& del_Ws, double l_rate);

double calc_error(double expected, const Vector<double>& predicted);

DNN::DNN(const Vector<int>& layer_sizes) { form_network(network, weights, layer_sizes); }

void form_network(Vector<Vector<Neuron>>& network, Vector<Vector<Vector<double>>>& weights, const Vector<int>& layer_sizes)
{
	network.resize(layer_sizes.size() - 1);
	weights.resize(layer_sizes.size() - 1);
	for (int l = 1; l < layer_sizes.size(); l++)
	{
		int n_neuron = layer_sizes[l];
		int n_weight = layer_sizes[l - 1] + 1;
		network[l - 1] = Vector<Neuron>(n_neuron);
		weights[l - 1] = Vector<Vector<double>>(n_neuron, Vector<double>(n_weight));
	}
}

void DNN::fit(const Vector<Vector<double>>& X, const Vector<int>& Y, double l_rate, int n_epoch, int batch)
{
	init_weights_xavier(weights);
	for (int epoch = 1; epoch <= n_epoch; epoch++)
	{
		double error = train_network(network, weights, X, Y, l_rate, batch);
		cout << "Error rate(epoch" << epoch << ") : " << error / (double)X.size() * 100 << "%" << endl;
	}
}

void init_weights_normal(Vector<Vector<Vector<double>>>& weights)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	normal_distribution<double> dist(0, 1);

	for (int l = 0; l < weights.size(); l++)
		for (int j = 0; j < weights[l].size(); j++)
			for (int i = 0; i < weights[l][j].size(); i++)
				weights[l][j][i] = dist(e);
}

void init_weights_lecun(Vector<Vector<Vector<double>>>& weights)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	for (int l = 0; l < weights.size(); l++)
		for (int j = 0; j < weights[l].size(); j++)
		{
			double var = sqrt((double)weights[l][j].size());
			normal_distribution<double> dist(0, var);
			for (int i = 0; i < weights[l][j].size(); i++)
				weights[l][j][i] = dist(e);
		}
}

void init_weights_xavier(Vector<Vector<Vector<double>>>& weights)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	for (int l = 0; l < weights.size(); l++)
	{
		int n_out = weights[l].size();
		for (int j = 0; j < weights[l].size(); j++)
		{
			int n_in = weights[l][j].size();
			double var = sqrt(2 / (double)(n_in + n_out));
			normal_distribution<double> dist(0, var);
			for (int i = 0; i < weights[l][j].size(); i++)
				weights[l][j][i] = dist(e);
		}
	}
}

double train_network(Vector<Vector<Neuron>>& network, Vector<Vector<Vector<double>>>& weights,
	const Vector<Vector<double>>& X, const Vector<int>& Y, double l_rate, int batch)
{
	Vector<Vector<Vector<double>>> del_weights;
	init_delta_weights(del_weights, weights);

	

	double sum_error = 0;
	for (int i = 0; i < X.size(); i++)
	{
		Vector<double> output;
		forward_propagate(network, weights, X[i], output);
		backward_propagate(network, weights, del_weights, X[i], Y[i]);

		if ((i + 1) % batch == 0)
		{
			update_weights(weights, del_weights, l_rate);
			init_delta_weights(del_weights, weights);
		}

		sum_error += calc_error(Y[i], output);
	}
	return sum_error;
}

void init_delta_weights(Vector<Vector<Vector<double>>>& del_weights, const Vector<Vector<Vector<double>>>& weights)
{
	del_weights.resize(weights.size());
	for (int l = 0; l < weights.size(); l++)
		for (int j = 0; j < weights[l].size(); j++)
			del_weights[l] = Vector<Vector<double>>(weights[l].size(), Vector<double>(weights[l][j].size(), 0));
}

void forward_propagate(Vector<Vector<Neuron>>& network, const Vector<Vector<Vector<double>>>& weights,
	Vector<double> input, Vector<double>& output)
{
	for (int l = 0; l < network.size(); l++)
	{
		Vector<double> temp(network[l].size());
		for (int j = 0; j < network[l].size(); j++)
		{
			Neuron& N = network[l][j];
			N.sum = dot(input, weights[l][j]);
			N.output = sigmoid(N.sum);
			temp[j] = N.output;
		}
		input = temp;
	}
	output = input;
}

double dot(const Vector<double>& u, const Vector<double>& v)
// v vector는 항상 u vector보다 하나의 원소를 더 갖는다.
// u vector의 첫 번째 원소는 항상 1이다(bias term).
{
	double sum = v[0];
	for (int i = 0; i < u.size(); i++)
		sum += u[i] * v[i + 1];
	return sum;
}

double sigmoid(double x) { return 2 / (1 + exp(-x)) - 1; }

void backward_propagate(Vector<Vector<Neuron>>& N, Vector<Vector<Vector<double>>>& Ws,
	Vector<Vector<Vector<double>>>& del_Ws, const Vector<double>& input, double expected)
{
	int last = N.size() - 1;

	for (int j = 0; j < N[last].size(); j++)
	{
		double E = (j == expected) ? 1 : 0;
		N[last][j].delta = sigmoid_prime(N[last][j].sum) * (E - N[last][j].output);
	}

	for (int j = 0; j < N[last].size(); j++)
	{
		del_Ws[last][j][0] -= N[last][j].delta;
		for (int i = 0; i < N[last - 1].size(); i++)
			del_Ws[last][j][i + 1] -= N[last][j].delta * N[last - 1][i].output;
	}

	for (int l = last - 1; l >= 0; l--)
	{
		for (int j = 0; j < N[l].size(); j++)
		{
			N[l][j].delta = 0;
			for (int p = 0; p < N[l + 1].size(); p++)
				N[l][j].delta += N[l + 1][p].delta * Ws[l + 1][p][j + 1];
			N[l][j].delta *= sigmoid_prime(N[l][j].sum);
		}

		int n_weight = (l == 0) ? input.size() : N[l - 1].size();

		for (int j = 0; j < N[l].size(); j++)
		{
			del_Ws[l][j][0] -= N[l][j].delta;
			for (int i = 0; i < n_weight; i++)
			{
				double activated = (l == 0) ? input[i] : N[l - 1][i].output;
				del_Ws[l][j][i + 1] -= N[l][j].delta * activated;
			}
		}
	}
}

double sigmoid_prime(double x) { return 0.5 * (1 + sigmoid(x)) * (1 - sigmoid(x)); }

void update_weights(Vector<Vector<Vector<double>>>& Ws, const Vector<Vector<Vector<double>>>& del_Ws, double l_rate)
{
	for (int l = Ws.size() - 1; l >= 0; l--)
		for (int j = 0; j < Ws[l].size(); j++)
			for (int i = 0; i < Ws[l][j].size(); i++)
				Ws[l][j][i] -= l_rate * del_Ws[l][j][i];
}

double calc_error(double expected, const Vector<double>& predicted)
{
	double sum_error = 0;
	for (int i = 0; i < predicted.size(); i++)
	{
		double E = (i == expected) ? 1 : 0;
		sum_error += pow((E - predicted[i]), 2);
	}
	return sum_error * 0.5;
}

Vector<int> DNN::predict(const Vector<Vector<double>>& X)
{
	Vector<int> predicts(X.size());
	for (int i = 0; i < X.size(); i++)
	{
		Vector<double> output;
		forward_propagate(network, weights, X[i], output);
		int label = (int)distance(output.begin(), max_element(output.begin(), output.end()));
		predicts[i] = label;
	}
	return predicts;
}

void DNN::print_network()
{
	for (int l = 0; l < network.size(); l++)
	{
		if (l != network.size() - 1)
			cout << "[ Hidden layer" << l << " ] : " << endl;
		else
			cout << " [ Output layer" << " ] : " << endl;
		for (int j = 0; j < network[l].size(); j++)
		{
			cout << "	[ neuron " << j + 1 << " ] :" << endl;
			cout << "	  - sum : " << network[l][j].sum << endl;
			cout << "	  - activated : " << network[l][j].output << endl;
			cout << "	  - delta : " << network[l][j].delta << endl;
			cout << "	  - weights : ";
			for (int i = 0; i < weights[l][j].size(); i++)
				cout << weights[l][j][i] << ' ';
			cout << endl;
		}
	}
}