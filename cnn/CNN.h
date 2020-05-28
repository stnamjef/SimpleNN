#pragma once
#include "file_manage.h"
#include "Dense.h"
#include "Conv2D.h"
#include "Pool2D.h"
using namespace std;

// ******************************************************** //
// 1. 첫 번째 layer는 언제나 Convolutional layer이다.
// 2. Convolutional layer 다음은 언제나 Pooling layer이다.
// 3. 마지막 layer는 언제나 Dense layer이다.
// ******************************************************** //


struct Layer
{
	string type;
	Dense* dense;
	Conv2D* conv;
	Pool2D* pool;
	Layer(string type, Dense* dense, Conv2D* conv2d, Pool2D* pool2d) : 
		type(type), dense(dense), conv(conv2d), pool(pool2d) {}
	~Layer() {}
};

class CNN
{
public:
	vector<Layer> net;
public:
	CNN() {}
	void add(Dense* layer);
	void add(Conv2D* layer);
	void add(Pool2D* layer);
	void fit(const vector<Matrix>& X, const Vector& Y, double l_rate = 0.001, int n_epoch = 100, int batch = 30);
	Vector predict(const vector<Matrix>& X);
};

void CNN::add(Dense* layer) { net.push_back(Layer("Dense", layer, nullptr, nullptr)); }

void CNN::add(Conv2D* layer) { net.push_back(Layer("Conv2D", nullptr, layer, nullptr)); }

void CNN::add(Pool2D* layer) { net.push_back(Layer("Pool2D", nullptr, nullptr, layer)); }

void CNN::fit(const vector<Matrix>& X, const Vector& Y, double l_rate, int n_epoch, int batch)
{
	Vector training_errors(n_epoch);
	for (int epoch = 0; epoch < n_epoch; epoch++)
	{
		for (int n = 0; n < X.size(); n++)
		{
			// forward propagate
			for (int l = 0; l < net.size(); l++)
			{
				int cur = l, prev = l - 1;
				if (l == 0)
					net[cur].conv->forward_propagate(X[n]);
				else if (net[prev].type == "Conv2D" && net[cur].type == "Pool2D")
				{
					net[cur].pool->forward_propagate(net[prev].conv->output);
				}
				else if (net[prev].type == "Pool2D" && net[cur].type == "Conv2D")
				{
					net[cur].conv->forward_propagate(net[prev].pool->output);
				}
				else if (net[prev].type == "Pool2D" && net[cur].type == "Dense")
				{
					net[cur].dense->forward_propagate(flatten(net[prev].pool->output));
				}
				else if (net[prev].type == "Dense" && net[cur].type == "Dense")
				{
					net[cur].dense->forward_propagate(net[prev].dense->output);
				}
				else
				{
					cout << "CNN::fit(): Unexpected forward propagation." << endl;
					exit(100);
				}
			}

			// backward propagate
			for (int l = net.size() - 1; l >= 0; l--)
			{
				int cur = l, prev = l - 1;
				if (l == net.size() - 1)
				{
					if (net[prev].type == "Dense")
					{
						net[prev].dense->delta = net[cur].dense->backward_propagate(Y[n], net[prev].dense->output);
					}
					else if (net[prev].type == "Pool2D")
					{
						const Vector& delta = net[cur].dense->backward_propagate(Y[n], flatten(net[prev].pool->output));
						net[prev].pool->delta = unflatten(delta, net[prev].pool->n_out, net[prev].pool->out_size);
					}
					else
					{
						cout << "CNN::fit(): Unexpected backward propagation." << endl;
						exit(100);
					}
				}
				else if (l == 0)
				{
					net[cur].conv->backward_propagate(X[n]);
				}
				else if (net[cur].type == "Dense" && net[prev].type == "Dense")
				{
					net[prev].dense->delta = net[cur].dense->backward_propagate(net[prev].dense->output);
				}
				else if (net[cur].type == "Dense" && net[prev].type == "Pool2D")
				{
					const Vector& delta = net[cur].dense->backward_propagate(flatten(net[prev].pool->output));
					net[prev].pool->delta = unflatten(delta, net[prev].pool->n_out, net[prev].pool->out_size);
				}
				else if (net[cur].type == "Pool2D" && net[prev].type == "Conv2D")
				{
					net[prev].conv->delta = net[cur].pool->backward_propagate(net[prev].conv->output);
				}
				else if (net[cur].type == "Conv2D" && net[prev].type == "Pool2D")
				{
					//cout << net[cur].conv->Ws.size() << ' ' << net[cur].conv->dWs.size() << endl;
					net[prev].pool->delta = net[cur].conv->backward_propagate(net[prev].pool->output);
				}
				else
				{
					cout << "CNN::fit(): Unexpected backward propagation." << endl;
					exit(100);
				}
			}

			// update weight
			if ((n + 1) % batch == 0)
			{
				for (int l = 0; l < net.size(); l++)
				{
					if (net[l].type == "Dense")
						net[l].dense->update_weight(l_rate);
					else if (net[l].type == "Conv2D")
						net[l].conv->update_weight(l_rate);
				}
			}
		}

		double error = calc_error(Y, predict(X));
		training_errors[epoch] = error;
		cout << "Error rate(epoch" << epoch + 1 << ") : " << error * 100 << "%" << endl;
	}

	ofstream error_out("training_error.csv");
	write_vector(error_out, training_errors);
	error_out.close();

	ofstream model_out("model_out.csv");
	for (int l = 0; l < net.size(); l++)
	{
		if (net[l].type == "Conv2D")
		{
			for (int n = 0; n < net[l].conv->Ws.size(); n++)
				write_matrix(model_out, net[l].conv->Ws[n]);
			write_vector(model_out, net[l].conv->b);
		}
		else if (net[l].type == "Dense")
		{
			write_matrix(model_out, net[l].dense->W);
			write_vector(model_out, net[l].dense->b);
		}
	}
	model_out.close();
}

Vector CNN::predict(const vector<Matrix>& X)
{
	Vector predicted(X.size());
	for (int n = 0; n < X.size(); n++)
	{
		for (int l = 0; l < net.size(); l++)
		{
			int cur = l, prev = l - 1;
			if (l == 0)
				net[cur].conv->forward_propagate(X[n]);
			else if (net[prev].type == "Conv2D" && net[cur].type == "Pool2D")
			{
				net[cur].pool->forward_propagate(net[prev].conv->output);
			}
			else if (net[prev].type == "Pool2D" && net[cur].type == "Conv2D")
			{
				net[cur].conv->forward_propagate(net[prev].pool->output);
			}
			else if (net[prev].type == "Pool2D" && net[cur].type == "Dense")
			{
				net[cur].dense->forward_propagate(flatten(net[prev].pool->output));
			}
			else if (net[prev].type == "Dense" && net[cur].type == "Dense")
			{
				net[cur].dense->forward_propagate(net[prev].dense->output);
			}
			else
			{
				cout << "CNN::fit(): Unexpected forward propagation." << endl;
				exit(100);
			}
		}
		predicted[n] = (double)max_idx(net.back().dense->output);
	}
	return predicted;
}