#pragma once
#include <time.h>
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "dense_layer.h"
#include "activation_layer.h"
#include "output_layer.h"
#include "flatten_layer.h"
#include "batch_normalization_layer.h"

namespace simple_nn
{
	class SimpleNN
	{
	private:
		int n_batch;
		int batch_size;
		int in_channels;
		int in_h;
		int in_w;
		vector<Layer*> net;
		vector<vector<int>> batch_indices;
		Tensor batch_X;
		Tensor batch_Y;
	public:
		SimpleNN();
		void add(Layer* layer);
		void fit(const Tensor& X,
			const Vector& Y,
			double l_rate,
			int n_epoch,
			int batch_size,
			double lambda,
			const Tensor& test_X = {},
			const Vector& test_Y = {});
		Vector predict(const Tensor& X);
	private:
		void set_network(int n_batch, int batch_size);
		void generate_batch_indices();
		void get_batch(const Tensor& X, const vector<int>& batch_idx);
		void get_batch(const Vector& Y, const vector<int>& batch_idx);
		void forward_propagate(const Tensor& X, bool isPrediction = false);
		void backward_propagate(const Tensor& X, const Tensor& Y);
		void update_weight(double l_rate, double lambda);
		void reset_batch(int new_batch_size);
		void print_loss_error(const Vector& expected, const Vector& predicted, bool isTraining);
	};
	void print_progress(int epoch, int n_epoch, int count);

	SimpleNN::SimpleNN() :
		n_batch(0),
		batch_size(0),
		in_channels(0),
		in_h(0),
		in_w(0) {}

	void SimpleNN::add(Layer* layer) { net.push_back(layer); }

	void SimpleNN::fit(const Tensor& X,
		const Vector& Y,
		double l_rate,
		int n_epoch,
		int batch_size,
		double lambda,
		const Tensor& test_X,
		const Vector& test_Y)
	{
		set_network(X.batches() / batch_size, batch_size);
		for (int epoch = 0; epoch < n_epoch; epoch++) {
			int count = 0;
			int range = n_batch / 30;
			clock_t start = clock();
			for (int n = 0; n < n_batch; n++) {
				get_batch(X, batch_indices[n]);
				get_batch(Y, batch_indices[n]);
				forward_propagate(batch_X);
				backward_propagate(batch_X, batch_Y);
				update_weight(l_rate, lambda);
				if ((n % range == 0 || n == n_batch - 1) && count <= 30) {
					print_progress(epoch, n_epoch, count);
					count++;
				}
			}
			cout << " - t: " << (double)(clock() - start) / CLOCKS_PER_SEC;
			reset_batch(X.batches());
			print_loss_error(Y, predict(X), true);
			reset_batch(test_X.batches());
			print_loss_error(test_Y, predict(test_X), false);
			reset_batch(batch_size);
		}
	}

	void SimpleNN::set_network(int n_batch, int batch_size)
	{
		vector<int>in_shape = net.front()->input_shape();
		this->n_batch = n_batch;
		this->batch_size = batch_size;
		if (in_shape.size() == 3) {
			in_h = in_shape[0];
			in_w = in_shape[1];
			in_channels = in_shape[2];
		}
		else {
			in_h = in_shape[0];
			in_w = 1;
			in_channels = 1;
		}
		batch_X.resize(batch_size, in_channels, in_h, in_w);
		batch_Y.resize(batch_size, 1, 1);
		generate_batch_indices();

		for (int l = 0; l < net.size(); l++) {
			if (l == 0) {
				net[l]->set_layer(batch_size, in_shape);
			}
			else {
				net[l]->set_layer(batch_size, net[l - 1]->output_shape());
			}
		}
	}

	void SimpleNN::generate_batch_indices()
	{
		vector<int> ranNum((__int64)n_batch * batch_size);
		for (int i = 0; i < ranNum.size(); i++) {
			ranNum[i] = i;
		}

		/*unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(ranNum.begin(), ranNum.end(), std::default_random_engine(seed));*/

		batch_indices.resize(n_batch, vector<int>(batch_size));
		for (int i = 0; i < n_batch; i++) {
			for (int j = 0; j < batch_size; j++) {
				batch_indices[i][j] = ranNum[(__int64)i * batch_size + j];
			}
		}
	}

	void SimpleNN::get_batch(const Tensor& X, const vector<int>& batch_idx)
	{
		for (int n = 0; n < batch_size; n++) {
			for (int c = 0; c < in_channels; c++) {
				batch_X[n][c] = X[batch_idx[n]][c];
			}
		}
	}

	void SimpleNN::get_batch(const Vector& Y, const vector<int>& batch_idx)
	{
		for (int n = 0; n < batch_size; n++) {
			batch_Y[n][0](0) = Y(batch_idx[n]);
		}
	}

	void SimpleNN::forward_propagate(const Tensor& X, bool isPrediction)
	{
		for (int l = 0; l < net.size(); l++)
		{
			if (l == 0) {
				net[l]->forward_propagate(X, isPrediction);
			}
			else {
				net[l]->forward_propagate(net[l - 1]->output, isPrediction);
			}
		}
	}

	void SimpleNN::backward_propagate(const Tensor& X, const Tensor& Y)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--)
		{
			if (l == (int)net.size() - 1) {
				net[l]->backward_propagate(Y, net[l - 1]->delta);
			}
			else if (l == 0) {
				Tensor empty;
				net[l]->backward_propagate(X, empty, true);
			}
			else {
				net[l]->backward_propagate(net[l - 1]->output, net[l - 1]->delta);
			}
		}
	}

	void SimpleNN::update_weight(double l_rate, double lambda)
	{
		for (const auto& layer : net) {
			if (layer->type == "conv2d" ||
				layer->type == "dense" ||
				layer->type == "batchnorm") {
				layer->update_weight(l_rate, lambda);
			}
		}
	}

	void print_progress(int epoch, int n_epoch, int count)
	{
		int bar_length = 30;
		string bar = "";
		for (int i = 0; i < count; i++) {
			bar.push_back('=');
		}
		cout << "Epoch " << epoch + 1 << "/" << n_epoch << "  ";
		cout << "[" << std::setw(bar_length) << std::left << bar << "] - ";
		cout << fixed << setprecision(2) << setw(4) << std::right;
		cout << count / (double)bar_length * 100 << "%";
		if (count < bar_length) {
			cout << '\r';
		}
	}

	void SimpleNN::reset_batch(int new_batch_size)
	{
		batch_size = new_batch_size;
		batch_X.resize(batch_size, in_channels, in_h, in_w);
		batch_Y.resize(batch_size, 1, 1);
		for (const auto& layer : net) {
			layer->reset_batch(batch_size);
		}
	}

	void SimpleNN::print_loss_error(const Vector& expected, const Vector& predicted, bool isTraining)
	{
		double loss = net.back()->calc_loss(expected);
		double error = 0.0;
		for (int n = 0; n < expected.size(); n++) {
			if (expected(n) != predicted(n)) {
				error += 1.0;
			}
		}
		cout << fixed << setprecision(4) << setw(6) << std::right;
		if (isTraining) {
			cout << " - loss: " << loss;
			cout << ", error: " << error / batch_size * 100 << "%";
		}
		else {
			cout << " - loss(validation): " << loss;
			cout << ", error(validation): " << error / batch_size * 100 << "%" << endl;
		}
	}

	Vector SimpleNN::predict(const Tensor& X)
	{
		forward_propagate(X, true);
		Vector output(batch_size);
		const Tensor& last = net.back()->output;
		for (int n = 0; n < batch_size; n++) {
			output(n) = (int)std::distance(last[n][0].begin(), 
				std::max_element(last[n][0].begin(), last[n][0].end()));
		}
		return output;
	}
}