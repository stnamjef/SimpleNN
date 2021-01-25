#pragma once
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "average_pooling_layer.h"
#include "activation_layer.h"
#include "batch_normalization_1d_layer.h"
#include "batch_normalization_2d_layer.h"
#include "sgd_optimizer.h"


namespace simple_nn
{
	int PROGRESS_BAR_LEGNTH = 30;

	class SimpleNN
	{
	private:
		int batch;
		int n_batch;
		int in_h;
		int in_w;
		int in_channels;
		SGD* optim;
		vector<Layer*> net;
		vector<vector<int>> batch_indices;
	public:
		SimpleNN();
		void add(Layer* layer);
		void fit(const float* X,
			int n_data,
			const int* Y,
			int n_label,
			int n_epoch,
			int batch,
			SGD* optim,
			const float* valid_X = nullptr,
			int n_data_valid = 0,
			const int* valid_Y = nullptr,
			int n_label_valid = 0);
		void predict(const float* X, float* predicted);
	private:
		void set_network(int n_data, int n_label, int batch, SGD* optim);
		void generate_batch_indices(int batch, int n_batch, vector<vector<int>>& batch_indices, bool shuffle = true);
		void get_batch_X(const float* X, int batch_idx, float* batch_X);
		void get_batch_Y(const int* Y, int batch_idx, int* batch_Y);
		void forward_propagate(const float* X, bool isEval);
		void classify(const float* prev_out, int n_label, float* classified);
		void error_criterion(const float* classified, const int* labels, float& running_error);
		void loss_criterion(const float* prev_out, const int* labels, float& running_loss);
		void zero_grad();
		void backward_propagate(const float* X);
		void update_weight();
	};

	void print_progress(int epoch, int n_epoch, int n_bar);
	void print_time_loss_error(float loss, float error, int n_batch, duration<float> sec,
							   float loss_valid, float error_valid, int n_batch_valid);

	// function definition

	SimpleNN::SimpleNN() :
		batch(0),
		n_batch(0),
		in_h(0),
		in_w(0),
		in_channels(0) {}

	void SimpleNN::add(Layer* layer) { net.push_back(layer); }

	void SimpleNN::fit(const float* X,
		int n_data,
		const int* Y,
		int n_label,
		int n_epoch,
		int batch,
		SGD* optim,
		const float* valid_X,
		int n_data_valid,
		const int* valid_Y,
		int n_label_valid)
	{
		set_network(n_data, n_label, batch, optim);

		float* classified;
		float* batch_X;
		int* batch_Y;

		allocate_memory(classified, n_batch);
		allocate_memory(batch_X, batch * in_h * in_w);
		allocate_memory(batch_Y, batch);

		for (int e = 0; e < n_epoch; e++) {

			float running_loss = 0.0F;
			float running_error = 0.0F;

			int n_bar = 0;
			int print = n_batch / PROGRESS_BAR_LEGNTH;
			system_clock::time_point start = system_clock::now();

			for (int n = 0; n < n_batch; n++) {
				get_batch_X(X, n, batch_X);
				get_batch_Y(Y, n, batch_Y);

				forward_propagate(batch_X, false);
				classify(net.back()->output, n_label, classified);
				error_criterion(classified, batch_Y, running_error);

				zero_grad();
				loss_criterion(net.back()->output, batch_Y, running_loss);
				backward_propagate(batch_X);
				update_weight();
				if (n % print == 0 && n_bar <= PROGRESS_BAR_LEGNTH) {
					print_progress(e, n_epoch, n_bar);
					n_bar++;
				}
			}

			system_clock::time_point end = system_clock::now();

			float running_loss_valid = 0.0F;
			float running_error_valid = 0.0F;

			// temporary code, must be modified
			int n_batch_valid = 0;
			if (valid_X != nullptr && valid_Y != nullptr) {
				n_batch_valid = n_data_valid / batch;
				vector<vector<int>> batch_indices_valid;
				generate_batch_indices(batch, n_batch_valid, batch_indices_valid, false);
				for (int n = 0; n < n_batch_valid; n++) {
					get_batch_X(valid_X, n, batch_X);
					get_batch_Y(valid_Y, n, batch_Y);

					forward_propagate(batch_X, true);
					classify(net.back()->output, n_label, classified);
					error_criterion(classified, batch_Y, running_error_valid);
					loss_criterion(net.back()->output, batch_Y, running_loss_valid);
				}
			}
			print_time_loss_error(running_loss, running_error, n_batch, end - start,
								  running_loss_valid, running_error_valid, n_batch_valid);
		}
		delete_memory(classified);
		delete_memory(batch_X);
		delete_memory(batch_Y);
	}

	void SimpleNN::set_network(int n_data, int n_label, int batch, SGD* optim)
	{
		this->batch = batch;
		vector<int> in_shape = net.front()->input_shape();
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

		n_batch = n_data / batch;
		generate_batch_indices(batch, n_batch, batch_indices, false);

		for (int l = 0; l < net.size(); l++) {
			if (l == 0) {
				net[l]->set_layer(batch, in_shape);
			}
			else {
				net[l]->set_layer(batch, net[l - 1]->output_shape());
			}
		}

		this->optim = optim;
		this->optim->set(batch, n_label);

		if (net.back()->output_shape().size() > 1) {
			throw logic_error("SimpleNN::set_network(int, int): The last layer must be 1d.");
		}
	}

	void SimpleNN::generate_batch_indices(int batch, int n_batch, vector<vector<int>>& batch_indices, bool shuffle)
	{
		vector<int> ran_num(batch * n_batch);
		for (int i = 0; i < ran_num.size(); i++) {
			ran_num[i] = i;
		}

		if (shuffle) {
			unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(ran_num.begin(), ran_num.end(), std::default_random_engine(seed));
		}

		batch_indices.resize(n_batch, vector<int>(batch));
		for (int i = 0; i < n_batch; i++) {
			for (int j = 0; j < batch; j++) {
				batch_indices[i][j] = ran_num[i * batch + j];
			}
		}
	}

	void SimpleNN::get_batch_X(const float* X, int batch_idx, float* batch_X)
	{
		// assume that the channel of input is always one
		int im_size = in_h * in_w;
		const vector<int>& indices = batch_indices[batch_idx];
		for (int i = 0; i < indices.size(); i++) {
			const float* src = X + im_size * indices[i];
			float* dst = batch_X + im_size * i;
			std::copy(src, src + im_size, dst);
		}
	}

	void SimpleNN::get_batch_Y(const int* Y, int batch_idx, int* batch_Y)
	{
		const vector<int>& indices = batch_indices[batch_idx];
		for (int i = 0; i < indices.size(); i++) {
			batch_Y[i] = Y[indices[i]];
		}
	}

	void SimpleNN::forward_propagate(const float* X, bool isEval)
	{
		for (int l = 0; l < net.size(); l++) {
			set_zero(net[l]->output, net[l]->get_out_block_size());
			if (l == 0) {
				net[l]->forward_propagate(X, isEval);
			}
			else {
				net[l]->forward_propagate(net[l - 1]->output, isEval);
			}
		}
	}

	void SimpleNN::classify(const float* prev_out, int n_label, float* classified)
	{
		for (int i = 0; i < batch; i++) {
			const float* src = prev_out + n_label * i;
			float* dst = classified + i;
			*dst = (float)std::distance(src, std::max_element(src, src + n_label));
		}
	}

	void SimpleNN::error_criterion(const float* classified, const int* labels, float& running_error)
	{
		running_error += optim->error_criterion(classified, labels);
	}

	void SimpleNN::loss_criterion(const float* prev_out, const int* labels, float& running_loss)
	{
		running_loss += optim->loss_criterion(prev_out, labels, net.back()->delta);
	}

	void SimpleNN::zero_grad()
	{
		for (const auto& l : net) {
			set_zero(l->delta, l->get_out_block_size());
		}
	}

	void SimpleNN::backward_propagate(const float* X)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--) {
			if (l == (int)net.size() - 1) {
				net[l]->backward_propagate(net[l - 1]->output, net[l - 1]->delta, false);
			}
			else if (l == 0) {
				net[l]->backward_propagate(X, nullptr, true);
			}
			else {
				net[l]->backward_propagate(net[l - 1]->output, net[l - 1]->delta, false);
			}
		}
	}

	void SimpleNN::update_weight()
	{
		float lr = optim->lr();
		float decay = optim->decay();
		for (const auto& l : net) {
			if (l->type == CONV2D ||
				l->type == LINEAR ||
				l->type == BATCHNORM1D ||
				l->type == BATCHNORM2D) {
				l->update_weight(lr, decay);
			}
		}
	}

	void print_progress(int epoch, int n_epoch, int n_bar)
	{
		string bar(n_bar, ' ');
		std::for_each(bar.begin(), bar.end(), [](char& c) { c = '='; });

		cout << "Epoch" << setw(3) << epoch + 1 << "/" << n_epoch;
		cout << " [" << setw(PROGRESS_BAR_LEGNTH) << std::left << bar << "] - ";
		cout << fixed << setprecision(2) << setw(5) << std::right;
		cout << n_bar / (float)PROGRESS_BAR_LEGNTH * 100 << "%";

		if (n_bar < PROGRESS_BAR_LEGNTH) {
			cout << '\r';
		}
	}

	void print_time_loss_error(float loss, float error, int n_batch, duration<float> sec,
							   float loss_valid, float error_valid, int n_batch_valid)
	{
		cout << fixed << setprecision(4);
		cout << " - t: " << sec.count() << 's';
		cout << " - loss: " << loss / n_batch;
		cout << ", error: " << error / n_batch * 100 << "%";
		if (n_batch_valid != 0) {
			cout << " - loss(valid): " << loss_valid / n_batch_valid;
			cout << ", error(valid): " << error_valid / n_batch_valid * 100 << "%";
		}
		cout << '\n';
	}

	void SimpleNN::predict(const float* X, float* predicted)
	{
		// do not use it yet
		exit(1);
		forward_propagate(X, true);
		predicted = net.back()->output;
	}
}