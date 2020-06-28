#pragma once
#include "file_manage.h"
#include "dense_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "activation_layer.h"
#include "output_layer.h"
#include "batch_normalization_layer.h"

namespace simple_nn
{
	class SimpleNN
	{
	private:
		vector<Layer*> net;
	public:
		SimpleNN() {}
		~SimpleNN() {}
		void add(Layer* layer);
		void fit(const vector<Matrix>& X,
			const Vector& Y,
			double l_rate,
			int n_epoch,
			int batch_size,
			double lambda,
			const vector<Matrix>& test_X,
			const Vector& test_Y);
		Vector predict(const vector<Matrix>& X, const Vector& Y, double& loss);
	};

	vector<vector<int>> split_into_batches(int batch_size, int n_batch);

	vector<vector<Matrix>> get_batch(const vector<Matrix>& X, vector<int>& batch_idx);

	vector<vector<Vector>> get_batch(const Vector& Y, vector<int>& batch_idx);

	void set_batch(const vector<Layer*>& net, int batch_size, int n_batch);

	void forward_propagate(const vector<Layer*>& net, const vector<vector<Matrix>>& X, int batch_size, bool isPrediction = false);

	void backward_propagate(const vector<Layer*>& net, const vector<vector<Matrix>>& X, const vector<vector<Vector>>& Y, int batch_size);

	void update_weight(const vector<Layer*>& net, double l_rate, double lambda, int batch_size);

	//---------------------------------------------- function definition ----------------------------------------------

	void SimpleNN::add(Layer* layer) { net.push_back(layer); }

	void SimpleNN::fit(const vector<Matrix>& X,
					   const Vector& Y,
					   double l_rate,
					   int n_epoch,
					   int batch_size,
					   double lambda,
					   const vector<Matrix>& test_X,
					   const Vector& test_Y)
	{
		int n_batch = (int)X.size() / batch_size;
		vector<vector<int>> batches = split_into_batches(batch_size, n_batch);

		for (int epoch = 0; epoch < n_epoch; epoch++)
		{
			set_batch(net, batch_size, n_batch);

			double train_loss = 0.0, test_loss = 0.0;
			double train_error = 0.0, test_error = 0.0;

			for (int n = 0; n < n_batch; n++)
			{
				forward_propagate(net, get_batch(X, batches[n]), batch_size);
				backward_propagate(net, get_batch(X, batches[n]), get_batch(Y, batches[n]), batch_size);
				update_weight(net, l_rate, lambda, batch_size);
			}

			train_error = calc_error(Y, predict(X, Y, train_loss));
			test_error = calc_error(test_Y, predict(test_X, test_Y, test_loss));

			cout << "Epoch" << epoch + 1;
			cout << " : training loss -> " << train_loss;
			cout << ", testing loss -> " << test_loss;
			cout << ", training error -> " << train_error * 100 << "%";
			cout << ", testing error -> " << test_error * 100 << "%" << endl;
		}
	}

	vector<vector<int>> split_into_batches(int batch_size, int n_batch)
	{
		vector<int> ranNum((__int64)n_batch * batch_size);
		for (int i = 0; i < ranNum.size(); i++)
			ranNum[i] = i;

		unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(ranNum.begin(), ranNum.end(), std::default_random_engine(seed));

		vector<vector<int>> batches(n_batch, vector<int>(batch_size));
		for (int i = 0; i < n_batch; i++)
			for (int j = 0; j < batch_size; j++)
				batches[i][j] = ranNum[(__int64)i * batch_size + j];

		return batches;
	}

	vector<vector<Matrix>> get_batch(const vector<Matrix>& X, vector<int>& batch_idx)
		// an input img has only one channel
	{
		vector<vector<Matrix>> batch(batch_idx.size(), vector<Matrix>(1));
		for (unsigned int i = 0; i < batch_idx.size(); i++)
			batch[i][0] = X[batch_idx[i]]; 
		return batch;
	}

	vector<vector<Vector>> get_batch(const Vector& Y, vector<int>& batch_idx)
		// 첫 번째 batch의 첫 번째 channel에 모든 데이터가 다 있음
	{
		vector<vector<Vector>> batch(1, vector<Vector>(1, Vector((int)batch_idx.size())));
		for (unsigned int i = 0; i < batch_idx.size(); i++)
			batch[0][0][i] = Y[batch_idx[i]];
		return batch;
	}

	void set_batch(const vector<Layer*>& net, int batch_size, int n_batch)
	{
		for (const auto& layer : net)
			layer->set_batch(batch_size, n_batch);
	}

	void forward_propagate(const vector<Layer*>& net, const vector<vector<Matrix>>& X, int batch_size, bool isPrediction)
	{
		for (unsigned int l = 0; l < net.size(); l++)
		{
			Layer* curr = net[l];
			Layer* prev = (l == 0) ? nullptr : net[l - 1];

			if (l == 0)
			{
				curr->forward_propagate(X, isPrediction);
			}
			else if (prev->type == LayerType::POOL2D &&
					 curr->type == LayerType::DENSE)
			{
				curr->forward_propagate(flatten(prev->output, batch_size), isPrediction);
			}
			else
			{
				curr->forward_propagate(prev->output, isPrediction);
			}
		}
	}

	void backward_propagate(const vector<Layer*>& net, const vector<vector<Matrix>>& X, const vector<vector<Vector>>& Y, int batch_size)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--)
		{
			Layer* curr = net[l];
			Layer* prev = (l == 0) ? nullptr : net[l - 1];

			if (l == (int)net.size() - 1)
			{
				prev->delta = curr->backward_propagate(Y);
			}
			else if (l == 0)
			{
				curr->backward_propagate(X, true);
			}
			else if (curr->type == LayerType::DENSE &&
					 prev->type == LayerType::POOL2D)
			{
				const vector<vector<Vector>>& delta = curr->backward_propagate(flatten(prev->output, batch_size));
				int prev_out_channels = (int)prev->output.at(0).size();
				int prev_out_size = (int)prev->output.at(0).at(0).rows();
				prev->delta = unflatten(delta, prev_out_channels, prev_out_size);
			}
			else
			{
				prev->delta = curr->backward_propagate(prev->output);
			}
		}
	}

	void update_weight(const vector<Layer*>& net, double l_rate, double lambda, int batch_size)
	{
		for (const auto& layer : net)
		{
			if (layer->type != LayerType::POOL2D &&
				layer->type != LayerType::ACTIVATION &&
				layer->type != LayerType::OUTPUT)
			{
				layer->update_weight(l_rate, lambda, batch_size);
			}
		}
	}

	Vector SimpleNN::predict(const vector<Matrix>& X, const Vector& Y, double& loss)
	{
		int n_data = (int)X.size();
		set_batch(net, 1, n_data);

		Vector predicts(n_data);
		for (int i = 0; i < n_data; i++)
		{
			vector<vector<Matrix>> temp(1, vector<Matrix>(1, X[i]));
			forward_propagate(net, temp, 1, true);
			
			const Vector& output = net.back()->output[0][0];
			predicts[i] = max_idx(output);

			if (net.back()->get_loss_opt() == Loss::MSE)
				loss += 0.5 * pow2(as_vector((int)Y[i], 10) - output).sum();
			else
				loss += -log(softmax(output)[(int)Y[i]]);
		}
		loss /= (double)X.size();

		return predicts;
	}
}