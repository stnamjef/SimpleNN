#include "simple_nn.h"
using namespace std;
using namespace simple_nn;

int main()
{
	int n_img_train = 60000, n_img_test = 10000, img_size = 784;
	vector<Matrix> train_X, test_X;
	Vector train_Y, test_Y;

	ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
	ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
	ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
	ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

	vector<vector<int>> indices(6, vector<int>(16));
	indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
	indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
	indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
	indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
	indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
	indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };

	SimpleNN model;
	model.add(new Conv2D({ 28, 28, 1 }, { 5, 5, 6 }, 2, Init::UNIFORM));
	model.add(new BatchNorm({ 28, 28, 6 }, true));
	model.add(new Activation(6, Activate::RELU));
	model.add(new Pool2D({ 28, 28, 6 }, { 2, 2, 6 }, 2, Pool::AVG));
	model.add(new Conv2D({ 14, 14, 6 }, { 5, 5, 16 }, 0, Init::UNIFORM, indices));
	model.add(new BatchNorm({ 10, 10, 16 }, true));
	model.add(new Activation(16, Activate::RELU));
	model.add(new Pool2D({ 10, 10, 16 }, { 2, 2, 16 }, 2, Pool::AVG));
	model.add(new Dense(400, 120, Init::UNIFORM));
	model.add(new BatchNorm({ 120, 120, 1 }, false));
	model.add(new Activation(1, Activate::RELU));
	model.add(new Dense(120, 84, Init::UNIFORM));
	model.add(new BatchNorm({ 84, 84, 1 }, false));
	model.add(new Activation(1, Activate::RELU));
	model.add(new Dense(84, 10, Init::UNIFORM));
	model.add(new BatchNorm({ 10, 10, 1 }, false));
	model.add(new Activation(1, Activate::SOFTMAX));
	model.add(new Output(10, Loss::CROSS_ENTROPY));

	int n_epoch = 30, batch = 20;
	double l_rate = 0.01, lambda = 0.0;

	model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);

	return 0;
}

