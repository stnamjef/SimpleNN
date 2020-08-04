#include "file_manage.h"
#include "simple_nn.h"
using namespace simple_nn;

//int main()
//{
//	int n_img_train = 60000, n_img_test = 10000, img_size = 784;
//	Tensor train_X, test_X;
//	Vector train_Y, test_Y;
//
//	ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X, true);
//	ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
//	ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X, true);
//	ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);
//
//	SimpleNN model;
//	model.add(new Dense(500, "uniform", 784));
//	model.add(new BatchNorm);
//	model.add(new Activation("tanh"));
//	model.add(new Dense(150, "uniform"));
//	model.add(new BatchNorm);
//	model.add(new Activation("tanh"));
//	model.add(new Dense(10, "uniform"));
//	model.add(new BatchNorm);
//	model.add(new Activation("tanh"));
//	model.add(new Output(10, "mse"));
//
//	int n_epoch = 30, batch = 30;
//	double l_rate = 0.1, lambda = 0.0;
//
//	model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);
//
//
//	return 0;
//}

int main()
{
	int n_img_train = 60000, n_img_test = 10000, img_size = 784;
	Tensor train_X, test_X;
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
	model.add(new Conv2D(6, { 5, 5 }, 2, "uniform", { 28, 28, 1 }));
	model.add(new BatchNorm);
	model.add(new Activation("tanh"));
	model.add(new Pool2D({ 2, 2 }, 2, "avg"));
	model.add(new Conv2D(16, { 5, 5 }, 0, "uniform", indices));
	model.add(new BatchNorm);
	model.add(new Activation("tanh"));
	model.add(new Pool2D({ 2, 2 }, 2, "avg"));
	model.add(new Flatten);
	model.add(new Dense(120, "uniform"));
	model.add(new BatchNorm);
	model.add(new Activation("tanh"));
	model.add(new Dense(84, "uniform"));
	model.add(new BatchNorm);
	model.add(new Activation("tanh"));
	model.add(new Dense(10, "uniform"));
	model.add(new BatchNorm);
	model.add(new Activation("tanh"));
	model.add(new Output(10, "mse"));

	int n_epoch = 20, batch = 16;
	double l_rate = 0.5, lambda = 0;

	model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);

	return 0;
}