# SimpleNN

- SimpleNN is a C++ implementation of convolutional neural network. 
- Dependency free, pure c++ implemented.
- Only use contiguous memory to increase the processing speed.
- Provides GEMM(general matrix multiplication), im2col operation.
- Can be used to deeply understand how deep learning works inside a convolutional neural network.

## Supported networks

### layer-types

- fully connected
- convolutional
- average pooling
- max pooling
- batch normalization

### activation functions

- tanh
- relu
- softmax

### loss functions

- mean squared error
- cross-entropy

### optimization algorithms

- stochastic gradient descent

## Examples

- Download [MNIST](http://yann.lecun.com/exdb/mnist/) first, and put them in the project folder where "main.cpp" is located.

- Construct LeNet-5 with batch normalization.

```c++
#pragma once
#include "simple_nn.h"
#include "file_manage.h"
using namespace std;
using namespace simple_nn;

int main()
{
	int n_img_train = 60000;
	int n_img_test = 10000;
	int n_label = 10;
	int img_size = 784;

	float* train_X;
	float* test_X;
	int* train_Y;
	int* test_Y;

	allocate_memory(train_X, n_img_train * img_size);
	allocate_memory(test_X, n_img_test * img_size);
	allocate_memory(train_Y, n_img_train);
	allocate_memory(test_Y, n_img_test);

	ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
	ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
	ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
	ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

	SimpleNN model;
	model.add(new Conv2d(6, 5, 2, { 28, 28, 1 }, "uniform"));
	model.add(new BatchNorm2d);
	model.add(new Activation("tanh"));
	model.add(new AvgPool2d(2, 2));
	model.add(new Conv2d(16, 5, 0, "uniform"));
	model.add(new BatchNorm2d);
	model.add(new Activation("tanh"));
	model.add(new AvgPool2d(2, 2));
	model.add(new Linear(120, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("tanh"));
	model.add(new Linear(84, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("tanh"));
	model.add(new Linear(10, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("softmax"));

	int n_epoch = 30, batch = 32;
	float lr = 0.1F, decay = 0.005F;

	SGD* optim = new SGD(lr, decay, "MSE");

	model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
			  test_X, n_img_test, test_Y, n_label);

	delete_memory(train_X);
	delete_memory(test_X);
	delete_memory(train_Y);
	delete_memory(test_Y);

	return 0;
}
```

- Construct DNN(500 x 150 x 10) with batch normalization.

```c++
#pragma once
#include "simple_nn.h"
#include "file_manage.h"
using namespace std;
using namespace simple_nn;

int main()
{
	int n_img_train = 60000;
	int n_img_test = 10000;
	int n_label = 10;
	int img_size = 784;

	float* train_X;
	float* test_X;
	int* train_Y;
	int* test_Y;

	allocate_memory(train_X, n_img_train * img_size);
	allocate_memory(test_X, n_img_test * img_size);
	allocate_memory(train_Y, n_img_train);
	allocate_memory(test_Y, n_img_test);

	ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
	ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
	ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
	ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

	SimpleNN model;

	model.add(new Linear(500, 28 * 28, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("relu"));
	model.add(new Linear(150, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("relu"));
	model.add(new Linear(10, "uniform"));
	model.add(new BatchNorm1d);
	model.add(new Activation("softmax"));

	int n_epoch = 30, batch = 32;
	float lr = 0.01F, decay = 0.0F;

	SGD* optim = new SGD(lr, decay, "cross entropy");

	model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
			  test_X, n_img_test, test_Y, n_label);

	delete_memory(train_X);
	delete_memory(test_X);
	delete_memory(train_Y);
	delete_memory(test_Y);

	return 0;
}
```

## Ongoing list

- cpu parallel processing
- optimize GEMM with open source BLAS library(OpenBlas etc)
- various activation functions and optimizers
- 1x1 convolution
- data loader for PASCAL VOC
