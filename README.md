# SimpleNN

- SimpleNN is a C++ implementation of convolutional neural network. 
- Provides ~~GEMM(GEneral Matrix Multiplication)~~, im2col, and col2im operations
- [Update: 2021-06-24]
  - GEMM has been replaced by Eigen library.

## Requirements

- C++17
- Eigen 3.3.9

## Supported networks

### layer-types

- fully connected
- convolutional
- average pooling
- max pooling
- batch normalization

### activation functions

- sigmoid
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
#include "simple_nn.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

int main()
{
	int n_train = 60000, n_test = 10000;
	int batch = 32, channels = 1, height = 28, width = 28, n_label = 10;

	MatXf train_X = read_mnist("train-images.idx3-ubyte", n_train);
	VecXi train_Y = read_mnist_label("train-labels.idx1-ubyte", n_train);
	MatXf test_X = read_mnist("test-images.idx3-ubyte", n_test);
	VecXi test_Y = read_mnist_label("test-labels.idx1-ubyte", n_test);

	DataLoader train_loader(train_X, train_Y, batch, channels, height, width, true);
	DataLoader test_loader(test_X, test_Y, batch, channels, height, width, false);

	SimpleNN model;

	model.add(new Conv2d(1, 6, 5, 2, Init::LecunUniform));
	model.add(new BatchNorm2d);
	model.add(new ReLU);
	model.add(new MaxPool2d(2, 2));
	model.add(new Conv2d(6, 16, 5, 0, Init::LecunUniform));
	model.add(new BatchNorm2d);
	model.add(new ReLU);
	model.add(new MaxPool2d(2, 2));
	model.add(new Flatten);
	model.add(new Linear(400, 120, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(120, 84, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(84, 10, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new Softmax);

	int epochs = 30;
	float lr = 0.01f, decay = 0.f;

	model.compile({ batch, channels, height, width }, new SGD(lr, decay), new CrossEntropyLoss);
	model.fit(train_loader, epochs, test_loader);
	model.save("./model_zoo", "lenet5");

	return 0;
}
```

- Construct DNN(500 x 150 x 10) with batch normalization.

```c++
#include "simple_nn.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

int main()
{
	int n_train = 60000, n_test = 10000;
	int batch = 32, channels = 1, height = 28, width = 28, n_label = 10;

	MatXf train_X = read_mnist("train-images.idx3-ubyte", n_train);
	VecXi train_Y = read_mnist_label("train-labels.idx1-ubyte", n_train);
	MatXf test_X = read_mnist("test-images.idx3-ubyte", n_test);
	VecXi test_Y = read_mnist_label("test-labels.idx1-ubyte", n_test);

	DataLoader train_loader(train_X, train_Y, batch, channels, height, width, true);
	DataLoader test_loader(test_X, test_Y, batch, channels, height, width, false);

	SimpleNN model;

	model.add(new Linear(784, 500, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new Tanh);
	model.add(new Linear(500, 150, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new Tanh);
	model.add(new Linear(150, 10, Init::LecunUniform));
	model.add(new BatchNorm1d);
	model.add(new Softmax);

	int epochs = 30;
	float lr = 0.01f, decay = 0.f;

	model.compile({ batch, channels, height, width }, new SGD(lr, decay), new CrossEntropyLoss);
	model.fit(train_loader, epochs, test_loader);
	model.save("./model_zoo", "linear");

	return 0;
}
```

## Ongoing list

- various activation functions and optimizers
- 1x1 convolution
- data loader for PASCAL VOC
