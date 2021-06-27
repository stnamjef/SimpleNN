# SimpleNN

- SimpleNN is a C++ implementation of convolutional neural network (CNN). 
- Provides ~~GEMM(GEneral Matrix Multiplication)~~, im2col, and col2im operations.
- Reasonably fast, without GPU. 
  - It takes about 13 seconds per epoch (MNIST 60,000 images).
  - Test env = model: lenet5; batch size: 32;  CPU: i7-10750H.
- [Update: 2021-06-24]
  - GEMM has been replaced by [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library.
  - [A dockerized test environment](https://hub.docker.com/repository/docker/stnamjef/eigen_3.3.9) is available.

## 1. Requirements

- Eigen 3.3.9
- g++ 9.0 or higher

## 2. Supported networks

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

### weight decay

- L2 regularization

### optimization algorithms

- stochastic gradient descent

## 3. Usage

### 3.1. Data preparation

- Download [MNIST datasets](http://yann.lecun.com/exdb/mnist/) and extract them to the directory named "dataset".
- It should have a directory structure as below. **Please do not change the file names.**

```shell
SimpleNN
    ├── dataset
    │   ├── t10k-images.idx3-ubyte		# testing images
    │   ├── t10k-labels.idx1-ubyte		# testing labels
    │   ├── train-images.idx3-ubyte		# training images
    │   └── train-labels.idx1-ubyte		# training labels
    ├── headers
    │   ├── activation_layer.h
    │   ├── average_pooling_layer.h
    │   ├── batch_normalization_1d_layer.h
    │   ├── batch_normalization_2d_layer.h
    │   ├── col2im.h
    │   ├── common.h
    │   ├── config.h
    │   ├── convolutional_layer.h
    │   ├── data_loader.h
    │   ├── file_manage.h
    │   ├── flatten_layer.h
    │   ├── fully_connected_layer.h
    │   ├── im2col.h
    │   ├── layer.h
    │   ├── loss_layer.h
    │   ├── max_pooling_layer.h
    │   ├── optimizers.h
    │   └── simple_nn.h
    └── main.cpp
```

- If the dataset directory is different, --data_dir must be specified.

### 3.2. Compile and build

- Pull the docker image.

```shell
# host shell
docker pull stnamjef/eigen_3.3.9:1.0
```

- Run the docker image

```shell
# pwd -> the project directory (SimpleNN)
docker run -it -v $(pwd):/usr/build stnamjef/eigen_3.3.9:1.0
```

- Compile

```shell
# container shell at /usr/build
g++ main.cpp --std=c++17 -I ../include -O2 -o simplenn
```

### 3.3. Train predefined models

- SimpleNN provides two predefined models: [lenet5](https://ieeexplore.ieee.org/abstract/document/726791) and linear.
- Ex 1) model: lenet5, pool: max, hidden layer activation: relu, loss: cross entropy, weight initialization: lecun uniform

```shell
# the command below is the same as ./simplenn (the default setting)
./simplenn --mode=train --model=lenet5 --pool=max --activ=relu --loss=cross_entropy --init=lecun_uniform
```

- Ex 2) The model above with batch normalization adopted

```shell
# the command below is the same as ./simplenn --use_batchnorm=1
./simplenn --mode=train --model=lenet5 --pool=max --activ=relu --loss=cross_entropy --init=lecun_uniform --use_batchnorm=1
```

- Ex 3) model: lenet5, pool: average, hidden layer activation: tanh, loss: mean squared error, weight initialization: xavier uniform

```shell
./simplenn --mode=train --model=lenet5 --pool=avg --activ=tanh --loss=mse --init=xavier_uniform
```

- Ex 4) model: linear, hidden layer activation: tanh, loss: cross entropy, weight initialization: xavier uniform

```shell
./simplenn --mode=train --model=linear --activ=tanh --loss=cross_entropy --init=xavier_uniform
```

### 3.4. Test pretrained models

- SimpleNN provides one pretrained weight: lenet5
- Ex) test default model (error rate: 1.07%)

```shell
# if pretained weights are not in ./model_zoo, --save_dir should be changed
./simplenn --mode=test --save_dir=./model_zoo --pretrained=lenet5.pth
```

## 4. Build custom models

- If you want to build your own model, write it in main.cpp file and follow the same process as in 3.1. Since CLI options are not available for custom models, we strongly recommend setting parameters (e.g. batch size, learning rate, decay...) manually before compiling.
- Ex 1) Train a simple three-layer DNN model. Note that this model is already defined in SimpleNN and named "linear".

```c++
#include "headers/simple_nn.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

int main()
{
	int n_train = 60000, n_test = 10000;
	int batch = 32, channels = 1, height = 28, width = 28, n_label = 10;

	MatXf train_X = read_mnist("./dataset", "train-images.idx3-ubyte", n_train);
	VecXi train_Y = read_mnist_label("./dataset", "train-labels.idx1-ubyte", n_train);
	MatXf test_X = read_mnist("./dataset", "t10k-images.idx3-ubyte", n_test);
	VecXi test_Y = read_mnist_label("./dataset", "t10k-labels.idx1-ubyte", n_test);

	DataLoader train_loader(train_X, train_Y, batch, channels, height, width, true);
	DataLoader test_loader(test_X, test_Y, batch, channels, height, width, false);

	SimpleNN model;

	model.add(new Linear(784, 500, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(500, 150, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(150, 10, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new Softmax);

	int epochs = 30;
	float lr = 0.01f, decay = 0.f;

	model.compile({ batch, channels, height, width }, new SGD(lr, decay), new CrossEntropyLoss);
	model.fit(train_loader, epochs, test_loader);
	model.save("./model_zoo", "linear");

	return 0;
}
```

- Ex 2) Test the above model.

```c++
#include "headers/simple_nn.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

int main()
{
	int n_train = 60000, n_test = 10000;
	int batch = 32, channels = 1, height = 28, width = 28, n_label = 10;

	MatXf test_X = read_mnist("./dataset", "t10k-images.idx3-ubyte", n_test);
	VecXi test_Y = read_mnist_label("./dataset", "t10k-labels.idx1-ubyte", n_test);

	DataLoader test_loader(test_X, test_Y, batch, channels, height, width, false);

	SimpleNN model;

	model.add(new Linear(784, 500, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(500, 150, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new ReLU);
	model.add(new Linear(150, 10, "lecun_uniform"));
	//model.add(new BatchNorm1d);
	model.add(new Softmax);

	model.compile({ batch, channels, height, width });
	model.load("./model_zoo", "linear.pth");
	model.evaluate(test_loader);

	return 0;
}
```

## 5. CLI options

| Command         | Data type | Description                                                  |
| --------------- | --------- | ------------------------------------------------------------ |
| --mode          | string    | Program mode (options: train, test; default: train)          |
| --model         | string    | Model name (options: lenet5, linear; default: lenet5)        |
| --data_dir      | string    | Dataset directory (default: ./dataset)                       |
| --save_dir      | string    | Saving directory (default: ./model_zoo)                      |
| --pretrained    | string    | Pretrained file name (default: None)                         |
| --pool          | string    | Pooling method (options: max, avg; default: max)             |
| --activ         | string    | Activation function for hidden layer (options: tanh, relu; default: relu) |
| --init          | string    | Weight initialization (options: uniform, normal, lecun_uniform, lecun_normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal; default: lecun_uniform) |
| --loss          | string    | Loss function for training (options: cross_entropy, mse; default: cross_entropy) |
| --batch         | int       | Batch size (default: 32)                                     |
| --epoch         | int       | Total epochs (default: 30)                                   |
| --lr            | float     | Learning rate (default: 0.01)                                |
| --decay         | float     | L2 regularization (default: 0)                               |
| --use_batchnorm | bool      | Use batch normalization (options: 0, 1; default: 0)          |
| --shuffle_train | bool      | Shuffle training dataset (options: 0, 1; default: 1)         |
| --shuffle_test  | bool      | Shuffle testing dataset (options: 0, 1; default: 0)          |

