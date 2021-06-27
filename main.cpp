#include "headers/simple_nn.h"
#include "headers/config.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

void load_model(const Config& cfg, SimpleNN& model);

int main(int argc, char** argv)
{
	Config cfg;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28;

	MatXf train_X, test_X;
	VecXi train_Y, test_Y;

	DataLoader train_loader, test_loader;

	if (cfg.mode == "train") {
		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train);
		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train);
		train_loader.load(train_X, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train);
	}

	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test);
	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test);
	test_loader.load(test_X, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;

	SimpleNN model;
	load_model(cfg, model);

	cout << "Model construction completed." << endl;

	if (cfg.mode == "train") {
		if (cfg.loss == "cross_entropy") {
			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss);
		}
		else {
			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss);
		}
		model.fit(train_loader, cfg.epoch, test_loader);
		model.save("./model_zoo", cfg.model + ".pth");
	}
	else {
		model.compile({ cfg.batch, ch, h, w });
		model.load(cfg.save_dir, cfg.pretrained);
		model.evaluate(test_loader);
	}

	return 0;
}

void load_model(const Config& cfg, SimpleNN& model)
{
	if (cfg.model == "lenet5") {
		for (int i = 0; i < 6; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Conv2d(1, 6, 5, 2, cfg.init));
				}
				else {
					model.add(new Conv2d(6, 16, 5, 0, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm2d);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU);
				}
				else {
					model.add(new Tanh);
				}
				if (cfg.pool == "max") {
					model.add(new MaxPool2d(2, 2));
				}
				else {
					model.add(new AvgPool2d(2, 2));
				}
			}
			else if (i == 2) {
				model.add(new Flatten);
			}
			else if (i < 5) {
				if (i == 3) {
					model.add(new Linear(400, 120, cfg.init));
				}
				else {
					model.add(new Linear(120, 84, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU);
				}
				else {
					model.add(new Tanh);
				}
			}
			else {
				model.add(new Linear(84, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d);
				}
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax);
				}
				else {
					model.add(new Sigmoid);
				}
			}
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Linear(784, 500, cfg.init));
				}
				else {
					model.add(new Linear(500, 150, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU);
				}
				else {
					model.add(new Tanh);
				}
			}
			else {
				model.add(new Linear(150, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d);
				}
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax);
				}
				else {
					model.add(new Sigmoid);
				}
			}
		}
	}
}