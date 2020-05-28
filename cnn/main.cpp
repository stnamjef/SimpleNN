#include "CNN.h"
using namespace std;

void write_weight(ofstream& out, const Matrix& W)
{
	for (int i = 0; i < W.rows(); i++)
	{
		for (int j = 0; j < W.cols(); j++)
			out << W(i, j) << ',';
		out << endl;
	}
}

void write_bias(ofstream& out, const Vector& b)
{
	for (int i = 0; i < b.size(); i++)
		out << b[i] << ',';
	out << endl;
}

int main()
{
	int n_img = 50000, img_size = 784;
	vector<Matrix> train_X;
	Vector train_Y;

	ReadMNIST("train-images.idx3-ubyte", n_img, img_size, train_X);
	ReadMNISTLabel("train-labels.idx1-ubyte", n_img, train_Y);

	vector<vector<int>> indices(6, vector<int>(16));
	indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
	indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
	indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
	indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
	indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
	indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };

	int init_normal = 0, init_uniform = 1, max_pool = 0, avg_pool = 1;

	CNN model;
	model.add(new Conv2D({ 28, 28, 1 }, { 5, 5, 6 }, 2, init_normal));
	model.add(new Pool2D({ 28, 28, 6 }, { 2, 2, 6 }, 2, max_pool));
	model.add(new Conv2D({ 14, 14, 6 }, { 5, 5, 16 }, 0, init_normal, indices));
	model.add(new Pool2D({ 10, 10, 16 }, { 2, 2, 16 }, 2, max_pool));
	model.add(new Dense(400, 120, init_normal));
	model.add(new Dense(120, 84, init_normal));
	model.add(new Dense(84, 10, init_normal));

	int n_epoch = 100, batch = 32;
	double l_rate = 0.03;

	model.fit(train_X, train_Y, l_rate, n_epoch, batch);

	return 0;
}

