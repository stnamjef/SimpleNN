#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <Eigen/Dense>
#include "file_manage.h"
using namespace std;
using namespace Eigen;


MatrixXd conv2d(const MatrixXd& input, const MatrixXd& filter, int padding, int stride, double bias, int out_size)
{
	MatrixXd output(out_size, out_size);
	for (int j = 0; j < out_size; j++)
		for (int i = 0; i < out_size; i++)
		{
			output(j, i) = bias;
			for (int y = 0; y < filter.rows(); y++)
				for (int x = 0; x < filter.cols(); x++)
				{
					int jj = j + y - padding;
					int ii = i + x - padding;

					if (j != 0)
						jj += (stride - 1);
					if (i != 0)
						ii += (stride - 1);

					if (0 <= jj && jj < input.rows() && 0 <= ii && ii < input.cols())
						output(j, i) += input(jj, ii) * filter(y, x);
				}
		}
	return output;
}

MatrixXd pool2d(const MatrixXd& input, int kernel, int S = 1)
// input은 정방행렬이다.
{
	int out_size = (int)(floor(input.rows() - kernel) / S) + 1;
	MatrixXd output(out_size, out_size);

	for (int j = 0; j < out_size; j++)
		for (int i = 0; i < out_size; i++)
		{
			vector<double> temp(kernel * kernel);
			for (int y = 0; y < kernel; y++)
				for (int x = 0; x < kernel; x++)
				{
					int img_j = j + y;
					int img_i = i + x;

					if (j != 0)
						img_j += (S - 1);
					if (i != 0)
						img_i += (S - 1);

					temp[y * kernel + x] = input(img_j, img_i);
				}
			output(j, i) = *max_element(temp.begin(), temp.end());
		}

	return output;
}

VectorXd flatten(const vector<MatrixXd>& inputs)
{
	VectorXd flattened(inputs.size() * inputs[0].size());

	int k = 0;
	for (const auto& featmap : inputs)
	{
		for (int i = 0; i < featmap.rows(); i++)
			for (int j = 0; j < featmap.cols(); j++)
				flattened[k++] = featmap(i, j);
	}
	return flattened;
}

double tan_h(double x) { return 2 / (1 + exp(-x)) - 1; }

double relu(double x) { return max(0.0, x); }

MatrixXd activate(const MatrixXd& sum, string type)
{
	MatrixXd output(sum.rows(), sum.cols());
	for (int j = 0; j < sum.rows(); j++)
		for (int i = 0; i < sum.cols(); i++)
		{
			if (type == "tanh")
				output(j, i) = tan_h(sum(j, i));
			else
				output(j, i) = relu(sum(j, i));
		}
	return output;
}

void init_filter_normal(vector<MatrixXd>& filter, VectorXd& bias, int n_in, int n_out)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double var = sqrt(2 / (double)(n_in + n_out));
	normal_distribution<double> dist(0, var);

	for (int i = 0; i < bias.size(); i++)
		bias[i] = dist(e);

	for (int n = 0; n < filter.size(); n++)
		for (int j = 0; j < filter[n].rows(); j++)
			for (int i = 0; i < filter[n].cols(); i++)
				filter[n](j, i) = dist(e);
}

void init_weight_normal(MatrixXd& weight, VectorXd& bias, int n_in, int n_out)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double var = sqrt(2 / (double)(n_in + n_out));
	normal_distribution<double> dist(0, var);

	for (int j = 0; j < weight.rows(); j++)
	{
		bias[j] = dist(e);
		for (int i = 0; i < weight.cols(); i++)
			weight(j, i) = dist(e);
	}
}

double tanh_prime(double x) { return 0.5 * (1 + tan_h(x)) * (1 - tan_h(x)); }

double relu_prime(double x) { return (x <= 0) ? 0 : 1; }

VectorXd activate_prime(const VectorXd& sum, string type)
{
	VectorXd output(sum.size());
	for (int i = 0; i < sum.size(); i++)
	{
		if (type == "tanh")
			output[i] = tanh_prime(sum[i]);
		else
			output[i] = relu_prime(sum[i]);
	}
	return output;
}

vector<MatrixXd> deflatten(const VectorXd& flattened, int n_delta, int delta_size)
{
	vector<MatrixXd> delta(n_delta, MatrixXd(delta_size, delta_size));
	for (int n = 0; n < n_delta; n++)
		for (int j = 0; j < delta_size; j++)
			for (int i = 0; i < delta_size; i++)
			{
				int idx = j * delta_size + i + (n * (int)delta[n].size());
				delta[n](j, i) = flattened[idx];
			}
	return delta;
}

MatrixXd calc_delta_featmap(const MatrixXd& sum, const MatrixXd& featmap, int this_out_size,
	const MatrixXd& next_delta, int next_filter, int next_stride, int next_out_size)
	// j, i : index of next layer featmap
	// jj, ii : index of this layer featmap
{
	MatrixXd delta(this_out_size, this_out_size);
	delta.setZero();
	for (int j = 0; j < next_out_size; j++)
		for (int i = 0; i < next_out_size; i++)
		{
			vector<double> temp(next_filter * next_filter);
			for (int y = 0; y < next_filter; y++)
				for (int x = 0; x < next_filter; x++)
				{
					int jj = j + y;
					int ii = i + x;

					if (j != 0)
						jj += (next_stride - 1);
					if (i != 0)
						ii += (next_stride - 1);

					temp[y * next_filter + x] = featmap(jj, ii);
				}
			int max = distance(temp.begin(), max_element(temp.begin(), temp.end()));
			int jj = j + max / next_filter;
			int ii = i + max % next_filter;

			if (j != 0)
				jj += (next_stride - 1);
			if (i != 0)
				ii += (next_stride - 1);

			delta(jj, ii) = next_delta(j, i) * tanh_prime(sum(jj, ii));
		}
	return delta;
}

MatrixXd rotate_180(const MatrixXd& filter)
{
	int m = filter.rows(), n = filter.cols();
	MatrixXd rotate(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			rotate(i, j) = filter(m - i - 1, n - j - 1);
	return rotate;
}

VectorXd asVector(double num)
{
	VectorXd output(10);
	output.setZero();
	output[num] = 1;
	return output;
}

double calc_error(double expected, const VectorXd& predicted)
{
	double max = (double)distance(predicted.data(), max_element(predicted.data(), predicted.data() + predicted.size()));
	return (expected != max) ? 1 : 0;
}

int main()
{
	int n_img = 5000;

	vector<MatrixXd> X;
	VectorXd Y;

	ReadMNIST(n_img, 784, X);
	ReadMNISTLabel(n_img, Y);

	// 1st layer: conv2d
	vector<MatrixXd> conv_sum1(6, MatrixXd(28, 28));
	vector<MatrixXd> conv_output1(6, MatrixXd(28, 28));
	vector<MatrixXd> conv_filter1(6, MatrixXd(5, 5));
	VectorXd conv_bias1(6);
	vector<MatrixXd> conv_delta1(6, MatrixXd(28, 28));
	vector<MatrixXd> conv_del_filter1(6, MatrixXd(5, 5));
	VectorXd conv_del_bias1(6);
	init_filter_normal(conv_filter1, conv_bias1, 5*5, 28*28);

	// 2nd layer: pool2d
	vector<MatrixXd> pool_output1(6, MatrixXd(14, 14));
	vector<MatrixXd> pool_delta1(6, MatrixXd(14, 14));

	// 3rd layer: conv2d
	vector<MatrixXd> conv_sum2(6, MatrixXd(10, 10));
	vector<MatrixXd> conv_output2(6, MatrixXd(10, 10));
	vector<MatrixXd> conv_filter2(6, MatrixXd(5, 5));
	VectorXd conv_bias2(6);
	vector<MatrixXd> conv_delta2(6, MatrixXd(10, 10));
	vector<MatrixXd> conv_del_filter2(6, MatrixXd(5, 5));
	VectorXd conv_del_bias2(6);
	init_filter_normal(conv_filter2, conv_bias2, 5*5, 10*10);
	
	// 4th layer: pool2d
	vector<MatrixXd> pool_output2(6, MatrixXd(5, 5));
	vector<MatrixXd> pool_delta2(6, MatrixXd(5, 5));

	// 5th layer: dense
	VectorXd dense_sum1(60);
	VectorXd dense_output1(60);
	MatrixXd dense_weight1(60, 150);
	VectorXd dense_bias1(60);
	VectorXd dense_delta1(60);
	MatrixXd dense_del_weight1(60, 150);
	VectorXd dense_del_bias1(60);
	init_weight_normal(dense_weight1, dense_bias1, 150, 60);

	// 6th layer: dense
	VectorXd dense_sum2(10);
	VectorXd dense_output2(10);
	MatrixXd dense_weight2(10, 60);
	VectorXd dense_bias2(10);
	VectorXd dense_delta2(10);
	MatrixXd dense_del_weight2(10, 60);
	VectorXd dense_del_bias2(10);
	init_weight_normal(dense_weight2, dense_bias2, 60, 10);


	int n_epoch = 5;
	double l_rate = 0.001;

	// model training start
	for (int epoch = 0; epoch < n_epoch; epoch++)
	{
		double error = 0;
		for (int n = 0; n < X.size(); n++)
		{
			//------------------------------- forward propagate start -------------------------------

			// input -> 1st conv2d
			for (int i = 0; i < conv_output1.size(); i++)
			{
				conv_sum1[i] = conv2d(X[n], conv_filter1[i], 2, 1, conv_bias1[i], 28);
				conv_output1[i] = activate(conv_sum1[i], "tanh");
			}

			// 1st conv2d -> 1st pool2d
			for (int i = 0; i < pool_output1.size(); i++)
				pool_output1[i] = pool2d(conv_output1[i], 2, 2);

			// 1st pool2d -> 2nd conv2d
			for (int i = 0; i < conv_output2.size(); i++)
			{
				conv_sum2[i] = conv2d(pool_output1[i], conv_filter2[i], 0, 1, conv_bias2[i], 10);
				conv_output2[i] = activate(conv_sum2[i], "tanh");
			}

			// 2nd conv2d -> 2nd pool2d
			for (int i = 0; i < pool_output2.size(); i++)
				pool_output2[i] = pool2d(conv_output2[i], 2, 2);

			// 2nd pool2d -> 1st dense
			dense_sum1 = dense_weight1 * flatten(pool_output2) + dense_bias1;
			dense_output1 = activate(dense_sum1, "tanh");

			// 1st dense -> 2nd dense
			dense_sum2 = dense_weight2 * dense_output1 + dense_bias2;
			dense_output2 = activate(dense_sum2, "tanh");

			//------------------------------- backward propagate start -------------------------------

			// output -> 2nd dense

			dense_delta2 = (asVector(Y[n]) - dense_output2).array() * activate_prime(dense_sum2, "tanh").array();
			dense_del_weight2 = dense_delta2 * dense_output1.transpose();
			dense_del_bias2 = dense_delta2;

			// 2nd dense -> 1st dense
			dense_delta1 = (dense_weight2.transpose() * dense_delta2).array() * activate_prime(dense_sum1, "tanh").array();
			dense_del_weight1 = dense_delta1 * flatten(pool_output2).transpose();
			dense_del_bias1 = dense_delta1;

			// 1st dense -> 2nd pool2d
			pool_delta2 = deflatten(dense_weight1.transpose() * dense_delta1, 6, 5);
			
			// 2nd pool2d -> 2nd conv2d
			for (int i = 0; i < conv_delta2.size(); i++)
				conv_delta2[i] = calc_delta_featmap(conv_sum2[i], conv_output2[i], 10, pool_delta2[i], 2, 2, 5);


			for (int i = 0; i < conv_del_filter2.size(); i++)
				conv_del_filter2[i] = conv2d(pool_output1[i], conv_output2[i], 0, 1, 0, 5);

			for (int i = 0; i < conv_del_bias2.size(); i++)
				conv_del_bias2[i] = conv_delta2[i].sum();

			// 2nd conv2d -> 1st pool2d
			for (int i = 0; i < pool_delta1.size(); i++)
			{
				int padding = conv_filter2[i].rows() - 1;
				pool_delta1[i] = conv2d(conv_delta2[i], rotate_180(conv_filter2[i]), padding, 1, 0, 14);
			}

			// 1st poo2d -> 1st conv2d
			for (int i = 0; i < conv_delta1.size(); i++)
				conv_delta1[i] = calc_delta_featmap(conv_sum1[i], conv_output1[i], 28, pool_delta1[i], 2, 2, 14);

			for (int i = 0; i < conv_del_filter1.size(); i++)
				conv_del_filter1[i] = conv2d(X[n], conv_output1[i], 2, 1, 0, 5); //bias가 있어야 하나?

			for (int i = 0; i < conv_del_bias1.size(); i++)
				conv_del_bias1[i] = conv_delta1[i].sum();

			//break;

			//------------------------------- update weight & filter start -------------------------------

			// 1st conv2d
			for (int i = 0; i < conv_filter1.size(); i++)
				conv_filter1[i] -= l_rate * conv_del_filter1[i];
			conv_bias1 -= l_rate * conv_del_bias1;

			// 2nd conv2d
			for (int i = 0; i < conv_filter2.size(); i++)
				conv_filter2[i] -= l_rate * conv_del_filter2[i];
			conv_bias2 -= l_rate * conv_del_bias2;

			// 1st dense
			dense_weight1 -= l_rate * dense_del_weight1;
			dense_bias1 -= l_rate * dense_del_bias1;

			// 2nd dense
			dense_weight2 -= l_rate * dense_del_weight2;
			dense_bias2 -= l_rate * dense_del_bias2;

			//------------------------------- update error -------------------------------

			error += calc_error(Y[n], dense_output2);
		}
		cout << "Error rate(epoch" << epoch + 1 << ") : " << error / (double)X.size() * 100 << "%" << endl;
	}

	/*cout << "-----------------------------------------------------" << endl << endl;
	cout << "Input image : " << endl;
	for (int i = 0; i < X.size(); i++)
		cout << X[i] << endl << endl;*/

	cout << "-----------------------------------------------------" << endl << endl;
	cout << "[ 1st layer: Convolutional layer featuremap ] " << endl;
	for (int i = 0; i < conv_output1.size(); i++)
		cout << conv_output1[i] << endl << endl;

	cout << "[ 1st layer: Convolutional layer filter ]" << endl;
	for (int i = 0; i < conv_filter1.size(); i++)
		cout << conv_filter1[i] << endl << endl;

	cout << "[ 1st layer: Convolutional layer bias ]" << endl;
	cout << conv_bias1.transpose() << endl << endl;

	cout << "[ 1st layer: Convolutional layer delta featuremap]" << endl;
	for (int i = 0; i < conv_delta1.size(); i++)
		cout << conv_delta1[i] << endl << endl;

	cout << "[ 1st layer: Convolutional layer delta filter ]" << endl;
	for (int i = 0; i < conv_del_filter1.size(); i++)
		cout << conv_del_filter1[i] << endl << endl;

	cout << "[ 1st layer: Convolutional layer delta bias ]" << endl;
	cout << conv_del_bias1.transpose() << endl << endl;

	cout << "-----------------------------------------------------" << endl << endl;
	cout << "[ 2nd layer: Pooling layer featuremap ]" << endl;
	for (int i = 0; i < pool_output1.size(); i++)
		cout << pool_output1[i] << endl << endl;

	cout << "[ 2nd layer: Pooling layer delta featuremap ]" << endl;
	for (int i = 0; i < pool_delta1.size(); i++)
		cout << pool_delta1[i] << endl << endl;

	cout << "-----------------------------------------------------" << endl << endl;
	cout << "[ 3rd layer: Convolutional layer featuremap ]" << endl;
	for (int i = 0; i < conv_output2.size(); i++)
		cout << conv_output2[i] << endl << endl;

	cout << "[ 3rd layer: Convolutional layer filter ]" << endl;
	for (int i = 0; i < conv_filter2.size(); i++)
		cout << conv_filter2[i] << endl << endl;

	cout << "[ 3rd layer: Convolutional layer bias ]" << endl;
	cout << conv_bias2.transpose() << endl << endl;

	cout << "[ 3rd layer: Convolutional layer delta featuremap ]" << endl;
	for (int i = 0; i < conv_delta2.size(); i++)
		cout << conv_delta2[i] << endl << endl;

	cout << "[ 3rd layer: Convolutional layer delta filter ]" << endl;
	for (int i = 0; i < conv_del_filter2.size(); i++)
		cout << conv_del_filter2[i] << endl << endl;

	cout << "[ 3rd layer: Convolutional layer delta bias ]" << endl;
	cout << conv_del_bias2.transpose() << endl << endl;

	cout << "-----------------------------------------------------" << endl << endl;
	cout << "[ 4th layer: Pooling layer featuremap ]" << endl;
	for (int i = 0; i < pool_output2.size(); i++)
		cout << pool_output2[i] << endl << endl;

	cout << "[ 4th layer: Pooling layer delta featuremap ]" << endl;
	for (int i = 0; i < pool_delta2.size(); i++)
		cout << pool_delta2[i] << endl << endl;

	cout << "-----------------------------------------------------" << endl << endl;

	cout << "[ 5th layer: Fully connected layer sum ]" << endl;
	cout << dense_sum1.transpose() << endl << endl;
	cout << "[ 5th layer: Fully connected layer output ]" << endl;
	cout << dense_output1.transpose() << endl << endl;
	cout << "[ 5th layer: Fully connected layer weight ]" << endl;
	cout << dense_weight1 << endl << endl;
	cout << "[ 5th layer: Fully connected layer bias ]" << endl;
	cout << dense_bias1.transpose() << endl << endl;
	cout << "[ 5th layer: Fully connected layer delta ]" << endl;
	cout << dense_delta1.transpose() << endl << endl;
	cout << "[ 5th layer: Fully connected layer delta weight ]" << endl;
	cout << dense_del_weight1 << endl << endl;
	cout << "[ 5th layer: Fully connected layer delta bias ]" << endl;
	cout << dense_del_bias1.transpose() << endl << endl;

	cout << "-----------------------------------------------------" << endl << endl;
	cout << "[ 6th layer: Fully connected layer sum ]" << endl;
	cout << dense_sum2.transpose() << endl << endl;
	cout << "[ 6th layer: Fully connected layer output ]" << endl;
	cout << dense_output2.transpose() << endl << endl;
	cout << "[ 6th layer: Fully connected layer weight ]" << endl;
	cout << dense_weight2 << endl << endl;
	cout << "[ 6th layer: Fully connected layer bias ]" << endl;
	cout << dense_bias2.transpose() << endl << endl;
	cout << "[ 6th layer: Fully connected layer delta ]" << endl;
	cout << dense_delta2.transpose() << endl << endl;
	cout << "[ 6th layer: Fully connected layer delta weight ]" << endl;
	cout << dense_del_weight2 << endl << endl;
	cout << "[ 6th layer: Fully connected layer delta bias ]" << endl;
	cout << dense_del_bias2.transpose() << endl << endl;

	return 0;
}