#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <numeric>
#include "Matrix.h"

//------------------------------- function declaration -------------------------------

void init_weight_normal(Matrix& W, Vector& b, int n_in, int n_out);

void init_weight_uniform(Matrix& W, Vector& b, int n_in);

void init_filter_normal(vector<Matrix>& Ws, Vector& b, int n_in, int n_out);

void init_filter_uniform(vector<Matrix>& Ws, Vector& b, int n_in);

int calc_outsize(int in_size, int filt_size, int stride, int pad);

Matrix conv2d(const Matrix& img, const Matrix& filt, int pad);

Matrix pool2d(const Matrix& img, int filt, int stride, int opt = 0);

// delta, filt, stride는 next layer의 delta, filt, stride이다.
Matrix delta_img_max(const Matrix& img, const Matrix& delta, int filt, int stride);

// delta, filt, stride는 next layer의 delta, filt, stride이다.
Matrix delta_img_avg(int img_size, const Matrix& delta, int filt, int stride);

double tan_h(double x);

double tanh_prime(double x);

double relu(double x);

double relu_prime(double x);

Matrix activate(const Matrix& sum, int opt = 0);

Matrix activate_prime(const Matrix& sum, int opt = 0);

Vector flatten(const vector<Matrix>& imgs);

vector<Matrix> unflatten(const Vector& input, int n_img, int img_size);

Matrix rotate_180(const Matrix& filt);

Vector as_vector(double num, int size);

int calc_error(double expected, const Vector& predicted);

double calc_error(const Vector& expected, const Vector& predicted);

int max_idx(const Vector& vec);


//------------------------------- function definition -------------------------------


void init_weight_normal(Matrix& W, Vector& b, int n_in, int n_out)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double var = sqrt(2 / (double)(n_in + n_out));
	normal_distribution<double> dist(0, var);

	for (int i = 0; i < W.rows(); i++)
	{
		b[i] = dist(e);
		for (int j = 0; j < W.cols(); j++)
			W(i, j) = dist(e);
	}
}

void init_weight_uniform(Matrix& W, Vector& b, int n_in)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double lower = -2.4 / n_in, upper = 2.4 / n_in;
	uniform_real_distribution<double> dist(lower, upper);

	for (int i = 0; i < W.rows(); i++)
	{
		b[i] = dist(e);
		for (int j = 0; j < W.cols(); j++)
			W(i, j) = dist(e);
	}
}

void init_filter_normal(vector<Matrix>& Ws, Vector& b, int n_in, int n_out)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double var = sqrt(2 / (double)(n_in + n_out));
	normal_distribution<double> dist(0, var);

	for (int n = 0; n < Ws.size(); n++)
	{
		b[n] = dist(e);
		for (int i = 0; i < Ws[n].rows(); i++)
			for (int j = 0; j < Ws[n].cols(); j++)
				Ws[n](i, j) = dist(e);
	}
}

void init_filter_uniform(vector<Matrix>& Ws, Vector& b, int n_in)
{
	unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
	default_random_engine e(seed);

	double lower = -2.4 / n_in, upper = 2.4 / n_in;
	uniform_real_distribution<double> dist(lower, upper);

	for (int n = 0; n < Ws.size(); n++)
	{
		b[n] = dist(e);
		for (int i = 0; i < Ws[n].rows(); i++)
			for (int j = 0; j < Ws[n].cols(); j++)
				Ws[n](i, j) = dist(e);
	}
}

int calc_outsize(int in_size, int filt_size, int stride, int pad)
{
	return (int)floor((in_size + 2 * pad - filt_size) / stride) + 1;
}

Matrix conv2d(const Matrix& img, const Matrix& filt, int pad)
{
	int out = calc_outsize(img.rows(), filt.rows(), 1, pad);
	Matrix output(out, out);
	for (int i = 0; i < out; i++)
		for (int j = 0; j < out; j++)
		{
			output(i, j) = 0;
			for (int x = 0; x < filt.rows(); x++)
				for (int y = 0; y < filt.cols(); y++)
				{
					int ii = i + x - pad;
					int jj = j + y - pad;

					if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
						output(i, j) += img(ii, jj) * filt(x, y);
				}
		}
	return output;
}

Matrix pool2d(const Matrix& img, int filt, int stride, int opt)
{
	int out = calc_outsize(img.rows(), filt, stride, 0);
	Matrix output(out, out);
	for (int i = 0; i < out; i++)
		for (int j = 0; j < out; j++)
		{
			vector<double> temp(filt * filt);
			for (int x = 0; x < filt; x++)
				for (int y = 0; y < filt; y++)
				{
					int ii = i + x;
					int jj = j + y;

					if (i != 0)
						ii += (stride - 1);
					if (j != 0)
						jj += (stride - 1);

					if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
						temp[x * filt + y] = img(ii, jj);
				}
			if (opt == 0)
				output(i, j) = *max_element(temp.begin(), temp.end());
			else
				output(i, j) = accumulate(temp.begin(), temp.end(), 0.0) / (double)temp.size();
		}
	return output;
}

Matrix delta_img_max(const Matrix& img, const Matrix& delta, int filt, int stride)
// delta, filt, stride는 next layer의 delta, filt, stride이다.
// out은 next layer의 output의 크기이다.
{
	Matrix output(img.rows(), img.cols());
	output.setZero();

	int out = calc_outsize(img.rows(), filt, stride, 0);
	for (int i = 0; i < out; i++)
		for (int j = 0; j < out; j++)
		{
			int ii = 0, jj = 0;
			vector<double> temp;
			for (int x = 0; x < filt; x++)
				for (int y = 0; y < filt; y++)
				{
					ii = i + x;
					jj = j + y;

					if (i != 0)
						ii += (stride - 1);
					if (j != 0)
						jj += (stride - 1);

					if (ii >= 0 && ii < img.rows() && jj >= 0 && jj < img.cols())
						temp.push_back(img(ii, jj));
				}
			int max = distance(temp.begin(), max_element(temp.begin(), temp.end()));
			ii = i + max / filt;
			jj = j + max % filt;

			if (i != 0)
				ii += (stride - 1);
			if (j != 0)
				jj += (stride - 1);

			output(ii, jj) = delta(i, j);
		}
	return output;
}

Matrix delta_img_avg(int img_size, const Matrix& delta, int filt, int stride)
// img_size는 this layer의 output의 크기이다.
// delta, filt, stride는 next layer의 delta, filt, stride이다.
// out은 next layer의 output의 크기이다.
{
	Matrix output(img_size, img_size);
	
	int out = calc_outsize(img_size, filt, stride, 0);
	for (int i = 0; i < out; i++)
		for (int j = 0; j < out; j++)
			for (int x = 0; x < filt; x++)
				for (int y = 0; y < filt; y++)
				{
					int ii = i + x;
					int jj = j + y;

					if (i != 0)
						ii += (stride - 1);
					if (j != 0)
						jj += (stride - 1);

					if (ii >= 0 && ii < delta.rows() && jj >= 0 && jj < delta.cols())
						delta(ii, jj) = delta(i, j) / pow((double)filt, 2);
				}
	return output;
}

double tan_h(double x) { return 2 / (1 + exp(-x)) - 1; }

double tanh_prime(double x) { return 0.5 * (1 - pow(tan_h(x), 2)); }

double relu(double x) { return max(0.0, x); }

double relu_prime(double x) { return (x <= 0) ? 0 : 1; }

Matrix activate(const Matrix& sum, int opt)
{
	Matrix output(sum.rows(), sum.cols());
	for (int i = 0; i < sum.rows(); i++)
		for (int j = 0; j < sum.cols(); j++)
		{
			if (opt == 0)
				output(i, j) = tan_h(sum(i, j));
			else
				output(i, j) = relu(sum(i, j));
		}
	return output;
}

Matrix activate_prime(const Matrix& sum, int opt)
{
	Matrix output(sum.rows(), sum.cols());
	for (int i = 0; i < sum.rows(); i++)
		for (int j = 0; j < sum.cols(); j++)
		{
			if (opt == 0)
				output(i, j) = tanh_prime(sum(i, j));
			else
				output(i, j) = relu_prime(sum(i, j));
		}
	return output;
}

Vector flatten(const vector<Matrix>& imgs)
{
	Vector flattened(imgs.size() * imgs[0].size());

	int k = 0;
	for (const auto& img : imgs)
		for (int i = 0; i < img.rows(); i++)
			for (int j = 0; j < img.cols(); j++)
				flattened[k++] = img(i, j);
	return flattened;
}

vector<Matrix> unflatten(const Vector& input, int n_img, int img_size)
{
	vector<Matrix> imgs(n_img, Matrix(img_size, img_size));
	for (int n = 0; n < n_img; n++)
		for (int i = 0; i < img_size; i++)
			for (int j = 0; j < img_size; j++)
			{
				int idx = i * img_size + j + (n * imgs[n].size());
				imgs[n](i, j) = input[idx];
			}
	return imgs;
}

Matrix rotate_180(const Matrix& filt)
{
	int r = filt.rows(), c = filt.cols();
	Matrix rotate(r, c);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			rotate(i, j) = filt(r - i - 1, c - j - 1);
	return rotate;
}

Vector as_vector(double num, int size)
{
	Vector output(size);
	output.setZero();
	output[(int)num] = 1;
	return output;
}

int calc_error(double expected, const Vector& predicted)
{
	double max = (double)max_idx(predicted);
	return (expected != max) ? 1 : 0;
}

double calc_error(const Vector& expected, const Vector& predicted)
{
	int error = 0;
	for (int i = 0; i < expected.size(); i++)
		if (expected[i] != predicted[i])
			error++;
	return error / (double)expected.size();
}

int max_idx(const Vector& vec)
{
	int max_i = 0;
	double max = 0;
	for (int i = 0; i < vec.size(); i++)
		if (max < vec[i])
		{
			max = vec[i];
			max_i = i;
		}
	return max_i;
}