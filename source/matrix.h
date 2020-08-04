#pragma once
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iomanip>
using namespace std;

namespace simple_nn
{
	class Matrix
	{
	private:
		double* _data;
		int _row;
		int _col;
		int _size;
	public:
		Matrix();
		Matrix(int row);
		Matrix(int row, int col);
		Matrix(const Matrix& other);
		~Matrix();
		double* begin();
		const double* begin() const;
		double* end();
		const double* end() const;
		int size() const;
		int rows() const;
		int cols() const;
		void resize(int size);
		void resize(int row, int col);
		void setZero();
		double max() const;
		double sum() const;
		double mean() const;
		double var(double mean) const;
		Matrix pow(int num) const;
		Matrix sqrt() const;
		Matrix elem_wise_mult(const Matrix& other) const;
		Matrix transpose() const;
		double& operator()(const int& i);
		const double& operator()(const int& i) const;
		double& operator()(const int& i, const int& j);
		const double& operator()(const int& i, const int& j) const;
		Matrix& operator=(const Matrix& other);

		friend Matrix operator+(const Matrix& one, const Matrix& other);
		friend Matrix operator+(const Matrix& mat, double num);
		friend Matrix operator+(double num, const Matrix& mat);
		friend Matrix operator-(const Matrix& one, const Matrix& other);
		friend Matrix operator-(const Matrix& mat, double num);
		friend Matrix operator*(const Matrix& one, const Matrix& other);
		friend Matrix operator*(double num, const Matrix& mat);
		friend Matrix operator*(const Matrix& mat, double num);
		friend Matrix operator/(const Matrix& one, const Matrix& other);
		friend Matrix operator/(const Matrix& mat, double num);
		friend Matrix operator/(double num, const Matrix& mat);
		friend void operator+=(const Matrix& one, const Matrix& other);
		friend void operator+=(const Matrix& mat, double num);
		friend void operator/=(const Matrix & one, const Matrix& other);
		friend void operator/=(const Matrix& mat, double num);
		friend ostream& operator<<(ostream& out, const Matrix& mat);
	};

	Matrix::Matrix() : 
		_data(nullptr), 
		_row(0), 
		_col(0), 
		_size(0) {}

	Matrix::Matrix(int row) : 
		_row(row),
		_col(1), 
		_size(row) { _data = new double[_size]; }

	Matrix::Matrix(int row, int col) : 
		_row(row),
		_col(col), 
		_size(row* col) { _data = new double[_size]; }

	Matrix::Matrix(const Matrix& other) : 
		_row(other._row),
		_col(other._col),
		_size(other._size)
	{
		_data = new double[_size];
		std::copy(other._data, other._data + other._size, _data);
	}

	Matrix::~Matrix() { delete[] _data; }

	double* Matrix::begin() { return _data; }

	const double* Matrix::begin() const { return _data; }

	double* Matrix::end() { return _data + _size; }

	const double* Matrix::end() const { return _data + _size; }

	int Matrix::size() const { return _size; }

	int Matrix::rows() const { return _row; }

	int Matrix::cols() const { return _col; }

	void Matrix::resize(int size)
	{
		if (_size != size) {
			delete[] _data;
			_data = new double[size];
			_size = size;
			if (_row == 0 && _col == 0) {
				_row = size;
				_col = 1;
			}
			else {
				_row = (_row == 1) ? 1 : size;
				_col = (_col == 1) ? 1 : size;
			}
		}
	}

	void Matrix::resize(int row, int col)
	{
		if (_row != row || _col != col) {
			_size = row * col;
			delete[] _data;
			_data = new double[_size];
			_row = row;
			_col = col;
		}
	}

	void Matrix::setZero() { std::for_each(_data, _data + _size, [](double& elem) { elem = 0; }); }

	double Matrix::max() const { return *std::max_element(_data, _data + _size); }

	double Matrix::sum() const { return std::accumulate(_data, _data + _size, 0.0); }

	double Matrix::mean() const { return std::accumulate(_data, _data + _size, 0.0) / _size; }

	double Matrix::var(double mean) const
	{
		double var = std::accumulate(_data, _data + _size, 0.0, [=](double sum, double x) {
			return sum + (x - mean) * (x - mean);
		});
		return var / _size;
	}

	Matrix Matrix::pow(int num) const
	{
		Matrix out(_row, _col);
		std::transform(_data, _data + _size, out._data, [&](const double& elem) { return std::pow(elem, num); });
		return out;
	}

	Matrix Matrix::sqrt() const
	{
		Matrix out(_row, _col);
		std::transform(_data, _data + _size, out._data, [](const double& elem) { return std::sqrt(elem); });
		return out;
	}

	Matrix Matrix::elem_wise_mult(const Matrix& other) const
	{
		if (_row != other._row || _col != other._col) {
			cout << "Matrix::elem_wise_mult(const Matrix&): Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(_row, _col);
		std::transform(_data, _data + _size, other._data, out._data, std::multiplies<double>());
		return out;
	}

	Matrix Matrix::transpose() const
	{
		Matrix out(_col, _row);
		if (_row > 1 && _col > 1) {
			for (int i = 0; i < _row; i++) {
				for (int j = 0; j < _col; j++) {
					out._data[i + j * _row] = _data[j + i * _col];
				}
			}
		}
		else {
			std::copy(_data, _data + _size, out._data);
		}
		return out;
	}

	double& Matrix::operator()(const int& i) { return _data[i]; }

	const double& Matrix::operator()(const int& i) const { return _data[i]; }

	double& Matrix::operator()(const int& i, const int& j) { return _data[i * _col + j]; }

	const double& Matrix::operator()(const int& i, const int& j) const { return _data[i * _col + j]; }

	Matrix& Matrix::operator=(const Matrix& other)
	{
		resize(other._row, other._col);
		_size = other._size;
		std::copy(other._data, other._data + other._size, _data);
		return *this;
	}

	Matrix operator+(const Matrix& one, const Matrix& other)
	{
		if (one._row != other._row || one._col != other._col) {
			cout << "Matrix::operator+: Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(one._row, one._col);
		transform(one._data, one._data + one._size, other._data, out._data, std::plus<double>());
		return out;
	}

	Matrix operator+(const Matrix& mat, double num)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::plus<double>(), num));
		return out;
	}

	Matrix operator+(double num, const Matrix& mat)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::plus<double>(), num));
		return out;
	}

	Matrix operator-(const Matrix& one, const Matrix& other)
	{
		if (one._row != other._row || one._col != other._col) {
			cout << "Matrix::operator-: Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(one._row, one._col);
		transform(one._data, one._data + one._size, other._data, out._data, std::minus<double>());
		return out;
	}

	Matrix operator-(const Matrix& mat, double num)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::minus<double>(), num));
		return out;
	}

	Matrix operator*(const Matrix& one, const Matrix& other)
	{
		if (one._col != other._row) {
			cout << "Matrix::opertor*: Matrices are incompatible." << endl;
			exit(100);
		}
		else if (one._row == 1 && other._col == 1) {
			cout << "Matrix::operator*: Use dot() function." << endl;
			exit(100);
		}
		Matrix out(one._row, other._col);
		out.setZero();
		for (int i = 0; i < one._row; i++) {
			for (int k = 0; k < one._col; k++) {
				double temp = one._data[i * one._col + k];
				for (int j = 0; j < other._col; j++) {
					out._data[i * other._col + j] += temp * other._data[k * other._col + j];
				}
			}
		}
		return out;
	}

	Matrix operator*(double num, const Matrix& mat)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::multiplies<double>(), num));
		return out;
	}

	Matrix operator*(const Matrix& mat, double num)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::multiplies<double>(), num));
		return out;
	}

	Matrix operator/(const Matrix& one, const Matrix& other)
	{
		if (one._row != other._row || one._col != other._col)
		{
			cout << "Matrix::operator/: Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(one._row, one._col);
		std::transform(one._data, one._data + one._size, other._data, out._data, std::divides<double>());
		return out;
	}

	Matrix operator/(const Matrix& mat, double num)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind2nd(std::divides<double>(), num));
		return out;
	}

	Matrix operator/(double num, const Matrix& mat)
	{
		Matrix out(mat._row, mat._col);
		std::transform(mat._data, mat._data + mat._size, out._data, std::bind1st(std::divides<double>(), num));
		return out;
	}

	void operator+=(const Matrix& one, const Matrix& other)
	{
		if (one._row != other._row || one._col != other._col)
		{
			cout << "Matrix::operator+=: Matrices are incompatible." << endl;
			exit(100);
		}
		std::transform(one._data, one._data + one._size, other._data, one._data, std::plus<double>());
	}

	void operator+=(const Matrix& mat, double num)
	{
		std::for_each(mat._data, mat._data + mat._size, [&](double& elem) { elem += num; });
	}

	void operator/=(const Matrix& one, const Matrix& other)
	{
		if (one._row != other._row || one._col != other._col)
		{
			cout << "Matrix::operator+=: Matrices are incompatible." << endl;
			exit(100);
		}
		std::transform(one._data, one._data + one._size, other._data, one._data, std::divides<double>());
	}

	void operator/=(const Matrix& mat, double num)
	{
		std::for_each(mat._data, mat._data + mat._size, [&](double& elem) { elem /= num; });
	}

	ostream& operator<<(ostream& out, const Matrix& mat)
	{
		if (mat._row == 1)
		{
			for (int i = 0; i < mat._size; i++)
				cout << std::setw(8) << mat._data[i];
			cout << endl;
		}
		else if (mat._col == 1)
		{
			for (int i = 0; i < mat._size; i++)
				cout << std::setw(8) << mat._data[i] << endl;
		}
		else
		{
			for (int i = 0; i < mat._row; i++)
			{
				for (int j = 0; j < mat._col; j++)
					cout << std::setw(8) << mat._data[i * mat._col + j];
				cout << endl;
			}
		}
		return out;
	}
}