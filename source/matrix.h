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
		double* mat;
		int row;
		int col;
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
		Matrix element_wise(const Matrix& other);
		Matrix transpose() const;


		double& operator[](const int& i);
		const double& operator[](const int& i) const;
		double& operator()(const int& i, const int& j);
		const double& operator()(const int& i, const int& j) const;
		Matrix& operator=(const Matrix& other);

		friend Matrix operator+(const Matrix& one, const Matrix& other);
		friend Matrix operator+(const Matrix& mat, double num);
		friend Matrix operator+(double num, const Matrix& mat);
		friend Matrix operator-(const Matrix& one, const Matrix& other);
		friend Matrix operator-(const Matrix& mat, double num);
		friend Matrix operator-(double num, const Matrix& mat);
		friend Matrix operator*(const Matrix& one, const Matrix& other);
		friend Matrix operator*(double num, const Matrix& mat);
		friend Matrix operator*(const Matrix& mat, double num);
		friend void operator+=(const Matrix& one, const Matrix& other);
		friend void operator+=(const Matrix& mat, double num);
		friend void operator-=(const Matrix& one, const Matrix& other);
		friend void operator-=(const Matrix& mat, double num);
		friend void operator*=(const Matrix& one, const Matrix& other);
		friend void operator*=(const Matrix& mat, double num);

		friend ostream& operator<<(ostream& out, const Matrix& mat);
	};

	typedef Matrix Vector;

	Matrix::Matrix() : mat(nullptr), row(0), col(0), _size(0) {}

	Matrix::Matrix(int row) : row(row), col(1), _size(row) { mat = new double[row]; }

	Matrix::Matrix(int row, int col) : row(row), col(col), _size((__int64)row* col) { mat = new double[_size]; }

	Matrix::Matrix(const Matrix& other) : row(other.row), col(other.col), _size((__int64)row* col)
	{
		mat = new double[_size];
		std::copy(other.mat, other.mat + other._size, mat);
	}

	Matrix::~Matrix() { delete[] mat; }

	double* Matrix::begin() { return mat; }

	const double* Matrix::begin() const { return mat; }

	double* Matrix::end() { return mat + _size; }

	const double* Matrix::end() const { return mat + _size; }

	int Matrix::size() const { return _size; }

	int Matrix::rows() const { return row; }

	int Matrix::cols() const { return col; }

	void Matrix::resize(int size)
	{
		if (_size != size)
		{
			delete[] mat;
			mat = new double[size];
			_size = size;
			if (row == 0 && col == 0)
			{
				row = size;
				col = 1;
			}
			else
			{
				row = (row == 1) ? 1 : size;
				col = (col == 1) ? 1 : size;
			}
		}
	}

	void Matrix::resize(int row, int col)
	{
		if (this->row != row || this->col != col)
		{
			_size = row * col;
			delete[] mat;
			mat = new double[_size];
			this->row = row;
			this->col = col;
		}
	}

	void Matrix::setZero() { std::for_each(mat, mat + _size, [](double& elem) { elem = 0; }); }

	double Matrix::max() const { return *std::max_element(mat, mat + _size); }

	double Matrix::sum() const { return accumulate(mat, mat + _size, 0.0); }

	double Matrix::mean() const { return std::accumulate(mat, mat + _size, 0.0) / _size; }

	double Matrix::var(double mean) const
	{
		double var = std::accumulate(mat, mat + _size, 0.0, [=](double sum, double x) {
			return sum + (x - mean) * (x - mean);
		});
		return var / _size;
	}

	Matrix Matrix::element_wise(const Matrix& other)
	{
		if (row != other.row || col != other.col)
		{
			cout << "Matrix::element_wise(const Matrix&): Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(row, col);
		std::transform(mat, mat + _size, other.mat, out.mat, std::multiplies<double>());
		return out;
	}

	Matrix Matrix::transpose() const
	{
		Matrix out(col, row);
		if (row != 1 && col != 1)
		{
			for (int i = 0; i < row; i++)
				for (int j = 0; j < col; j++)
					out.mat[j * row + i] = mat[i * col + j];
		}
		else
		{
			std::copy(mat, mat + _size, out.mat);
		}
		return out;
	}

	double& Matrix::operator[](const int& i) { return mat[i]; }

	const double& Matrix::operator[](const int& i) const { return mat[i]; }

	double& Matrix::operator()(const int& i, const int& j) { return mat[i * col + j]; }

	const double& Matrix::operator()(const int& i, const int& j) const { return mat[i * col + j]; }

	Matrix& Matrix::operator=(const Matrix& other)
	{
		resize(other.row, other.col);
		_size = other._size;
		std::copy(other.mat, other.mat + other._size, mat);
		return *this;
	}

	Matrix operator+(const Matrix& one, const Matrix& other)
	{
		if (one.row != other.row || one.col != other.col)
		{
			cout << "Matrix::operator+: Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(one.row, one.col);
		transform(one.mat, one.mat + one._size, other.mat, out.mat, std::plus<double>());
		return out;
	}

	Matrix operator+(const Matrix& mat, double num)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, std::bind2nd(std::plus<double>(), num));
		return out;
	}

	Matrix operator+(double num, const Matrix& mat)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, std::bind2nd(std::plus<double>(), num));
		return out;
	}

	Matrix operator-(const Matrix& one, const Matrix& other)
	{
		if (one.row != other.row || one.col != other.col)
		{
			cout << "Matrix::operator-: Matrices are incompatible." << endl;
			exit(100);
		}
		Matrix out(one.row, one.col);
		transform(one.mat, one.mat + one._size, other.mat, out.mat, std::minus<double>());
		return out;
	}

	Matrix operator-(const Matrix& mat, double num)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, std::bind2nd(std::minus<double>(), num));
		return out;
	}

	Matrix operator-(double num, const Matrix& mat)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, [&num](const double& elem) {
			return num - elem;
		});
		return out;
	}

	Matrix operator*(const Matrix& one, const Matrix& other)
	{
		if (one.col != other.row)
		{
			cout << "Matrix::opertor*: Matrices are incompatible." << endl;
			exit(100);
		}
		else if (one.row == 1 && other.col == 1)
		{
			cout << "Matrix::operator*: Use dot() function." << endl;
			exit(100);
		}

		Matrix out(one.row, other.col);

		//#pragma omp parallel for
		for (int i = 0; i < one.row; i++)
			for (int j = 0; j < other.col; j++)
			{
				out.mat[i * other.col + j] = 0;
				for (int k = 0; k < one.col; k++)
					out.mat[i * other.col + j] += one.mat[i * one.col + k] * other.mat[k * other.col + j];
			}
		
		return out;
	}

	Matrix operator*(double num, const Matrix& mat)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, std::bind2nd(std::multiplies<double>(), num));
		return out;
	}

	Matrix operator*(const Matrix& mat, double num)
	{
		Matrix out(mat.row, mat.col);
		std::transform(mat.mat, mat.mat + mat._size, out.mat, std::bind2nd(std::multiplies<double>(), num));
		return out;
	}

	void operator+=(const Matrix& one, const Matrix& other)
	{
		if (one.row != other.row || one.col != other.col)
		{
			cout << "Matrix::operator+=: Matrices are incompatible." << endl;
			exit(100);
		}
		std::transform(one.mat, one.mat + one._size, other.mat, one.mat, std::plus<double>());
	}

	void operator+=(const Matrix& mat, double num)
	{
		std::for_each(mat.mat, mat.mat + mat._size, [&](double& elem) { elem += num; });
	}

	void operator-=(const Matrix& one, const Matrix& other)
	{
		if (one.row != other.row || one.col != other.col)
		{
			cout << "Matrix::operator-=: Matrices are incompatible." << endl;
			exit(100);
		}
		std::transform(one.mat, one.mat + one._size, other.mat, one.mat, std::minus<double>());
	}

	void operator-=(const Matrix& mat, double num)
	{
		std::for_each(mat.mat, mat.mat + mat._size, [&](double& elem) { elem -= num; });
	}

	void operator*=(const Matrix& one, const Matrix& other)
	{
		if (one.row != other.row || one.col != other.col)
		{
			cout << "Matrix::operator*=: Matrices are incompatible." << endl;
			exit(100);
		}
		std::transform(one.mat, one.mat + one._size, other.mat, one.mat, std::multiplies<double>());
	}

	void operator*=(const Matrix& mat, double num)
	{
		std::for_each(mat.mat, mat.mat + mat._size, [&](double& elem) { elem *= num; });
	}

	ostream& operator<<(ostream& out, const Matrix& mat)
	{
		if (mat.row == 1)
		{
			for (int i = 0; i < mat._size; i++)
				cout << std::setw(8) << mat.mat[i];
			cout << endl;
		}
		else if (mat.col == 1)
		{
			for (int i = 0; i < mat._size; i++)
				cout << std::setw(8) << mat.mat[i] << endl;
		}
		else
		{
			for (int i = 0; i < mat.row; i++)
			{
				for (int j = 0; j < mat.col; j++)
					cout << std::setw(8) << mat.mat[i * mat.col + j];
				cout << endl;
			}
		}
		return out;
	}
}