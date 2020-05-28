#pragma once
#include <iostream>
#include <iomanip>
using namespace std;

class Matrix
{
private:
	double** mat;
	int row;
	int col;
public:
	Matrix();
	Matrix(int row);
	Matrix(int row, int col);
	Matrix(const Matrix& other);
	~Matrix();
	int size() const;
	int rows() const;
	int cols() const;
	void resize(int size);
	void resize(int row, int col);
	void setZero();
	double sum();
	void element_wise(const Matrix& other);
	Matrix transpose() const;


	double& operator[](const int& i);
	double& operator[](const int& i) const;
	double& operator()(const int& i, const int& j);
	double& operator()(const int& i, const int& j) const;
	Matrix& operator=(const Matrix& other);

	Matrix operator+(const Matrix& other);
	Matrix operator-(const Matrix& other);
	Matrix operator*(const Matrix& other);
	void operator+=(const Matrix& other);
	void operator+=(double num);
	void operator-=(const Matrix& other);

	friend Matrix operator*(double num, Matrix& mat);
	friend ostream& operator<<(ostream& out, const Matrix& mat);
};

typedef Matrix Vector;

Matrix operator*(double num, Matrix& mat)
{
	Matrix mult(mat.row, mat.col);
	for (int i = 0; i < mat.row; i++)
		for (int j = 0; j < mat.col; j++)
			mult.mat[i][j] = mat.mat[i][j] * num;
	return mult;
}

ostream& operator<<(ostream& out, const Matrix& mat)
{
	for (int i = 0; i < mat.row; i++)
	{
		for (int j = 0; j < mat.col; j++)
			out << setw(10) << setprecision(8) << mat.mat[i][j];
		if (i != mat.row - 1)
			out << endl;
	}

	return out;
}

namespace matrix
{
	void allocate_memory(double**& mat, int row, int col)
	{
		mat = new double* [row];
		for (int i = 0; i < row; i++)
			mat[i] = new double[col];
	}

	void deallocate_memory(double**& mat, int row)
	{
		for (int i = 0; i < row; i++)
			delete[] mat[i];
		delete[] mat;
	}

	void initialize_matrix(double**& mat, int row, int col, int init)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				mat[i][j] = init;
	}
}

Matrix::Matrix() : row(0), col(0), mat(nullptr) {}

Matrix::Matrix(int row) : row(row), col(1) { matrix::allocate_memory(mat, row, col); }

Matrix::Matrix(int row, int col) : row(row), col(col) { matrix::allocate_memory(mat, row, col); }

Matrix::Matrix(const Matrix& other) : row(other.row), col(other.col)
{
	matrix::allocate_memory(mat, row, col);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] = other.mat[i][j];
}

Matrix::~Matrix() { matrix::deallocate_memory(mat, row); }

int Matrix::size() const { return row * col; }

int Matrix::rows() const { return row; }

int Matrix::cols() const { return col; }

void Matrix::resize(int size)
{
	if (col > 1)
		cout << "Matrix::resize(int): Not a matrix function." << endl;
	else
	{
		matrix::deallocate_memory(mat, row);
		matrix::allocate_memory(mat, size, 1);
		row = size;
		col = 1;
	}
}

void Matrix::resize(int row, int col)
{
	if (this->row != row || this->col != col)
	{
		matrix::deallocate_memory(mat, this->row);
		matrix::allocate_memory(mat, row, col);
		this->row = row;
		this->col = col;
	}
}

void Matrix::setZero()
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] = 0;
}

double Matrix::sum()
{
	double sum = 0.0;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			sum += mat[i][j];
	return sum;
}

void Matrix::element_wise(const Matrix& other)
{
	if (row != other.row || col != other.col)
	{
		cout << "Matrix::element_wise(): Matrices are incompatible." << endl;
		return;
	}
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] *= other.mat[i][j];
}

Matrix Matrix::transpose() const
{
	Matrix transposed(col, row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			transposed.mat[j][i] = mat[i][j];
	return transposed;
}

double& Matrix::operator[](const int& i) { return mat[i][0]; }

double& Matrix::operator[](const int& i) const { return mat[i][0]; }

double& Matrix::operator()(const int& i, const int& j) { return mat[i][j]; }

double& Matrix::operator()(const int& i, const int& j) const { return mat[i][j]; }

Matrix& Matrix::operator=(const Matrix& other)
{
	resize(other.row, other.col);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] = other.mat[i][j];
	return *this;
}

Matrix Matrix::operator+(const Matrix& other)
{
	Matrix add(row, col);
	if (row != other.row || col != other.col)
		cout << "Matrix::operator+: Matrices are incompatible." << endl;
	else
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				add.mat[i][j] = mat[i][j] + other.mat[i][j];
	}
	return add;
}

Matrix Matrix::operator-(const Matrix& other)
{
	Matrix sub(row, col);
	if (row != other.row || col != other.col)
		cout << "Matrix::operator-: Matrices are incompatible." << endl;
	else
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				sub.mat[i][j] = mat[i][j] - other.mat[i][j];
	}
	return sub;
}

Matrix Matrix::operator*(const Matrix& other)
{
	Matrix mult(row, other.col);
	if (col != other.row)
		cout << "Matrix::opertor*: Matrices are incompatible." << endl;
	else
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < other.col; j++)
			{
				mult.mat[i][j] = 0;
				for (int k = 0; k < col; k++)
					mult.mat[i][j] += mat[i][k] * other.mat[k][j];
			}
	}
	return mult;
}

void Matrix::operator+=(const Matrix& other)
{
	if (row != other.row || col != other.col)
		cout << "Matrix::operator+=: Matrices are incompatible." << endl;
	else
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				mat[i][j] += other.mat[i][j];
	}
}

void Matrix::operator+=(double num)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] += num;
}

void Matrix::operator-=(const Matrix& other)
{
	if (row != other.row || col != other.col)
		cout << "Matrix::operator-=: Matrices are incompatible." << endl;
	else
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				mat[i][j] -= other.mat[i][j];
	}
}