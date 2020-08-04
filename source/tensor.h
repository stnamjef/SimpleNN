#pragma once
#include "matrix.h"

namespace simple_nn
{
	typedef Matrix Vector;

	class Tensor
	{
	private:
		Matrix** _data;
		int _batches;
		int _channels;
		int _height;
		int _width;
	public:
		Tensor();
		Tensor(int batches, int channels, int size);
		Tensor(int batches, int channels, int height, int width);
		Tensor(const Tensor& other);
		~Tensor();
		int batches() const;
		int channels() const;
		int height() const;
		int width() const;
		void resize(int batches, int channels, int size);
		void resize(int batches, int channels, int height, int width);
		Matrix*& operator[](const int& i);
		Matrix*& operator[](const int& i) const;
		Tensor& operator=(const Tensor& other);
	private:
		void allocate_memory(int batches, int channels, int height, int width);
		void deallocate_memory(int batches);
	};

	Tensor::Tensor() :
		_data(nullptr),
		_batches(0),
		_channels(0),
		_height(0),
		_width(0) {}

	Tensor::Tensor(int batches, int channels, int size) :
		_data(nullptr),
		_batches(batches),
		_channels(channels),
		_height(size),
		_width(1)
	{
		allocate_memory(_batches, _channels, _height, _width);
	}

	Tensor::Tensor(int batches, int channels, int height, int width) :
		_data(nullptr),
		_batches(batches),
		_channels(channels),
		_height(height),
		_width(width)
	{
		allocate_memory(_batches, _channels, _height, _width);
	}
	Tensor::Tensor(const Tensor& other) :
		_data(nullptr),
		_batches(other._batches),
		_channels(other._channels),
		_height(other._height),
		_width(other._width)
	{
		allocate_memory(_batches, _channels, _height, _width);
		for (int n = 0; n < _batches; n++) {
			for (int c = 0; c < _channels; c++) {
				_data[n][c] = other._data[n][c];
			}
		}
	}

	Tensor::~Tensor() { deallocate_memory(_batches); }

	int Tensor::batches() const { return _batches; }

	int Tensor::channels() const { return _channels; }

	int Tensor::height() const { return _height; }

	int Tensor::width() const { return _width; }

	void Tensor::resize(int batches, int channels, int size)
	{
		if (_batches != batches || _channels != channels ||
			_height != size || _width != 1) {
			deallocate_memory(_batches);
			allocate_memory(batches, channels, size, 1);
			_batches = batches;
			_channels = channels;
			_height = size;
			_width = 1;
		}
	}

	void Tensor::resize(int batches, int channels, int height, int width)
	{
		if (_batches != batches || _channels != channels ||
			_height != height || _width != width) {
			deallocate_memory(_batches);
			allocate_memory(batches, channels, height, width);
			_batches = batches;
			_channels = channels;
			_height = height;
			_width = width;
		}
	}

	Matrix*& Tensor::operator[](const int& i) { return _data[i]; }

	Matrix*& Tensor::operator[](const int& i) const { return _data[i]; }

	Tensor& Tensor::operator=(const Tensor& other)
	{
		resize(other._batches, other._channels, other._height, other._width);
		for (int n = 0; n < _batches; n++) {
			for (int c = 0; c < _channels; c++) {
				_data[n][c] = other._data[n][c];
			}
		}
		return *this;
	}

	void Tensor::allocate_memory(int batches, int channels, int height, int width)
	{
		_data = new Matrix * [batches];
		for (int n = 0; n < batches; n++) {
			_data[n] = new Matrix[channels];
			for (int c = 0; c < channels; c++) {
				_data[n][c] = Matrix(height, width);
			}
		}
	}

	void Tensor::deallocate_memory(int batches)
	{
		for (int n = 0; n < batches; n++) {
			delete[] _data[n];
		}
		delete[] _data;
	}
}