#pragma once
#include <iostream>
#include <algorithm>
using namespace std;

template<class T>
class vec
{
	int _size;
	T* _vec;
public:
	vec();
	vec(int size);
	vec(int size, const T& init);
	vec(initializer_list<T> init);
	vec(const vec<T>& other);
	~vec();
	int size() const;
	void resize(int size);
	void resize(int size, const T& init);
	vec<T> dot(const vec<T>& other);
	T* begin() const;
	T* end() const;

	T& operator[](const int& idx);
	T& operator[](const int& idx) const;
	vec<T>& operator=(const vec<T>& other);
};

template<class T>
vec<T>::vec() : _size(0), _vec(nullptr) {}

template<class T>
vec<T>::vec(int size) : _size(size) { _vec = new T[_size]; }

template<class T>
vec<T>::vec(int size, const T& init)
{
	_size = size;
	_vec = new T[_size];
	for (int i = 0; i < _size; i++)
		_vec[i] = init;
}

template<class T>
vec<T>::vec(initializer_list<T> init) : _size((int)init.size())
{
	_vec = new T[_size];
	copy(init.begin(), init.end(), _vec);
}

template<class T>
vec<T>::vec(const vec<T>& other) : _size(other._size)
{
	_vec = new T[_size];
	copy(other._vec, other._vec + _size, _vec);
}

template<class T>
vec<T>::~vec() { delete[] _vec; }

template<class T>
int vec<T>::size() const { return _size; }

template<class T>
void vec<T>::resize(int size)
{
	if (_size != size)
	{
		if (_size != 0)
			delete[] _vec;

		_size = size;
		_vec = new T[_size];
	}
}

template<class T>
void vec<T>::resize(int size, const T& init)
{
	if (_size != size)
	{
		if (_size != 0)
			delete[] _vec;

		_size = size;
		_vec = new T[_size];
		for (int i = 0; i < _size; i++)
			_vec[i] = init;
	}
}

template<class T>
vec<T> vec<T>::dot(const vec<T>& other)
{
	T sum = 0;
	for (int i = 0; i < _size; i++)
		sum += _vec[i] * other._vec[i];
	return sum;
}

template<class T>
T* vec<T>::begin() const { return _vec; }

template<class T>
T* vec<T>::end() const { return _vec + _size; }

template<class T>
T& vec<T>::operator[](const int& idx) { return _vec[idx]; }

template<class T>
T& vec<T>::operator[](const int& idx) const { return _vec[idx]; }

template<class T>
vec<T>& vec<T>::operator=(const vec<T>& other)
{
	resize(other._size);
	copy(other._vec, other._vec + _size, _vec);
	return *this;
}