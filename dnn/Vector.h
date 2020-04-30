#pragma once
#include <iostream>
#include <algorithm>
using namespace std;

template<class T>
class Vector
{
	int _size;
	T* vec;
public:
	Vector();
	Vector(int size);
	Vector(int size, const T& init);
	Vector(initializer_list<T> init);
	Vector(const Vector<T>& other);
	~Vector();
	int size() const;
	void resize(int size);
	T* begin() const;
	T* end() const;

	T& operator[](const int& idx);
	T& operator[](const int& idx) const;
	Vector<T>& operator=(const Vector<T>& other);
};

template<class T>
Vector<T>::Vector() : _size(0), vec(nullptr) {}

template<class T>
Vector<T>::Vector(int size) : _size(size) { vec = new T[_size]; }

template<class T>
Vector<T>::Vector(int size, const T& init)
{
	_size = size;
	vec = new T[_size];
	for (int i = 0; i < _size; i++)
		vec[i] = init;
}

template<class T>
Vector<T>::Vector(initializer_list<T> init) : _size((int)init.size())
{
	vec = new T[_size];
	copy(init.begin(), init.end(), vec);
}

template<class T>
Vector<T>::Vector(const Vector<T>& other) : _size(other._size)
{
	vec = new T[_size];
	copy(other.vec, other.vec + _size, vec);
}

template<class T>
Vector<T>::~Vector() { delete[] vec; }

template<class T>
int Vector<T>::size() const { return _size; }

template<class T>
void Vector<T>::resize(int size)
{
	if (_size != size)
	{
		if (_size != 0)
			delete[] vec;
		this->_size = size;
		vec = new T[_size];
	}
}

template<class T>
T* Vector<T>::begin() const { return vec; }

template<class T>
T* Vector<T>::end() const { return vec + _size; }

template<class T>
T& Vector<T>::operator[](const int& idx) { return vec[idx]; }

template<class T>
T& Vector<T>::operator[](const int& idx) const { return vec[idx]; }

template<class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& other)
{
	resize(other._size);
	copy(other.vec, other.vec + _size, vec);
	return *this;
}