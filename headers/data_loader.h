#pragma once
#include "common.h"

namespace simple_nn
{
	class DataLoader
	{
	private:
		int n_batch;
		int batch;
		int ch;
		int h;
		int w;
		int chhw;
		MatXf X;
		VecXi Y;
		vector<vector<int>> batch_indices;
	public:
		DataLoader();
		DataLoader(MatXf& X, VecXi& Y, int batch, int channels,
					int height, int width, bool shuffle);
		void load(MatXf& X, VecXi& Y, int batch, int channels,
			int height, int width, bool shuffle);
		int size() const;
		vector<int> input_shape() const;
		MatXf get_x(int i) const;
		VecXi get_y(int i) const;
	private:
		void generate_batch_indices(bool shuffle);
	};

	DataLoader::DataLoader() :
		n_batch(0), batch(0), ch(0), h(0), w(0), chhw(0) {}

	DataLoader::DataLoader(
		MatXf& X,
		VecXi& Y,
		int batch,
		int channels,
		int height,
		int width,
		bool shuffle
	) :
		X(std::move(X)),
		Y(std::move(Y)),
		n_batch(0),
		batch(batch),
		ch(channels),
		h(height),
		w(width),
		chhw(channels * height * width)
	{
		n_batch = (int)this->X.rows() / ch / batch;
		generate_batch_indices(shuffle);
	}

	void DataLoader::load(MatXf& X, VecXi& Y, int batch, int channels,
		int height, int width, bool shuffle)
	{
		this->X = std::move(X);
		this->Y = std::move(Y);
		this->batch = batch;
		ch = channels;
		h = height;
		w = width;
		chhw = ch * h * w;
		n_batch = (int)this->X.rows() / ch / batch;
		generate_batch_indices(shuffle);
	}

	int DataLoader::size() const { return n_batch; }

	vector<int> DataLoader::input_shape() const { return { batch, ch, h, w }; }

	void DataLoader::generate_batch_indices(bool shuffle)
	{
		vector<int> rand_num(batch * n_batch);
		std::iota(rand_num.begin(), rand_num.end(), 0);

		if (shuffle) {
			unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(rand_num.begin(), rand_num.end(), std::default_random_engine(seed));
		}

		batch_indices.resize(n_batch, vector<int>(batch));
		for (int i = 0; i < n_batch; i++) {
			for (int j = 0; j < batch; j++) {
				batch_indices[i][j] = rand_num[i * batch + j];
			}
		}
	}

	MatXf DataLoader::get_x(int i) const
	{
		MatXf batch_x(batch * ch, h * w);
		for (int j = 0; j < batch; j++) {
			const float* first = X.data() + batch_indices[i][j] * chhw;
			const float* last = first + chhw;
			float* dest = batch_x.data() + j * chhw;
			std::copy(first, last, dest);
		}
		return batch_x;
	}

	VecXi DataLoader::get_y(int i) const
	{
		VecXi batch_y(batch);
		for (int j = 0; j < batch_indices[i].size(); j++) {
			batch_y[j] = Y[batch_indices[i][j]];
		}
		return batch_y;
	}
}