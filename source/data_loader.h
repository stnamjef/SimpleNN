#pragma once
#include "common.h"

namespace simple_nn
{
	class DataLoader
	{
	private:
		int n_batch;
		int batch;
		int channels;
		int height;
		int width;
		MatXf X;
		VecXi Y;
		vector<vector<int>> batch_indices;
	public:
		DataLoader(MatXf& X, VecXi& Y, int batch, int channels,
					int height, int width, bool shuffle = true);
		int size() const;
		vector<int> input_shape() const;
		MatXf get_x(int i) const;
		VecXi get_y(int i) const;
	private:
		void generate_batch_indices(bool shuffle);
	};

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
		channels(channels),
		height(height),
		width(width)
	{
		int n_img = (int)this->X.size() / (channels * height * width);
		n_batch = n_img / batch;
		generate_batch_indices(shuffle);
	}

	int DataLoader::size() const { return n_batch; }

	vector<int> DataLoader::input_shape() const { return { batch, channels, height, width }; }

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
		MatXf batch_x(batch * channels, height * width);
		for (int j = 0; j < batch; j++) {
			for (int c = 0; c < channels; c++) {
				batch_x.row(c + channels * j) = X.row(c + channels * batch_indices[i][j]);
			}
		}
		return batch_x;

		/*MatXf batch_x(batch, channels * height * width);
		for (int j = 0; j < batch_indices[i].size(); j++) {
			batch_x.row(j) = X.row(batch_indices[i][j]);
		}
		return batch_x;*/
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