#include "simple_nn.h"
using namespace std;
using namespace simple_nn;



int main()
{
	int n_img_train = 60000, n_img_test = 10000, img_size = 784;
	vector<Matrix> train_X, test_X;
	Vector train_Y, test_Y;

	ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
	ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
	ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
	ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

	vector<vector<int>> indices(6, vector<int>(16));
	indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
	indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
	indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
	indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
	indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
	indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };

	SimpleNN model;
	model.add(new Conv2D({ 28, 28, 1 }, { 5, 5, 6 }, 2, Init::UNIFORM));
	//model.add(new BatchNorm({ 28, 28, 6 }));
	model.add(new Activation(6, Activate::TANH));
	model.add(new Pool2D({ 28, 28, 6 }, { 2, 2, 6 }, 2, Pool::AVG));
	model.add(new Conv2D({ 14, 14, 6 }, { 5, 5, 16 }, 0, Init::UNIFORM, indices));
	//model.add(new BatchNorm({ 10, 10, 16 }));
	model.add(new Activation(16, Activate::TANH));
	model.add(new Pool2D({ 10, 10, 16 }, { 2, 2, 16 }, 2, Pool::AVG));
	model.add(new Dense(400, 120, Init::UNIFORM));
	//model.add(new BatchNorm({ 120, 120, 1 }));
	model.add(new Activation(1, Activate::TANH));
	model.add(new Dense(120, 84, Init::UNIFORM));
	//model.add(new BatchNorm({ 84, 84, 1 }));
	model.add(new Activation(1, Activate::TANH));
	model.add(new Dense(84, 10, Init::UNIFORM));
	//model.add(new BatchNorm({ 10, 10, 1 }));
	model.add(new Activation(1, Activate::TANH));
	model.add(new Output(10, Loss::MSE));

	int n_epoch = 30, batch = 20;
	double l_rate = 0.5, lambda = 0.0;

	model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);

	return 0;
}




//int main()
//{
//	int batch_size = 3, in_channels = 2, fmap_size = 3;
//	vector<vector<Matrix>> inputs(batch_size, vector<Matrix>(in_channels, Matrix(fmap_size, fmap_size)));
//
//	for (int n = 0; n < batch_size; n++)
//		for (int ch = 0; ch < in_channels; ch++)
//			for (int i = 0; i < fmap_size; i++)
//				for (int j = 0; j < fmap_size; j++)
//					inputs[n][ch](i, j) = (n + 1) * (i + j);
//
//	int filt_size = 2, pad = 0, out_channels = 3;
//	vector<Matrix> filters(out_channels, Matrix(filt_size, filt_size));
//	for (int ch = 0; ch < out_channels; ch++)
//		for (int i = 0; i < filt_size; i++)
//			for (int j = 0; j < filt_size; j++)
//				filters[ch](i, j) = (ch + 1) * (i + j);
//
//
//	int out_size = 2;
//	vector<vector<Matrix>> output(batch_size, vector<Matrix>(out_channels, Matrix(out_size, out_size)));
//
//	//#pragma omp parallel for
//	for (int n = 0; n < batch_size; n++)
//	{
//		for (int out_ch = 0; out_ch < out_channels; out_ch++)
//		{
//			for (int in_ch = 0; in_ch < in_channels; in_ch++)
//			{
//				if (in_ch == 0)
//					output[n][out_ch] = conv2d(inputs[n][in_ch], filters[out_ch], pad);
//				else
//					output[n][out_ch] += conv2d(inputs[n][in_ch], filters[out_ch], pad);
//			}
//		}
//	}
//
//	cout << "____________________________INPUT____________________________" << endl;
//	for (int n = 0; n < batch_size; n++)
//	{
//		cout << "Batch " << n + 1 << endl;
//		for (int in_ch = 0; in_ch < in_channels; in_ch++)
//		{
//			cout << "Channel " << in_ch + 1 << endl << endl;
//			cout << inputs[n][in_ch] << endl << endl;
//		}
//	}
//
//	cout << "____________________________FILTER____________________________" << endl;
//	for (int out_ch = 0; out_ch < out_channels; out_ch++)
//	{
//		cout << "Channel " << out_ch + 1 << endl << endl;
//		cout << filters[out_ch] << endl << endl;
//	}
//
//	cout << "____________________________OUTPUT____________________________" << endl;
//	for (int n = 0; n < batch_size; n++)
//	{
//		cout << "Batch " << n + 1 << endl;
//		for (int out_ch = 0; out_ch < out_channels; out_ch++)
//		{
//			cout << "Channel " << out_ch + 1 << endl << endl;
//			cout << output[n][out_ch] << endl << endl;
//		}
//	}
//
//	return 0;
//}

//int main()
//{
//	/*vector<vector<Matrix>> batch(2, vector<Matrix>(2, Matrix(3, 3)));
//
//	int k = 0;
//	for (int n = 0; n < 2; n++)
//		for (int ch = 0; ch < 2; ch++)
//			for (int i = 0; i < 3; i++)
//				for (int j = 0; j < 3; j++)
//					batch[n][ch](i, j) = k++;*/
//
//	Matrix img1(3, 3), img2(3, 3);
//	int a = 0, b = 9;
//	for (int i = 0; i < 3; i++)
//		for (int j = 0; j < 3; j++)
//		{
//			img1(i, j) = a++;
//			img2(i, j) = b++;
//		}
//
//	vector<Matrix> imgs;
//	imgs.push_back(img1);
//	imgs.push_back(img2);
//
//	Matrix f1(2, 2);
//	f1(0, 0) = 0;
//	f1(0, 1) = 1;
//	f1(1, 0) = -1;
//	f1(1, 1) = 0;
//
//	Matrix f2(2, 2);
//	f2(0, 0) = 0;
//	f2(0, 1) = -1;
//	f2(1, 0) = 1;
//	f2(1, 1) = 0;
//
//	vector<Matrix> fs;
//	fs.push_back(f1);
//	fs.push_back(f2);
//
//	clock_t start = clock();
//
//	int ch_size = 2, filt_size = 2, out_size = 2;
//	/*Matrix out(8, 4);
//	for (int ch = 0; ch < 2; ch++)
//	{
//		for (int i = 0; i < out_size; i++)
//			for (int j = 0; j < out_size; j++)
//			{
//				int r = i * out_size + j;
//				for (int x = 0; x < filt_size; x++)
//					for (int y = 0; y < filt_size; y++)
//					{
//						int c = x * filt_size + y;
//						int ii = i + x;
//						int jj = j + y;
//
//						out(r, c) = imgs[ch](ii, jj);
//					}
//			}
//	}*/
//
//	Matrix out(4, 8);
//	for (int i = 0; i < out_size; i++)
//		for (int j = 0; j < out_size; j++)
//		{
//			int r = i * out_size + j;
//			for (int ch = 0; ch < 2; ch++)
//			{
//				for (int x = 0; x < filt_size; x++)
//					for (int y = 0; y < filt_size; y++)
//					{
//						//int c = x * filt_size + y + pow(filt_size, 2) * ch;
//						int c = (x + ch * filt_size) * filt_size + y;
//						int ii = i + x;
//						int jj = j + y;
//
//						out(r, c) = imgs[ch](ii, jj);
//					}
//			}
//		}
//
//	/*cout << imgs[0] << endl << endl;
//	cout << imgs[1] << endl << endl;*/
//	cout << out << endl << endl;
//
//	Vector flat(8, 2);
//	for (int ch = 0; ch < 2; ch++)
//		for (int i = 0; i < 2; i++)
//			for (int j = 0; j < 2; j++)
//			{
//				flat[i * 2 + j] = fs[ch](i, j);
//			}
//
//	/*cout << fs[0] << endl << endl;
//	cout << fs[1] << endl << endl;*/
//	cout << flat << endl << endl;
//
//	cout << out * flat << endl << endl;
//
//
//	vector<Matrix> out2(2, Matrix(3, 3));
//	for (int ch = 0; ch < 2; ch++)
//		out2[ch] = conv2d(imgs[ch], fs[ch], 0);
//
//	cout << out2[0] << endl << endl;
//	cout << out2[1] << endl << endl;
//
//	/*Vector flat(4);
//	for (int i = 0; i < 2; i++)
//		for (int j = 0; j < 2; j++)
//			flat[i * 2 + j] = f(i, j);
//
//	out * flat;
//
//	clock_t end = clock();
//	cout << double(end - start) << endl;*/
//
//	/*cout << img << endl << endl;
//	cout << out << endl << endl;
//	cout << out * flat << endl << endl;
//	cout << conv2d(img, f, 0) << endl << endl;*/
//}