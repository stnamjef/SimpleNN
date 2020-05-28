//#include "CNN.h"
//#include "file_manage.h"
//using namespace std;
//
//int main()
//{
//	int n_img = 3000, img_size = 784;
//
//	vector<Matrix> X;
//	Vector Y;
//	
//	ReadMNIST(n_img, img_size, X);
//	ReadMNISTLabel(n_img, Y);
//
//	vector<vector<int>> indices(6, vector<int>(16));
//	indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
//	indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
//	indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
//	indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
//	indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
//	indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };
//
//	Conv2D first({ 28, 28, 1 }, { 5, 5, 6 }, 2);
//	Pool2D second({ 28, 28, 6 }, { 2, 2, 6 }, 2);
//	Conv2D third({ 14, 14, 16 }, { 5, 5, 16 }, 0);
//	Pool2D fourth({ 10, 10, 16 }, { 2, 2, 16 }, 2);
//	Dense fifth(400, 120);
//	Dense sixth(120, 84);
//	Dense seventh(84, 10);
//
//	vector<double> training_errors;
//	int n_epoch = 1, batch = 30;
//	double l_rate = 0.01;
//
//	for (int epoch = 0; epoch < n_epoch; epoch++)
//	{
//		double error = 0;
//		for (int n = 0; n < X.size(); n++)
//		{
//			for (int i = 0; i < first.output.size(); i++)
//			{
//				first.sum[i] = conv2d(X[n], first.Ws[i], first.pad);
//				first.sum[i] += first.b[i];
//				first.output[i] = activate(first.sum[i]);
//			}
//
//			for (int i = 0; i < second.output.size(); i++)
//				second.output[i] = pool2d(first.output[i], second.filt_size, second.stride, 1);
//
//			for (int i = 0; i < third.output.size(); i++)
//			{
//				third.sum[i].setZero();
//				for (int c = 0; c < indices[0].size(); c++)
//				{
//					for (int r = 0; r < indices.size(); r++)
//						if (indices[r][c] != 0)
//							third.sum[i] += conv2d(second.output[r], third.Ws[i], third.pad);
//				}
//				third.sum[i] += third.b[i];
//				third.output[i] = activate(third.sum[i]);
//			}
//
//			for (int i = 0; i < fourth.output.size(); i++)
//				fourth.output[i] = pool2d(third.output[i], fourth.filt_size, fourth.stride, 1);
//
//			fifth.sum = fifth.W * flatten(fourth.output) + fifth.b;
//			fifth.output = activate(fifth.sum);
//
//			sixth.sum = sixth.W * fifth.output + sixth.b;
//			sixth.output = activate(sixth.sum);
//
//			seventh.sum = seventh.W * sixth.output + seventh.b;
//			seventh.output = activate(seventh.sum);
//
//			//------------------------------- backward propagate start -------------------------------
//
//			seventh.delta = as_vector(Y[n], 10) - seventh.output;
//			seventh.delta.element_wise(activate_prime(seventh.sum));
//			seventh.dW -= seventh.delta * sixth.output.transpose();
//			seventh.db -= seventh.delta;
//
//			sixth.delta = seventh.W.transpose() * seventh.delta;
//			sixth.delta.element_wise(activate_prime(sixth.sum));
//			sixth.dW -= sixth.delta * fifth.output.transpose();
//			sixth.db -= sixth.delta;
//
//			fifth.delta = sixth.W.transpose() * sixth.delta;
//			fifth.delta.element_wise(activate_prime(fifth.sum));
//			fifth.dW -= fifth.delta * flatten(fourth.output).transpose();
//			fifth.db -= fifth.delta;
//
//			fourth.delta = unflatten(fifth.W.transpose() * fifth.delta, 16, 5);
//
//			for (int i = 0; i < third.delta.size(); i++)
//			{
//				third.delta[i] = delta_img_avg(third.out_size, fourth.delta[i], fourth.filt_size, fourth.stride);
//				//third.delta[i] = delta_img_max(third.output[i], fourth.delta[i], fourth.filt_size, fourth.stride);
//				third.delta[i].element_wise(activate_prime(third.sum[i]));
//			}
//
//			for (int i = 0; i < third.output.size(); i++)
//			{
//				for (int c = 0; c < indices[0].size(); c++)
//					for (int r = 0; r < indices.size(); r++)
//					{
//						if (indices[r][c] != 0)
//						{
//							third.dWs[i] -= conv2d(second.output[r], third.delta[i], third.pad);
//						}
//					}
//				third.db[i] -= third.delta[i].sum(); // 델타를 이렇게 여러번 계산해야 하는지 아니면 한번만 해야하는지
//			}
//
//			int padding = third.filt_size - 1;
//			for (int i = 0; i < second.delta.size(); i++)
//			{
//				second.delta[i].setZero();
//				for (int r = 0; r < indices.size(); r++)
//					for (int c = 0; c < indices.size(); c++)
//						if (indices[r][c] != 0)
//							second.delta[i] += conv2d(third.delta[c], rotate_180(third.Ws[c]), padding);
//			}
//
//			for (int i = 0; i < first.delta.size(); i++)
//			{
//				first.delta[i] = delta_img_avg(first.out_size, second.delta[i], second.filt_size, second.stride);
//				//first.delta[i] = delta_img_max(first.output[i], second.delta[i], second.filt_size, second.stride);
//				first.delta[i].element_wise(activate_prime(first.sum[i]));
//			}
//
//			for (int i = 0; i < first.dWs.size(); i++)
//			{
//				first.dWs[i] -= conv2d(X[n], first.delta[i], first.pad);
//				first.db[i] -= first.delta[i].sum();
//			}
//
//			//------------------------------- update weight -------------------------------
//
//			if ((n + 1) % batch == 0)
//			{
//				for (int i = 0; i < first.Ws.size(); i++)
//				{
//					first.Ws[i] -= l_rate * first.dWs[i];
//					first.dWs[i].setZero();
//				}
//				first.b -= l_rate * first.db;
//				first.db.setZero();
//
//				for (int i = 0; i < third.Ws.size(); i++)
//				{
//					third.Ws[i] -= l_rate * third.dWs[i];
//					third.dWs[i].setZero();
//				}
//				third.b -= l_rate * third.db;
//				third.db.setZero();
//
//				fifth.W -= l_rate * fifth.dW;
//				fifth.b -= l_rate * fifth.db;
//				fifth.dW.setZero();
//				fifth.db.setZero();
//
//				sixth.W -= l_rate * sixth.dW;
//				sixth.b -= l_rate * sixth.db;
//				sixth.dW.setZero();
//				sixth.db.setZero();
//
//				seventh.W -= l_rate * seventh.dW;
//				seventh.b -= l_rate * seventh.db;
//				seventh.dW.setZero();
//				seventh.db.setZero();
//			}
//
//			error += calc_error(Y[n], seventh.output);
//		}
//
//		double sum_error = error / (double)X.size() * 100;
//		training_errors.push_back(sum_error);
//
//		cout << "Error rate(epoch" << epoch + 1 << ") : " << sum_error << "%" << endl;
//	}
//
//	cout << first.Ws[0] << endl << endl;
//	cout << first.output[0] << endl << endl;
//	cout << third.Ws[0] << endl << endl;
//	cout << third.output[0] << endl << endl;
//	cout << fifth.W << endl << endl;
//	cout << fifth.output << endl << endl;
//
//	/*ofstream error_out("training_error.csv");
//	for (int i = 0; i < training_errors.size(); i++)
//		error_out << training_errors[i] << endl;
//	error_out.close();
//
//	ofstream model_out("model_out.csv");
//
//	for (int l = 0; l < first.Ws.size(); l++)
//		write_weight(model_out, first.Ws[l]);
//
//	write_bias(model_out, first.b);
//
//	for (int l = 0; l < third.Ws.size(); l++)
//		write_weight(model_out, third.Ws[l]);
//
//	write_bias(model_out, third.b);
//
//	write_weight(model_out, fifth.W);
//	write_bias(model_out, fifth.b);
//
//	write_weight(model_out, sixth.W);
//	write_bias(model_out, sixth.b);
//
//	write_weight(model_out, seventh.W);
//	write_bias(model_out, seventh.b);
//
//	model_out.close();*/
//
//	return 0;
//}