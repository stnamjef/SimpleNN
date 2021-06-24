#pragma once
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "average_pooling_layer.h"
#include "activation_layer.h"
#include "batch_normalization_1d_layer.h"
#include "batch_normalization_2d_layer.h"
#include "flatten_layer.h"
#include "loss_layer.h"
#include "optimizers.h"
#include "data_loader.h"
#include "file_manage.h"

namespace simple_nn
{
	int PROGRESS_BAR_LEGNTH = 30;

	class SimpleNN
	{
	private:
		vector<Layer*> net;
		Optimizer* optim;
		Loss* loss;
	public:
		void add(Layer* layer);
		void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss* loss=nullptr);
		void fit(const DataLoader& train_loader, int epochs, const DataLoader& valid_loader);
		void save(string save_dir, string fname);
		void load(string save_dir, string fname);
		void evaluate(const DataLoader& data_loader);
	private:
		void forward(const MatXf& X, bool is_training);
		void classify(const MatXf& output, VecXi& classified);
		void error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc);
		void loss_criterion(const MatXf& output, const VecXi& labels, float& loss_acc);
		void zero_grad();
		void backward(const MatXf& X);
		void update_weight();
		int count_params();
		void write_or_read_params(fstream& fs, string mode);
	};

	void SimpleNN::add(Layer* layer) { net.push_back(layer); }

	void SimpleNN::compile(vector<int> input_shape, Optimizer* optim, Loss* loss)
	{
		// set optimizer & loss
		this->optim = optim;
		this->loss = loss;

		// set first & last layer
		net.front()->is_first = true;
		net.back()->is_last = true;

		// set network
		for (int l = 0; l < net.size(); l++) {
			if (l == 0) net[l]->set_layer(input_shape);
			else net[l]->set_layer(net[l - 1]->output_shape());
		}

		// set Loss layer
		loss->set_layer(net.back()->output_shape());
	}

	void SimpleNN::fit(const DataLoader& train_loader, int epochs, const DataLoader& valid_loader)
	{
		if (optim == nullptr || loss == nullptr) {
			cout << "The model must be compiled before fitting the data." << endl;
			exit(1);
		}

		int batch = train_loader.input_shape()[0];
		int n_batch = train_loader.size();

		MatXf X;
		VecXi Y;
		VecXi classified(batch);

		for (int e = 0; e < epochs; e++) {
			float loss = 0.f;
			float error = 0.f;

			system_clock::time_point start = system_clock::now();
			for (int n = 0; n < n_batch; n++) {
				X = train_loader.get_x(n);
				Y = train_loader.get_y(n);

				forward(X, true);
				classify(net.back()->output, classified);
				error_criterion(classified, Y, error);

				zero_grad();
				loss_criterion(net.back()->output, Y, loss);
				backward(X);
				update_weight();

				cout << "[Epoch:" << setw(3) << e + 1 << "/" << epochs << ", ";
				cout << "Batch: " << setw(4) << n + 1 << "/" << n_batch << "]";

				if (n + 1 < n_batch) {
					cout << "\r";
				}
			}
			system_clock::time_point end = system_clock::now();
			duration<float> sec = end - start;

			float loss_valid = 0.f;
			float error_valid = 0.f;

			int n_batch_valid = valid_loader.size();
			if (n_batch_valid != 0) {
				for (int n = 0; n < n_batch_valid; n++) {
					X = valid_loader.get_x(n);
					Y = valid_loader.get_y(n);

					forward(X, false);
					classify(net.back()->output, classified);
					error_criterion(classified, Y, error_valid);
					loss_criterion(net.back()->output, Y, loss_valid);
				}
			}

			cout << fixed << setprecision(2);
			cout << " - t: " << sec.count() << 's';
			cout << " - loss: " << loss / n_batch;
			cout << ", error: " << error / n_batch * 100 << "%";
			if (n_batch_valid != 0) {
				cout << " - loss(valid): " << loss_valid / n_batch_valid;
				cout << ", error(valid): " << error_valid / n_batch_valid * 100 << "%";
			}
			cout << endl;
		}
	}

	void SimpleNN::forward(const MatXf& X, bool is_training)
	{
		for (int l = 0; l < net.size(); l++) {
			if (l == 0) net[l]->forward(X, is_training);
			else net[l]->forward(net[l - 1]->output, is_training);
		}
	}

	void SimpleNN::classify(const MatXf& output, VecXi& classified)
	{
		// assume that the last layer is linear, not 2d.
		assert(output.rows() == classified.size());

		for (int i = 0; i < classified.size(); i++) {
			output.row(i).maxCoeff(&classified[i]);
		}
	}

	void SimpleNN::error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc)
	{
		int batch = (int)classified.size();

		float error = 0.f;
		for (int i = 0; i < batch; i++) {
			if (classified[i] != labels[i]) error++;
		}
		error_acc += error / batch;
	}

	void SimpleNN::loss_criterion(const MatXf& output, const VecXi& labels, float& loss_acc)
	{
		loss_acc += loss->calc_loss(output, labels, net.back()->delta);
	}

	void SimpleNN::zero_grad()
	{
		for (const auto& l : net) l->zero_grad();
	}

	void SimpleNN::backward(const MatXf& X)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--) {
			if (l == 0) {
				MatXf empty;
				net[l]->backward(X, empty);
			}
			else {
				net[l]->backward(net[l - 1]->output, net[l - 1]->delta);
			}
		}
	}

	void SimpleNN::update_weight()
	{
		float lr = optim->lr();
		float decay = optim->decay();
		for (const auto& l : net) {
			l->update_weight(lr, decay);
		}
	}

	void SimpleNN::save(string save_dir, string fname)
	{
		if (!std::filesystem::exists(save_dir)) {
			cout << "Save directory(" << save_dir << ") does not exist." << endl;
			cout << "Creating the directory..." << endl;
			std::filesystem::create_directories(save_dir);
		}

		string path = save_dir + "/" + fname + ".pth";
		fstream fout(path, ios::out | ios::binary);

		int total_params = count_params();
		fout.write((char*)&total_params, sizeof(int));

		write_or_read_params(fout, "write");
		cout << "Model parameters are saved in " << path << endl;

		fout.close();

		return;
	}

	void SimpleNN::load(string save_dir, string fname)
	{
		if (!std::filesystem::exists(save_dir)) {
			cout << "Save directory(" << save_dir << ") does not exist." << endl;
			exit(1);
		}

		string path = save_dir + "/" + fname + ".pth";
		fstream fin(path, ios::in | ios::binary);

		if (!fin) {
			cout << fname << ".pth does not exist." << endl;
			exit(1);
		}

		int total_params;
		fin.read((char*)&total_params, sizeof(int));

		if (total_params != count_params()) {
			cout << "The number of parameters does not match." << endl;
			fin.close();
			exit(1);
		}

		write_or_read_params(fin, "read");
		fin.close();

		cout << "Pretrained weights are loaded." << endl;

		return;
	}

	int SimpleNN::count_params()
	{
		int total_params = 0;
		for (const Layer* l : net) {
			if (l->type == LayerType::LINEAR) {
				const Linear* lc = dynamic_cast<const Linear*>(l);
				total_params += (int)lc->W.size();
				total_params += (int)lc->b.size();
			}
			else if (l->type == LayerType::CONV2D) {
				const Conv2d* lc = dynamic_cast<const Conv2d*>(l);
				total_params += (int)lc->kernel.size();
				total_params += (int)lc->bias.size();
			}
			else if (l->type == LayerType::BATCHNORM1D) {
				const BatchNorm1d* lc = dynamic_cast<const BatchNorm1d*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
			}
			else if (l->type == LayerType::BATCHNORM2D) {
				const BatchNorm2d* lc = dynamic_cast<const BatchNorm2d*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
			}
			else {
				continue;
			}
		}
		return total_params;
	}

	void SimpleNN::write_or_read_params(fstream& fs, string mode)
	{
		for (const Layer* l : net) {
			if (l->type == LayerType::LINEAR) {
				const Linear* lc = dynamic_cast<const Linear*>(l);
				int s1 = (int)lc->W.size();
				int s2 = (int)lc->b.size();
				if (mode == "write") {
					fs.write((char*)lc->W.data(), sizeof(float) * s1);
					fs.write((char*)lc->b.data(), sizeof(float) * s2);
				}
				else {
					fs.read((char*)lc->W.data(), sizeof(float) * s1);
					fs.read((char*)lc->b.data(), sizeof(float) * s2);
				}
			}
			else if (l->type == LayerType::CONV2D) {
				const Conv2d* lc = dynamic_cast<const Conv2d*>(l);
				int s1 = (int)lc->kernel.size();
				int s2 = (int)lc->bias.size();
				if (mode == "write") {
					fs.write((char*)lc->kernel.data(), sizeof(float) * s1);
					fs.write((char*)lc->bias.data(), sizeof(float) * s2);
				}
				else {
					fs.read((char*)lc->kernel.data(), sizeof(float) * s1);
					fs.read((char*)lc->bias.data(), sizeof(float) * s2);
				}
			}
			else if (l->type == LayerType::BATCHNORM1D) {
				const BatchNorm1d* lc = dynamic_cast<const BatchNorm1d*>(l);
				int s1 = (int)lc->move_mu.size();
				int s2 = (int)lc->move_var.size();
				int s3 = (int)lc->gamma.size();
				int s4 = (int)lc->beta.size();
				if (mode == "write") {
					fs.write((char*)lc->move_mu.data(), sizeof(float) * s1);
					fs.write((char*)lc->move_var.data(), sizeof(float) * s2);
					fs.write((char*)lc->gamma.data(), sizeof(float) * s3);
					fs.write((char*)lc->beta.data(), sizeof(float) * s4);
				}
				else {
					fs.read((char*)lc->move_mu.data(), sizeof(float) * s1);
					fs.read((char*)lc->move_var.data(), sizeof(float) * s2);
					fs.read((char*)lc->gamma.data(), sizeof(float) * s3);
					fs.read((char*)lc->beta.data(), sizeof(float) * s4);
				}
			}
			else if (l->type == LayerType::BATCHNORM2D) {
				const BatchNorm2d* lc = dynamic_cast<const BatchNorm2d*>(l);
				int s1 = (int)lc->move_mu.size();
				int s2 = (int)lc->move_var.size();
				int s3 = (int)lc->gamma.size();
				int s4 = (int)lc->beta.size();
				if (mode == "write") {
					fs.write((char*)lc->move_mu.data(), sizeof(float) * s1);
					fs.write((char*)lc->move_var.data(), sizeof(float) * s2);
					fs.write((char*)lc->gamma.data(), sizeof(float) * s3);
					fs.write((char*)lc->beta.data(), sizeof(float) * s4);
				}
				else {
					fs.read((char*)lc->move_mu.data(), sizeof(float) * s1);
					fs.read((char*)lc->move_var.data(), sizeof(float) * s2);
					fs.read((char*)lc->gamma.data(), sizeof(float) * s3);
					fs.read((char*)lc->beta.data(), sizeof(float) * s4);
				}
			}
			else {
				continue;
			}
		}
	}

	void SimpleNN::evaluate(const DataLoader& data_loader)
	{
		int batch = data_loader.input_shape()[0];
		int n_batch = data_loader.size();
		float error_acc = 0.f;

		MatXf X;
		VecXi Y;
		VecXi classified(batch);

		system_clock::time_point start = system_clock::now();
		for (int n = 0; n < n_batch; n++) {
			MatXf X = data_loader.get_x(n);
			VecXi Y = data_loader.get_y(n);

			forward(X, false);
			classify(net.back()->output, classified);
			error_criterion(classified, Y, error_acc);
			
			cout << "[Batch: " << setw(3) << n + 1 << "/" << n_batch << "]";
			if (n + 1 < n_batch) {
				cout << "\r";
			}
		}
		system_clock::time_point end = system_clock::now();
		duration<float> sec = end - start;

		cout << fixed << setprecision(2);
		cout << " - t: " << sec.count() << "s";
		cout << " - error(" << batch * n_batch << " images): ";
		cout << error_acc / n_batch * 100 << "%" << endl;
	}
}