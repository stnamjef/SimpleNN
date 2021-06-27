#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <filesystem>

namespace simple_nn
{
	class Config
	{
	public:
		std::string mode;
		std::string model;
		std::string data_dir;
		std::string save_dir;
		std::string pretrained;
		std::string pool;
		std::string activ;
		std::string init;
		std::string loss;
		int batch;
		int epoch;
		float lr;
		float decay;
		bool use_batchnorm;
		bool shuffle_train;
		bool shuffle_test;
		Config();
		void parse(int argc, char** argv);
		void print_config();
	private:
		void print_help();
		void check_if_args_valid();
	};

	Config::Config() :
		mode("train"),
		model("lenet5"),
		data_dir("./dataset"),
		save_dir("./model_zoo"),
		pretrained(""),
		pool("max"),
		activ("relu"),
		init("lecun_uniform"),
		loss("cross_entropy"),
		batch(32),
		epoch(30),
		lr(0.01f),
		decay(0.f),
		use_batchnorm(false),
		shuffle_train(true),
		shuffle_test(false) {}

	void Config::parse(int argc, char** argv)
	{
		std::regex pattern("--(.*)=(.*)");

		for (int i = 1; i < argc; i++) {
			std::string arg = argv[i];
			std::smatch matches;
			if (std::regex_match(arg, matches, pattern)) {
				auto it = matches.begin();
				it++;
				if ((*it) == "mode") {
					it++;
					mode = *it;
				}
				else if ((*it) == "model") {
					it++;
					model = *it;
				}
				else if ((*it) == "data_dir") {
					it++;
					data_dir = *it;
				}
				else if ((*it) == "save_dir") {
					it++;
					save_dir = *it;
				}
				else if ((*it) == "pretrained") {
					it++;
					pretrained = *it;
				}
				else if ((*it) == "pool") {
					it++;
					pool = *it;
				}
				else if ((*it) == "activ") {
					it++;
					activ = *it;
				}
				else if ((*it) == "init") {
					it++;
					init = *it;
				}
				else if ((*it) == "loss") {
					it++;
					loss = *it;
				}
				else if ((*it) == "batch") {
					it++;
					batch = std::stoi(*it);
				}
				else if ((*it) == "epoch") {
					it++;
					epoch = std::stoi(*it);
				}
				else if ((*it) == "lr") {
					it++;
					lr = std::stof(*it);
				}
				else if ((*it) == "decay") {
					it++;
					decay = std::stof(*it);
				}
				else if ((*it) == "use_batchnorm") {
					it++;
					use_batchnorm = !use_batchnorm;
				}
				else if ((*it) == "shuffle_train") {
					it++;
					shuffle_train = !shuffle_train;
				}
				else if ((*it) == "shuffle_test") {
					it++;
					shuffle_test = !shuffle_test;
				}
				else {
					std::cout << "Invalid arguments." << std::endl;
					print_help();
					exit(1);
				}
			}
			else if (arg == "--help") {
				print_help();
				exit(1);
			}
			else {
				std::cout << "Argument expression does not match." << std::endl;
				print_help();
				exit(1);
			}
		}

		check_if_args_valid();
	}

	void Config::print_config()
	{
		std::cout << "User Configurations:" << std::endl;
		std::cout << "  --mode          = " << mode << std::endl;
		std::cout << "  --model         = " << model << std::endl;
		std::cout << "  --data_dir      = " << data_dir << std::endl;
		std::cout << "  --save_dir      = " << save_dir << std::endl;
		std::cout << "  --pretrained    = " << pretrained << std::endl;
		std::cout << "  --pool          = " << pool << std::endl;
		std::cout << "  --activ         = " << activ << std::endl;
		std::cout << "  --init          = " << init << std::endl;
		std::cout << "  --loss          = " << loss << std::endl;
		std::cout << "  --batch         = " << batch << std::endl;
		std::cout << "  --epoch         = " << epoch << std::endl;
		std::cout << "  --lr            = " << lr << std::endl;
		std::cout << "  --decay         = " << decay << std::endl;
		std::cout << "  --use_batchnorm = " << use_batchnorm << std::endl;
		std::cout << "  --shuffle_train = " << shuffle_train << std::endl;
		std::cout << "  --shuffle_test  = " << shuffle_test << std::endl;
	}

	void Config::print_help()
	{
		std::cout << "CLI options:" << std::endl;
		std::cout << "  --mode          = Program mode (options: train, test; default: train)" << std::endl;
		std::cout << "  --model         = Model name (options: lenet5, linear; default: lenet5)" << std::endl;
		std::cout << "  --data_dir      = Dataset directory (default: ./dataset)" << std::endl;
		std::cout << "  --save_dir      = Saving directory (default: ./model_zoo)" << std::endl;
		std::cout << "  --pretrained    = Pretrained file name (default: None)" << std::endl;
		std::cout << "  --pool          = Pooling method (options: max, avg; default: max)" << std::endl;
		std::cout << "  --activ         = Activation function for hidden layer (options: tanh, relu; default: relu)" << std::endl;
		std::cout << "  --init          = Weight initialization (default: lecun_uniform)" << std::endl;
		std::cout << "                    (options: lecun_uniform, lecun_normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)" << std::endl;
		std::cout << "  --loss          = Loss function for training (options: cross_entropy, mse; default: cross_entropy)" << std::endl;
		std::cout << "  --batch         = Batch size (default: 32)" << std::endl;
		std::cout << "  --epoch         = Total epochs (default: 30)" << std::endl;
		std::cout << "  --lr            = Learning rate (default: 0.01)" << std::endl;
		std::cout << "  --decay         = L2 regularization (default: 0)" << std::endl;
		std::cout << "  --use_batchnorm = Use batch normalization (options: 0, 1; default: 0)" << std::endl;
		std::cout << "  --shuffle_train = Shuffle training dataset (options: 0, 1; default: 1)" << std::endl;
		std::cout << "  --shuffle_test  = Shuffle testing dataset (options: 0, 1; default: 0)" << std::endl;
	}

	void Config::check_if_args_valid()
	{
		if (!std::filesystem::exists(data_dir)) {
			std::cout << "Dataset directory (" << data_dir << ")" << std::endl;
			std::cout << "does not exist." << std::endl;
			exit(1);
		}

		if (!std::filesystem::exists(save_dir)) {
			std::cout << "Creating saving derectory (" << save_dir << ")" << std::endl;
			std::filesystem::create_directories(save_dir);
		}

		if (mode != "train" && mode != "test") {
			std::cout << "Invalid mode." << std::endl;
			exit(1);
		}

		if (mode == "test" && pretrained == "") {
			std::cout << "Pretrained weights should be given in test mode." << std::endl;
			std::cout << "Use the following CLI option: --pretrained=file_name." << std::endl;
			exit(1);
		}

		if (model != "lenet5" && model != "linear") {
			std::cout << "Invalid model." << std::endl;
			exit(1);
		}

		if (pool!= "max" && pool != "avg") {
			std::cout << "Invalid pooling." << std::endl;
			exit(1);
		}

		if (activ != "relu" && activ != "tanh") {
			std::cout << "Invalid activation." << std::endl;
			exit(1);
		}

		if (init != "normal" && init != "uniform" &&
			init != "lecun_normal" && init != "lecun_uniform" &&
			init != "xavier_normal" && init != "xavier_uniform" &&
			init != "kaiming_normal" && init != "kaiming_uniform") {
			std::cout << "Invalid initialization." << std::endl;
			exit(1);
		}

		if (loss != "cross_entropy" && loss != "mse") {
			std::cout << "Invalid loss function." << std::endl;
			exit(1);
		}
	}
}