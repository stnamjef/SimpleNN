#pragma once
#include "layer.h"

namespace simple_nn
{
	class Linear : public Layer
	{
	public:
		int batch;
		int n_input;
		int n_node;
		int out_block_size;
		int weight_block_size;
		string init_opt;
		float* W;
		float* dW;
		float* b;
		float* db;
	public:
		Linear(int n_node, string init_opt = "normal");
		Linear(int n_node, int n_input = 0, string init_opt = "normal");
		~Linear();
		void set_layer(int batch, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isEval) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		void update_weight(float lr, float decay) override;
		vector<int> input_shape() override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	};

	Linear::Linear(int n_node, string init_opt) :
		Layer(LINEAR),
		batch(0),
		n_input(0),
		n_node(n_node),
		out_block_size(0),
		weight_block_size(0),
		init_opt(init_opt)
	{
		if (init_opt != "normal" && init_opt != "uniform") {
			throw logic_error("Linear::Linear(int, string, int): Invalid init option.");
		}
	}

	Linear::Linear(int n_node, int n_input, string init_opt) :
		Layer(LINEAR),
		batch(0),
		n_input(n_input),
		n_node(n_node),
		out_block_size(0),
		weight_block_size(0),
		init_opt(init_opt)
	{
		if (init_opt != "normal" && init_opt != "uniform") {
			throw logic_error("Linear::Linear(int, string, int): Invalid init option.");
		}
	}

	Linear::~Linear()
	{
		delete_memory(output);
		delete_memory(delta);
		delete_memory(W);
		delete_memory(dW);
		delete_memory(b);
		delete_memory(db);
	}

	void Linear::set_layer(int batch, const vector<int>& input_shape)
	{
		this->batch = batch;
		if (input_shape.size() == 1) {
			n_input = input_shape[0];
		}
		else {
			int in_h = input_shape[0];
			int in_w = input_shape[1];
			int channels = input_shape[2];
			n_input = in_h * in_w * channels;
		}
		out_block_size = batch * n_node;
		weight_block_size = n_node * n_input;
		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
		allocate_memory(W, weight_block_size);
		allocate_memory(dW, weight_block_size);
		allocate_memory(b, n_node);
		allocate_memory(db, n_node);
		init_weight(W, weight_block_size, n_input, n_node, init_opt);
		set_zero(dW, weight_block_size);
		set_zero(b, n_node);
		set_zero(db, n_node);
	}

	void Linear::forward_propagate(const float* prev_out, bool isEval)
	{
		int M = n_node;
		int N = 1;
		int K = n_input;
		for (int n = 0; n < batch; n++) {
			const float* A = W;
			const float* B = prev_out + K * n;
			float* C = output + M * n;
			gemm_nn(M, N, K, 1.0F, A, K, B, N, C, N);
			//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0F, A, K, B, N, 0.0F, C, N);
			std::transform(C, C + M, b, C, std::plus<float>());
		}
	}

	void Linear::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		int M = n_node;
		int N = n_input;
		int K = 1;
		for (int n = 0; n < batch; n++) {
			const float* A = delta + M * n;
			const float* B = prev_out + N * n;
			float* C = dW;
			gemm_nt(M, N, K, 1.0F, A, K, B, K, C, N);
			//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0F, A, K, B, K, 0.0F, C, M);
			std::transform(db, db + n_node, A, db,
				[](float& _db, const float& delta) { return _db + delta; });
		}

		if (!isFirst) {
			M = n_input;
			N = 1;
			K = n_node;
			for (int n = 0; n < batch; n++) {
				const float* A = W;
				const float* B = delta + K * n;
				float* C = prev_delta + M * n;
				gemm_tn(M, N, K, 1.0F, A, M, B, N, C, N);
				//cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0F, A, M, B, N, 0.0F, C, K);
			}
		}
	}

	void Linear::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		for (int i = 0; i < weight_block_size; i++) {
			W[i] = t1 * W[i] - t2 * dW[i];
			dW[i] = 0.0F;
		}
		for (int i = 0; i < n_node; i++) {
			b[i] = t1 * b[i] - t2 * db[i];
			db[i] = 0.0F;
		}
	}

	vector<int> Linear::input_shape()
	{
		if (n_input == 0) {
			throw logic_error("Linear::input_shape(): Input shape is empty.");
		}
		return { n_input };
	}

	vector<int> Linear::output_shape() { return { n_node }; }

	int Linear::get_out_block_size() { return out_block_size; }
}