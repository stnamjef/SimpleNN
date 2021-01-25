#pragma once

namespace simple_nn
{
	void gemm_nn(int M, int N, int K, float alpha,
		const float* A, int lda,
		const float* B, int ldb,
		float* C, int ldc)
	{
		int i, j, k;
		#pragma omp parallel for
		for (i = 0; i < M; i++) {
			for (k = 0; k < K; k++) {
				register float temp = alpha * A[i * lda + k];
				for (j = 0; j < N; j++) {
					C[i * ldc + j] += temp * B[k * ldb + j];
				}
			}
		}
	}

	void gemm_nt(int M, int N, int K, float alpha,
		const float* A, int lda,
		const float* B, int ldb,
		float* C, int ldc)
	{
		int i, j, k;
		#pragma omp parallel for
		for (i = 0; i < M; i++) {
			for (j = 0; j < N; j++) {
				register float sum = 0;
				for (k = 0; k < K; k++) {
					sum += alpha * A[i * lda + k] * B[j * ldb + k];
				}
				C[i * ldc + j] += sum;
			}
		}
	}

	void gemm_tn(int M, int N, int K, float alpha,
		const float* A, int lda,
		const float* B, int ldb,
		float* C, int ldc)
	{
		int i, j, k;
		#pragma omp parallel for
		for (i = 0; i < M; i++) {
			for (k = 0; k < K; k++) {
				register float temp = alpha * A[k * lda + i];
				for (j = 0; j < N; j++) {
					C[i * ldc + j] += temp * B[k * ldb + j];
				}
			}
		}
	}
}