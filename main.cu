#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include <cuda.h>
#include <cstdio>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static void learn();
static void clear();
static unsigned int classify(double data[28][28]);

static void loaddata()
{
	mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte",
		&test_set, &test_cnt);
}

static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	if (cuInit(0) != CUDA_SUCCESS) {
		fprintf(stderr, "cuInit failed\n");
		return 1;
	}

	loaddata();
	learn();
	test();
	clear();

	return 0;
}

///////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-04f;

struct Layer {
	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	const int M, N;

	Layer(int M, int N)
		: M(M), N(N)
	{
		float h_bias[N];
		float h_weight[N][M];

		output = NULL;
		preact = NULL;
		bias   = NULL;
		weight = NULL;

		for (int i = 0; i < N; ++i) {
			h_bias[i] = 0.5 - float(rand()) / float(RAND_MAX);
			/*h_bias[i] = 0.0f;*/

			for (int j = 0; j < M; ++j) {
				h_weight[i][j] = 0.5 - float(rand()) / float(RAND_MAX);
			}
		}

		cudaMalloc(&output, sizeof(float) * N);
		cudaMalloc(&preact, sizeof(float) * N);

		cudaMalloc(&bias, sizeof(float) * N);

		cudaMalloc(&weight, sizeof(float) * M * N);

		cudaMalloc(&d_output, sizeof(float) * N);
		cudaMalloc(&d_preact, sizeof(float) * N);
		cudaMalloc(&d_weight, sizeof(float) * M * N);

		cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

		cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	}

	~Layer()
	{
		cudaFree(output);
		cudaFree(preact);

		cudaFree(bias);

		cudaFree(weight);

		cudaFree(d_output);
		cudaFree(d_preact);
		cudaFree(d_weight);
	}

	void setOutput(float *data)
	{
		cudaMemcpy(output, data, sizeof(float) * N, cudaMemcpyHostToDevice);
	}
};

static std::vector<Layer *> layers;
static cublasHandle_t handle;

__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *output, float *preact, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(preact[idx]);
	}
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void grad_preact(float *d_preact, float *d_output, float *preact, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		const float g = step_function(preact[idx]);

		d_preact[idx] = d_output[idx] * g * (1 - g);
	}
}

static void propagate(Layer *l1, Layer *l2)
{
	static const float alpha = 1.0f;
	static const float beta = 0.0f;

	cublasSgemv(handle, CUBLAS_OP_N, l2->N, l2->M,
		&alpha, l2->weight, l2->N, l1->output, 1, &beta, l2->preact, 1);
	cublasSaxpy(handle, l2->N, &alpha, l2->bias, 1, l2->preact, 1);
	apply_step_function<<<64, 64>>>(l2->output, l2->preact, l2->N);
}

static void backpropagate(Layer *l1, Layer *l2)
{
	float alpha = 1.0;
	float beta = 0.0;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, l2->N, l2->M, 1, &alpha,
		l2->d_preact, l2->N, l1->output, l2->M, &beta, l2->d_weight, l2->N);
	cublasSgemv(handle, CUBLAS_OP_T, l2->N, l2->M, &alpha, l2->weight, l2->N,
		l2->d_preact, 1, &beta, l1->d_output, 1);
	grad_preact<<<64, 64>>>(l1->d_preact, l1->d_output, l1->preact, l1->N);

	cublasSaxpy(handle, l2->N, &dt, l2->d_preact, 1, l2->bias, 1);
	cublasSaxpy(handle, l2->N * l2->M, &dt, l2->d_weight, 1, l2->weight, 1);
}

static void propagate(double data[28][28])
{
	float input[28*28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			const int idx = i * 28 + j;

			input[idx] = data[j][i];
		}
	}

	layers[0]->setOutput(input);

	for (int i = 1; i < layers.size(); ++i) {
		propagate(layers[i - 1], layers[i]);
	}
}

static void clear()
{
	for (int i = 0; i < layers.size(); ++i) {
		delete layers[i];
	}

	layers.clear();
}

static void learn()
{
	cublasCreate(&handle);

	layers.push_back(new Layer(0, 28*28));
	layers.push_back(new Layer(28*28, 500));
	layers.push_back(new Layer(500, 300));
	layers.push_back(new Layer(300, 10));

	float err;

	while (true) {
		err = 0.0f;
		
		for (int i = 0; i < train_cnt; ++i) {
			/*float test[10];*/
			float tmp;

			Layer *ol = layers[layers.size() - 1];

			propagate(train_set[i].data);

			makeError<<<10, 1>>>(ol->d_preact, ol->output, train_set[i].label, 10);

			cublasSnrm2(handle, 10, ol->d_preact, 1, &tmp);

			err += tmp;
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e\n", err);

		if (err < threshold)
			break;

		for (int i = 0; i < train_cnt; ++i) {
			propagate(train_set[i].data);

			Layer *ol = layers[layers.size() - 1];
			makeError<<<10, 1>>>(ol->d_preact, ol->output, train_set[i].label, 10);

			for (int j = layers.size() - 1; j > 0; --j) {
				backpropagate(layers[j - 1], layers[j]);
			}
		}
	}
}

static unsigned int classify(double data[28][28])
{
	float res[10];
	propagate(data);

	unsigned int max = 0;

	cudaMemcpy(res, layers[layers.size() - 1]->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

