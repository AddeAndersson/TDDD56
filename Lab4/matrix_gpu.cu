// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>

__global__ void add_matrix_cuda(float *a, float *b, float *c) {
	/*
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
	*/

	// Part C
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.x * blockDim.x) + threadIdx.y;

	c[idx] = a[idx] + b[idx];
}

int main()
{
	const int N = 1024;
	const int blocksize = 16;
	const int size = N*N*sizeof(float);

	float *a = new float[N*N];
	float *b = new float[N*N];
	float *res = new float[N*N];
	float *a_dev;
	float *b_dev;
	float *c;

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	
	//add_matrix(a, b, c, N);
	cudaMalloc( (void**)&a_dev, size );
	cudaMalloc( (void**)&b_dev, size );
	cudaMalloc( (void**)&c, size );

	float milliseconds = 0;
	cudaMemcpy(a_dev, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, size, cudaMemcpyHostToDevice);
	dim3 numOfThreads( blocksize, blocksize);
	dim3 numOfBlocks( N / numOfThreads.x, N / numOfThreads.y );
	
	cudaEventRecord(start, 0); // Start timer
	add_matrix_cuda<<<numOfBlocks, numOfThreads>>>(a_dev, b_dev, c);
	cudaEventRecord(stop, 0); // End timer
	cudaEventSynchronize(stop);
	cudaMemcpy( res, c, size, cudaMemcpyDeviceToHost ); 
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c );

	// for (int i = 0; i < N; i++)
	// {
	// 	for (int j = 0; j < N; j++)
	// 	{
	// 		printf("%0.2f ", res[i+j*N]);
	// 	}
	// 	printf("\n");
	// }
	printf("Time elapsed: %fms\n", milliseconds);

	delete[] a;
	delete[] b;
	delete[] res;
}
n