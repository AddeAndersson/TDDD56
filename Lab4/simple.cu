// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
	  cudaDeviceProp prop;
	  cudaGetDeviceProperties(&prop, i);
	  printf("Device Number: %d\n", i);
	  printf("  Device name: %s\n", prop.name);
	  printf("  Memory Clock Rate (KHz): %d\n",
			 prop.memoryClockRate);
	  printf("  Memory Bus Width (bits): %d\n",
			 prop.memoryBusWidth);
	  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}


	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);

	for(int i = 0; i < N; i++) {
		c[i] = i*i;
	}

	cudaMalloc( (void**)&cd, size );
	cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
	dim3 dimBlock( blocksize, 1 ); // max 1024
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");
	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
