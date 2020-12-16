// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// 
// 
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

#include <chrono>
#include <algorithm>
#include <vector>

// Use these for setting shared memory size.
#define maxKernelSizeX 32
#define maxKernelSizeY 32
#define block_size_x 32
#define block_size_y 32
#define kernel_size_x 2
#define kernel_size_y 2

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, int kernelsizex, int kernelsizey)
{ 
	__shared__ unsigned char shared_mem[maxKernelSizeY][maxKernelSizeX * 3]; // shared memory

	// For separable
	if(kernelsizex == 1) kernelsizex = 0; 
	if(kernelsizey == 1) kernelsizey = 0;

	int tile_w = blockDim.x - 2*kernelsizex;
	int tile_h = blockDim.y - 2*kernelsizey;
	// map from blockIdx to pixel position
	int x = blockIdx.x * tile_w + threadIdx.x - kernelsizex;
	int y = blockIdx.y * tile_h + threadIdx.y - kernelsizey;			
	
	// clamp to edge of image
	y = min(max(y, 0), imagesizey-1);
	x = min(max(x, 0), imagesizex-1);


    // Each thread copies its pixel of the block to shared memory
	shared_mem[threadIdx.y][threadIdx.x*3+0] = image[(y*imagesizex+x)*3+0]; // r
	shared_mem[threadIdx.y][threadIdx.x*3+1] = image[(y*imagesizex+x)*3+1]; // g
	shared_mem[threadIdx.y][threadIdx.x*3+2] = image[(y*imagesizex+x)*3+2]; // b
	__syncthreads();

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	// Filter kernel (simple box filter)
	if(threadIdx.x >= kernelsizex && threadIdx.x < (blockDim.x - kernelsizex) &&
		threadIdx.y >= kernelsizey && threadIdx.y < (blockDim.y - kernelsizey)) {
		sumx=0;sumy=0;sumz=0;
		for(dy=-kernelsizey;dy<=kernelsizey;dy++)
			for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
			{
				
				// Clamp inside of kernel
				int idx_x = threadIdx.x + dx;
				int idx_y = threadIdx.y + dy;
				
				// Sum r,g,b channels
				sumx += shared_mem[idx_y][idx_x*3+0]; // r
				sumy += shared_mem[idx_y][idx_x*3+1]; // g
				sumz += shared_mem[idx_y][idx_x*3+2]; // b
			}
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void filter_gaussian(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, int kernelsizex, int kernelsizey)
{ 
	__shared__ unsigned char shared_mem[maxKernelSizeY][maxKernelSizeX * 3]; // shared memory

	// For separable
	if(kernelsizex == 1) kernelsizex = 0; 
	if(kernelsizey == 1) kernelsizey = 0;

	int tile_w = blockDim.x - 2*kernelsizex;
	int tile_h = blockDim.y - 2*kernelsizey;
	// map from blockIdx to pixel position
	int x = blockIdx.x * tile_w + threadIdx.x - kernelsizex;
	int y = blockIdx.y * tile_h + threadIdx.y - kernelsizey;			

	// clamp to edge of image
	y = min(max(y, 0), imagesizey-1);
	x = min(max(x, 0), imagesizex-1);

    // Each thread copies its pixel of the block to shared memory
	shared_mem[threadIdx.y][threadIdx.x*3+0] = image[(y*imagesizex+x)*3+0]; // r
	shared_mem[threadIdx.y][threadIdx.x*3+1] = image[(y*imagesizex+x)*3+1]; // g
	shared_mem[threadIdx.y][threadIdx.x*3+2] = image[(y*imagesizex+x)*3+2]; // b
	__syncthreads();

	int dy, dx;
	unsigned int sumx, sumy, sumz;
	int weights[5] = { 1, 4, 6, 4, 1};
	int divby = 16;

	// Filter kernel (simple gaussian filter)
	if(threadIdx.x >= kernelsizex && threadIdx.x < (blockDim.x - kernelsizex) &&
		threadIdx.y >= kernelsizey && threadIdx.y < (blockDim.y - kernelsizey)) {
		sumx=0;sumy=0;sumz=0;
		for(dy=-kernelsizey;dy<=kernelsizey;dy++)
			for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
			{
				// Index weights
				int i = (kernelsizex > 1) ? dx : dy;
				// Clamp inside of kernel
				int idx_x = threadIdx.x + dx;
				int idx_y = threadIdx.y + dy;
				// Sum r,g,b channels
				sumx += shared_mem[idx_y][idx_x*3+0] * weights[i + 2]; // r
				sumy += shared_mem[idx_y][idx_x*3+1] * weights[i + 2]; // g
				sumz += shared_mem[idx_y][idx_x*3+2] * weights[i + 2]; // b
			}
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void filter_median(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, int kernelsizex, int kernelsizey)
{ 
	__shared__ unsigned char shared_mem[maxKernelSizeY][maxKernelSizeX * 3]; // shared memory

	// For separable
	if(kernelsizex == 1) kernelsizex = 0; 
	if(kernelsizey == 1) kernelsizey = 0;

	int tile_w = blockDim.x - 2*kernelsizex;
	int tile_h = blockDim.y - 2*kernelsizey;
	// map from blockIdx to pixel position
	int x = blockIdx.x * tile_w + threadIdx.x - kernelsizex;
	int y = blockIdx.y * tile_h + threadIdx.y - kernelsizey;			

	// clamp to edge of image
	y = min(max(y, 0), imagesizey-1);
	x = min(max(x, 0), imagesizex-1);

    // Each thread copies its pixel of the block to shared memory
	shared_mem[threadIdx.y][threadIdx.x*3+0] = image[(y*imagesizex+x)*3+0]; // r
	shared_mem[threadIdx.y][threadIdx.x*3+1] = image[(y*imagesizex+x)*3+1]; // g
	shared_mem[threadIdx.y][threadIdx.x*3+2] = image[(y*imagesizex+x)*3+2]; // b
	__syncthreads();

	
	int dy, dx;
	const int N = maxKernelSizeX*maxKernelSizeY+1;
	unsigned char medianx[N];
	unsigned char mediany[N];
	unsigned char medianz[N];
	int counter = 0;
	
	// Filter kernel (simple gaussian filter)
	if(threadIdx.x >= kernelsizex && threadIdx.x < (blockDim.x - kernelsizex) &&
	threadIdx.y >= kernelsizey && threadIdx.y < (blockDim.y - kernelsizey)) {
		for(dy=-kernelsizey;dy<=kernelsizey;dy++)
			for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
			{
				// Clamp inside of kernel
				int idx_x = threadIdx.x + dx;
				int idx_y = threadIdx.y + dy;
				// Sum r,g,b channels
				medianx[counter] = shared_mem[idx_y][idx_x*3+0]; // r
				mediany[counter] = shared_mem[idx_y][idx_x*3+1]; // g
				medianz[counter] = shared_mem[idx_y][idx_x*3+2]; // b
				counter++;
			}

		// Sort
		float tempx = 0;
		float tempy = 0;
		float tempz = 0;
		for(int i = 0; i < counter; i++) {
			for(int j = 0; j < counter-i-1; j++) {
				if(medianx[j] > medianx[j+1]) {
					tempx = medianx[j];
					medianx[j] = medianx[j+1];
					medianx[j+1] = tempx;
				}
				if(mediany[j] > mediany[j+1]) {
					tempy = mediany[j];
					mediany[j] = mediany[j+1];
					mediany[j+1] = tempy;
				}
				if(medianz[j] > medianz[j+1]) {
					tempz = medianz[j];
					medianz[j] = medianz[j+1];
					medianz[j+1] = tempz;
				}
			}
		}

		out[(y*imagesizex+x)*3+0] = medianx[(counter-1)/2];
		out[(y*imagesizex+x)*3+1] = mediany[(counter-1)/2];
		out[(y*imagesizex+x)*3+2] = medianz[(counter-1)/2];
	}
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // Start time
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

	dim3 numOfThreads( block_size_x, block_size_y);
	dim3 grid((imagesizex)/(numOfThreads.x - (2*kernel_size_x+2)),(imagesizey)/(numOfThreads.y - (2*kernel_size_y+2)));

	filter_median<<<grid, numOfThreads>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, 1); // Awful load balance
	filter_median<<<grid, numOfThreads>>>(dev_bitmap, dev_input, imagesizex, imagesizey, 1, kernelsizey); // Awful load balance

	
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_input, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); // End time
	std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
	printf("Time elapsed: %fms\n", time_span.count()*1000);

	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros-noisy.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(kernel_size_x, kernel_size_y);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
