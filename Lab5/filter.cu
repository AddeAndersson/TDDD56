// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
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

// Use these for setting shared memory size.
#define maxKernelSizeX 32
#define maxKernelSizeY 32
#define block_size_x 32
#define block_size_y 32
#define kernel_size_x 3
#define kernel_size_y 3

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
	__shared__ unsigned char shared_mem[maxKernelSizeX * 3][maxKernelSizeY]; // shared memory
	
	// map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// clamp to edge of image
	y = min(max(y, 0), imagesizey-1);
	x = min(max(x, 0), imagesizex-1);

    // Each thread copies its pixel of the block to shared memory
	shared_mem[threadIdx.x*3+0][threadIdx.y] = image[(y*imagesizex+x)*3+0]; // r
	shared_mem[threadIdx.x*3+1][threadIdx.y] = image[(y*imagesizex+x)*3+1]; // g
	shared_mem[threadIdx.x*3+2][threadIdx.y] = image[(y*imagesizex+x)*3+2]; // b
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
				// Use max and min to avoid branching!
				//int yy = min(max(y+dy, 0), kernelsizey-1);
				//int xx = min(max(x+dx, 0), kernelsizex-1);

				//int thread_nr = min(max(0, (thread_idx + ((y+dy)*blockDim.x) + x+dx)), divby);
				//int kernel = (maxKernelSizeX * 2 + 1) * (maxKernelSizeY * 2 + 1) * 3;
				
				//int idx = max(min(thread_nr, divby), 0);
				
				// Clamp inside of kernel
				int idx_x = threadIdx.x + dx;
				int idx_y = threadIdx.y + dy;
				
				// Sum r,g,b channels
				sumx += shared_mem[idx_x*3+0][idx_y]; // r
				sumy += shared_mem[idx_x*3+1][idx_y]; // g
				sumz += shared_mem[idx_x*3+2][idx_y]; // b
				//printf("shared mem r %i\n", shared_mem[idx_x*3+1][idx_y]);
			}
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
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
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	dim3 numOfThreads( block_size_x, block_size_y);
	dim3 grid((imagesizex+kernelsizex-1)/kernelsizex,(imagesizey+kernelsizey-1)/kernelsizey); // Maybe bad
	filter<<<grid,numOfThreads>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
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
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

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
