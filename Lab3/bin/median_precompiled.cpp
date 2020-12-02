#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
#define SKEPU_OPENCL
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"


unsigned char median_kernel(skepu::Region2D<unsigned char> image, skepu::Vec<unsigned char> sortedasd, size_t elemPerPx)
{
	// your code here
	unsigned int idx = 0;
	unsigned char sorted[1000];
	for(int y = -image.oi; y <= image.oi; y++) {
		for(int x = -image.oj; x <= image.oj; x += elemPerPx) {
			sorted[idx] = image(y,x);
			idx++;
		}
	}
	float temp = 0;
	unsigned int n = 2*image.oi+1;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n-i-1; j++) {
			if(sorted[j] > sorted[j+1]) {
				temp = sorted[j];
				sorted[j] = sorted[j+1];
				sorted[j+1] = temp;
			}
		}
	}

	return sorted[image.oi];
}




struct skepu_userfunction_skepu_skel_0calculateMedian_median_kernel
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<skepu::Vec<unsigned char>>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::ReadWrite, };

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region2D<unsigned char> image, skepu::Vec<unsigned char> sortedasd, unsigned long elemPerPx)
{
	// your code here
	unsigned int idx = 0;
	unsigned char sorted[1000];
	for(int y = -image.oi; y <= image.oi; y++) {
		for(int x = -image.oj; x <= image.oj; x += elemPerPx) {
			sorted[idx] = image(y,x);
			idx++;
		}
	}
	float temp = 0;
	unsigned int n = 2*image.oi+1;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n-i-1; j++) {
			if(sorted[j] > sorted[j+1]) {
				temp = sorted[j];
				sorted[j] = sorted[j+1];
				sorted[j+1] = temp;
			}
		}
	}

	return sorted[image.oi];
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region2D<unsigned char> image, skepu::Vec<unsigned char> sortedasd, unsigned long elemPerPx)
{
	// your code here
	unsigned int idx = 0;
	unsigned char sorted[1000];
	for(int y = -image.oi; y <= image.oi; y++) {
		for(int x = -image.oj; x <= image.oj; x += elemPerPx) {
			sorted[idx] = image(y,x);
			idx++;
		}
	}
	float temp = 0;
	unsigned int n = 2*image.oi+1;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n-i-1; j++) {
			if(sorted[j] > sorted[j+1]) {
				temp = sorted[j];
				sorted[j] = sorted[j+1];
				sorted[j+1] = temp;
			}
		}
	}

	return sorted[image.oi];
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	skepu::Vector<unsigned char> sorted(radius+1+(radius+1)*imageInfo.elementsPerPixel);
	skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_0calculateMedian_median_kernel, bool, CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(false);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, sorted, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


