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


unsigned char median_kernel(skepu::Region2D<unsigned char> image, skepu::Vec<unsigned char> sorted, size_t elemPerPx)
{
	// your code here
	unsigned int idx = 0;
	for(int y = -image.oi; y <= image.oi; y++) {
		for(int x = -image.oj; x <= image.oj; x += elemPerPx) {
			sorted[idx] = image(y,x);
			idx++;
		}
	}
	float temp = 0;
	/*unsigned int n = 2*image.oi+1;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n-i-1; j++) {
			if(sorted.data[j] > sorted.data[j+1]) {
				temp = sorted.data[j];
				sorted.data[j] = sorted.data[j+1];
				sorted.data[j+1] = temp;
			}
		}
	}*/

	return sorted[image.oi];
}



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
	auto calculateMedian = skepu::MapOverlap(median_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, sorted, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


