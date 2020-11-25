/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */
float multiply(float a, float b)
{
	return a * b;
}

float add(float a, float b)
{
	return a + b;
}


int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
//	spec.setCPUThreads(<integer value>);
	skepu::setGlobalBackendSpec(spec);


	/* Skeleton instances */
	auto CombMap = skepu::MapReduce<2>(multiply, add);
	auto MultMap = skepu::Map<2>(multiply);
	auto AddReduce = skepu::Reduce(add);

	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f), v3(size);

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu::benchmark::measureExecTime([&]
	{
		resComb = CombMap(v1, v2);
	});

	auto timeSep = skepu::benchmark::measureExecTime([&]
	{
		MultMap(v3, v1, v2);
		resSep = AddReduce(v3);
	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << (timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
