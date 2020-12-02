# Lab 3: High-level parallelism with skeleton programming

## Dot product

* _Question 1.1_: Why does SkePU have a fused "fused" ``MapReduce`` when there already are separate ``Map`` and ``Reduce`` skeletons?

It is generally faster to combine multiple operations in a single kernel, and since Map + Reduce is a common sequence, SkePU has a fused version.

* _Question 1.2_: Is there any practical reason to ever use separate ``Map`` and ``Reduce`` in sequence?

Yes. For instance, if you want to use another operation between the map and the reduce calls.

* _Question 1.3_: Is there a SkePU backend which is always more efficient to use, or does this depend on the problem size? Why? Either show with measurements or provide a valid reasoning

It strongly depends on the implementation of SkePU, which backend is most efficient. If the compiler can make better optimiziations for a certain backend it will generally be better but probably not for every task. In terms of backend performance, OpenCL and CUDA are very similar. For the CPU it is generally faster to hand-optimize the code rather than relying on OpenMP. However that would require a lot more work and for many cases the multithreding by the OpenMP backend is sufficient.

* _Question 1.4_: Try measuring the parallel backends with ``measureExecTime`` exchanged for ``measureExecTimeIdempotent``. This measurement does a "cold run" of the lambda expression before running the proper measurement. Do you see a difference for some backends, and if so, why?

Yes, we see a time reduction by a magnitude of about 25 for the GPU-backends, this is due to the caching caused by the cold run. After the cold run the data is already in memory which means that for the next run the memcpy can return instantly. Since this problem is very memory-bound, we see a huge boost in performance. The same goes for the CPU-backend OpenMP but the performance gain is smaller since the data does not have to go thorugh the PCI-bus to the GPU, which is relatively slow.

## Averaging filters
### Box filter
![Average filter](result33x33-separable.png)
### Gaussian filter
![Gaussian filter](result33x33-gaussian.png)

* _Question 2.1_: Which version of the averaging filter (unified, separable) is the most efficient? Why?

Seperable is more efficient since it only has to access the radius*2 elements per pixel compared to radius^2.

## Median filtering

![Median filter](result33x33-median.png)

* _Question 3.1_: In data-parallel skeletons like ``MapOverlap``, all elements are processed independently of each other. Is this a good fit for the median filter? Why/why not?

Yes, beacuse we only want to take the median of the elements inside of the filter-kernel, therefore we have to sort the elements differently in each kernel call.


* _Question 3.2_:  Describe the sequence of instructions executed in your userfunction. Is it data dependent? What does this mean for e.g., automatic
vectorization, or the GPU backend?

It is data dependent since we first have to store the data in a local array then sort it, therefore it can't be vectorized automatically.
