# Lab 4: Introduction to CUDA
## Part 1. Trying out CUDA
### a) Compile and run simple.cu
* How many cores will simple.cu use, max, as written? How many SMs?

    It will use as many cores needed for the specified block size (multiple of available threads), it will launch as many SMs as needed to fit all the blocks. This example utilizes only one SM.

### b) Modifying simple.cu
* Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?

    Yes it is, it could maybe vary if the floating point precision is different between the CPU and the device.

## Part 2. Performance and block size
### a) Array computation from C to CUDA
* How do you calculate the index in the array, using 2-dimensional blocks?

    The index can be calculated by using the code snippet:
    `int idx = (threadIdx.x + (threadIdx.y * blockDim.x));`
### b) Larger data set and timing with CUDA Events
* What happens if you use too many threads per block?

It is unpredictable since there is a limit to the number of threads per block, since all threads of a block are expected to redide on the same CUDA core.

* At what data size is the GPU faster than the CPU?

For 16x16 threads/block the GPU is faster at N>32 without memcpy and N>64 with memcpys, however the CPU is not multithreaded.

* What block size seems like a good choice? Compared to what?

Somewhere in the middle, its program specific.

* Write down your data size, block size and timing data for the best GPU performance you can get.

N=256, 16x16 threads, (N/16)x(N/16) blocks aka 16x16, 0.0097ms