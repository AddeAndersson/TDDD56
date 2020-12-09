# Lab 5: Image filtering with CUDA and using shared memory
## Part 1. Make a low-pass box filter with shared memory

* How much data did you put in shared memory?

* How much data does each thread copy to shared memory?

* How did you handle the necessary overlap between the blocks?

* If we would like to increase the block size, about how big blocks would be safe to use in this case? Why?

* How much speedup did you get over the naive version? For what filter size?

* Is your access to global memory coalesced? What should you do to get that?

## Part 2. Separable LP filter

* How much speedup did you get over the non-separated? For what filter size?

## Part 3. Convolution filters with kernel weights, gaussian filters

* Compare the visual result to that of the box filter. Is the image LP-filtered with the weighted kernel noticeably better?

* What was the difference in time to a box filter of the same size (5x5)?

* If you want to make a weighted kernel customizable by weights from the host, how would you deliver the weights to the GPU?

## Part 4. Median filter

* What kind of algorithm did you implement for finding the median?

* What filter size was best for reducing noise?

* (non-mandatory): Compare the result of the separable and full median filters.

* (non-mandatory): Compare the difference in performance of the separable and full median filters.