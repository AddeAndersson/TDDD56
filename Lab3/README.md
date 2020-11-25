# Lab 3: High-level parallelism with skeleton programming

## Dot product

* _Question 1.1_: Why does SkePU have a fused "fused" ``MapReduce`` when there already are separate ``Map`` and ``Reduce`` skeletons?

* _Question 1.2_: Is there any practical reason to ever use separate ``Map`` and ``Reduce`` in sequence?

* _Question 1.3_: Is there a SkePU backend which is always more efficient to use, or does this depend on the problem size? Why? Either show with measurements or provide a valid reasoning

* _Question 1.4_: Try measuring the parallel backends with ``measureExecTime`` exchanged for ``measureExecTimeIdempotent``. This measurement does a "cold run" of the lambda expression before running the proper measurement. Do you see a difference for some backends, and if so, why?


## Averaging filters

* _Question 2.1_: Which version of the averaging filter (unified, separable) is the most efficient? Why?

## Median filtering

* _Question 3.1_: In data-parallel skeletons like ``MapOverlap``, all elements are processed independently of each other. Is this a good fit for the median filter? Why/why not?


* _Question 3.2_:  Describe the sequence of instructions executed in your userfunction. Is it data dependent? What does this mean for e.g., automatic
vectorization, or the GPU backend?
