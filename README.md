# GPU Arch Microbenchmark


## Prerequisites
1. install `turingas` compiler
    > `git clone git@github.com:daadaada/turingas.git`  
    > `python setup.py install`


## Usage    
1. `mkdir build && cd build`
2. `cmake .. && make`
3. `python ../compile_sass.py -arch=<70|75|80>`

## Benchmark

### 1. Latency

|Device                      |           |  RTX-2070 |
|:--------------------------:|:---------:|:---------:|
|Global Latency              |cycle      | TBD       |
|L2 Latency                  |cycle      | 236       |
|L1 Latency                  |cycle      | 32        |  
|Constant Latency            |cycle      | 465       |
|Constant L2 Latency         |cycle      | 65        |
|Constant L1 Latency         |cycle      | 7         |  


### 2. Cache Linesize


|Device                      |           |  RTX-2070 |
|:--------------------------:|:---------:|:---------:|
|L2 Linesise                 |bytes      | TBD       |
|L1 Linesize                 |bytes      | 32        |
|Constant L2 Linesise        |bytes      | TBD       |
|Constant L1 Linesize        |bytes      | TBD       |



### 3. Reg Bankconflict

| Instruction | conflict | without conflict |
|:-----------:|:--------:|:----------------:|
|FFMA         | 1.484    | 1.758            |



# Citation
- Jia, Zhe, et al. "Dissecting the NVIDIA volta GPU architecture via microbenchmarking." arXiv preprint arXiv:1804.06826 (2018).
- Jia, Zhe, et al. "Dissecting the NVidia Turing T4 GPU via microbenchmarking." arXiv preprint arXiv:1903.07486 (2019).
- Yan, Da, Wei Wang, and Xiaowen Chu. "Optimizing batched winograd convolution on GPUs." Proceedings of the 25th ACM SIGPLAN symposium on principles and practice of parallel programming. 2020. [**(turingas)**](https://github.com/daadaada/turingas)
