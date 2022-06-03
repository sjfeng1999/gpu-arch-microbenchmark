# GPU Arch Microbenchmark


## Prerequisites
1. install `turingas` compiler
    > `git clone --recursive git@github.com:sjfeng1999/gpu-arch-microbenchmark.git`  
    > `cd turingas`  
    > `python setup.py install`  

## Usage    
1. `mkdir build && cd build`
2. `cmake .. && make`
3. `python ../compile_sass.py -arch=<70|75|80>`

## Microbenchmark

### 1. Memory Latency

|Device                      |Latency    |  Turing RTX-2070 |
|:--------------------------:|:---------:|:----------------:|
|Global Latency              |cycle      | 1000 ~ 1200      |
|TLB Latency                 |cycle      | 472              |
|L2 Latency                  |cycle      | 236              |
|L1 Latency                  |cycle      | 32               |  
|Shared Latency              |cycle      | 23               |  
|Constant Latency            |cycle      | 448              |
|Constant L2 Latency         |cycle      | 62               |
|Constant L1 Latency         |cycle      | 4                |  

- const L1-cache is as fast as register.

### 2. Memory Bandwidth  

1. Single Thread
   
|Device          | Bandwidth   | Turing RTX-2070 |
|:--------------:|:-----------:|:---------------:|
|Global  LDG.128 | GB/s        |8.12             |
|Global  LDG.64  | GB/s        |8.78             |
|Global  LDG.32  | GB/s        |4.18             |
|Shared  LDS.128 | GB/s        |19.12            |
|Shared  LDS.64  | GB/s        |8.51             |
|Shared  LDS.32  | GB/s        |4.21             |

- LDG.64 > LDG.128 ?

### 3. Cache Linesize

|Device                      | Linesize  | Turing RTX-2070  |
|:--------------------------:|:---------:|:----------------:|
|L2 Linesise                 |bytes      | 64               |
|L1 Linesize                 |bytes      | 32               |
|Constant L2 Linesise        |bytes      | 256              |
|Constant L1 Linesize        |bytes      | 32               |

### 4. Reg Bankconflict

| Instruction |CPI      | conflict | without conflict | reg reuse | double reuse |
|:-----------:|:-------:|:--------:|:----------------:|:---------:|:------------:|
|FFMA         |  cycle  | 3.516    | 2.969            |  2.938    |  2.938       |
|IADD3        |  cycle  | 3.031    | 2.062            |  2.031    |  2.031       |


### 5. Shared Bankconflict

| Memory Load            | Latency   | Turing RTX-2070  |
|:----------------------:|:---------:|:----------------:|
| Single                 | cycle     |  23              |
| Vector2 X 2            | cycle     |  27              |
| Conflict Strided       | cycle     |  41              |
| Conlict-Free Strided   | cycle     |  32              |



## Roadmap

- [ ] total memory bandwidth
- [ ] warp schedule
- [ ] L1/L2 cache n-way k-set

# Citation
- Jia, Zhe, et al. "Dissecting the NVIDIA volta GPU architecture via microbenchmarking." arXiv preprint arXiv:1804.06826 (2018).
- Jia, Zhe, et al. "Dissecting the NVidia Turing T4 GPU via microbenchmarking." arXiv preprint arXiv:1903.07486 (2019).
- Yan, Da, Wei Wang, and Xiaowen Chu. "Optimizing batched winograd convolution on GPUs." Proceedings of the 25th ACM SIGPLAN symposium on principles and practice of parallel programming. 2020. [**(turingas)**](https://github.com/daadaada/turingas)
