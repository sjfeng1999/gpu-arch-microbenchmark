# GPU Arch Microbenchmark


## Prerequisites
1. install `turingas` compiler
    > `git clone git@github.com:daadaada/turingas.git`  
    > `python setup.py install`


## Usage    
1. `mkdir build && cd build`
2. `cmake .. && make`
3. `python ../compile_sass.py -arch=<70|75|80>`

## Microbenchmark

### 1. Latency

|Device                      |           |  Turing RTX-2070 |
|:--------------------------:|:---------:|:----------------:|
|Global Latency              |cycle      | TBD              |
|L2 Latency                  |cycle      | 236              |
|L1 Latency                  |cycle      | 32               |  
|Shared Latency              |cycle      | 23               |  
|Constant Latency            |cycle      | 448              |
|Constant L2 Latency         |cycle      | 62               |
|Constant L1 Latency         |cycle      | 4                |  

- const L1-cache is as fast as register.

### 2. Cache Linesize


|Device                      |           | Turing RTX-2070  |
|:--------------------------:|:---------:|:----------------:|
|L2 Linesise                 |bytes      | 64               |
|L1 Linesize                 |bytes      | 32               |
|Constant L2 Linesise        |bytes      | 256              |
|Constant L1 Linesize        |bytes      | 32               |



### 3. Reg Bankconflict

| Instruction |         | conflict | without conflict | reg reuse |
|:-----------:|:-------:|:--------:|:----------------:|:---------:|
|FFMA         |  CPI    | 3.516    | 2.969            |  2.938    |
|IADD3        |  CPI    | 3.031    | 2.062            |  2.031    |


### 4. Shared Bankconflict

| Memory Load            |           | Turing RTX-2070  |
|:----------------------:|:---------:|:----------------:|
| Single                 | cycle     |  23              |
| Vector2 X 2            | cycle     |  27              |
| Conflict Strided       | cycle     |  41              |
| Conlict-Free Strided   | cycle     |  32              |



# Citation
- Jia, Zhe, et al. "Dissecting the NVIDIA volta GPU architecture via microbenchmarking." arXiv preprint arXiv:1804.06826 (2018).
- Jia, Zhe, et al. "Dissecting the NVidia Turing T4 GPU via microbenchmarking." arXiv preprint arXiv:1903.07486 (2019).
- Yan, Da, Wei Wang, and Xiaowen Chu. "Optimizing batched winograd convolution on GPUs." Proceedings of the 25th ACM SIGPLAN symposium on principles and practice of parallel programming. 2020. [**(turingas)**](https://github.com/daadaada/turingas)
