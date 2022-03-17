//
// CUDA
// Created by sjfeng
//

#pragma once 

#define UPPER_DIV(x, y)        ((x + y - 1) / y)

constexpr int kWarpSize = 32