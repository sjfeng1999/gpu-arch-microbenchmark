//
//
// Created by sjfeng
//
//
#pragma once
#include <cstdint>

template <typename T>
void format_array(T *array, size_t size);


template <>
void format_array<float>(float *array, size_t size){
    for (size_t i = 0; i < size; ++i){
        printf("%.3f, ", array[i]);
        if (i % 10 == 9){
            printf("\n");
        }
        if (i % 100 == 99){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}

template <>
void format_array<uint32_t>(uint32_t *array, size_t size){
    for (size_t i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % 10 == 9){
            printf("\n");
        }
        if (i % 100 == 99){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}
