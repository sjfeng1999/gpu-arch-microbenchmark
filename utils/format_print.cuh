//
// C++
// Created by sjfeng
//
//
#pragma once

#include "stdio.h"

void formatArray(float* array, int size, int newline=10){
    for (int i = 0; i < size; ++i){
        printf("%.3f, ", array[i]);
        if (i % newline == newline - 1){
            printf("\n");
        }
    }
    printf("\n\t");
}

void formatArray(uint* array, int size, int newline=10){
    for (int i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % newline == newline - 1){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}

void formatArray(int* array, int size, int newline=10){
    for (int i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % newline == newline - 1){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}
