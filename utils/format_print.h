//
// C++
// Created by sjfeng
//
//
#pragma once

void format_array(float *array, int size, int newline=10){
    for (int i = 0; i < size; ++i){
        printf("%.3f, ", array[i]);
        if (i % newline == newline - 1){
            printf("\n");
        }
    }
    printf("\n\t");
}

void format_array(uint32_t *array, int size, int newline=10){
    for (int i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % newline == newline - 1){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}
