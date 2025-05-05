#include "mnist.h"

#include <stdint.h>
#include <stdio.h>

void prepare_input(uint8_t *raw, double *prepared, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        prepared[i] = raw[i] / 255.0;
    }
}

void display_image(double *image) {
    for (uint8_t i = 0; i < IMAGE_HEIGHT; i++) {
        for (uint8_t j = 0; j < IMAGE_WIDTH; j++) {
            printf("%c", image[i * IMAGE_WIDTH + j] == 0 ? '.' : 'X');
        }
        printf("\n");
    }
}
