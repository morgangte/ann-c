#ifndef FASHION_MNIST_H
#define FASHION_MNIST_H

#include <stdint.h>

#define NUMBER_OF_IMAGES_TRAIN 60000
#define NUMBER_OF_IMAGES_TEST 10000

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE (IMAGE_HEIGHT * IMAGE_WIDTH)
#define IMAGE_DIMENSION 1

#define INPUT_SIZE (IMAGE_SIZE * IMAGE_DIMENSION)
#define HIDDEN_SIZE 120
#define OUTPUT_SIZE 10

void prepare_input(uint8_t *raw, double *prepared, uint32_t size);

#endif  // FASHION_MNIST_H
