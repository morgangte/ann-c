#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

#define NUMBER_OF_IMAGES_TRAIN 60000
#define NUMBER_OF_IMAGES_TEST 10000

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE (IMAGE_HEIGHT * IMAGE_WIDTH)

#define INPUT_SIZE IMAGE_SIZE
#define HIDDEN_SIZE 89
#define OUTPUT_SIZE 10

void prepare_input(uint8_t *raw, double *prepared, uint32_t size);
void display_image(double *image);

#endif  // MNIST_H
