#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>

uint32_t read_uint32(FILE *f);

uint8_t *load_mnist_images(const char *filename, uint32_t *num_images, uint32_t *image_size);
uint8_t *load_mnist_labels(const char *filename, uint32_t *num_labels);

#endif
