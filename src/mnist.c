#include "mnist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t read_uint32(FILE *f) {
    uint32_t result;
    if (fread(&result, sizeof(result), 1, f) != 1) {
        perror("Failed to read uint32_t");
        exit(EXIT_FAILURE);
    }
    return __builtin_bswap32(result);
}

uint8_t *load_mnist_images(const char *filename, uint32_t *num_images, uint32_t *image_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open images file");
        exit(EXIT_FAILURE);
    }
    printf("Opened image file: %s\n", filename);

    uint32_t magic = read_uint32(f);
    magic = magic; // remove warning
    *num_images = read_uint32(f);
    uint32_t rows = read_uint32(f);
    uint32_t cols = read_uint32(f);
    *image_size = rows * cols;

    uint8_t *images = (uint8_t *)malloc((*num_images) * (*image_size));
    if (!images) {
        perror("Failed to allocate memory for images");
        exit(EXIT_FAILURE);
    }

    if (fread(images, *image_size, *num_images, f) != (size_t)(*num_images)) {
        perror("Failed to read images");
        free(images);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    return images;
}

uint8_t *load_mnist_labels(const char *filename, uint32_t *num_labels) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open labels file");
        exit(EXIT_FAILURE);
    }
    printf("Opened label file: %s\n", filename);

    uint32_t magic = read_uint32(f);
    magic = magic; // remove warning
    *num_labels = read_uint32(f);

    uint8_t *labels = (uint8_t *)malloc(*num_labels);
    if (!labels) {
        perror("Failed to allocate memory for labels");
        exit(EXIT_FAILURE);
    }

    if (fread(labels, 1, *num_labels, f) != (size_t)(*num_labels)) {
        perror("Failed to read labels");
        free(labels);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    return labels;
}
