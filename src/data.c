#include "data.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t read_uint32(FILE *file) {
    uint32_t integer;
    if (fread(&integer, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read uint32_t at read_uint32()");
        exit(EXIT_FAILURE);
    }
    return __builtin_bswap32(integer);
}

uint8_t *load_images(const char *filename, uint32_t *num_images, uint32_t *image_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open images file at load_images()");
        exit(EXIT_FAILURE);
    }

    *num_images = read_uint32(f);
    uint32_t rows = read_uint32(f);
    uint32_t cols = read_uint32(f);
    *image_size = rows * cols;

    uint8_t *images = (uint8_t *)malloc((*num_images) * (*image_size));
    if (!images) {
        perror("Failed to allocate memory for images at load_images()");
        exit(EXIT_FAILURE);
    }

    if (fread(images, *image_size, *num_images, f) != (size_t)(*num_images)) {
        perror("Failed to read images at load_images()");
        free(images);
        exit(EXIT_FAILURE);
    }

    printf("Successfully loaded images from '%s'\n", filename);
    fclose(f);
    return images;
}

uint8_t *load_labels(const char *filename, uint32_t *num_labels) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open labels file at load_labels()");
        exit(EXIT_FAILURE);
    }

    *num_labels = read_uint32(f);

    uint8_t *labels = (uint8_t *)malloc(*num_labels);
    if (!labels) {
        perror("Failed to allocate memory for labels at load_labels()");
        exit(EXIT_FAILURE);
    }

    if (fread(labels, 1, *num_labels, f) != (size_t)(*num_labels)) {
        perror("Failed to read labels at load_labels()");
        free(labels);
        exit(EXIT_FAILURE);
    }

    printf("Successfully loaded labels from '%s'\n", filename);
    fclose(f);
    return labels;
}
