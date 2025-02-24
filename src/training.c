#include "training.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void trainingcontext_save(TrainingContext *context, const char *filename, bool verbose) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen() failed at linearlayer_save()");
        exit(EXIT_FAILURE);
    }

    size_t res = 1;
    res = (res == 1) ? fwrite(&context->learning_rate, sizeof(double), 1, f) : res;
    res = (res == 1) ? fwrite(&context->number_of_epochs, sizeof(uint32_t), 1, f) : res;
    fclose(f);

    if (res != 1) {
        perror("fwrite() failed at linearlayer_save()");
        exit(EXIT_FAILURE);
    }
    if (verbose) {
        printf("Successfully saved NeuralNetwork training context to '%s'\n", filename);
    }
}

void trainingcontext_load(TrainingContext *context, const char *filename, bool verbose) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("fopen() failed at linearlayer_load()");
        exit(EXIT_FAILURE);
    }

    size_t res = 1;
    res = (res == 1) ? fread(&context->learning_rate, sizeof(double), 1, f) : res;
    res = (res == 1) ? fread(&context->number_of_epochs, sizeof(uint32_t), 1, f) : res;
    fclose(f);

    if (res != 1) {
        perror("fread() failed at linearlayer_load()");
        exit(EXIT_FAILURE);
    }
    if (verbose) {
        printf("Successfully loaded NeuralNetwork training context from '%s'\n", filename);
    }
}