#include "training.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int trainingcontext_save(TrainingContext *context, FILE *file) {
    size_t res = 1;
    res = (res == 1) ? fwrite(&context->learning_rate, sizeof(double), 1, file) : res;
    res = (res == 1) ? fwrite(&context->number_of_epochs, sizeof(uint32_t), 1, file) : res;
    res = (res == 1) ? fwrite(&context->number_of_examples, sizeof(uint32_t), 1, file) : res;

    if (res != 1) {
        perror("fwrite() failed at linearlayer_save()");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int trainingcontext_load(TrainingContext *context, FILE *file) {
    size_t res = 1;
    res = (res == 1) ? fread(&context->learning_rate, sizeof(double), 1, file) : res;
    res = (res == 1) ? fread(&context->number_of_epochs, sizeof(uint32_t), 1, file) : res;
    res = (res == 1) ? fread(&context->number_of_examples, sizeof(uint32_t), 1, file) : res;

    if (res != 1) {
        perror("fread() failed at linearlayer_load()");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}