#ifndef TRAINING_CONTEXT_H
#define TRAINING_CONTEXT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct trainingcontext {
    double learning_rate;
    uint32_t number_of_epochs;
    uint32_t number_of_examples;
} TrainingContext;

int trainingcontext_save(TrainingContext *context, FILE *file);
int trainingcontext_load(TrainingContext *context, FILE *file);

#endif  // TRAINING_CONTEXT_H