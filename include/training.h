#ifndef TRAINING_CONTEXT_H
#define TRAINING_CONTEXT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct trainingcontext {
    double learning_rate;
    uint32_t number_of_epochs;
} TrainingContext;

void trainingcontext_save(TrainingContext *context, const char *filename, bool verbose);
void trainingcontext_load(TrainingContext *context, const char *filename, bool verbose);

#endif  // TRAINING_CONTEXT_H