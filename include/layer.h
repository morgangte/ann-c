#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define RANDOM(min, max) (((max) - (min)) * (double)rand() / RAND_MAX + (min))

typedef struct linearlayer {
    uint32_t input_size;
    uint32_t output_size;
    double **weights;
    double *biases;
} LinearLayer;

typedef enum activationfunction {
    IDENTITY,
    SIGMOID,
} ActivationFunction;

typedef struct sigmoidlayer {
    uint32_t size;
} SigmoidLayer;

LinearLayer linearlayer_create(uint32_t input_size, uint32_t output_size);
void linearlayer_initialize(LinearLayer *layer);
void linearlayer_forward(LinearLayer *layer, double *input, double *output);
void linearlayer_destroy(LinearLayer *layer);

void linearlayer_save(LinearLayer *layer, const char *filename, bool verbose);
void linearlayer_load(LinearLayer *layer, const char *filename, bool verbose);

SigmoidLayer sigmoidlayer_create(uint32_t size);

double sigmoid(double x);
double sigmoid_derivative(double x);

void sigmoidlayer_forward(SigmoidLayer *layer, double *input, double *output);

#endif
