#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "training.h"

#define RANDOM(min, max) (((max) - (min)) * (double)rand() / RAND_MAX + (min))

typedef struct layerbackwardcontext {
    bool hidden_layer;
    double learning_rate;
    uint32_t label;

    double *input;
    double *output;

    double *layer_errors;
    uint32_t next_layer_output_size;
    double *next_layer_errors;
} LayerBackwardContext;

typedef enum activationfunction {
    LINEAR_ACTIVATION,
    SIGMOID_ACTIVATION,
    SOFTMAX_ACTIVATION,
} ActivationFunction;

typedef struct layer {
    uint32_t input_size;
    double *biases;
    double **weights;
    uint32_t output_size;
    ActivationFunction activation_function;
} Layer;

Layer layer_create(uint32_t input_size, ActivationFunction activation_function, uint32_t output_size);
void layer_initialize(Layer *layer);

void layer_forward_sigmoid(Layer *layer, double *input, double *output);
void layer_forward_softmax(Layer *layer, double *input, double *output);
void layer_forward(Layer *layer, double *input, double *output);

void layer_backward_sigmoid(Layer *layer, LayerBackwardContext *context);
void layer_backward_softmax(Layer *layer, LayerBackwardContext *context);
void layer_backward(Layer *layer, LayerBackwardContext *context);

void layer_destroy(Layer *layer);

int layer_save(Layer *layer, FILE *file);
int layer_load(Layer *layer, FILE *file);

double sigmoid(double x);
double sigmoid_derivative(double sigmoid_x);

#endif
