#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define RANDOM(min, max) (((max) - (min)) * (double)rand() / RAND_MAX + (min))

typedef enum activationfunction {
    SIGMOID_ACTIVATION,
} ActivationFunction;

typedef struct layertrainingcontext {
    bool hidden_layer;
    double learning_rate;
    uint32_t label;

    double *input;
    double *output;

    double *layer_errors;
    uint32_t next_layer_output_size;
    double *next_layer_errors;
} LayerTrainingContext;

typedef struct layer {
    uint32_t input_size;
    double *biases;
    double **weights;
    uint32_t output_size;
    ActivationFunction activation_function;
    double (*activate)(double x);
} Layer;

Layer layer_create(uint32_t input_size, ActivationFunction activation_function, uint32_t output_size);
void layer_initialize(Layer *layer);
void layer_forward(Layer *layer, double *input, double *output);
void layer_backward_sigmoid(Layer *layer, LayerTrainingContext *context);
void layer_backward(Layer *layer, LayerTrainingContext *context);
void layer_destroy(Layer *layer);

void layer_save(Layer *layer, const char *filename, bool verbose);
void layer_load(Layer *layer, const char *filename, bool verbose);

double sigmoid(double x);
double sigmoid_derivative(double sigmoid_x);

#endif
