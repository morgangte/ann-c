#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>

#include "layer.h"

typedef struct backwardcontext {
    double learning_rate;
    uint32_t label;
    uint16_t number_of_layers;
    double **layers_errors;
} BackwardContext;

typedef struct neuralnetwork {
    uint16_t layers_capacity;
    uint16_t layers_size;
    Layer *layers;
    double **layers_outputs;
} NeuralNetwork;

uint8_t max_index(double *array, uint8_t size);

BackwardContext backwardcontext_create(NeuralNetwork *network, double learning_rate);
void backwardcontext_destroy(BackwardContext *context);

NeuralNetwork neuralnetwork_create(uint16_t number_of_layers);
void neuralnetwork_add_layer(NeuralNetwork *network, uint32_t input_size, ActivationFunction activation_function, uint32_t output_size);
void neuralnetwork_initialize(NeuralNetwork *network);

void neuralnetwork_forward(NeuralNetwork *network, double *input);
void neuralnetwork_backward(NeuralNetwork *network, double *input, BackwardContext *backward_context);
void neuralnetwork_train(NeuralNetwork *network, double *inputs, uint8_t *labels, TrainingContext *context);

uint8_t neuralnetwork_ask(NeuralNetwork *network, double *input);
double neuralnetwork_benchmark(NeuralNetwork *network, double *inputs, uint8_t *labels, uint32_t number_of_inputs);

uint32_t neuralnetwork_input_size(NeuralNetwork *network);
uint32_t neuralnetwok_output_size(NeuralNetwork *network);
double *neuralnetwork_output(NeuralNetwork *network);

void neuralnetwork_destroy(NeuralNetwork *network);

void neuralnetwork_save(NeuralNetwork *network, TrainingContext *context, const char *filename);
void neuralnetwork_load(NeuralNetwork *network, TrainingContext *context, const char *filename);

#endif  // NEURAL_NETWORK_H
