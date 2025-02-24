#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>

#include "layer.h"
#include "training.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

typedef struct neuralnetwork {
    LinearLayer linear_layer0;
    double hidden0[HIDDEN_SIZE];
    SigmoidLayer sigmoid_layer0;
    double hidden1[HIDDEN_SIZE];
    LinearLayer linear_layer1;
    double hidden2[OUTPUT_SIZE];
    SigmoidLayer sigmoid_layer1;
    double output[OUTPUT_SIZE];
} NeuralNetwork;

NeuralNetwork neuralnetwork_create();
void neuralnetwork_initialize(NeuralNetwork *network);

void prepare_input(uint8_t *raw, double *prepared, uint32_t size);
void neuralnetwork_forward(NeuralNetwork *network, double *input);
void neuralnetwork_backward(NeuralNetwork *network, double *input, uint8_t label, double learning_rate);

void neuralnetwork_train(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images, TrainingContext *context);

uint8_t max_index(double *array, uint8_t size);
uint8_t neuralnetwork_ask(NeuralNetwork *network, uint8_t *image);
double neuralnetwork_benchmark(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images);

void neuralnetwork_destroy(NeuralNetwork *network);

void neuralnetwork_save(NeuralNetwork *network, TrainingContext *context, const char *base_filename);
void neuralnetwork_load(NeuralNetwork *network, TrainingContext *context, const char *base_filename);

#endif  // NEURAL_NETWORK_H
