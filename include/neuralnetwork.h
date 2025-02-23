#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>

#include "layer.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define LEARNING_RATE 0.01
#define NUMBER_OF_EPOCHS 10

typedef struct {
    double weights[INPUT_SIZE][HIDDEN_SIZE];
    double biases[HIDDEN_SIZE];
} HiddenLayer;

typedef struct {
    double weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double biases[OUTPUT_SIZE];
} OutputLayer;

typedef struct neuralnetwork {
    LinearLayer linear_layer0;
    double *hidden0;
    SigmoidLayer sigmoid_layer0;
    double *hidden1;
    LinearLayer linear_layer1;
    double *hidden2;
    SigmoidLayer sigmoid_layer1;
    double *output;
} NeuralNetwork;

NeuralNetwork neuralnetwork_create();

void neuralnetwork_initialize(NeuralNetwork *network);
void initialize_layer(HiddenLayer *hidden, OutputLayer *output);

void prepare_input(uint8_t *raw, double *prepared, uint32_t size);

void neuralnetwork_forward(NeuralNetwork *network, double *input);
void forward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output);

void neuralnetwork_backward(NeuralNetwork *network, double *input, uint8_t label);
void backward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output, uint8_t label);

void neuralnetwork_train(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images, uint32_t number_of_epochs);
void train(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images, int num_epochs);

uint8_t max_index(double *array, uint8_t size);
uint8_t neuralnetwork_ask(NeuralNetwork *network, uint8_t *image);
uint8_t recognize(HiddenLayer *hidden, OutputLayer *output, uint8_t *image);

double neuralnetwork_benchmark(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images);
double calculate_accuracy(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images);

void neuralnetwork_destroy(NeuralNetwork *network);

void neuralnetwork_save(NeuralNetwork *network, const char *base_filename);
void neuralnetwork_load(NeuralNetwork *network, const char *base_filename);

void save_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output);
void load_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output);

#endif  // NEURAL_NETWORK_H
