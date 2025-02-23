#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define LEARNING_RATE 0.01
#define EPOCHS 10

typedef struct {
    double weights[INPUT_SIZE][HIDDEN_SIZE];
    double biases[HIDDEN_SIZE];
} HiddenLayer;

typedef struct {
    double weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double biases[OUTPUT_SIZE];
} OutputLayer;

void initialize_layer(HiddenLayer *hidden, OutputLayer *output);

double sigmoid(double x);
double sigmoid_derivative(double x);

void forward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output);
void backward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output, uint8_t label);

void train(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images, int num_epochs);

uint8_t recognize(HiddenLayer *hidden, OutputLayer *output, uint8_t *image);
double calculate_accuracy(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images);

void save_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output);
void load_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output);

#endif  // NEURAL_NETWORK_H
