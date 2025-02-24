#include "neuralnetwork.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

NeuralNetwork neuralnetwork_create() {
    return (NeuralNetwork){
        .linear_layer0 = linearlayer_create(INPUT_SIZE, HIDDEN_SIZE),
        .sigmoid_layer0 = sigmoidlayer_create(HIDDEN_SIZE),
        .linear_layer1 = linearlayer_create(HIDDEN_SIZE, OUTPUT_SIZE),
        .sigmoid_layer1 = sigmoidlayer_create(OUTPUT_SIZE),
    };
}

void neuralnetwork_initialize(NeuralNetwork *network) {
    srand(time(NULL));
    linearlayer_initialize(&network->linear_layer0);
    linearlayer_initialize(&network->linear_layer1);
}

void prepare_input(uint8_t *raw, double *prepared, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        prepared[i] = raw[i] / 255.0;
    }
}

void neuralnetwork_forward(NeuralNetwork *network, double *input) {
    linearlayer_forward(&network->linear_layer0, input, network->hidden0);
    sigmoidlayer_forward(&network->sigmoid_layer0, network->hidden0, network->hidden1);
    linearlayer_forward(&network->linear_layer1, network->hidden1, network->hidden2);
    sigmoidlayer_forward(&network->sigmoid_layer1, network->hidden2, network->output);
}

void neuralnetwork_backward(NeuralNetwork *network, double *input, uint8_t label, double learning_rate) {
    double output_errors[OUTPUT_SIZE];
    double hidden_errors[HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double target = (i == label) ? 1.0 : 0.0;
        output_errors[i] = (network->output[i] - target) * sigmoid_derivative(network->output[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_errors[i] += network->linear_layer1.weights[i][j] * output_errors[j];
        }
        hidden_errors[i] *= sigmoid_derivative(network->hidden1[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->linear_layer1.biases[i] += learning_rate * output_errors[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            network->linear_layer1.weights[j][i] -= learning_rate * network->hidden1[j] * output_errors[i];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->linear_layer0.biases[i] += learning_rate * hidden_errors[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            network->linear_layer0.weights[j][i] -= learning_rate * input[j] * hidden_errors[i];
        }
    }
}

void neuralnetwork_train(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images, TrainingContext *context) {
    double prepared_input[INPUT_SIZE];
    for (uint32_t epoch = 0; epoch < context->number_of_epochs; epoch++) {
        printf("Running epoch %d/%d...\n", epoch + 1, context->number_of_epochs);
        for (uint32_t i = 0; i < number_of_images; i++) {
            prepare_input(&images[i * INPUT_SIZE], prepared_input, INPUT_SIZE);
            neuralnetwork_forward(network, prepared_input);
            neuralnetwork_backward(network, prepared_input, labels[i], context->learning_rate);
        }
    }
}

uint8_t max_index(double *array, uint8_t size) {
    uint8_t max_index = 0;
    for (uint8_t i = 0; i < size; i++) {
        if (array[i] > array[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

uint8_t neuralnetwork_ask(NeuralNetwork *network, uint8_t *image) {
    double prepared_input[INPUT_SIZE];

    prepare_input(image, prepared_input, INPUT_SIZE);
    neuralnetwork_forward(network, prepared_input);

    return max_index(network->output, OUTPUT_SIZE);
}

double neuralnetwork_benchmark(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images) {
    uint32_t correct_predictions = 0;
    for (uint32_t i = 0; i < number_of_images; i++) {
        uint8_t answer = neuralnetwork_ask(network, &images[i * INPUT_SIZE]);
        correct_predictions += (answer == labels[i]) ? 1 : 0;
    }

    return (double)correct_predictions / number_of_images;
}

void neuralnetwork_destroy(NeuralNetwork *network) {
    linearlayer_destroy(&network->linear_layer0);
    linearlayer_destroy(&network->linear_layer1);
}

void neuralnetwork_save(NeuralNetwork *network, TrainingContext *context, const char *base_filename) {
    char *filename = (char *)malloc(strlen(base_filename) + 8);

    strcpy(filename, base_filename);
    strcat(filename, "_ll0.bin");
    linearlayer_save(&network->linear_layer0, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_ll1.bin");
    linearlayer_save(&network->linear_layer1, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_trainingcontext.bin");
    trainingcontext_save(context, filename, true);

    free(filename);
}

void neuralnetwork_load(NeuralNetwork *network, TrainingContext *context, const char *base_filename) {
    char *filename = (char *)malloc(strlen(base_filename) + 8);

    strcpy(filename, base_filename);
    strcat(filename, "_ll0.bin");
    linearlayer_load(&network->linear_layer0, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_ll1.bin");
    linearlayer_load(&network->linear_layer1, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_trainingcontext.bin");
    trainingcontext_load(context, filename, true);

    free(filename);
}
