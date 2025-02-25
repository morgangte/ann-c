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

    LayerBackwardContext output_context = {
        .hidden_layer = false,
        .learning_rate = learning_rate,
        .label = label,
        .layer_input = network->hidden1,
        .activation_function = SIGMOID_ACTIVATION,
        .activation_output = network->output,
        .layer_errors = output_errors,
        .next_layer_output_size = 0,
        .next_layer_errors = NULL,
    };
    linearlayer_backward(&network->linear_layer1, &output_context);

    LayerBackwardContext hidden_context = {
        .hidden_layer = true,
        .learning_rate = learning_rate,
        .label = 0,
        .layer_input = input,
        .activation_function = SIGMOID_ACTIVATION,
        .activation_output = network->hidden1,
        .layer_errors = hidden_errors,
        .next_layer_output_size = OUTPUT_SIZE,
        .next_layer_errors = output_errors,
    };
    linearlayer_backward(&network->linear_layer0, &hidden_context);
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
