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
        .hidden0 = (double *)malloc(sizeof(double) * HIDDEN_SIZE),
        .sigmoid_layer0 = sigmoidlayer_create(HIDDEN_SIZE),
        .hidden1 = (double *)malloc(sizeof(double) * HIDDEN_SIZE),
        .linear_layer1 = linearlayer_create(HIDDEN_SIZE, OUTPUT_SIZE),
        .hidden2 = (double *)malloc(sizeof(double) * OUTPUT_SIZE),
        .sigmoid_layer1 = sigmoidlayer_create(OUTPUT_SIZE),
        .output = (double *)malloc(sizeof(double) * OUTPUT_SIZE),
    };
}

void neuralnetwork_initialize(NeuralNetwork *network) {
    srand(time(NULL));
    linearlayer_initialize(&network->linear_layer0);
    linearlayer_initialize(&network->linear_layer1);
}

void initialize_layer(HiddenLayer *hidden, OutputLayer *output) {
    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden->biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output->biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
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

void forward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = hidden->biases[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] / 255.0 * hidden->weights[j][i];
        }
        hidden_output[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = output->biases[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_output[j] * output->weights[j][i];
        }
        output_output[i] = sigmoid(sum);
    }
}

void neuralnetwork_backward(NeuralNetwork *network, double *input, uint8_t label) {
    double output_errors[OUTPUT_SIZE];
    double hidden_errors[HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double target = (i == label) ? 1.0 : 0.0;
        output_errors[i] = (target - network->output[i]) * sigmoid_derivative(network->output[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_errors[i] += output_errors[j] * network->linear_layer1.weights[i][j];
        }
        hidden_errors[i] *= sigmoid_derivative(network->hidden1[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->linear_layer1.biases[i] += LEARNING_RATE * output_errors[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            network->linear_layer1.weights[j][i] += LEARNING_RATE * output_errors[i] * network->hidden1[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->linear_layer0.biases[i] += LEARNING_RATE * hidden_errors[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            network->linear_layer0.weights[j][i] += LEARNING_RATE * hidden_errors[i] * input[j];
        }
    }
}

void backward_pass(HiddenLayer *hidden, OutputLayer *output, uint8_t *input, double *hidden_output, double *output_output, uint8_t label) {
    double output_errors[OUTPUT_SIZE];
    double hidden_errors[HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double target = (i == label) ? 1.0 : 0.0;
        output_errors[i] = (target - output_output[i]) * sigmoid_derivative(output_output[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_errors[i] += output_errors[j] * output->weights[i][j];
        }
        hidden_errors[i] *= sigmoid_derivative(hidden_output[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output->biases[i] += LEARNING_RATE * output_errors[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output->weights[j][i] += LEARNING_RATE * output_errors[i] * hidden_output[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden->biases[i] += LEARNING_RATE * hidden_errors[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden->weights[j][i] += LEARNING_RATE * hidden_errors[i] * (input[j] / 255.0);
        }
    }
}

void neuralnetwork_train(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images, uint32_t number_of_epochs) {
    double prepared_input[INPUT_SIZE];

    for (uint32_t epoch = 0; epoch < number_of_epochs; epoch++) {
        printf("Running epoch %d/%d...\n", epoch + 1, number_of_epochs);
        for (uint32_t i = 0; i < number_of_images; i++) {
            prepare_input(&images[i * INPUT_SIZE], prepared_input, INPUT_SIZE);
            neuralnetwork_forward(network, prepared_input);
            neuralnetwork_backward(network, prepared_input, labels[i]);
        }
    }
}

void train(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images, int num_epochs) {
    double hidden_output[HIDDEN_SIZE];
    double output_output[OUTPUT_SIZE];

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int i = 0; i < num_images; i++) {
            forward_pass(hidden, output, &images[i * INPUT_SIZE], hidden_output, output_output);
            backward_pass(hidden, output, &images[i * INPUT_SIZE], hidden_output, output_output, labels[i]);
        }
        printf("Epoch %d/%d completed\n", epoch + 1, num_epochs);
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

uint8_t recognize(HiddenLayer *hidden, OutputLayer *output, uint8_t *image) {
    double hidden_output[HIDDEN_SIZE];
    double output_output[OUTPUT_SIZE];

    forward_pass(hidden, output, image, hidden_output, output_output);

    uint8_t recognized_digit = 0;
    double max_output = output_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output_output[i] > max_output) {
            max_output = output_output[i];
            recognized_digit = i;
        }
    }
    return recognized_digit;
}

double neuralnetwork_benchmark(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images) {
    uint32_t correct_predictions = 0;
    for (uint32_t i = 0; i < number_of_images; i++) {
        uint8_t answer = neuralnetwork_ask(network, &images[i * INPUT_SIZE]);
        correct_predictions += (answer == labels[i]) ? 1 : 0;
    }

    return (double)correct_predictions / number_of_images;
}

double calculate_accuracy(HiddenLayer *hidden, OutputLayer *output, uint8_t *images, uint8_t *labels, int num_images) {
    int correct_predictions = 0;
    for (int i = 0; i < num_images; i++) {
        uint8_t recognized_digit = recognize(hidden, output, &images[i * INPUT_SIZE]);
        if (recognized_digit == labels[i]) {
            correct_predictions++;
        }
    }
    return (double)correct_predictions / num_images;
}

void neuralnetwork_destroy(NeuralNetwork *network) {
    linearlayer_destroy(&network->linear_layer0);
    free(network->hidden0);
    free(network->hidden1);
    linearlayer_destroy(&network->linear_layer1);
    free(network->hidden2);
    free(network->output);
}

void neuralnetwork_save(NeuralNetwork *network, const char *base_filename) {
    char *filename = (char *)malloc(strlen(base_filename) + 8);

    strcpy(filename, base_filename);
    strcat(filename, "_ll0.bin");
    linearlayer_save(&network->linear_layer0, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_ll1.bin");
    linearlayer_save(&network->linear_layer1, filename, true);

    free(filename);
}

void neuralnetwork_load(NeuralNetwork *network, const char *base_filename) {
    char *filename = (char *)malloc(strlen(base_filename) + 8);

    strcpy(filename, base_filename);
    strcat(filename, "_ll0.bin");
    linearlayer_load(&network->linear_layer0, filename, true);

    strcpy(filename, base_filename);
    strcat(filename, "_ll1.bin");
    linearlayer_load(&network->linear_layer1, filename, true);

    free(filename);
}

void save_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output) {
    FILE *f_hidden = fopen(hidden_file, "wb");
    if (!f_hidden) {
        perror("Failed to open hidden layer file");
        exit(1);
    }

    if (fwrite(hidden, sizeof(HiddenLayer), 1, f_hidden) != 1) {
        perror("Failed to write hidden layer");
        fclose(f_hidden);
        exit(1);
    }
    fclose(f_hidden);

    FILE *f_output = fopen(output_file, "wb");
    if (!f_output) {
        perror("Failed to open output layer file");
        exit(1);
    }

    if (fwrite(output, sizeof(OutputLayer), 1, f_output) != 1) {
        perror("Failed to write output layer");
        fclose(f_output);
        exit(1);
    }
    fclose(f_output);
}

void load_model(const char *hidden_file, const char *output_file, HiddenLayer *hidden, OutputLayer *output) {
    printf("Loading hidden layer model from %s\n", hidden_file);
    FILE *f_hidden = fopen(hidden_file, "rb");
    if (!f_hidden) {
        perror("Failed to open hidden layer file");
        exit(1);
    }

    if (fread(hidden, sizeof(HiddenLayer), 1, f_hidden) != 1) {
        perror("Failed to read hidden layer");
        fclose(f_hidden);
        exit(1);
    }
    fclose(f_hidden);

    printf("Loading output layer model from %s\n", output_file);
    FILE *f_output = fopen(output_file, "rb");
    if (!f_output) {
        perror("Failed to open output layer file");
        exit(1);
    }

    if (fread(output, sizeof(OutputLayer), 1, f_output) != 1) {
        perror("Failed to read output layer");
        fclose(f_output);
        exit(1);
    }
    fclose(f_output);
}
