#include "neuralnetwork.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
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
