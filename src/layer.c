#include "layer.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

Layer layer_create(uint32_t input_size, ActivationFunction activation_function, uint32_t output_size) {
    Layer layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation_function = activation_function,
    };

    switch (activation_function) {
        case SIGMOID_ACTIVATION:
            layer.activate = sigmoid;
            break;
        default:
            fprintf(stderr, "ERROR at layer_create(): Unknown/Not supported activation function\n");
            exit(EXIT_FAILURE);
    }

    layer.biases = (double *)malloc(sizeof(double) * output_size);
    double **weights = (double **)malloc(sizeof(double *) * input_size);
    if (!weights || !layer.biases) {
        fprintf(stderr, "ERROR: malloc() failed at layer_create()\n");
        exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < input_size; i++) {
        weights[i] = (double *)malloc(sizeof(double) * output_size);
        if (!weights[i]) {
            fprintf(stderr, "ERROR: malloc() failed at layer_create()\n");
            exit(EXIT_FAILURE);
        }
    }
    layer.weights = weights;

    return layer;
}

void layer_initialize(Layer *layer) {
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            layer->weights[j][i] = RANDOM(-1.0, 1.0);
        }
        layer->biases[i] = RANDOM(-1.0, 1.0);
    }
}

void layer_forward(Layer *layer, double *input, double *output) {
    double sum = 0.0;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        sum = layer->biases[i];
        for (uint32_t j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[j][i];
        }
        output[i] = layer->activate(sum);
    }
}

void layer_backward_sigmoid(Layer *layer, LayerTrainingContext *context) {
    // printf("entered layer bacward sigmoid\n");
    if (context->hidden_layer) {
        // printf("  hidden layer\n");
        for (uint32_t i = 0; i < layer->output_size; i++) {
            context->layer_errors[i] = 0.0;
            for (uint32_t j = 0; j < context->next_layer_output_size; j++) {
                context->layer_errors[i] += layer->weights[i][j] * context->next_layer_errors[j];
            }
            context->layer_errors[i] *= sigmoid_derivative(context->output[i]);
        }
    } else {
        // printf("  output layer\n");
        for (uint32_t i = 0; i < layer->output_size; i++) {
            // // printf("  a\n");
            double target = (i == context->label) ? 1.0 : 0.0;
            // // printf("  b\n");
            context->layer_errors[i] = (context->output[i] - target) * sigmoid_derivative(context->output[i]);
            // // printf("  c\n");
        }
    }

    // printf("  bias and weight update\n");
    for (uint32_t i = 0; i < layer->output_size; i++) {
        // printf("  a\n");
        layer->biases[i] += context->learning_rate * context->layer_errors[i];
        // printf("  b\n");
        for (uint32_t j = 0; j < layer->input_size; j++) {
            // printf("  c\n");
            layer->weights[j][i] -= context->learning_rate * context->input[j] * context->layer_errors[i];
        }
        // printf("  d\n");
    }
}

void layer_backward(Layer *layer, LayerTrainingContext *context) {
    switch (layer->activation_function) {
        case SIGMOID_ACTIVATION:
            layer_backward_sigmoid(layer, context);
            // printf("layer_backward ended\n");
            return;
        default:
            printf("ERROR at layer_backward(): Unsupported activation function\n");
            exit(EXIT_FAILURE);
    }
}

void layer_destroy(Layer *layer) {
    free(layer->biases);
    for (uint32_t i = 0; i < layer->input_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
}

int layer_save(Layer *layer, FILE *file) {
    size_t res = 1;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            res = (res == 1) ? fwrite(&layer->weights[j][i], sizeof(double), 1, file) : res;
        }
        res = (res == 1) ? fwrite(&layer->biases[i], sizeof(double), 1, file) : res;
    }

    if (res != 1) {
        perror("fwrite() failed at layer_save()");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int layer_load(Layer *layer, FILE *file) {
    size_t res = 1;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            res = (res == 1) ? fread(&layer->weights[j][i], sizeof(double), 1, file) : res;
        }
        res = (res == 1) ? fread(&layer->biases[i], sizeof(double), 1, file) : res;
    }

    if (res != 1) {
        perror("fread() failed at layer_load()");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double sigmoid_x) {
    return sigmoid_x * (1.0 - sigmoid_x);
}
