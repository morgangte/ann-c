#include "layer.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

LinearLayer linearlayer_create(uint32_t input_size, uint32_t output_size) {
    LinearLayer layer = {
        .input_size = input_size,
        .output_size = output_size,
    };

    layer.biases = (double *)malloc(sizeof(double) * output_size);
    double **weights = (double **)malloc(sizeof(double *) * input_size);
    if (!weights || !layer.biases) {
        fprintf(stderr, "ERROR: malloc() failed at linearlayer_create()\n");
        exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < input_size; i++) {
        weights[i] = (double *)malloc(sizeof(double) * output_size);
        if (!weights[i]) {
            fprintf(stderr, "ERROR: malloc() failed at linearlayer_create()\n");
            exit(EXIT_FAILURE);
        }
    }

    layer.weights = weights;
    return layer;
}

void linearlayer_initialize(LinearLayer *layer) {
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            layer->weights[j][i] = RANDOM(-1.0, 1.0);
        }
        layer->biases[i] = RANDOM(-1.0, 1.0);
    }
}

void linearlayer_forward(LinearLayer *layer, double *input, double *output) {
    double sum = 0.0;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        sum = layer->biases[i];
        for (uint32_t j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[j][i];
        }
        output[i] = sum;
    }
}

void linearlayer_backward_sigmoid_activation(LinearLayer *layer, LayerBackwardContext *context) {
    if (context->hidden_layer) {
        for (uint32_t i = 0; i < layer->output_size; i++) {
            context->layer_errors[i] = 0.0;
            for (uint32_t j = 0; j < context->next_layer_output_size; j++) {
                context->layer_errors[i] += layer->weights[i][j] * context->next_layer_errors[j];
            }
            context->layer_errors[i] *= sigmoid_derivative(context->activation_output[i]);
        }
    } else {
        for (uint32_t i = 0; i < layer->output_size; i++) {
            double target = (i == context->label) ? 1.0 : 0.0;
            context->layer_errors[i] = (context->activation_output[i] - target) * sigmoid_derivative(context->activation_output[i]);
        }
    }

    for (uint32_t i = 0; i < layer->output_size; i++) {
        layer->biases[i] += context->learning_rate * context->layer_errors[i];
        for (uint32_t j = 0; j < layer->input_size; j++) {
            layer->weights[j][i] -= context->learning_rate * context->layer_input[j] * context->layer_errors[i];
        }
    }
}

void linearlayer_backward(LinearLayer *layer, LayerBackwardContext *context) {
    switch (context->activation_function) {
        case SIGMOID_ACTIVATION:
            linearlayer_backward_sigmoid_activation(layer, context);
            return;
        default:
            printf("ERROR at linearlayer_backward(): Unsupported activation function\n");
            exit(EXIT_FAILURE);
    }
}

void linearlayer_destroy(LinearLayer *layer) {
    free(layer->biases);
    for (uint32_t i = 0; i < layer->input_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
}

void linearlayer_save(LinearLayer *layer, const char *filename, bool verbose) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen() failed at linearlayer_save()");
        exit(EXIT_FAILURE);
    }

    size_t res = 1;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            res = (res == 1) ? fwrite(&layer->weights[j][i], sizeof(double), 1, f) : res;
        }
        res = (res == 1) ? fwrite(&layer->biases[i], sizeof(double), 1, f) : res;
    }
    fclose(f);

    if (res != 1) {
        perror("fwrite() failed at linearlayer_save()");
        exit(EXIT_FAILURE);
    }
    if (verbose) {
        printf("Successfully saved LinearLayer to '%s'\n", filename);
    }
}

void linearlayer_load(LinearLayer *layer, const char *filename, bool verbose) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("fopen() failed at linearlayer_load()");
        exit(EXIT_FAILURE);
    }

    size_t res = 1;
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            res = (res == 1) ? fread(&layer->weights[j][i], sizeof(double), 1, f) : res;
        }
        res = (res == 1) ? fread(&layer->biases[i], sizeof(double), 1, f) : res;
    }
    fclose(f);

    if (res != 1) {
        perror("fread() failed at linearlayer_load()");
        exit(EXIT_FAILURE);
    }
    if (verbose) {
        printf("Successfully loaded LinearLayer from '%s'\n", filename);
    }
}

SigmoidLayer sigmoidlayer_create(uint32_t size) {
    return (SigmoidLayer){
        .size = size,
    };
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double sigmoid_x) {
    return sigmoid_x * (1.0 - sigmoid_x);
}

void sigmoidlayer_forward(SigmoidLayer *layer, double *input, double *output) {
    for (uint32_t i = 0; i < layer->size; i++) {
        output[i] = sigmoid(input[i]);
    }
}
