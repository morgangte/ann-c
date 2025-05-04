#include "layer.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

Layer layer_create(uint32_t input_size, ActivationFunction activation_function, uint32_t output_size) {
    Layer layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation_function = activation_function,
    };

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
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < layer->output_size; i++) {
        for (uint32_t j = 0; j < layer->input_size; j++) {
            layer->weights[j][i] = RANDOM(-1.0, 1.0);
        }
        layer->biases[i] = RANDOM(-1.0, 1.0);
    }
}

void layer_forward_linear(Layer *layer, double *input, double *output) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < layer->output_size; i++) {
        double sum = layer->biases[i];
        for (uint32_t j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[j][i];
        }
        output[i] = sum;
    }
}

void layer_forward_sigmoid(Layer *layer, double *input, double *output) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < layer->output_size; i++) {
        double sum = layer->biases[i];
        for (uint32_t j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[j][i];
        }
        output[i] = sigmoid(sum);
    }
}

void layer_forward_softmax(Layer *layer, double *input, double *output) {
    double sum_exp = 0.0;
    
#pragma omp parallel
    {
#pragma omp for schedule(static) reduction(+ : sum_exp)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            double sum = layer->biases[i];
            for (uint32_t j = 0; j < layer->input_size; j++) {
                sum += input[j] * layer->weights[j][i];
            }

            output[i] = exp(sum);
            sum_exp += output[i];
        }

#pragma omp barrier

#pragma omp for schedule(static)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            output[i] = output[i] / sum_exp;
        }
    }
}

void layer_forward(Layer *layer, double *input, double *output) {
    switch (layer->activation_function) {
        case LINEAR_ACTIVATION:
            layer_forward_linear(layer, input, output);
            return;
        case SIGMOID_ACTIVATION:
            layer_forward_sigmoid(layer, input, output);
            return;
        case SOFTMAX_ACTIVATION:
            layer_forward_softmax(layer, input, output);
            return;
        default:
            printf("ERROR at layer_forward(): Unsupported activation function\n");
            exit(EXIT_FAILURE);
    }
}

void layer_backward_linear(Layer *layer, LayerBackwardContext *context) {
#pragma omp parallel
    {
        if (context->hidden_layer) {
#pragma omp for schedule(static)
            for (uint32_t i = 0; i < layer->output_size; i++) {
                context->layer_errors[i] = 0.0;
                for (uint32_t j = 0; j < context->next_layer_output_size; j++) {
                    context->layer_errors[i] += layer->weights[i][j] * context->next_layer_errors[j];
                }
            }
        } else {
#pragma omp for schedule(static)
            for (uint32_t i = 0; i < layer->output_size; i++) {
                double target = (i == context->label) ? 1.0 : 0.0;
                context->layer_errors[i] = (context->output[i] - target);
            }
        }

#pragma omp barrier

#pragma omp for schedule(static)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            layer->biases[i] += context->learning_rate * context->layer_errors[i];
            for (uint32_t j = 0; j < layer->input_size; j++) {
                layer->weights[j][i] -= context->learning_rate * context->input[j] * context->layer_errors[i];
            }
        }
    }
}

void layer_backward_sigmoid(Layer *layer, LayerBackwardContext *context) {
#pragma omp parallel
    {
        if (context->hidden_layer) {
#pragma omp for schedule(static)
            for (uint32_t i = 0; i < layer->output_size; i++) {
                context->layer_errors[i] = 0.0;
                for (uint32_t j = 0; j < context->next_layer_output_size; j++) {
                    context->layer_errors[i] += layer->weights[i][j] * context->next_layer_errors[j];
                }
                context->layer_errors[i] *= sigmoid_derivative(context->output[i]);
            }
        } else {
#pragma omp for schedule(static)
            for (uint32_t i = 0; i < layer->output_size; i++) {
                double target = (i == context->label) ? 1.0 : 0.0;
                context->layer_errors[i] = (context->output[i] - target) * sigmoid_derivative(context->output[i]);
            }
        }

#pragma omp barrier

#pragma omp for schedule(static)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            layer->biases[i] += context->learning_rate * context->layer_errors[i];
            for (uint32_t j = 0; j < layer->input_size; j++) {
                layer->weights[j][i] -= context->learning_rate * context->input[j] * context->layer_errors[i];
            }
        }
    }
}

void layer_backward_softmax(Layer *layer, LayerBackwardContext *context) {
    if (context->hidden_layer) {
        fprintf(stderr, "ERROR: Softmax not supported for hidden layers\n");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            double target = (i == context->label) ? 1.0 : 0.0;
            context->layer_errors[i] = (context->output[i] - target);
        }

#pragma omp barrier

#pragma omp for schedule(static)
        for (uint32_t i = 0; i < layer->output_size; i++) {
            layer->biases[i] += context->learning_rate * context->layer_errors[i];
            for (uint32_t j = 0; j < layer->input_size; j++) {
                layer->weights[j][i] -= context->learning_rate * context->input[j] * context->layer_errors[i];
            }
        }
    }
}

void layer_backward(Layer *layer, LayerBackwardContext *context) {
    switch (layer->activation_function) {
        case LINEAR_ACTIVATION:
            layer_backward_linear(layer, context);
            return;
        case SIGMOID_ACTIVATION:
            layer_backward_sigmoid(layer, context);
            return;
        case SOFTMAX_ACTIVATION:
            layer_backward_softmax(layer, context);
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
