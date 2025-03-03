#include "neuralnetwork.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

NeuralNetwork neuralnetwork_create(uint16_t number_of_layers) {
    return (NeuralNetwork){
        .layers_capacity = number_of_layers,
        .layers_size = 0,
        .layers = NULL,
        .layers_outputs = NULL,
    };
}

void neuralnetwork_add_layer(NeuralNetwork *network, uint32_t input_size, ActivationFunction activation_function, uint32_t output_size) {
    assert(network->layers_size < network->layers_capacity);

    if (!network->layers) {
        network->layers = (Layer *)malloc(network->layers_capacity * sizeof(Layer));
        if (!network->layers) {
            fprintf(stderr, "ERROR: malloc() failed at neuralnetwork_add_layer()\n");
            exit(EXIT_FAILURE);
        }
    }
    if (!network->layers_outputs) {
        network->layers_outputs = (double **)malloc(network->layers_capacity * sizeof(double *));
        if (!network->layers_outputs) {
            fprintf(stderr, "ERROR: malloc() failed at neuralnetwork_add_layer()\n");
            exit(EXIT_FAILURE);
        }
    }

    network->layers[network->layers_size] = layer_create(input_size, activation_function, output_size);
    network->layers_outputs[network->layers_size] = (double *)malloc(output_size * sizeof(double));
    if (!network->layers_outputs[network->layers_size]) {
        fprintf(stderr, "ERROR: malloc() failed at neuralnetwork_add_layer()\n");
        exit(EXIT_FAILURE);
    }
    network->layers_size += 1;
}

void neuralnetwork_initialize(NeuralNetwork *network) {
    srand(time(NULL));
    for (uint16_t i = 0; i < network->layers_size; i++) {
        layer_initialize(&network->layers[i]);
    }
}

void prepare_input(uint8_t *raw, double *prepared, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        prepared[i] = raw[i] / 255.0;
    }
}

void neuralnetwork_forward(NeuralNetwork *network, double *input) {
    double *layer_input = input;
    double *layer_output;
    for (uint16_t i = 0; i < network->layers_size; i++) {
        layer_output = network->layers_outputs[i];
        layer_forward(&network->layers[i], layer_input, layer_output);
        layer_input = network->layers_outputs[i];
    }
}

void neuralnetwork_backward(NeuralNetwork *network, double *input, BackwardContext *backward_context) {
    LayerBackwardContext layer_backward_context = {
        .learning_rate = backward_context->learning_rate,
        .label = backward_context->label,
    };

    for (uint16_t i = 0; i < network->layers_size; i++) {
        uint16_t layer_index = network->layers_size - 1 - i;
        layer_backward_context.hidden_layer = (layer_index < network->layers_size - 1);
        layer_backward_context.input = (layer_index == 0) ? input : network->layers_outputs[layer_index - 1];
        layer_backward_context.output = network->layers_outputs[layer_index];
        layer_backward_context.layer_errors = backward_context->layers_errors[layer_index];
        layer_backward_context.next_layer_output_size = (layer_index == network->layers_size - 1) ? 0 : network->layers[layer_index + 1].output_size;
        layer_backward_context.next_layer_errors = (layer_index == network->layers_size - 1) ? NULL : backward_context->layers_errors[layer_index + 1];

        layer_backward(&network->layers[layer_index], &layer_backward_context);
    }
}

BackwardContext backwardcontext_create(NeuralNetwork *network, double learning_rate) {
    BackwardContext backward_context = {
        .learning_rate = learning_rate,
        .label = 0,
        .number_of_layers = network->layers_size,
    };
    backward_context.layers_errors = (double **)malloc(network->layers_size * sizeof(double *));
    for (uint16_t i = 0; i < network->layers_size; i++) {
        backward_context.layers_errors[i] = (double *)malloc(network->layers[i].output_size * sizeof(double));
    }

    return backward_context;
}

void backwardcontext_destroy(BackwardContext *context) {
    for (uint16_t i = 0; i < context->number_of_layers; i++) {
        free(context->layers_errors[i]);
    }
    free(context->layers_errors);
}

void neuralnetwork_train(NeuralNetwork *network, uint8_t *images, uint8_t *labels, TrainingContext *training_context) {
    double prepared_input[INPUT_SIZE];
    BackwardContext backward_context = backwardcontext_create(network, training_context->learning_rate);

    for (uint32_t epoch = 0; epoch < training_context->number_of_epochs; epoch++) {
        printf("Running epoch %d/%d...\n", epoch + 1, training_context->number_of_epochs);
        for (uint32_t i = 0; i < training_context->number_of_examples; i++) {
            prepare_input(&images[i * INPUT_SIZE], prepared_input, INPUT_SIZE);
            backward_context.label = labels[i];
            neuralnetwork_forward(network, prepared_input);
            neuralnetwork_backward(network, prepared_input, &backward_context);
        }
    }

    backwardcontext_destroy(&backward_context);
}

uint8_t max_index(double *array, uint8_t size) {
    assert((size > 0 && array) || size == 0);

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

    return max_index(neuralnetwork_output(network), OUTPUT_SIZE);
}

double neuralnetwork_benchmark(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t number_of_images) {
    uint32_t correct_predictions = 0;
    for (uint32_t i = 0; i < number_of_images; i++) {
        uint8_t answer = neuralnetwork_ask(network, &images[i * INPUT_SIZE]);
        correct_predictions += (answer == labels[i]) ? 1 : 0;
    }

    return (double)correct_predictions / number_of_images;
}

double *neuralnetwork_output(NeuralNetwork *network) {
    if (network->layers_size == 0) {
        return NULL;
    }
    return network->layers_outputs[network->layers_size - 1];
}

void neuralnetwork_destroy(NeuralNetwork *network) {
    for (uint16_t i = 0; i < network->layers_size; i++) {
        free(network->layers_outputs[i]);
        layer_destroy(&network->layers[i]);
    }
    free(network->layers_outputs);
    free(network->layers);
}

void neuralnetwork_save(NeuralNetwork *network, TrainingContext *context, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("fopen() failed at neuralnetwork_save()");
        exit(EXIT_FAILURE);
    }

    if (fwrite(&network->layers_size, sizeof(uint16_t), 1, file) != 1) {
        fclose(file);
        perror("fwrite() failed at neuralnetwork_save()");
        exit(EXIT_FAILURE);
    }

    size_t res = 1;
    for (uint16_t i = 0; i < network->layers_size; i++) {
        res = (res == 1) ? fwrite(&network->layers[i].input_size, sizeof(uint32_t), 1, file) : res;
        res = (res == 1) ? fwrite(&network->layers[i].activation_function, sizeof(ActivationFunction), 1, file) : res;
        res = (res == 1) ? fwrite(&network->layers[i].output_size, sizeof(uint32_t), 1, file) : res;
        if (res != 1) {
            fclose(file);
            perror("fwrite() failed at neuralnetwork_save()");
            exit(EXIT_FAILURE);
        }

        if (layer_save(&network->layers[i], file)) {
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    if (trainingcontext_save(context, file)) {
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

void neuralnetwork_load(NeuralNetwork *network, TrainingContext *context, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("fopen() failed at neuralnetwork_load()");
        exit(EXIT_FAILURE);
    }

    uint16_t number_of_layers;
    if (fread(&number_of_layers, sizeof(uint16_t), 1, file) != 1) {
        fclose(file);
        perror("fread() failed at neuralnetwork_load()");
        exit(EXIT_FAILURE);
    }

    *network = neuralnetwork_create(number_of_layers);
    size_t res = 1;
    uint32_t input_size;
    ActivationFunction activation_function;
    uint32_t output_size;
    for (uint16_t i = 0; i < number_of_layers; i++) {
        res = (res == 1) ? fread(&input_size, sizeof(uint32_t), 1, file) : res;
        res = (res == 1) ? fread(&activation_function, sizeof(ActivationFunction), 1, file) : res;
        res = (res == 1) ? fread(&output_size, sizeof(uint32_t), 1, file) : res;
        if (res != 1) {
            fclose(file);
            perror("fwrite() failed at neuralnetwork_save()");
            exit(EXIT_FAILURE);
        }

        neuralnetwork_add_layer(network, input_size, activation_function, output_size);
        if (layer_load(&network->layers[i], file)) {
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    if (trainingcontext_load(context, file)) {
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}
