#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "neuralnetwork.h"

int main(void) {
    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_images("data/train-images.bin", &number_of_images, &image_size);
    uint8_t *labels = load_labels("data/train-labels.bin", &number_of_labels);
    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }

    NeuralNetwork network = neuralnetwork_create(2);
    neuralnetwork_add_layer(&network, INPUT_SIZE, SIGMOID_ACTIVATION, HIDDEN_SIZE);
    neuralnetwork_add_layer(&network, HIDDEN_SIZE, SOFTMAX_ACTIVATION, OUTPUT_SIZE);
    neuralnetwork_initialize(&network);

    TrainingContext context = {
        .learning_rate = 0.10,
        .number_of_epochs = 5,
        .number_of_examples = number_of_images,
    };
    neuralnetwork_train(&network, images, labels, &context);

    neuralnetwork_save(&network, &context, "model/nn_mnist.bin");

    neuralnetwork_destroy(&network);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
