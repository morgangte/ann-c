#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

int main(void) {
    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_mnist_images("data/train-images-idx3-ubyte", &number_of_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/train-labels-idx1-ubyte", &number_of_labels);

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }

    NeuralNetwork network = neuralnetwork_create();
    TrainingContext context = {
        .learning_rate = 0.05,
        .number_of_epochs = 10,
    };
    neuralnetwork_initialize(&network);
    neuralnetwork_train(&network, images, labels, number_of_images, &context);

    neuralnetwork_save(&network, &context, "model/my_nn_model");

    neuralnetwork_destroy(&network);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
