#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

#define TEST_IMAGES 10

int main(void) {
    NeuralNetwork network = neuralnetwork_create();
    neuralnetwork_load(&network, "model/my_nn_model");

    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_mnist_images("data/train-images-idx3-ubyte", &number_of_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/train-labels-idx1-ubyte", &number_of_labels);

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }

    printf("1\n");
    double performance = neuralnetwork_benchmark(&network, images, labels, number_of_images);
    printf("Neural Network results:\n   Accuracy: %.3f%%\n", performance * 100);

    printf("Neural Network prediction examples:\n");
    for (int i = 0; i < TEST_IMAGES; i++) {
        uint8_t answer = neuralnetwork_ask(&network, &images[i * INPUT_SIZE]);
        printf("   %d recognized as a %d\n", labels[i], answer);
    }

    neuralnetwork_destroy(&network);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
