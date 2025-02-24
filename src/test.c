#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

#define TEST_IMAGES 10

void print_results(NeuralNetwork *network, uint8_t *images, uint8_t *labels, double performance, TrainingContext *context) {
    printf(
        "Neural Network results:\n"
        "   Training context:\n"
        "      %d epochs\n"
        "      learning rate: %.3f\n"
        "   Accuracy: %.3f%%\n",
        context->number_of_epochs,
        context->learning_rate,
        performance * 100);
    printf("   Prediction examples:\n");
    for (int i = 0; i < TEST_IMAGES; i++) {
        uint8_t answer = neuralnetwork_ask(network, &images[i * INPUT_SIZE]);
        printf("      %d recognized as a %d\n", labels[i], answer);
    }
}

int main(void) {
    NeuralNetwork network = neuralnetwork_create();
    TrainingContext context;
    neuralnetwork_load(&network, &context, "model/my_nn_model");

    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_mnist_images("data/train-images-idx3-ubyte", &number_of_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/train-labels-idx1-ubyte", &number_of_labels);

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        neuralnetwork_destroy(&network);
        free(images);
        free(labels);
        exit(EXIT_FAILURE);
    }

    double performance = neuralnetwork_benchmark(&network, images, labels, number_of_images);
    print_results(&network, images, labels, performance, &context);

    neuralnetwork_destroy(&network);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
