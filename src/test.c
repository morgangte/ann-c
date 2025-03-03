#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

#define TEST_IMAGES 10

void print_results(NeuralNetwork *network, uint8_t *images, uint8_t *labels, uint32_t test_examples, double performance, TrainingContext *context) {
    printf(
        "Neural Network results:\n"
        "   Training context:\n"
        "      %d epochs\n"
        "      learning rate: %.3f\n"
        "      number of examples: %d images\n"
        "   Accuracy: %.3f%% on %d test examples\n",
        context->number_of_epochs,
        context->learning_rate,
        context->number_of_examples,
        performance * 100,
        test_examples);
    printf("   Prediction examples:\n");
    for (int i = 0; i < TEST_IMAGES; i++) {
        uint8_t answer = neuralnetwork_ask(network, &images[i * INPUT_SIZE]);
        printf("      %d recognized as a %d\n", labels[i], answer);
    }
}

int main(void) {
    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_mnist_images("data/test-images", &number_of_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/test-labels", &number_of_labels);
    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }

    NeuralNetwork network;
    TrainingContext context;
    neuralnetwork_load(&network, &context, "model/nn_split_datasets.bin");

    double accuracy = neuralnetwork_benchmark(&network, images, labels, number_of_images);
    print_results(&network, images, labels, number_of_images, accuracy, &context);

    neuralnetwork_destroy(&network);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
