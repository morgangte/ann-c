#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "mnist.h"
#include "neuralnetwork.h"

void print_results(uint32_t test_examples, double performance, TrainingContext *context) {
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
}

int main(void) {
    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_images("data/test-images.bin", IMAGE_DIMENSION, &number_of_images, &image_size);
    uint8_t *labels = load_labels("data/test-labels.bin", &number_of_labels);
    if (number_of_images != NUMBER_OF_IMAGES_TEST) {
        fprintf(stderr, "ERROR: Unexpected number of images (expected %d, loaded %d)\n", NUMBER_OF_IMAGES_TEST, number_of_images);
        exit(EXIT_FAILURE);
    }
    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }
    double *prepared_images = (double *)malloc(sizeof(double) * IMAGE_SIZE * NUMBER_OF_IMAGES_TEST);
    prepare_input(images, prepared_images, IMAGE_SIZE * NUMBER_OF_IMAGES_TEST);

    NeuralNetwork network;
    TrainingContext context;
    neuralnetwork_load(&network, &context, "model/nn_mnist.bin");

    double accuracy = neuralnetwork_benchmark(&network, prepared_images, labels, number_of_images);
    print_results(number_of_images, accuracy, &context);

    neuralnetwork_destroy(&network);
    free(prepared_images);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
