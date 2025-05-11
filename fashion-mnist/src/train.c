#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "mnist.h"
#include "neuralnetwork.h"

int main(void) {
    uint32_t number_of_images, image_size, number_of_labels;
    uint8_t *images = load_images("data/train-images.bin", IMAGE_DIMENSION, &number_of_images, &image_size);
    uint8_t *labels = load_labels("data/train-labels.bin", &number_of_labels);
    if (number_of_images != NUMBER_OF_IMAGES_TRAIN) {
        fprintf(stderr, "ERROR: Unexpected number of images (expected %d, loaded %d)\n", NUMBER_OF_IMAGES_TRAIN, number_of_images);
        exit(EXIT_FAILURE);
    }
    if (number_of_images != number_of_labels) {
        fprintf(stderr, "ERROR: The number of images and labels don't match\n");
        exit(EXIT_FAILURE);
    }
    double *prepared_images = (double *)malloc(sizeof(double) * IMAGE_SIZE * NUMBER_OF_IMAGES_TRAIN);
    prepare_input(images, prepared_images, IMAGE_SIZE * NUMBER_OF_IMAGES_TRAIN);

    NeuralNetwork network = neuralnetwork_create(2);
    neuralnetwork_add_layer(&network, INPUT_SIZE, SIGMOID_ACTIVATION, HIDDEN_SIZE);
    neuralnetwork_add_layer(&network, HIDDEN_SIZE, SOFTMAX_ACTIVATION, OUTPUT_SIZE);
    neuralnetwork_initialize(&network);

    TrainingContext context = {
        .learning_rate = 0.125,
        .number_of_epochs = 5,
        .number_of_examples = number_of_images,
    };
    neuralnetwork_train(&network, prepared_images, labels, &context);

    neuralnetwork_save(&network, &context, "model/nn_fashion.bin");

    neuralnetwork_destroy(&network);
    free(prepared_images);
    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
