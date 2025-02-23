#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

#define TEST_IMAGES 10

int main() {
    HiddenLayer hidden;
    OutputLayer output;

    load_model("hidden_layer.bin", "output_layer.bin", &hidden, &output);

    int num_images, image_size, num_labels;
    uint8_t *images = load_mnist_images("data/train-images-idx3-ubyte", &num_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/train-labels-idx1-ubyte", &num_labels);

    if (num_images != num_labels) {
        fprintf(stderr, "Number of images and labels do not match\n");
        exit(EXIT_FAILURE);
    }

    double accuracy = calculate_accuracy(&hidden, &output, images, labels, num_images);
    printf("Accuracy: %.2f%%\n", accuracy * 100);

    for (int i = 0; i < TEST_IMAGES; i++) {
        uint8_t recognized_digit = recognize(&hidden, &output, &images[i * INPUT_SIZE]);
        printf("Image %d: Recognized as %d, Actual %d\n", i + 1, recognized_digit, labels[i]);
    }

    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
