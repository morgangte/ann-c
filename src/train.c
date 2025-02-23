#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "neuralnetwork.h"

int main() {
    int num_images, image_size, num_labels;
    uint8_t *images = load_mnist_images("data/train-images-idx3-ubyte", &num_images, &image_size);
    uint8_t *labels = load_mnist_labels("data/train-labels-idx1-ubyte", &num_labels);

    if (num_images != num_labels) {
        fprintf(stderr, "Number of images and labels do not match\n");
        exit(EXIT_FAILURE);
    }

    HiddenLayer hidden;
    OutputLayer output;
    initialize_layer(&hidden, &output);

    train(&hidden, &output, images, labels, num_images, EPOCHS);

    save_model("hidden_layer.bin", "output_layer.bin", &hidden, &output);

    free(images);
    free(labels);
    return EXIT_SUCCESS;
}
