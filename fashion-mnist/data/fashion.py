import numpy as np
import struct
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import sys

TRAIN_IMAGES = "train-images.bin"
TRAIN_LABELS = "train-labels.bin"
TEST_IMAGES = "test-images.bin"
TEST_LABELS = "test-labels.bin"

EXAMPLE_IMAGES_DIR = "images"

IMAGE_WIDTH = 28

def load_mnist_data():
    """
    See https://keras.io/api/datasets/fashion_mnist/
    """
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    assert train_images.shape == (60000, IMAGE_WIDTH, IMAGE_WIDTH)
    assert test_images.shape == (10000, IMAGE_WIDTH, IMAGE_WIDTH)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)
    return (train_images, train_labels), (test_images, test_labels)

def save_mnist_binary(images, labels, images_path, labels_path):
    with open(images_path, 'wb') as images_file, open(labels_path, 'wb') as labels_file:
        images_file.write(struct.pack('>III', len(images), IMAGE_WIDTH, IMAGE_WIDTH))
        labels_file.write(struct.pack('>I', len(labels)))
        for image, label in zip(images, labels):
            images_file.write(image.tobytes())
            labels_file.write(label.tobytes())

def save_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
    train_images = (train_images / 255.0 * 255).astype(np.uint8)
    test_images = (test_images / 255.0 * 255).astype(np.uint8)

    save_mnist_binary(train_images, train_labels, TRAIN_IMAGES, TRAIN_LABELS)
    save_mnist_binary(test_images, test_labels, TEST_IMAGES, TEST_LABELS)

def save_example_image(n: int, images, labels):
    if not os.path.exists(EXAMPLE_IMAGES_DIR):
        os.makedirs(EXAMPLE_IMAGES_DIR)
    filename = EXAMPLE_IMAGES_DIR + '/fashion_' + str(n) + '.png'

    try:
        index = next(i for i, label in enumerate(labels) if label == n)
        plt.imsave(filename, images[index], cmap='gray')
    except:
        print(">>> Error while save example image")
        return
    print(f">>> Successfully saved example image as {filename}")

def print_help_args():
    print(">>> INVALID COMMAND LINE: Usage:\n"
          "       save: saves the MNIST dataset\n"
          "       example [n]: saves an image labeled as n (integer)\n"
          "       examples: saves an example of each class")

def print_help_label():
    print(">>> INVALID INPUT: The label must be an integer between 0 and 9")

def handle_argv(argv):
    if (len(argv) == 2) and (argv[1] == "save"):
        print(">>> Saving the MNIST dataset...")
        save_dataset()
        return
    
    if (len(argv) == 2) and (argv[1] == "examples"):
        (images, labels), (_, _) = load_mnist_data()
        for n in range(0, 10):
            save_example_image(n, images, labels)
        return

    if (len(argv) == 3) and (argv[1] == "example"):
        try:
            n = int(argv[2])
        except:
            print_help_label()
            return
        if n < 0 or n >= 10:
            print_help_label()
        else:
            (images, labels), (_, _) = load_mnist_data()
            save_example_image(n, images, labels)
        return
    
    print_help_args()

if __name__ == "__main__":
    handle_argv(sys.argv)