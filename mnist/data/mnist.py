import numpy as np
import struct
import tensorflow as tf

TRAIN_IMAGES = "train-images.bin"
TRAIN_LABELS = "train-labels.bin"

TEST_IMAGES = "test-images.bin"
TEST_LABELS = "test-labels.bin"

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    assert train_images.shape == (60000, 28, 28)
    assert train_labels.shape == (60000, )
    assert test_images.shape == (10000, 28, 28)
    assert test_labels.shape == (10000, )
    return (train_images, train_labels), (test_images, test_labels)

def save_mnist_binary(images, labels, images_path, labels_path):
    with open(images_path, 'wb') as images_file, open(labels_path, 'wb') as labels_file:
        images_file.write(struct.pack('>III', len(images), 28, 28))
        labels_file.write(struct.pack('>I', len(labels)))
        for image, label in zip(images, labels):
            images_file.write(image.tobytes())
            labels_file.write(label.tobytes())

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
    train_images = (train_images / 255.0 * 255).astype(np.uint8)
    test_images = (test_images / 255.0 * 255).astype(np.uint8)

    save_mnist_binary(train_images, train_labels, TRAIN_IMAGES, TRAIN_LABELS)
    save_mnist_binary(test_images, test_labels, TEST_IMAGES, TEST_LABELS)
