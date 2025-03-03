import numpy as np
import tensorflow as tf
import os
import struct

DATA_FOLDER = "./data/"

TRAIN_IMAGES = "train-images"
TRAIN_LABELS = "train-labels"

TEST_IMAGES = "test-images"
TEST_LABELS = "test-labels"

def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    assert train_images.shape == (60000, 28, 28)
    assert train_labels.shape == (60000, )
    assert test_images.shape == (10000, 28, 28)
    assert test_labels.shape == (10000, )
    return (train_images, train_labels), (test_images, test_labels)

def save_mnist_binary(images, labels, images_path, labels_path):
    with open(images_path, 'wb') as images_file, open(labels_path, 'wb') as labels_file:
        images_file.write(struct.pack('>IIII', 2051, len(images), 28, 28))
        labels_file.write(struct.pack('>II', 2049, len(labels)))
        for image, label in zip(images, labels):
            images_file.write(image.tobytes())
            labels_file.write(struct.pack('B', label))

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images = (train_images / 255.0 * 255).astype(np.uint8)
    test_images = (test_images / 255.0 * 255).astype(np.uint8)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    save_mnist_binary(train_images, train_labels, 
                      os.path.join(DATA_FOLDER, TRAIN_IMAGES), os.path.join(DATA_FOLDER, TRAIN_LABELS))
    save_mnist_binary(test_images, test_labels, 
                      os.path.join(DATA_FOLDER, TEST_IMAGES), os.path.join(DATA_FOLDER, TEST_LABELS))
    
if __name__ == "__main__":
    main()
