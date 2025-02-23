import numpy as np
import tensorflow as tf
import os
import struct

def save_mnist_binary(images, labels, image_path, label_path):
    with open(image_path, 'wb') as img_file, open(label_path, 'wb') as lbl_file:
        img_file.write(struct.pack('>IIII', 2051, len(images), 28, 28))
        lbl_file.write(struct.pack('>II', 2049, len(labels)))
        for image, label in zip(images, labels):
            img_file.write(image.tobytes())
            lbl_file.write(struct.pack('B', label))

def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    images = (images / 255.0 * 255).astype(np.uint8)
    save_path = './data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_mnist_binary(images, labels, os.path.join(save_path, 'train-images-idx3-ubyte'),
                      os.path.join(save_path, 'train-labels-idx1-ubyte'))
    
if __name__ == "__main__":
    main()
