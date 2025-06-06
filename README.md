# ann-c

Artificial Neural Network (ANN) in C. 

## Example: MNIST hand-written digits dataset

You will first need to download the MNIST Dataset. See `mnist/data/README.md`.

```
$ make mnist
$ cd mnist
$ ./train
$ ./test
```

## How to use?

Create an ANN with 2 layers (one hidden, and the output layer):

```c
NeuralNetwork network = neuralnetwork_create(2);
```

Configure each layers, and initialize your ANN (initialization will set random biases and weights to each layer):

```c
neuralnetwork_add_layer(&network, INPUT_SIZE, SIGMOID_ACTIVATION, HIDDEN_SIZE);
neuralnetwork_add_layer(&network, HIDDEN_SIZE, SOFTMAX_ACTIVATION, OUTPUT_SIZE);
neuralnetwork_initialize(&network);
```

Available activation functions are: 
- Linear function (`LINEAR_ACTIVATION`)
- Sigmoid function (`SIGMOID_ACTIVATION`)
- Softmax function (`SOFTMAX_ACTIVATION`) for the output layer

Provide training parameters and train your model:

```c
TrainingContext context = {
  .learning_rate = 0.10,
  .number_of_epochs = 10,
  .number_of_examples = number_of_images,
};
neuralnetwork_train(&network, inputs, labels, &context);
```

In the case of a classifier, ask the ANN for the class of a given input:

```c
uint8_t answer = neuralnetwork_ask(&network, input);
```

Once you are done, destroy the ANN:

```c
neuralnetwork_destroy(&network);
```

## Dependencies

- C OpenMP
- Python Tensorflow library

## Author

- Morgan Gillette <morgan.gte@gmail.com>
