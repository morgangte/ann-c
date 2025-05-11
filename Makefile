CC=gcc
CFLAGS=-Wall -Wextra -pedantic -march=native -fopenmp
CPPFLAGS=-I./$(INC_DIR)
LIB=-lm -fopenmp

SRC_DIR=src
INC_DIR=include
MODEL_DIR=model
DATA_DIR=data
IMG_DIR=images

MNIST_DIR=mnist
FASHION_MNIST_DIR=fashion-mnist

TRAIN_EXEC=train
TEST_EXEC=test

.PHONY: all mnist fashion cifar10 clean distclean cleanmodel

all: 
	@echo "Available targets:\n\
	   mnist: the famous hand-written digits MNIST dataset\n\
	   fashion: an alternative to the MNIST dataset\n\
	Cleaning:\n\
	   clean\n\
	   distclean\n\
	   clobber"

mnist: $(MNIST_DIR)/$(TRAIN_EXEC) $(MNIST_DIR)/$(TEST_EXEC)
fashion: $(FASHION_MNIST_DIR)/$(TRAIN_EXEC) $(FASHION_MNIST_DIR)/$(TEST_EXEC)

# ************************ Neural network **************************

$(SRC_DIR)/layer.o: $(SRC_DIR)/layer.c $(INC_DIR)/layer.h
$(SRC_DIR)/training.o: $(SRC_DIR)/training.c $(INC_DIR)/training.h
$(SRC_DIR)/neuralnetwork.o: $(SRC_DIR)/neuralnetwork.c $(INC_DIR)/neuralnetwork.h
$(SRC_DIR)/data.o: $(SRC_DIR)/data.c $(INC_DIR)/data.h

$(SRC_DIR)/%.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# **************************** MNIST *******************************

$(MNIST_DIR)/$(TRAIN_EXEC): $(MNIST_DIR)/$(SRC_DIR)/train.o $(MNIST_DIR)/$(SRC_DIR)/mnist.o $(SRC_DIR)/data.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o $(SRC_DIR)/training.o
	$(CC) $^ -o $@ $(LIB)

$(MNIST_DIR)/$(TEST_EXEC): $(MNIST_DIR)/$(SRC_DIR)/test.o $(MNIST_DIR)/$(SRC_DIR)/mnist.o $(SRC_DIR)/data.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o $(SRC_DIR)/training.o
	$(CC) $^ -o $@ $(LIB)

$(MNIST_DIR)/$(SRC_DIR)/mnist.o: $(MNIST_DIR)/$(SRC_DIR)/mnist.c $(MNIST_DIR)/$(INC_DIR)/mnist.h
$(MNIST_DIR)/$(SRC_DIR)/train.o: $(MNIST_DIR)/$(SRC_DIR)/train.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/data.h
$(MNIST_DIR)/$(SRC_DIR)/test.o: $(MNIST_DIR)/$(SRC_DIR)/test.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/data.h

$(MNIST_DIR)/$(SRC_DIR)/%.o:
	$(CC) $(CPPFLAGS) -I./$(MNIST_DIR)/$(INC_DIR) $(CFLAGS) -c $< -o $@	

# ************************ FASHION MNIST ***************************

$(FASHION_MNIST_DIR)/$(TRAIN_EXEC): $(FASHION_MNIST_DIR)/$(SRC_DIR)/train.o $(FASHION_MNIST_DIR)/$(SRC_DIR)/mnist.o $(SRC_DIR)/data.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o $(SRC_DIR)/training.o
	$(CC) $^ -o $@ $(LIB)

$(FASHION_MNIST_DIR)/$(TEST_EXEC): $(FASHION_MNIST_DIR)/$(SRC_DIR)/test.o $(FASHION_MNIST_DIR)/$(SRC_DIR)/mnist.o $(SRC_DIR)/data.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o $(SRC_DIR)/training.o
	$(CC) $^ -o $@ $(LIB)

$(FASHION_MNIST_DIR)/$(SRC_DIR)/mnist.o: $(FASHION_MNIST_DIR)/$(SRC_DIR)/mnist.c $(FASHION_MNIST_DIR)/$(INC_DIR)/mnist.h
$(FASHION_MNIST_DIR)/$(SRC_DIR)/train.o: $(FASHION_MNIST_DIR)/$(SRC_DIR)/train.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/data.h
$(FASHION_MNIST_DIR)/$(SRC_DIR)/test.o: $(FASHION_MNIST_DIR)/$(SRC_DIR)/test.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/data.h

$(FASHION_MNIST_DIR)/$(SRC_DIR)/%.o:
	$(CC) $(CPPFLAGS) -I./$(FASHION_MNIST_DIR)/$(INC_DIR) $(CFLAGS) -c $< -o $@	

# *************************** Cleaning *****************************

clean:
	rm -f $(SRC_DIR)/*.o
	rm -f $(MNIST_DIR)/$(SRC_DIR)/*.o
	rm -f $(FASHION_MNIST_DIR)/$(SRC_DIR)/*.o

distclean: clean
	rm -f $(MNIST_DIR)/$(TRAIN_EXEC)
	rm -f $(MNIST_DIR)/$(TEST_EXEC)
	rm -f $(FASHION_MNIST_DIR)/$(TRAIN_EXEC)
	rm -f $(FASHION_MNIST_DIR)/$(TEST_EXEC)

clobber: distclean
	rm -f $(MNIST_DIR)/$(MODEL_DIR)/*.bin
	rm -f $(FASHION_MNIST_DIR)/$(MODEL_DIR)/*.bin
	rm -f $(MNIST_DIR)/$(DATA_DIR)/*.bin
	rm -f $(FASHION_MNIST_DIR)/$(DATA_DIR)/*.bin
	rm -rf $(MNIST_DIR)/$(DATA_DIR)/$(IMG_DIR)
	rm -rf $(FASHION_MNIST_DIR)/$(DATA_DIR)/$(IMG_DIR)
