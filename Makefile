CC=gcc
CFLAGS=-Wall -Wextra -pedantic
CPPFLAGS=-I./$(INC_DIR)
LIB=-lm

SRC_DIR=src
INC_DIR=include

TRAIN_EXEC=train
TEST_EXEC=test

.PHONY: all clean distclean

all: $(TRAIN_EXEC) $(TEST_EXEC)

# ************************** Executable ****************************

$(TRAIN_EXEC): $(SRC_DIR)/train.o $(SRC_DIR)/mnist.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o
	$(CC) $^ -o $@ $(LIB)

$(TEST_EXEC): $(SRC_DIR)/test.o $(SRC_DIR)/mnist.o $(SRC_DIR)/neuralnetwork.o $(SRC_DIR)/layer.o
	$(CC) $^ -o $@ $(LIB)

# ************************* Object files ***************************

$(SRC_DIR)/layer.o: $(SRC_DIR)/layer.c $(INC_DIR)/layer.h
$(SRC_DIR)/neuralnetwork.o: $(SRC_DIR)/neuralnetwork.c $(INC_DIR)/neuralnetwork.h
$(SRC_DIR)/mnist.o: $(SRC_DIR)/mnist.c $(INC_DIR)/mnist.h
$(SRC_DIR)/train.o: $(SRC_DIR)/train.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/mnist.h
$(SRC_DIR)/test.o: $(SRC_DIR)/test.c $(INC_DIR)/neuralnetwork.h $(INC_DIR)/mnist.h

$(SRC_DIR)/%.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@	

# *************************** Cleaning *****************************

clean:
	rm -f $(SRC_DIR)/*.o

distclean: clean
	rm -f $(TRAIN_EXEC)
	rm -f $(TEST_EXEC)

