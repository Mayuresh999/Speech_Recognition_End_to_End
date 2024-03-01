import os
import sys
import json
import numpy as np
import keras
# import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from exception import CustomException
from logger import logging


DATA_PATH = os.path.join("data", "data.json")
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 64
SAVED_MODEL_PATH = os.path.join("assets", "model.h5")
NUM_KEYWORDS = 30

def load_dataset(data_path):
    with open(data_path, "r") as jf:
        data = json.load(jf)

    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return x, y

def get_data_splits(data_path, test_size=0.1, validation_size=0.1):
    # load data
    x, y = load_dataset(data_path)
    logging.info("X and Y loaded from json file")

    # create splits
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=42)

    # convert inputs from 2D to 3D array
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_val, x_test, y_train, y_val, y_test

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape,
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

    model.add(keras.layers.Conv2D(32, (3,3), activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

    model.add(keras.layers.Conv2D(32, (2,2), activation="relu", 
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(NUM_KEYWORDS, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=['accuracy'])

    model.summary()

    return model

def main():
    # load splits 
    x_train, x_val, x_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)
    logging.info('Data splitted in train test splits')

    # build the CNN
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]) # dim1 = no. of segments, dim2 = no. of coefficients (mfcc), dim3 = no. of channels (1 as this is audio data )
    model=build_model(input_shape, LEARNING_RATE)
    logging.info("Model built")

    # train the network
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    # evaluate the network
    test_error, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test error : {test_error}, Test accuracy : {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)
    logging.info("model saved")

if __name__ == "__main__":
    main()