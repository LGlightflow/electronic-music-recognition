from unittest import TestCase
import librosa
import os
import math
from distutils.log import error
import json
from signal import signal
from more_itertools import sample
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from extract_data import SAMPLE_RATE, SAMPLES_PER_TRACK


MODEL_PATH="EMR.h5"
SONG_PATH = "test.wav"
DATA_PATH = "data.json"


def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates RNN-LSTM model

    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)  #获取最大值索引，二维数组中axis0为外层 1为内层

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def predict_one_song(model,song_path):
    hop_length=512
    num_segments = 10
    num_mfcc=13
    n_fft=2048
    
    if(os.path.exists(song_path)):
        signal,sample_rate = librosa.load(song_path,SAMPLE_RATE)  
        samples_per_segment = int(SAMPLES_PER_TRACK /num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)#向上取整
        for d in range(num_segments):

            start = samples_per_segment * d     #分段音频中每一段音频的起始时间
            finish = start + samples_per_segment

            # extract mfcc
            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

        mfcc_=mfcc[np.newaxis,:]
        prediction=model.predict(mfcc_)
        predicted_index = np.argmax(prediction, axis=1)
        print("Predicted label: {}".format(predicted_index))

    else:
        print("error!")


if __name__ == "__main__":
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    # get train, validation, test splits
    if os.path.exists(MODEL_PATH):
        model=keras.models.load_model("EMR.h5")
        predict_one_song(model,SONG_PATH)
        #testX, testy = load_data(SONG_PATH)
        #predict(model,testX[7],y_test[0])
        """
        x=X_test[300]
        y=y_test[300]
        predict(model,x,y)
        """
    else:    
        # create network
        input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
        model = build_model(input_shape)

        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

        # plot accuracy/error for training and validation
        plot_history(history)

        # evaluate model on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

    