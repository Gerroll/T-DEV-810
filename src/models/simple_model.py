from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras

import tensorflow as tf
import datetime


class SimpleModel:
    def __init__(self, modelType=type file=fileNameNeural, shape=inputShape, class=classNumber, active_log=True):
        self.fileNameNeural = fileNameNeural
        self.inputShape = inputShape
        self.classNumber = classNumber
        self.active_log = active_log
        # set simple model
        self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=self.inputShape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(self.classNumber)
        ])
        # compile model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        if self.active_log:
            # complete logs
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def train(self, train_data, train_label, test_data, test_label, epochNumber):
        if self.active_log:
            # train model
            self.model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=epochNumber, callbacks=[self.tensorboard_callback])
        else:
            self.model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=epochNumber)

    def load(self, fileName):
        # load model
        self.model = keras.models.load_model(fileName + '.h5')
        # compile model
        self.model.compile(optimizer='adam',
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy'])

    def save(self):
        # save initialize neural network
        self.model.save(self.fileNameNeural + '.h5')

    def evaluate(self, test_images, test_labels):
        # evaluate model
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=3)
        print('\nTest accuracy:', test_acc)
        print('\nLoss: ', test_loss)
