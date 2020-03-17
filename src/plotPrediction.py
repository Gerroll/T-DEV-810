from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Display:
    def __init__(self, className, fileNameNeural, imageTest):
        self.className = className
        model = keras.models.load_model(fileNameNeural + '.h5')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        self.predictions = probability_model.predict(imageTest)

    def plot_image(self, i, predictions_array, true_label, img):
        img = img.reshape(400, 224, 224)
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.className[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.className[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(len(self.className)))
        plt.yticks([])
        thisplot = plt.bar(range(len(self.className)), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def predict(self, start, test_labels, test_images):
        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 10
        num_cols = 5
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(start + i, self.predictions[start + i], test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(start + i, self.predictions[start + i], test_labels)
        plt.tight_layout()
        plt.show()
