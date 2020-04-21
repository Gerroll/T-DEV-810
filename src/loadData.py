from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os


class Loader:
    def __init__(self, pathToData, batchSize, height, width, className):
        # loading parameter
        self.BATCH_SIZE = batchSize
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        # load data dir
        self.data_dir = pathlib.Path(pathToData)
        # display class name
        self.CLASS_NAMES = className
        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))

    def print_list_ds_n(self, n):
        for f in self.list_ds.take(n):
            print(f)

    def show_batch(self, image_batch, label_batch, n):
        plt.figure(figsize=(10, 10))
        for n in range(n):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(self.CLASS_NAMES[label_batch[n] == 1][0].title())
            plt.axis('off')
        plt.show()

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.equal(self.CLASS_NAMES, parts[-2])

    def decode_img(self, img):
        # channels = 3 for rgb image channels for grayscale
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def process_path_with_data_augmentation(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        # Data augmentation example
        # img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_flip_up_down(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=40):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(self.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def resize_label(self, data):
        # resize test labels
        x = []
        for item in data:
            for index in range(len(item)):
                if item[index]:
                    x.append(index)

        return np.array(x)

    def load_data(self, data_augmentation=False):
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        if data_augmentation:
            labeled_ds = self.list_ds.map(self.process_path_with_data_augmentation,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = self.prepare_for_training(labeled_ds, cache=False)
        image_batch, label_batch = next(iter(train_ds))
        print("image batch:", type(image_batch.numpy()))
        print("label batch:", type(label_batch.numpy()))

        """
        # See data as ndarray
        show_batch(image_batch.numpy(), label_batch.numpy())
        """
        return image_batch.numpy(), self.resize_label(label_batch.numpy())

    def load_both_data(self):
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

        DATASET_SIZE = 814

        train_size = int(0.7 * DATASET_SIZE)
        test_size = int(0.3 * DATASET_SIZE)

        labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_ds = labeled_ds.take(train_size)
        test_ds = labeled_ds.skip(train_size)
        test_ds = test_ds.take(test_size)

        train_ds = self.prepare_for_training(train_ds, cache=False)
        test_ds = self.prepare_for_training(test_ds, cache=False)
        image_batch, label_batch = next(iter(train_ds))
        image_batch_test, label_batch_test = next(iter(test_ds))
        print("image batch:", type(image_batch.numpy()))
        print("label batch:", type(label_batch.numpy()))

        """
        # See data as ndarray
        show_batch(image_batch.numpy(), label_batch.numpy())
        """
        return (image_batch.numpy(), self.resize_label(label_batch.numpy())), (image_batch_test.numpy(), self.resize_label(label_batch_test.numpy()))