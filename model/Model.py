import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Model:

    def __init__(self):
        self.model = tf.__version__

    def train(self, epoch):
        print(self.model)
        print("LOG : nb epoch : " + epoch)
