from models.cnn_model import CnnModel
from loadData import Loader

import sys

if __name__ == "__main__":

    fileName = 'neural_network'
    DATA_PATH_TRAIN = './resource/data/train'
    DATA_PATH_TEST = './resource/data/test'
    BATCH_SIZE = 400
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    inputShape = (IMG_WIDTH, IMG_HEIGHT, 1)
    EPOCHS = 50
    CLASS_NAME = ['NORMAL', 'BACTERIA', 'VIRUS']

    # init model
    if len(sys.argv) > 1:
        model = CnnModel(fileName, inputShape, 3, True, sys.argv[1])
    else:
        model = CnnModel(fileName, inputShape, 3, True)

    # replace the actual model with an existing one from h5 format file
    # model.load('neural_network')

    # load train data
    loader_train = Loader(DATA_PATH_TRAIN, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAME)
    train_data, train_label = loader_train.load_data()
    train_data = train_data.reshape((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 1))

    # load test data
    loader_test = Loader(DATA_PATH_TEST, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAME)
    test_data, test_label = loader_test.load_data()
    test_data = test_data.reshape((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 1))

    # train model then evaluate with test data
    model.train(train_data, train_label, test_data, test_label, EPOCHS)
    # save model in a .h5 file
    model.save()
