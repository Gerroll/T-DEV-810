from models.cnn_model import CnnModel
from loadData import Loader

if __name__ == "__main__":
    fileName = 'neural_network'
    inputShape = (224, 224, 1)
    DATA_PATH_TRAIN = './resource/data/train'
    DATA_PATH_TEST = './resource/data/test'
    BATCH_SIZE = 400
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CLASS_NAME = ['NORMAL', 'BACTERIA', 'VIRUS']

    # init model
    model = CnnModel(fileName, inputShape, 3, active_log=False)

    # replace the actual model with an existing one from h5 format file
    # model.load('neural_network')

    # load train data
    loader_train = Loader(DATA_PATH_TRAIN, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAME)
    train_data, train_label = loader_train.load_data()
    train_data = train_data.reshape((400, 224, 224, 1))

    # load test data
    loader_test = Loader(DATA_PATH_TEST, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAME)
    test_data, test_label = loader_test.load_data()
    test_data = test_data.reshape((400, 224, 224, 1))

    # train model then evaluate with test data
    model.train(train_data, train_label, test_data, test_label, 1)
    # save model in a .h5 file
    model.save()
