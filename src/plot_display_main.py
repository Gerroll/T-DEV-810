from loadData import Loader
from plotPrediction import Display

if __name__ == "__main__":
    # config param
    DATA_PATH = './resource/data/test'
    BATCH_SIZE = 400
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CLASS_NAME = ['NORMAL', 'BACTERIA', 'VIRUS']
    
    # load test data
    loader = Loader(DATA_PATH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAME)
    test_data, test_label = loader.load_data()
    test_data = test_data.reshape((400, 224, 224, 1))

    # init Display class
    d = Display(CLASS_NAME, 'neural_network', test_data)

    # display mutiple windows with the predict result
    for i in range(6):
        d.predict(i * 50, test_label, test_data)
