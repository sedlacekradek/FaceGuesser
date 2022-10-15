import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from keras_preprocessing.image import load_img
from keras.models import load_model
import numpy as np

### CONSTANTS ###
LOADED_MODEL = load_model('models/smile_model.h5')
# pprint(loaded_model.get_config())  # to get hyperparameters
# print(loaded_model.loss)  # to get loss function used for training
# print(loaded_model.optimizer)  # to get optimizer function used for training


### FEATURE EXTRACTION ###
def extract_features(image):
    features = []
    img = load_img(image, grayscale=True)
    img = img.resize((64, 64))
    img = np.array(img)  # convert to array with RGB values for each pixel
    features.append(img)
    features = np.array(features)  # convert to np. as file format suitable for neuro network
    # np.reshape - gives a new shape to an array without changing its data
    # 1) number of entries, 2) dimensions (rows cols), 3) format (3 for RGB, 1 for grayscale)
    features = features.reshape(len(features), 64, 64, 1)
    return features


def eval_smile(picture_path):
    picture = extract_features(picture_path)
    picture = picture / 255
    ### MAKE PREDICTION ###
    pred = LOADED_MODEL.predict(picture.reshape(1, 64, 64, 1))
    print(pred[0][0])
    if pred[0][0] > 0.3:
        return "The person is smiling."
    else:
        return "The person is not smiling."
