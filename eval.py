import os
import warnings
from keras_preprocessing.image import load_img
from keras.models import load_model
import numpy as np


### ERROR MESSAGE PARAMETERS ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


### CONSTANTS ###
GENDER_DICT = {0: "Male", 1: "Female"}
FACE_MODEL = load_model('models/face_model.h5')
SMILE_MODEL = load_model('models/smile_model.h5')


### PRINT MODEL CONFIG ###
# from pprint import pprint
# pprint(loaded_model.get_config())  # to get hyper-parameters
# print(loaded_model.loss)  # to get loss function used for training
# print(loaded_model.optimizer)  # to get optimizer function used for training


### FEATURE EXTRACTION ###
def extract_features(image, size):
    """
    takes image and image size and returns np.array of pixel values suitable for CNN
    """
    features = []
    img = load_img(image, grayscale=True)
    img = img.resize((size, size))  # 64x64 or 128x128
    img = np.array(img)  # converts to array with 0-255 values for each pixel
    features.append(img)
    features = np.array(features)  # converts to np. as a file format suitable for CNN
    return features


def eval_face(picture_path):
    picture = extract_features(picture_path, 128)
    picture = picture / 255
    ### MAKE PREDICTION ###
    pred = FACE_MODEL.predict(picture.reshape(1, 128, 128, 1))
    pred_gender = GENDER_DICT[round(pred[0][0][0])]
    if pred_gender == "Male":
        pred_age = round((pred[1][0][0]) * 0.9)  # manual adjustment of values based on user experience
    else:
        pred_age = round((pred[1][0][0]) * 1.25)
    # "|safe" tag used in template to allow html tags
    return f"Predicted gender: {pred_gender} <br> Predicted age: {pred_age}"


def eval_smile(picture_path):
    picture = extract_features(picture_path, 64)
    picture = picture / 255
    ### MAKE PREDICTION ###
    pred = SMILE_MODEL.predict(picture.reshape(1, 64, 64, 1))
    if pred[0][0] > 0.1:
        return "The person is smiling."
    else:
        return "The person is not smiling."