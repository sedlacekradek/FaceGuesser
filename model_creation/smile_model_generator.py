import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from keras_preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input


### ERROR MESSAGE PARAMETERS ###
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


### CONSTANTS ###
BASE_DIR = "../static/smile_databank"


### PICTURE DATA ###
image_paths = []
smile_labels = []

for filename in os.listdir(BASE_DIR):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split("_")
    smile = int(temp[0])
    image_paths.append(image_path)
    smile_labels.append(smile)


### CONVERT TO DATAFRAME ###
df = pd.DataFrame()
df["image"], df["smile"] = image_paths, smile_labels
df = df.sample(frac=1)  # randomize order


### FEATURE EXTRACTION ###
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)
        img = img.resize((64, 64))
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    return features  # returns features in np.array


x = extract_features(df["image"])
x = x / 255.0  # normalization


### MODEL CREATION ###
y_smile = np.array(df["smile"])  # convert "correct answers" to np.array
input_shape = (64, 64, 1)
inputs = Input(input_shape)  # to define expected input shape for the filter(=kernel)

conv_1 = Conv2D(256, kernel_size=(3, 3), activation="relu")(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(512, kernel_size=(3, 3), activation="relu")(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
dropout_1 = Dropout(0.8)(maxp_2)
conv_3 = Conv2D(512, kernel_size=(3, 3), activation="relu")(dropout_1)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
dropout_2 = Dropout(0.8)(maxp_3)
conv_4 = Conv2D(512, kernel_size=(3, 3), activation="relu")(dropout_2)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)  # flatten -> matrix structure to single-dimensional values
dense_1 = Dense(1024, activation='relu')(flatten)
dropout_3 = Dropout(0.8)(dense_1)
output_1 = Dense(1, activation='sigmoid', name='smile_out')(dropout_3)

model = Model(inputs=[inputs], outputs=[output_1])
model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])


### MODEL TRAINING ###
history = model.fit(x=x, y=y_smile, batch_size=32, epochs=32, validation_split=0.25, verbose=1)


### VISUALISE THE RESULTS ###
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training Accuracy')  # axis x, y, color, label
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.savefig("smile_acc_v02.jpg")


### SAVE MODEL ###
model.save('models/smile_v02.h5')
