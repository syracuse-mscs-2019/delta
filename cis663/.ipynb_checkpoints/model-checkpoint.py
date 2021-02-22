import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import pandas as pd
import delta.compat as tf

from sklearn.preprocessing import LabelEncoder

BATCH_SIZE=2

def get_fbank(item):
    a = np.load(item)
    a = np.resize(a, (5500,40,1))
    return a


def load_data(input_file):
    dfData = pd.read_csv(input_file)
    dfLabels = dfData['Subject'].astype('category')
    dfFiles = dfData['FileName']

    le = LabelEncoder()
    targets = le.fit_transform(dfLabels) # transforms subject names into a unique id, in all cases

    inputs = []
    for i in range(0, dfFiles.shape[0]):
        inputs.append(get_fbank(dfFiles.at[i]))
    inputs = np.stack(inputs)
    # create categorical inputs for fit function
    return inputs, tf.keras.utils.to_categorical(targets), len(np.unique(targets))

# Load up the config that Delta uses when creating the model
#config = utils.load_config('./speaker_verifier.yml')

# Load the data
inputs, targets, categories = load_data('./trainset.csv')

# Create the model
ilayer = tf.keras.layers.Input(shape=(5500,40,1), name="input")
x = tf.keras.layers.Conv2D(10, (10,10))(ilayer)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
olayer = tf.keras.layers.Dense(categories, activation="softmax", name="output")(x)
model = tf.keras.models.Model(inputs=ilayer, outputs=olayer)
model.summary()

model.compile(optimizer="Adam", loss="categorical_crossentropy")
model.fit(inputs, targets, epochs=10, batch_size=1, verbose=1)
