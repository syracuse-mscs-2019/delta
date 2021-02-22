import os
import numpy as np
import pandas as pd
import delta.compat as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

BATCH_SIZE=2

def get_fbank(item):
    a = np.load(item)
    a = np.resize(a, (2900,10,1))
    return a

def load_data(input_file):
    dfData = pd.read_csv(input_file)
    dfLabels = dfData['Subject'].astype('category')
    dfFiles = dfData['FileName']

    le = LabelEncoder()
    targets = le.fit_transform(dfLabels)

    inputs = []
    for i in range(0, dfFiles.shape[0]):
        inputs.append(get_fbank(dfFiles.at[i]))
    inputs = np.stack(inputs)

    #xmin = inputs.min()
    #xmax = inputs.max()
    #inputs = (inputs-xmin)/(xmax-xmin)

    return inputs, tf.keras.utils.to_categorical(targets), len(np.unique(targets))

# def genData():
#     dfData = pd.read_csv('./trainset.csv')
#     dfLabels = dfData['Subject'].astype('category')
#     le = LabelEncoder()
#     targets = le.fit_transform(dfLabels)
#     targets = tf.keras.utils.to_categorical(targets))

#     x = 0
#     for rows in dfData.iterrows():
#         x = x + 1
#         yield {'input': get_fbank(rows[1].FileName)}, targets[1]

# def input_gen():
#     dataset = tf.data.Dataset.from_generator(genData, ({'input': tf.float32}, tf.int16),
#                                             ({'input': tf.TensorShape([5500,40,1])}, tf.TensorShape([78])))
#     dataset = dataset.shuffle(10).repeat(10).batch(2).prefetch(1)
#     return dataset

# # Load up the config that Delta uses when creating the model
# data = input_gen()

# ilayer = tf.keras.layers.Input(shape=(5500,40,1), name="input")
# x = tf.keras.layers.Dense(512, activation="relu")(ilayer)
# x = tf.keras.layers.Dropout(0.2)(x)
# olayer = tf.keras.layers.Dense(78, activation="softmax", name="output")(x)
# model = tf.keras.models.Model(inputs=ilayer, outputs=olayer)
# model.summary()

# model.compile(optimizer="Adam", loss="categorical_crossentropy")
# model.fit(data, epochs=10, verbose=1)


# Load the data
inputs, targets, categories = load_data('./trainset.csv')
#dataset, categories = load_data('./trainset.csv')

# Create the model
ilayer = tf.keras.layers.Input(shape=(2900,10,1), name="input")
x = tf.keras.layers.Conv2D(8, (3,3))(ilayer)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Conv2D(16, (3,3))(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
olayer = tf.keras.layers.Dense(categories, activation="softmax", name="output")(x)
model = tf.keras.models.Model(inputs=ilayer, outputs=olayer)
model.summary()

model.compile(optimizer="Adam", loss="categorical_crossentropy")
model.fit(inputs, targets, epochs=10, batch_size=1, verbose=1)
