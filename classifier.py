from keras.preprocessing.image import ImageDataGenerator
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys

import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers import Dense
import os
import sys


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(
    32, 32, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dropout(0.2))

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=5, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# using imageDataGenerator to preprocess our data
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
evaluate_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'dataset/training', target_size=(32, 32), batch_size=25, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    'dataset/test', target_size=(32, 32), batch_size=25, class_mode='categorical')

# fit generate our model
score_fit = classifier.fit_generator(train_generator, steps_per_epoch=15,
                                     epochs=25, validation_data=test_generator, validation_steps=15)

# print("Train generator indices : ", train_generator.class_indices)
score = classifier.evaluate_generator(test_generator, 32)

print("Evaluation :  ===============================================================")
print("Loss : ", score[0])
print("Accuracy : ", score[1])
