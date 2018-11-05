import pandas as pd
import numpy as np
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(300, 300, 3)))
model.add(Dense(64, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory("data/Train_data",
                                                 target_size=(300, 300),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('data/Validation_data',
                                            target_size=(300, 300),
                                            batch_size=32,
                                            class_mode='categorical')
model.fit_generator(training_set,
                    steps_per_epoch=3600,  # number of images in Train_data
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=900  # number in test i think

                    )

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory("data/Train_data",
                                                 target_size=(300, 300),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory("data/test",
                                            target_size=(300, 300),
                                            batch_size=32,
                                            class_mode='categorical')
model.fit_generator(training_set,
                    steps_per_epoch=100,  # number of images in Train_data
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=15  # number in test i think
                    )

