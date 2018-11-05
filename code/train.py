#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import math
import cv2
from keras.utils.np_utils import to_categorical

from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

# dimensions of our images.
img_width, img_height = 300, 300
top_model_weights_path = 'model/bottleneck_fc_model.h5'
train_data_dir = 'data/Train_data'
validation_data_dir = 'data/Validation_data'
epochs = 50
batch_size = 16


def save_bottlebeck():
    """
      In this function we create the VGG16 model, without the fully connected layers,
      by specifying include_top=False and load using ImageNet weights
    """
    model = applications.VGG16(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames);

    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))  # for fixing a bug, we cal it ourselfs
    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)
    np.save('model/bottleneck_features_train.npy',
            bottleneck_features_train)

    # Validation datas turn
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('model/bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    '''
      Now its time to actually train the top layers of the model aka. the last layers
      We have our bottleneck extracted features using our pre trained model, now we use those to
      finish off our classification.



    '''
    # 1) in order to train the top model, we need the class labels for each training/val instance
    datagen_top = ImageDataGenerator('''preprocessing_function = exposure.equalize_hist''')
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('model/bottleneck_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    '''Repeat for validation set'''

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load(
        'model/bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    '''Now we train a small fully connected network as our top model, 
       using the bottle neck features as inputs'''

    model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)  # saving our model
   # model.save_weights(top_model_weights_path)  # saving our model

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    # '''Plotting training history'''
    #
    # plt.figure(1)
    #
    # # summarize history for accuracy
    #
    # plt.subplot(211)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    #
    # # summarize history for loss
    #
    # plt.subplot(212)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


if __name__ == '__main__':
    save_bottlebeck()#loads in transfer model and save into file, comment this out if you have already done it.
    train_top_model()
