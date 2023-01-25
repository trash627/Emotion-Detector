import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.python.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50


train_dir = "Live-Emotion-Detection/inputs/train"
test_dir = "Live-Emotion-Detection/inputs/test"


img_size = 48
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   rescale=1./255,
                                   validation_split=0.2
                                   )

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(
                                                        img_size, img_size),
                                                    batch_size=64,
                                                    color_mode="grayscale",
                                                    class_mode="categorical",
                                                    subset="training"
                                                    )


validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

validation_generator = validation_datagen.flow_from_directory(directory=test_dir,
                                                              target_size=(
                                                                  img_size, img_size),
                                                              batch_size=64,
                                                              color_mode="grayscale",
                                                              class_mode="categorical",
                                                              subset="validation"
                                                              )


# model= tf.keras.models.Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
# model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256,activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

# model.add(Dense(512,activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

# model.add(Dense(7, activation='softmax'))

# model.compile(
#     optimizer = Adam(lr=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
#   )


# model.summary()


model = tf.keras.models.Sequential()
pretrained_model = ResNet50(input_shape=(
    48, 48, 3), include_top=False, weights='imagenet')
pretrained_model.trainable = True

model.add(Conv2D(3, (1, 1), padding='same', input_shape=(48, 48, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(7))
model.summary()

# call-backs

# early stopping if no improvement
early_stop = EarlyStopping(
    monitor='loss', patience=30, mode='min', baseline=None)

# reduce learning rate when a metric has stop improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.7, patience=5, min_lr=1e-15, mode='min', verbose=1)


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 200
batch_szie = 60


history = model.fit(x=train_generator, epochs=epochs, validation_data=validation_generator,
                    callbacks=[early_stop, reduce_lr])


fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12, 4)
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')
plt.show()


# save model and weights
model.save('model_resnet1.h5')
model.save_weights('model_resnet1_weights.h5')
