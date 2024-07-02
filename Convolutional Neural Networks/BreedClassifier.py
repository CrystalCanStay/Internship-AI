import os
import random

import cv2
import keras as k
import numpy as np

import tensorflow as tf
import sys

from matplotlib import pyplot as plt

# this red line will fix itself at run time
sys.path.insert(1, '/home/crystal/PycharmProjects/Internship-AI-3/Data and Data Tools')
import image_converter


# Data initialization
dataInitialization = True
modelInitialization = True

# Data paths
training_data_path = '/home/crystal/Downloads/Cats-and-Dogs-Breed-Dataset-main/TRAIN'
validation_data_path = '/home/crystal/Downloads/Cats-and-Dogs-Breed-Dataset-main/VAL'
testing_data_path = '/home/crystal/Downloads/Cats-and-Dogs-Breed-Dataset-main/TEST'

if not dataInitialization:

    # Doing this next step due to the fact that the image converter can't deal with subdirectories yet
    # Basically its just running through each data path and converting all the images in every subdirectory
    # Commented out because I only had to run it once, time took to fully resize 18000 images (3 hours)
    '''data_array = [training_data_path, validation_data_path, testing_data_path]
    for i in range(len(data_array)):
        files = image_converter.files_to_array((data_array[i]))
        for j in range(len(files)):
            image_converter.convert_data(files[j], 224, 224)
    '''
    image_converter.convert_data(testing_data_path, 224, 224)
# There has to be a better way of doing this, ignore the spaghetti code
training_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                      '/Cats-and-Dogs-Breed-Dataset'
                                                                                      '-main/TRAIN',
                                                                                      target_size=(224, 224),
                                                                                      classes=['abyssinian',
                                                                                               'american_bulldog',
                                                                                               'american_pit_bull_terrier',
                                                                                               'basset_hound',
                                                                                               'beagle',
                                                                                               'bengal',
                                                                                               'birman',
                                                                                               'bombay',
                                                                                               'boxer',
                                                                                               'british_shorthair',
                                                                                               'chihuahua',
                                                                                               'egyptian_mau',
                                                                                               'english_cocker_spaniel',
                                                                                               'english_setter',
                                                                                               'german_shorthaired',
                                                                                               'great_pyrenees',
                                                                                               'havanese',
                                                                                               'japanese_chin',
                                                                                               'keeshond',
                                                                                               'leonberger',
                                                                                               'maine_coon',
                                                                                               'miniature_pinscher',
                                                                                               'newfoundland',
                                                                                               'persian',
                                                                                               'pomeranian',
                                                                                               'pug',
                                                                                               'ragdoll',
                                                                                               'russian_blue',
                                                                                               'saint_bernard',
                                                                                               'samoyed',
                                                                                               'scottish_terrier',
                                                                                               'shiba_inu',
                                                                                               'siamese',
                                                                                               'sphynx',
                                                                                               'staffordshire_bull_terrier',
                                                                                               'wheaten terrier',
                                                                                               'yorkshire_terrier'],
                                                                                      batch_size=10)
validation_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                        '/Cats-and-Dogs-Breed-Dataset'
                                                                                        '-main/VAL',
                                                                                        target_size=(224, 224),
                                                                                        classes=['abyssinian',
                                                                                                 'american_bulldog',
                                                                                                 'american_pit_bull_terrier',
                                                                                                 'basset_hound',
                                                                                                 'beagle',
                                                                                                 'bengal',
                                                                                                 'birman',
                                                                                                 'bombay',
                                                                                                 'boxer',
                                                                                                 'british_shorthair',
                                                                                                 'chihuahua',
                                                                                                 'egyptian_mau',
                                                                                                 'english_cocker_spaniel',
                                                                                                 'english_setter',
                                                                                                 'german_shorthaired',
                                                                                                 'great_pyrenees',
                                                                                                 'havanese',
                                                                                                 'japanese_chin',
                                                                                                 'keeshond',
                                                                                                 'leonberger',
                                                                                                 'maine_coon',
                                                                                                 'miniature_pinscher',
                                                                                                 'newfoundland',
                                                                                                 'persian',
                                                                                                 'pomeranian',
                                                                                                 'pug',
                                                                                                 'ragdoll',
                                                                                                 'russian_blue',
                                                                                                 'saint_bernard',
                                                                                                 'samoyed',
                                                                                                 'scottish_terrier',
                                                                                                 'shiba_inu',
                                                                                                 'siamese',
                                                                                                 'sphynx',
                                                                                                 'staffordshire_bull_terrier',
                                                                                                 'wheaten terrier',
                                                                                                 'yorkshire_terrier'],
                                                                                        batch_size=10)
test_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                        '/Cats-and-Dogs-Breed-Dataset'
                                                                                        '-main/VAL',
                                                                                        target_size=(224, 224),
                                                                                        classes=['abyssinian',
                                                                                                 'american_bulldog',
                                                                                                 'american_pit_bull_terrier',
                                                                                                 'basset_hound',
                                                                                                 'beagle',
                                                                                                 'bengal',
                                                                                                 'birman',
                                                                                                 'bombay',
                                                                                                 'boxer',
                                                                                                 'british_shorthair',
                                                                                                 'chihuahua',
                                                                                                 'egyptian_mau',
                                                                                                 'english_cocker_spaniel',
                                                                                                 'english_setter',
                                                                                                 'german_shorthaired',
                                                                                                 'great_pyrenees',
                                                                                                 'havanese',
                                                                                                 'japanese_chin',
                                                                                                 'keeshond',
                                                                                                 'leonberger',
                                                                                                 'maine_coon',
                                                                                                 'miniature_pinscher',
                                                                                                 'newfoundland',
                                                                                                 'persian',
                                                                                                 'pomeranian',
                                                                                                 'pug',
                                                                                                 'ragdoll',
                                                                                                 'russian_blue',
                                                                                                 'saint_bernard',
                                                                                                 'samoyed',
                                                                                                 'scottish_terrier',
                                                                                                 'shiba_inu',
                                                                                                 'siamese',
                                                                                                 'sphynx',
                                                                                                 'staffordshire_bull_terrier',
                                                                                                 'wheaten terrier',
                                                                                                 'yorkshire_terrier'],
                                                                                        batch_size=10)



model = k.models.Sequential([k.layers.Conv2D(150, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Conv2D(80, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Conv2D(40, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Flatten(), k.layers.Dense(37, activation='softmax')])
model.compile(k.optimizers.RMSprop(learning_rate=.00001), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint_filepath = '/home/crystal/Downloads/Models/Cat_And_Dog_Breeds_Model.keras'
model_checkpoint_callback = k.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
if not (modelInitialization):
    model.fit(training_data, validation_data=validation_data, validation_steps=5,
              steps_per_epoch=5, epochs=100, verbose=2, callbacks=[model_checkpoint_callback])
else:

    # I'm so lost
    k.models.load_model(checkpoint_filepath)
    #temp_path = '/home/crystal/Downloads/Cats-and-Dogs-Breed-Dataset-main/TRAIN/american_bulldog'
    predictions = model.predict(test_data, steps=1, verbose=0)
    files = image_converter.files_to_array(testing_data_path)

    i = 1
    while i < 15:
        try:
            # Cat images
            img = cv2.imread(files[random.randint(1, 1499)])
            unfiltered_image = img
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            #print(prediction)
            print(f"This image is probably a {np.argmax(prediction)}")
            plt.imshow(unfiltered_image, cmap=plt.cm.binary)
            plt.show()
        finally:
            i = (i + 1)