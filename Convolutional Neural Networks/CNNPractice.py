import os.path

import keras as k
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

# this red line will fix itself at run time
sys.path.insert(1, '/home/crystal/PycharmProjects/Internship-AI-3/Data and Data Tools')
import image_converter
import tensorflow as tf

data_initialization = True
Model_initialization = True
#Initializes the data, only needs to be done once

# Checks if data is initialized, if not it rescales the data you are using to be a certain size
if not data_initialization:
    source_cats = '/home/crystal/Downloads/cats_and_dogs/train/cat'
    source_dogs = '/home/crystal/Downloads/cats_and_dogs/train/dog'
    val_cats =  '/home/crystal/Downloads/cats_and_dogs/val/cat'
    val_dogs = '/home/crystal/Downloads/cats_and_dogs/val/dog'
    test_cats = '/home/crystal/Downloads/real_test/cat'
    test_dogs = '/home/crystal/Downloads/real_test/dog'
    image_converter.convert_data(source_cats, 224, 224)
    image_converter.convert_data(source_dogs, 224, 224)
    image_converter.convert_data(val_cats, 224, 224)
    image_converter.convert_data(val_dogs, 224, 224)
    image_converter.convert_data(test_cats, 224, 224)
    image_converter.convert_data(test_dogs, 224, 224)


# Creates the training data
training_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                      '/cats_and_dogs/train',
                                                                                      target_size=(224, 224),
                                                                                      classes=['dog', 'cat'],
                                                                                      batch_size=8)
# Creates the validation data
validation_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                        '/cats_and_dogs/val',
                                                                                        target_size=(224, 224),
                                                                                        classes=['dog', 'cat'],
                                                                                        batch_size=1)
# Creates the test data
test_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('/home/crystal/Downloads'
                                                                                        '/real_test',
                                                                                        target_size=(224, 224),
                                                                                        classes=['dog', 'cat'],
                                                                                        batch_size=1)

# Creates the model with 3 Convolutional layers, then flattens out the data with a flatten layer and then gives an output layer
model = k.models.Sequential([k.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Conv2D(15, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             k.layers.Flatten(), k.layers.Dense(2, activation='softmax')])
# Makes the model with the RMSprop model
model.compile(k.optimizers.RMSprop(
    learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = '/home/crystal/Downloads/Models/Checkpoints_cats_vs_dogs_Model.keras'
model_checkpoint_callback = k.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
if not (Model_initialization):
    model.fit(training_data, validation_data=validation_data, validation_steps=1,
          steps_per_epoch=2, epochs=50, verbose=2, callbacks=[model_checkpoint_callback])
else:
    A = ['dog', 'cat']
    k.models.load_model(checkpoint_filepath)
    predictions = model.predict(test_data, steps=1, verbose=0)
    i = 1
    while os.path.isfile(f"/home/crystal/Downloads/real_test/cat/cat-{i}.jpeg"):
        try:
            # Cat images
            img = cv2.imread(f"/home/crystal/Downloads/real_test/cat/cat-{i}.jpeg")
            unfiltered_image = cv2.imread(f"/home/crystal/Downloads/real_test/cat/cat-{i}.jpeg")
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(prediction)
            if (np.argmax(prediction) == 1):
                result = 'cat'
            elif (np.argmax(prediction) == 0):
                result = 'dog'
            print(f"This image is probably a {A[np.argmax(prediction)]}")
            plt.imshow(unfiltered_image, cmap=plt.cm.binary)
            plt.show()

            # Dog images
            img2 = cv2.imread(f"/home/crystal/Downloads/real_test/dog/dog-{i}.jpeg")
            unfiltered_image2 = cv2.imread(f"/home/crystal/Downloads/real_test/dog/dog-{i}.jpeg")
            img2 = np.invert(np.array([img2]))
            prediction2 = model.predict(img2)
            #if np.argmax(prediction2) == 0:
            #    result2 = 'cat'
            #elif np.argmax(prediction2) == 1:
            #    result2 = 'dog'
            print(f"This image is probably a {A[np.argmax(prediction2)]}")
            plt.imshow(unfiltered_image2, cmap=plt.cm.binary)
            plt.show()

        finally:
            i = (i + 1)







