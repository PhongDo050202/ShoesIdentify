# import the modules needed for testing
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob # returns all file paths that match a specific pattern
import cv2
import imghdr #identifies different image file formats
import warnings
warnings.filterwarnings("ignore")
import unittest
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


# define the data directory and the image extensions
data_dir = r"C:\Users\DELL\Pictures\Shoes"
image_exts = ['jpeg','jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path) #loads an image from the specified file
            tip = imghdr.what(image_path) #tests the image data contained in the file
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
#                 os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory(data_dir)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


# define the image height and width
batch_size = 32
img_height = 224
img_width = 224

# define the class names
class_names = ['Adidas', 'Balenciaga', 'Nike', 'Puma']

# define the number of classes
num_classes = len(class_names)

# define the test class
class TestTrainModel(unittest.TestCase):

    # define the test case for training the model
    def test_train_model(self):
        # create a dataset from the data directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        # create a model with the given architecture
        # re-size all the images to this
        IMAGE_SIZE = [224, 224]

        # add preprocessing layer to the front of VGG
        vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        # don't train existing weights
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(4, activation='softmax')(x)

        # create a model object
        vgg16 = Model(inputs=vgg.input, outputs=prediction)

        checkpointVgg16 = ModelCheckpoint("C:/Users/DELL/Pictures",
                                          monitor="val_loss",
                                          mode="min",
                                          save_best_only=True,
                                          verbose=1)

        # compile the model with the given optimizer, loss and metrics
        vgg16.compile(
            loss='SparseCategoricalCrossentropy',
            optimizer='adam',
            metrics=['accuracy']

        )
        # view the structure of the model
        tf.keras.utils.plot_model(
            vgg16,
            to_file="model.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=128,
            layer_range=None,
            show_layer_activations=True,
            show_trainable=False,
        )
        # train the model for 15 epochs
        epochs = 10
        history = vgg16.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=checkpointVgg16
        )
        # check if the model is trained correctly
        self.assertIsNotNone(history, "Model is not trained")
        # check if the model parameters are updated correctly after each epoch
        self.assertEqual(len(history.history['loss']), epochs, "Model parameters are not updated correctly")
        self.assertEqual(len(history.history['accuracy']), epochs, "Model parameters are not updated correctly")
        self.assertEqual(len(history.history['val_loss']), epochs, "Model parameters are not updated correctly")
        self.assertEqual(len(history.history['val_accuracy']), epochs, "Model parameters are not updated correctly")

# run the tests
if __name__ == "__main__":
    unittest.main()
