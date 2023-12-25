# import the modules needed for testing
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from keras.src.applications import MobileNetV2
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
from keras.applications import mobilenet_v2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# define the data directory and the image extensions
data_dir = r"C:\Users\DELL\ShoesIdentify\data\test_data"
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
class TestEvaluateModelWithMobileNetV2(unittest.TestCase):

    # define the test case for evaluating the model with MobileNetV2
    def test_evaluate_model_with_MobileNetV2(self):
        # create a dataset from the data directory
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            seed=324893,
            image_size=(img_height, img_width),
            batch_size=32)

        # create a model with the given architecture
        # re-size all the images to this
        IMAGE_SIZE = [224, 224]

        # create a MobileNetV2 model with the given input shape and weights
        base_model = MobileNetV2(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
        # don't train existing weights
        for layer in base_model.layers:
            layer.trainable = False
        # add a flatten layer and a dense layer with softmax activation to the MobileNetV2 model
        x = layers.Flatten()(base_model.output)
        prediction = layers.Dense(num_classes, activation='softmax')(x)
        # create a model object
        model = tf.keras.Model(inputs=base_model.input, outputs=prediction)
        # load the best model from the checkpoint
        model.load_weights("C:/Users/DELL/ShoesIdentify/model/Mobilenet.h5")
        # compile the model with the given optimizer, loss and metrics
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        # evaluate the model on the test dataset
        loss, accuracy = model.evaluate(test_ds, verbose=1, return_dict=True)
        # check if the evaluation returns the correct performance metrics
        self.assertIsNotNone(loss, "Evaluation does not return the loss")
        self.assertIsNotNone(accuracy, "Evaluation does not return the accuracy")
        # check if the performance metrics are in the expected range
        self.assertTrue(loss >= 0, "Loss is negative")
        self.assertTrue(accuracy >= 0 and accuracy <= 1, "Accuracy is out of range")
        # predict the model on the test dataset
        predictions = model.predict(test_ds)
        # check if the predictions have the correct shape
        self.assertEqual(predictions.shape, (len(test_ds) * 32, num_classes), "Predictions have incorrect shape")
        # extract the labels and the predictions for each class
        labels = []
        for _, y in test_ds:
            labels.extend(y.numpy())
        labels_adidas = [1 if i[0] == 1 else 0 for i in labels]
        labels_balenciaga = [1 if i[1] == 1 else 0 for i in labels]
        labels_nike = [1 if i[2] == 1 else 0 for i in labels]
        labels_puma = [1 if i[3] == 1 else 0 for i in labels]
        predic_adidas = [i[0] for i in predictions]
        predic_balenciaga = [i[1] for i in predictions]
        predic_nike = [i[2] for i in predictions]
        predic_puma = [i[3] for i in predictions]
        labels_for_each_class = [[labels_adidas, predic_adidas], [labels_balenciaga, predic_balenciaga],
                                 [labels_nike, predic_nike], [labels_puma, predic_puma]]
        # plot the precision-recall curve for each class
        plt.figure()
        for i in range(len(labels_for_each_class)):
            precision, recall, _ = precision_recall_curve(labels_for_each_class[i][0],
                                                          labels_for_each_class[i][1])
            plt.plot(recall, precision, lw=2, label='class {}'.format(i))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        plt.show()
        # plot the confusion matrix for the model
        new_predictions = [np.argmax(i) for i in predictions]
        ConfusionMatrixDisplay.from_predictions(
            labels, new_predictions,
            display_labels=class_names,
            cmap=plt.cm.Blues)
        plt.show()

    # run the tests
    if __name__ == "__main__":
        unittest.main()
