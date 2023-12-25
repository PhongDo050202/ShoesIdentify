# import the modules needed for testing
import unittest
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# define the model path and the input/output paths
model_dir = "C:/Users/DELL/ShoesIdentify/model" # use the absolute path of the model directory
model_path = os.path.join(model_dir, "Mobilenet.h5") # use os.path.join to concatenate the model directory and the model file name
output_dir = "C:/Users/DELL/ShoesIdentify/static/output" # use the absolute path of the output directory
output_image_path = os.path.join(output_dir, "bar.png") # use os.path.join to concatenate the output directory and the output image file name
input_image_path = os.path.join(output_dir, "origin_image.png") # use os.path.join to concatenate the output directory and the input image file name

# define the class names
class_name = ["Adidas", "Balenciaga", "Nike", "Puma"]

# define the test class
class TestPredictImage(unittest.TestCase):

    # define the test case for predicting an image
    def test_predict_image(self):
        # load the model from the model path
        model = keras.models.load_model(model_path)
        # load the image from the input image path
        img = Image.open(input_image_path)
        # resize the image to (224, 224)
        img = img.resize((224, 224))
        # convert the image to a numpy array
        img_array = keras.utils.img_to_array(img)
        # expand the dimensions of the array to (1, 224, 224, 3)
        img_array = tf.expand_dims(img_array, 0)
        # predict the model on the image array
        predict = model.predict(img_array)
        # get the score and the brand from the prediction
        score = predict
        brand = class_name[np.argmax(score)]
        # check if the prediction returns the correct score and brand
        self.assertIsNotNone(score, "Prediction does not return the score")
        self.assertIsNotNone(brand, "Prediction does not return the brand")
        # check if the score is a numpy array with the shape (1, 4)
        self.assertIsInstance(score, np.ndarray, "Score is not a numpy array")
        self.assertEqual(score.shape, (1, 4), "Score has incorrect shape")
        # check if the brand is a string in the class name list
        self.assertIsInstance(brand, str, "Brand is not a string")
        self.assertIn(brand, class_name, "Brand is not in the class name list")

    # define the test case for handling invalid image
    def test_handle_invalid_image(self):
        # load the model from the model path
        model = keras.models.load_model(model_path)
        # create a fake image path
        fake_image_path = "path/to/fake/image.png"
        # try to load the image from the fake image path
        try:
            img = Image.open(fake_image_path)
        # catch the exception
        except Exception as e:
            # check if the exception message is correct
            self.assertEqual(str(e), f"[Errno 2] No such file or directory: '{fake_image_path}'", "Exception message is incorrect")

# run the tests
if __name__ == "__main__":
    unittest.main()
