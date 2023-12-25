# import the modules needed for testing
import unittest
import tensorflow as tf
import numpy as np
from keras import layers

# define the data directory and the image extensions
data_dir = r"C:\Users\DELL\Pictures\TestImage"
image_exts = ["jpg", "png", "gif"]

# define the test class
class TestPreprocessingData(unittest.TestCase):

    # define the test case for normalizing the input data
    def test_normalize_input_data(self):
        # create a dataset from the data directory
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(28, 28),
            batch_size=32)
        # create a normalization layer
        normalization_layer = layers.Rescaling(1./255)
        # apply the normalization layer to the dataset
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        # get the first batch of images and labels
        image_batch, labels_batch = next(iter(normalized_ds))
        # get the first image
        first_image = image_batch[0]
        # check if the image values are in the range [0, 1]
        self.assertTrue(np.all(first_image >= 0) and np.all(first_image <= 1), "Image values are not normalized")

    # define the test case for handling invalid data
    def test_handle_invalid_data(self):
        # create a fake data directory
        fake_data_dir = r"C:\Users\DELL\Pictures\fake"
        # try to create a dataset from the fake data directory
        try:
            fake_ds = tf.keras.preprocessing.image_dataset_from_directory(
                fake_data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(28, 28),
                batch_size=32)
        # catch the exception
        except Exception as e:
            # check if the exception message is correct
            self.assertEqual(str(e), f"FileNotFoundError: [Errno 2] No such file or directory: '{fake_data_dir}'", "Exception message is incorrect")

# run the tests
if __name__ == "__main__":
    unittest.main()
