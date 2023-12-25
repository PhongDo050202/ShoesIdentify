# import the modules needed for testing
import unittest
import os
import cv2
import imghdr

# define the data directory and the image extensions
data_dir = r"D:\Pictures\Screenshots\Đồ Án Chuyên Ngành\TestImage"
image_exts = ["jpg", "png", "gif"]

# define the test class
class TestImageProcessing(unittest.TestCase):
    def remove_invalid_images(data_dir):
        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        for image_class in os.listdir(data_dir):
            class_path = os.path.join(data_dir, image_class)
            if os.path.isdir(class_path):  # Check if it's a directory
                for image in os.listdir(class_path):
                    image_path = os.path.join(class_path, image)
                    try:
                        img = cv2.imread(image_path)
                        tip = imghdr.what(image_path)
                        if tip not in image_exts:
                            os.remove(image_path)
                    except Exception as e:
                        os.remove(image_path)

    # define the test case for removing invalid images
    def test_remove_invalid_images(self):
        # loop through the image classes
        for image_class in os.listdir(data_dir):
            # loop through the images in each class
            for image in os.listdir(os.path.join(data_dir, image_class)):
                image_path = os.path.join(data_dir, image_class, image)
                # get the image type
                tip = imghdr.what(image_path)
                # check if the image type is in the image extensions
                self.assertIn(tip, image_exts, f"Image {image_path} is not in ext list")

    # define the test case for reading images
    def test_read_images(self):
        # loop through the image classes
        for image_class in os.listdir(data_dir):
            # loop through the images in each class
            for image in os.listdir(os.path.join(data_dir, image_class)):
                image_path = os.path.join(data_dir, image_class, image)
                # read the image using cv2
                img = cv2.imread(image_path)
                # check if the image is not None
                self.assertIsNotNone(img, f"Image {image_path} is None")
                # check if the image shape is valid
                self.assertEqual(len(img.shape), 3, f"Image {image_path} has invalid shape")

    # define the test case for handling non-existing images
    def test_handle_non_existing_images(self):
        # create a fake image path
        fake_image_path = os.path.join(data_dir, "fake", "fake.jpg")
        # try to read the image using cv2
        try:
            img = cv2.imread(fake_image_path)
        # catch the exception
        except Exception as e:
            # check if the exception message is correct
            self.assertEqual(str(e), f"Issue with image {fake_image_path}", f"Exception message is incorrect")

# run the tests
if __name__ == "__main__":
    unittest.main()
