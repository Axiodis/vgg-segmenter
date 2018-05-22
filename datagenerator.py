import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import random

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, mode, batch_size = 1, shuffle=True,buffer_size=1000):
        
        self._read_txt_file(txt_file)

        self.data_size = len(self.images)

        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.images = convert_to_tensor(self.images, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.string)

        # create dataset
        data = Dataset.from_tensor_slices((self.images, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8, output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8, output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))
            
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self, txt_file):
        self.images = []
        self.labels = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.images.append(items[0].strip("\r\n"))
                self.labels.append(items[1].strip("\r\n"))

    def _shuffle_lists(self):
        images = self.images
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.images = []
        self.labels = []
        for i in permutation:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, image, label):
        
        # load and preprocess the image
        img_string = tf.read_file(image)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        
        # load and preprocess the label
        label_string = tf.read_file(label)
        label_decoded = tf.image.decode_png(label_string)
        
        """
        Data augmentation
        """
        if random.random() < 0.5:
            img_decoded = tf.image.flip_left_right(img_decoded)
            label_decoded = tf.image.flip_left_right(label_decoded)
        
        img_centered = tf.subtract(tf.to_float(img_decoded), IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        
        return img_bgr, label_decoded

    def _parse_function_inference(self, image, label):
        
        # load and preprocess the image
        img_string = tf.read_file(image)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        
        # load and preprocess the label
        label_string = tf.read_file(label)
        label_decoded = tf.image.decode_png(label_string)
        
        """
        Data augmentation
        """
        img_centered = tf.subtract(tf.to_float(img_decoded), IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label_decoded