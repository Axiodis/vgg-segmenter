import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from processlabel import VocLabelsProcessor

tf.reset_default_graph()
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
NP_IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)


class Predictor:
    def __init__(self, num_classes=22, model_path="frozen_model.pb"):
        graph = self.load_graph(model_path)

        self.x = graph.get_tensor_by_name("prefix/input_image:0")
        self.keep_prob = graph.get_tensor_by_name("prefix/Placeholder:0")
        self.y = graph.get_tensor_by_name('prefix/ArgMax_1:0')

        #self.label_processor = VocLabelsProcessor(num_classes)
        #self.image_from_path = self.prepare_image_from_path()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(graph=graph, config=sess_config)

    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def __enter__(self):
        return self

    def predict_image(self, image):
        image = image.astype(np.float32)
        image -= NP_IMAGENET_MEAN
        image = np.expand_dims(image, axis=0)

        result_label = self.sess.run(self.y, feed_dict={self.x: image, self.keep_prob: 1.0})[0]

        return result_label

    # def predict_image_from_path(self, image_path):
    #     image = self.sess.run(self.image_from_path, feed_dict={self.x_path: image_path})
    #
    #     label_pred = self.sess.run(self.y, feed_dict={self.x: image, self.keep_prob: 1.0})
    #
    #     result_label = self.label_processor.class_label_to_color(label_pred[0])
    #
    #     return result_label
    #
    # def prepare_image_from_path(self):
    #     image_path = convert_to_tensor(self.x_path, dtype=dtypes.string)
    #
    #     img_string = tf.read_file(image_path)
    #
    #     img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    #
    #     img_decoded = tf.to_float(img_decoded)
    #
    #     img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)
    #
    #     img_bgr = img_centered[:, :, ::-1]
    #
    #     image = tf.expand_dims(img_bgr, 0)
    #
    #     return image

    def __del__(self):
        self.sess.close()
