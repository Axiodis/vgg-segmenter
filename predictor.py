import tensorflow as tf
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

tf.reset_default_graph()
NP_IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)


class Predictor:
    def __init__(self, num_classes=22, model_path="model/frozen_model.pb"):
        graph = self.load_graph(model_path)

        self.x = graph.get_tensor_by_name("prefix/input_image:0")
        self.keep_prob = graph.get_tensor_by_name("prefix/Placeholder:0")
        self.y = graph.get_tensor_by_name('prefix/ArgMax_1:0')
        self.prob = tf.nn.softmax(graph.get_tensor_by_name('prefix/upscore8/conv2d_transpose:0'))

        # self.label_processor = VocLabelsProcessor(num_classes)
        # self.image_from_path = self.prepare_image_from_path()

        self.sess = tf.Session(graph=graph)

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
        # image_flat_shape = image.shape[0] * image.shape[1]

        image = image.astype(np.float32)
        image -= NP_IMAGENET_MEAN

        prob = self.sess.run(self.prob, feed_dict={self.x: np.expand_dims(image, axis=0), self.keep_prob: 1.0})
        processed_probabilities = prob.squeeze()
        softmax = processed_probabilities.transpose((2, 0, 1))

        unary = unary_from_softmax(softmax)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 22)
        d.setUnaryEnergy(unary)

        # This potential penalizes small pieces of segmentation that are
        # spatially isolated -- enforces more spatially consistent segmentations
        feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features --
        # because the segmentation that we get from CNN are too coarse
        # and we can use local color features to refine them
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=image, chdim=2)

        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(5)

        res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

        return res

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        self.sess.close()
