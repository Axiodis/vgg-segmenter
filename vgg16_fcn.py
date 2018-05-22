import numpy as np
import tensorflow as tf
import inspect
import os


class VGG16_FCN:
    def __init__(self, x, num_classes, keep_prob):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        self.SKIP_LAYERS = ['fc8']

        self.build()

    def build(self):

        # ------------Build VGG16-FCN normal layers--------------------------

        # Layer 1
        self.conv1_1 = self.conv(self.X, 3, 3, 64, 1, 1, "conv1_1")
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 2, 2, 2, 2, 'pool1')

        # Layer 2
        self.conv2_1 = self.conv(self.pool1, 3, 3, 128, 1, 1, "conv2_1")
        self.conv2_2 = self.conv(self.conv2_1, 3, 3, 128, 1, 1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 2, 2, 2, 2, 'pool2')

        # Layer 3
        self.conv3_1 = self.conv(self.pool2, 3, 3, 256, 1, 1, "conv3_1")
        self.conv3_2 = self.conv(self.conv3_1, 3, 3, 256, 1, 1, "conv3_2")
        self.conv3_3 = self.conv(self.conv3_2, 3, 3, 256, 1, 1, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 2, 2, 2, 2, 'pool3')

        # Layer 4
        self.conv4_1 = self.conv(self.pool3, 3, 3, 512, 1, 1, "conv4_1")
        self.conv4_2 = self.conv(self.conv4_1, 3, 3, 512, 1, 1, "conv4_2")
        self.conv4_3 = self.conv(self.conv4_2, 3, 3, 512, 1, 1, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 2, 2, 2, 2, 'pool4')

        # Layer 5
        self.conv5_1 = self.conv(self.pool4, 3, 3, 512, 1, 1, "conv5_1")
        self.conv5_2 = self.conv(self.conv5_1, 3, 3, 512, 1, 1, "conv5_2")
        self.conv5_3 = self.conv(self.conv5_2, 3, 3, 512, 1, 1, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 2, 2, 2, 2, 'pool5')

        # -----------------------Build VGG16-FCN fully connvolutional layers---------

        self.conv6 = self.conv(self.pool5, 7, 7, 4096, 1, 1, "fc6")
        self.drop6 = self.dropout(self.conv6, self.KEEP_PROB)

        self.conv7 = self.conv(self.drop6, 1, 1, 4096, 1, 1, "fc7")
        self.drop7 = self.dropout(self.conv7, self.KEEP_PROB)

        self.score_fr = self.conv(self.drop7, 1, 1, self.NUM_CLASSES, 1, 1, "fc8",
                                  relu=False)

        # -----------------------Build VGG16-FCN upsample layers--------------------

        self.upscore2 = self.deconv(self.score_fr,
                                    shape=tf.shape(self.pool4),
                                    num_classes=self.NUM_CLASSES,
                                    name='upscore2',
                                    ksize=4, stride=2)
        self.score_pool4 = self.conv(self.pool4, 1, 1, self.NUM_CLASSES, 1, 1, "score_pool4",
                                     relu=False)
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore4 = self.deconv(self.fuse_pool4,
                                    shape=tf.shape(self.pool3),
                                    num_classes=self.NUM_CLASSES,
                                    name='upscore4',
                                    ksize=4, stride=2)
        self.score_pool3 = self.conv(self.pool3, 1, 1, self.NUM_CLASSES, 1, 1, "score_pool3",
                                     relu=False)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        self.upscore8 = self.deconv(self.fuse_pool3,
                                    shape=tf.shape(self.X),
                                    num_classes=self.NUM_CLASSES,
                                    name='upscore8',
                                    ksize=16, stride=8)

        self.pred_up = tf.cast(tf.argmax(self.upscore8, axis=3), tf.uint8, name='prediction')

    def deconv(self, x, shape, num_classes, name, ksize=4, stride=2):

        strides = [1, stride, stride, 1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            in_features = x.get_shape()[3].value

            if shape is None:
                in_shape = tf.shape(x)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]

            output_shape = tf.stack(new_shape)
            weights = tf.get_variable('up_filter', shape=[ksize,
                                                          ksize,
                                                          num_classes,
                                                          in_features])

            deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=strides, padding='SAME')

        return deconv

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             relu=True, padding='SAME'):

        input_channels = int(x.get_shape()[-1])

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels,
                                                        num_filters])

            biases = tf.get_variable('biases', shape=[num_filters], )

        conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding)

        bias = tf.nn.bias_add(conv, biases)

        if relu:
            relu = tf.nn.relu(bias, name=scope.name)

            return relu

        return bias

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):

        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def load_initial_weights(self, session, vgg16_npy_path=None):

        if vgg16_npy_path is None:
            path = inspect.getfile(VGG16_FCN)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path

        weights_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        for op_name in weights_dict:

            if op_name not in self.SKIP_LAYERS:

                with tf.variable_scope(op_name, reuse=tf.AUTO_REUSE):

                    """Need to reshape fc layers weights"""
                    if (op_name == 'fc6' or op_name == 'fc7'):
                        var = tf.get_variable('weights')
                        shape = var.get_shape();

                        weights = weights_dict[op_name][0]
                        weights = weights.reshape(shape)

                        init = tf.constant_initializer(value=weights, dtype=tf.float32)
                        weights = tf.get_variable(name="weights", initializer=init, shape=shape)
                        session.run(var.assign(weights))

                        var2 = tf.get_variable('biases')
                        session.run(var2.assign(weights_dict[op_name][1]))
                    else:
                        for data in weights_dict[op_name]:

                            if len(data.shape) == 1:
                                var = tf.get_variable('biases')
                                session.run(var.assign(data))

                            else:
                                var = tf.get_variable('weights')
                                session.run(var.assign(data))
