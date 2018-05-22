import scipy.misc as misc
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from vgg16_fcn import VGG16_FCN
from colormap import color_map

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

NUM_CLASSES = 22

checkpoint_path = "checkpoints"

x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input_image")
keep_prob = tf.placeholder(tf.float32)

"""Build Model"""
model = VGG16_FCN(x, NUM_CLASSES, keep_prob)

image_path = "cat1.jpg"
image_path = convert_to_tensor(image_path, dtype=dtypes.string)

img_string = tf.read_file(image_path)

img_decoded = tf.image.decode_jpeg(img_string, channels=3)

img_decoded = tf.to_float(img_decoded)

img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)

img_bgr = img_centered[:, :, ::-1]

image = tf.expand_dims(img_bgr, 0)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    image = sess.run(image)
    
    label_pred = sess.run(model.pred_up, feed_dict={x: image, keep_prob: 1.0})
    
img = label_pred[0]

"""
cmap = color_map(21)
    
result = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        ok = False
        
        for index in range(len(cmap)):
            if np.array_equal(label[i,j],cmap[index]):
                result[i,j] = index
                ok = True
                
        if not ok:
            result[i,j] = 21
    """
#img = np.asarray(label_pred[0], dtype=np.uint8)

#np.savetxt('pred_up.txt', img)

misc.imsave('fcn8_upsampled.png', label_pred[0])