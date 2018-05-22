import tensorflow as tf
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator
import numpy as np
from vgg16_fcn import VGG16_FCN
import matplotlib.pyplot as plt
import itertools

def confusion_matrix_label(result_label, label, num_classes=22):
    confusion = np.zeros([num_classes,num_classes], dtype=np.uint32)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            confusion[result_label[i,j],label[i,j]] = confusion[result_label[i,j],label[i,j]] + 1
    
    return confusion

def confusion_matrix(val_file, num_classes, checkpoint_path):
    
    with tf.device('/cpu:0'):
        val_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      shuffle=True)
    
        iterator = Iterator.from_structure(val_data.data.output_types,
                                           val_data.data.output_shapes)
        next_batch = iterator.get_next()
        
    val_init_op = iterator.make_initializer(val_data.data)
    
    
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input_image")
    keep_prob = tf.placeholder(tf.float32)
    model = VGG16_FCN(x, num_classes, keep_prob)
    
    cm = np.zeros([num_classes,num_classes], dtype=np.uint32)
    
    with tf.Session() as sess:
     
        sess.run(tf.global_variables_initializer())
        
        sess.run(val_init_op)
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        for step in range(val_data.data_size):
        
            image, label = sess.run(next_batch)
        
            result_label = sess.run(model.pred_up, feed_dict={x: image, keep_prob: 1.0})
        
            cml = confusion_matrix_label(result_label[0], label[0], num_classes)
            
            cm += cml
            
    return cm

def plot_confusion_matrix(cm, target_names, save_plot = False):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    if(save_plot):
        plt.savefig('confusion_matrix.png',format='png')
    else:
        plt.show()
