from trainer.vgg16_fcn import VGG16_FCN
from trainer.datagenerator import ImageDataGenerator

import tensorflow as tf
from tensorflow.contrib.data import Iterator
from datetime import datetime
import os
import logging
import subprocess

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('main_dir', 'main', 'Main Directory.')

log_file = "{}.log".format(datetime.now().strftime("%d-%m-%Y"))
logging.basicConfig(filename = log_file, format='%(levelname)s (%(asctime)s): %(message)s', level = logging.INFO)

num_epochs = 150
NUM_CLASSES = 22
learning_rate = 1e-6
batch_size = 1

filewriter_path = os.path.join(FLAGS.main_dir,"vgg_fcn/tensorboard")
checkpoint_path = os.path.join(FLAGS.main_dir,"vgg_fcn/checkpoints")

train_file = 'train.txt'

subprocess.check_call(['gsutil', '-m' , 'cp', '-r', os.path.join(FLAGS.main_dir, "vgg_fcn", train_file), '/tmp'])

train_file = os.path.join('/tmp',train_file)

""" Initialize datagenerator """
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 shuffle=True)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name="input_image")
y = tf.placeholder(tf.int32, shape=[batch_size, None, None, 1], name="input_label")
keep_prob = tf.placeholder(tf.float32)

"""Build Model"""
model = VGG16_FCN(x, NUM_CLASSES, keep_prob)
logging.info("Model built")


"""Define loss function"""
with tf.name_scope("loss"):
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.squeeze(y, axis=3), 
                                                                          logits = model.upscore8, 
                                                                          name="loss")))

"""Define training op"""
with tf.name_scope("train_op"):
    
    trainable_var = tf.trainable_variables()
    
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grads = optimizer.compute_gradients(loss, var_list=trainable_var)
    
    train_op = optimizer.apply_gradients(grads)
    
   
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize saver for store model checkpoints
saver = tf.train.Saver()
    
""" Summaries """
# Add gradients to summary
for gradient, var in grads:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in trainable_var:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

"""Start Tensorflow session"""
logging.info("Session started")

try:
    with tf.Session() as sess:
     
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
      
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logging.info("Model restored")
        else:
            model.load_initial_weights(sess,os.path.join(FLAGS.main_dir,"vgg_fcn/vgg16.npy"))
            logging.info("Npy file loaded")
      
        print("[TENSORBOARD] => Open Tensorboard at --logdir {}".format(filewriter_path))
        logging.info("Training started")
    
        for epoch in range(num_epochs):
            
            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            logging.info("Epoch: {}".format(epoch+1))
            
                
            for step in range(tr_data.data_size):
                
                batch_xs, batch_ys = sess.run(next_batch)            
                
                if ((step + 1) % 250 == 0):
                    t_loss = sess.run(loss, feed_dict={x: batch_xs, 
                                                       y: batch_ys, 
                                                       keep_prob: 0.5})
        
                    s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
        
                    writer.add_summary(s, epoch*tr_data.data_size + step)
                    
                    logging.info("Step {} of {} | Loss: {}".format(step, tr_data.data_size,t_loss))
                
                
                sess.run(train_op, feed_dict={x: batch_xs, 
                                              y: batch_ys, 
                                              keep_prob: 0.5})
            
            if((epoch+1) % 15 == 0): 
                logging.info("Saving Model")
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(datetime.now())+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)  
                
        logging.info("Training finished")
    
        checkpoint_name = os.path.join(checkpoint_path, 'final_model'+str(datetime.now())+'.ckpt')
        save_path = saver.save(sess, checkpoint_name) 

except Exception as e:
    print("[ERROR] => Time: {} Unexpected error encountered. Please check the log file.".format(datetime.now()))
    logging.error("Error message: {}".format(e))
    logging.info("Terminating...".format(e))