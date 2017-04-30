# Image Segmentation (Face Detection) using Deep Learning
# =======================================================
# Authors: Raghav Gupta, Akshay Khadse, Soumya Dutta, Ashish Sukhwani
# Date: 31/03/2017

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
from PIL import Image
import random
import tensorflow as tf
from glob import glob
from time import time
from math import ceil

# Reading Data
# ------------

# Load Label Data
test_path = 'test_image/'

def read_label_dir(path):
    filepaths = []
    for filepath in glob(path + '*.png'):
        filepaths.append(filepath)
    return filepaths


# Start Building Pipeline

filepaths = read_label_dir(test_path)
print('Test Examples: %d' % len(filepaths))

# Convert string into tensors
images = ops.convert_to_tensor(filepaths, dtype=dtypes.string)

# Build the Input Queues and Define How to Load Images

NUM_CHANNELS = 1

# Create input queues
test_queue = tf.train.slice_input_producer([images],
                                           shuffle=False)

# Process path and string tensor into an image and a label
file_content = tf.read_file(test_queue[0])
test_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)

#Added
test_image = tf.image.resize_images(test_image, [128, 128])

# Group Samples into Batches

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = len(filepaths)

# Define tensor shape
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# Collect batches of images before processing
test_image_batch = tf.train.batch([test_image],
                                  batch_size=BATCH_SIZE
                                  # ,num_threads=1
                                  )

# Neural Network Model
# --------------------

# Define Placeholders and Variables

x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# Define Model

def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# Input
x_image = tf.reshape(x, [-1, 128, 128, 1])

# Weights of CNN & Layers

W_conv1 = weight_variable([18, 18, 1, 60])
b_conv1 = bias_variable([60])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool_a = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([12, 12, 60, 30])
b_conv2 = bias_variable([30])

h_conv2 = tf.nn.relu(conv2d(h_pool_a, W_conv2) + b_conv2)
h_pool_b = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([6, 6, 30, 15])
b_conv3 = bias_variable([15])

h_conv3 = tf.nn.relu(conv2d(h_pool_b, W_conv3) + b_conv3)
h_pool_c = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([16 * 16 * 15, 4096])
b_fc1 = bias_variable([4096])

h_pool_c_flat = tf.reshape(h_pool_c, [-1, 16 * 16 * 15])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_c_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([4096, 256])
b_fc2 = bias_variable([256])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([256, 1])
b_fc3 = bias_variable([1])

y_conv = tf.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
y_out = tf.round(y_conv)
# Train and Evaluate
# ------------------

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    # Restore variables from disk.
    saver.restore(sess, "./saved_model_bak/model_0.942857_1181.ckpt")
    print("Model restored.")

    # Initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    size_box = 32
    print("Testing")
    print('xyz ' + str(sess.run(tf.shape(test_image_batch))))
    print('xyz ' + str(sess.run(test_image_batch)))
    #print('x: ' + str(sess.run(tf.shape(x), feed_dict = {x: test_image_batch.eval(), keep_prob: 1.0})))
    #print('x: ' + str(sess.run(x, feed_dict = {x: test_image_batch.eval(), keep_prob: 1.0})))
    #print(sess.run(y_conv, feed_dict = {x: test_image_batch.eval(), keep_prob: 1.0}))
    #print(tf.round(y_conv))
    #print(sess.run(tf.shape(y_conv), feed_dict = {x: test_image_batch.eval(), keep_prob: 1}))
    face_file = open('faces.txt', 'w')
    for i in range(0, 128-size_box, 8):
        for j in range(0, 128-size_box, 8):
            s_out = ''
            #if(i < 384 & j < 384):
            #test_image = tf.slice(test_image_batch, [0, i, j, 0], [1, 128, 128, 1])
            test_image1 = tf.image.crop_to_bounding_box(test_image, j, i, size_box, size_box)
            test_image1 = tf.image.resize_images(test_image1, [128, 128])
            test_image2 = tf.expand_dims(test_image1, 0)
            y_res = sess.run(y_conv, feed_dict = {x: test_image2.eval(), keep_prob: 1.0})
            print("#_" + str(y_res))
            if ((y_res) > 0.5):
                #boxes = [float(j), float(i), float((j+31)/512), float((i+31)/512)]
                boxes = tf.constant([1, 1, float(j), float(i), float((j+31)/512), float((i+31)/512)])
                s_out = str(j) + "," + str(i) + "," + str(float((j+size_box-1))) + "," + str(float((i+size_box-1))) + "," + str(y_res)
                face_file.write(s_out + "\n")
                print('#'+str(type(boxes)))
                #l = [boxes, y_conv]
                #faces.append(l)
                print("#_" + str(i) + str(j))
                #boxes = [0.1, 0.2, 0.5, 0.9]
                #xyz = tf.image.draw_bounding_boxes(tf.expand_dims(test_image, boxes, name=None)
                #print('#'+str(type(xyz)))
                #out = xyz.eval(session = sess)
                #print("#_" + str(xyz.shape))
    # Stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()
