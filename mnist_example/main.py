"""
Input Piplene to load your own data structures
==============================================

Author: Akshay Khadse
Date: 21/03/2017

Ref: http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
"""

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import tensorflow as tf

"""
Load Label Data
---------------
"""
dataset_path = 'mnist_data/'
test_labels_file = 'test/test.csv'
train_labels_file = 'train/train.csv'


def encode_label(label):
    label_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label_list[int(label)] = 1
    return label_list


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


# Reading labels and file path
train_filepaths, train_labels =\
    read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels =\
    read_label_file(dataset_path + test_labels_file)

"""
Optional Processing on string lists
-----------------------------------
"""
# Transform relative path into full path
train_filepaths = [dataset_path + 'train/' + fp for fp in train_filepaths]
test_filepaths = [dataset_path + 'test/' + fp for fp in test_filepaths]

# For this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

# print(all_filepaths)

"""
Start Building Pipeline
-----------------------
"""
# Convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)
"""
NOTE: ops.convert_to_tensor will create constants of your data in your graph.
Not Suitable for datasets with large datasets (i.e large individual images)
"""

"""
Partitioning Data
-----------------
"""
test_set_size = 10

# Create a partition vector
partitions = [0] * len(all_filepaths)
# print(partitions)
partitions[:test_set_size] = [1] * test_set_size
# print(partitions)
random.shuffle(partitions)
# print(partitions)

# Partition data into a test and train set according to partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

"""
Build the Input Queues and Define How to Load Images
----------------------------------------------------
"""
NUM_CHANNELS = 1

# Create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=False)

# Process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

"""
Group Samples into Batches
--------------------------
"""
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
BATCH_SIZE = 5

# Define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# Collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    # ,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    # ,num_threads=1
                                    )

"""
Run the Queue Runners and Start a Session
-----------------------------------------

with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    print(sess.run(all_images))
    # Initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("From the train set:")
    for i in range(20):
        print(sess.run(train_label_batch))

    print("From the test set:")
    for i in range(10):
        print(sess.run(test_label_batch))

    # Stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()
"""

"""
Define Placeholders and Variables
---------------------------------
"""

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([28, 28, 1]))
b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


"""
Define Model
------------
"""


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

"""
Train and Evaluate
------------------
"""
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),  tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,  tf.float32))

with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # Initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("Training")
    for i in range(200):
        feed_dict = {x: train_image_batch.eval(),
                     y_: train_label_batch.eval(),
                     keep_prob: 1.0}
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict)
            print("Step %d, Training accuracy %g" % (i, train_accuracy))
        feed_dict = {x: train_image_batch.eval(),
                     y_: train_label_batch.eval(),
                     keep_prob: 0.5}
        train_step.run(feed_dict)

    print("test accuracy %g" % accuracy.eval(feed_dict={
          x: test_image_batch.eval(), y_: test_label_batch.eval(),
          keep_prob: 1.0}))

    # Stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()
