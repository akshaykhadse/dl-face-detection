
# coding: utf-8

# # Image Segmentation (Face Detection) using Deep Learning

# In[1]:

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import tensorflow as tf
from glob import glob
from time import time
from math import ceil


# ## Reading Data

# ### Load Label Data

# In[2]:

dataset_path = 'face_detection_dataset/'
positive_eg = 'positive_bw/'
negative_eg = 'negative_bw/'


def encode_label(path):
    # path_segments = path.split('/')
    if 'positive' in path:
        label = 1
    else:
        label = 0
    return [int(label)]


def read_label_dir(path):
    # print(path)
    filepaths = []
    labels = []
    # print(path + '*.png')
    # print(glob(path + '*.png'))
    for filepath in glob(path + '*.png'):
        filepaths.append(filepath)
        labels.append(encode_label(filepath))
    return filepaths, labels

# print(dataset_path + positive_eg)
# print(read_label_dir(dataset_path + positive_eg))
# print(read_label_dir(dataset_path + negative_eg))


# ### Start Building Pipeline

# In[3]:

pos_filepaths, pos_labels =    read_label_dir(dataset_path + positive_eg)
print('Positive Examples: %d' % len(pos_labels))
neg_filepaths, neg_labels =    read_label_dir(dataset_path + negative_eg)
print('Negative Examples: %d' % len(neg_labels))

# all_filepaths = pos_filepaths + neg_filepaths
# all_labels = pos_labels + neg_labels

# Convert string into tensors
pos_images = ops.convert_to_tensor(pos_filepaths, dtype=dtypes.string)
pos_labels = ops.convert_to_tensor(pos_labels, dtype=dtypes.int32)

neg_images = ops.convert_to_tensor(neg_filepaths, dtype=dtypes.string)
neg_labels = ops.convert_to_tensor(neg_labels, dtype=dtypes.int32)


# ### Partitioning Data

# In[4]:

test_set_size = 1200
pos_test_size = ceil(test_set_size / 6)
neg_test_size = test_set_size - pos_test_size

# Positive Examples
# Create a partition vector
pos_partitions = [0] * len(pos_filepaths)
# print(partitions)
pos_partitions[:pos_test_size] = [1] * pos_test_size
# print(partitions)
random.shuffle(pos_partitions)
# print(partitions)

# Partition data into a test and train set according to partition vector
pos_train_images, pos_test_images = tf.dynamic_partition(pos_images, pos_partitions, 2)
pos_train_labels, pos_test_labels = tf.dynamic_partition(pos_labels, pos_partitions, 2)

# Negative Examples
# Create a partition vector
neg_partitions = [0] * len(neg_filepaths)
# print(partitions)
neg_partitions[:neg_test_size] = [1] * neg_test_size
# print(partitions)
random.shuffle(neg_partitions)
# print(partitions)

# Partition data into a test and train set according to partition vector
neg_train_images, neg_test_images = tf.dynamic_partition(neg_images, neg_partitions, 2)
neg_train_labels, neg_test_labels = tf.dynamic_partition(neg_labels, neg_partitions, 2)


# ### Build the Input Queues and Define How to Load Images

# In[5]:

NUM_CHANNELS = 1

# Create input queues
pos_train_queue = tf.train.slice_input_producer(
                                    [pos_train_images, pos_train_labels],
                                    shuffle=False)
pos_test_queue = tf.train.slice_input_producer(
                                    [pos_test_images, pos_test_labels],
                                    shuffle=False)

# Process path and string tensor into an image and a label
pos_file_content = tf.read_file(pos_train_queue[0])
pos_train_image = tf.image.decode_png(pos_file_content, channels=NUM_CHANNELS)
pos_train_label = pos_train_queue[1]

pos_file_content = tf.read_file(pos_test_queue[0])
pos_test_image = tf.image.decode_png(pos_file_content, channels=NUM_CHANNELS)
pos_test_label = pos_test_queue[1]

# Create negative input queues
neg_train_queue = tf.train.slice_input_producer(
                                    [neg_train_images, neg_train_labels],
                                    shuffle=False)
neg_test_queue = tf.train.slice_input_producer(
                                    [neg_test_images, neg_test_labels],
                                    shuffle=False)

# Process path and string tensor into an image and a label
neg_file_content = tf.read_file(neg_train_queue[0])
neg_train_image = tf.image.decode_png(neg_file_content, channels=NUM_CHANNELS)
neg_train_label = neg_train_queue[1]

neg_file_content = tf.read_file(neg_test_queue[0])
neg_test_image = tf.image.decode_png(neg_file_content, channels=NUM_CHANNELS)
neg_test_label = neg_test_queue[1]


# ### Group Samples into Batches

# In[6]:

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 240
POS_BATCH_SIZE = ceil(BATCH_SIZE / 6)
NEG_BATCH_SIZE = BATCH_SIZE - POS_BATCH_SIZE

# Define tensor shape
pos_train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
pos_test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

neg_train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
neg_test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# Collect batches of images before processing
pos_train_image_batch, pos_train_label_batch = tf.train.batch(
                                    [pos_train_image, pos_train_label],
                                    batch_size=POS_BATCH_SIZE
                                    # ,num_threads=1
                                    )
pos_test_image_batch, pos_test_label_batch = tf.train.batch(
                                    [pos_test_image, pos_test_label],
                                    batch_size=POS_BATCH_SIZE
                                    # ,num_threads=1
                                    )

neg_train_image_batch, neg_train_label_batch = tf.train.batch(
                                    [neg_train_image, neg_train_label],
                                    batch_size=NEG_BATCH_SIZE
                                    # ,num_threads=1
                                    )
neg_test_image_batch, neg_test_label_batch = tf.train.batch(
                                    [neg_test_image, neg_test_label],
                                    batch_size=NEG_BATCH_SIZE
                                    # ,num_threads=1
                                    )

# Join the postive and negative batches
train_image_batch = tf.concat([pos_train_image_batch, neg_train_image_batch], 0)
train_label_batch = tf.concat([pos_train_label_batch, neg_train_label_batch], 0)
test_image_batch = tf.concat([pos_test_image_batch, neg_test_image_batch], 0)
test_label_batch = tf.concat([pos_test_label_batch, neg_test_label_batch], 0)


# ### Run the Queue Runners and Start a Session
# - **Note:** This section is meant for testing only. Do not run during main code.
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # print(sess.run(all_images))

    # Initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("From the train set:")
    for i in range(1):
        print(sess.run(train_label_batch))

    print("From the test set:")
    for i in range(1):
        print(sess.run(test_label_batch))

    # Stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()
# ## Neural Network Model

# ### Define Placeholders and Variables

# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([5, 5, 1]))
b = tf.Variable(tf.zeros([1]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ### Define Model

# In[ ]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 128, 128, 1])

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
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# ## Train and Evaluate

# In[ ]:

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),  tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,  tf.float32))

train_iterations = 2
test_iterations = 2

with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # Initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("Training")
    for i in range(train_iterations):
        start_time = time()
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
        end_time = time()
        print("Step %d, Training time %f" % (i, start_time - end_time))

    for i in range(test_iterations):
        print("test accuracy %g" % accuracy.eval(feed_dict={
              x: test_image_batch.eval(), y_: test_label_batch.eval(),
              keep_prob: 1.0}))

    # Stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()

