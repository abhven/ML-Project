from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Weights
weights = {
    # 5x5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])), #[filter_depth, filter_height, filter_width, in_channels, out_channels]
    # 3x3x3 conv, 32 inputs, 32 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 32, 32])),
    # 8*8*8*32 inputs, 128 outputs
    'wfc1': tf.Variable(tf.random_normal([8*8*8*32, 128])),
    # 128 inputs, 10 outputs (class prediction)
    'wfc2': tf.Variable(tf.random_normal([128, 10])),
}

# building the network
# size of x: [batch, in_depth, in_height, in_width, in_channels]: [None,32,32,32,1]
def conv3d(x, W, ss):
    # Conv3D wrapper, with leaky relu activation
    x = tf.nn.conv3d(x, W, strides=[1, ss, ss, ss, 1], padding='SAME')
    return tf.maximum(x, 0.1 * x)

# Network flow
x = tf.Variable(tf.random_normal([1, 32, 32, 32, 1])) #temporaray input
#two Conv layers
conv1 = conv3d(x, weights['wc1'], 2) 
print(conv1)
conv2 = conv3d(conv1, weights['wc2'], 1)
print(conv2)

# Pool layer
pool = tf.nn.max_pool3d(conv2, [1,2,2,2,1], [1,2,2,2,1],  padding='SAME')
print(pool)

#two fully connected layers
fc1 = tf.reshape(pool, [-1, weights['wfc1'].get_shape().as_list()[0]])
fc1 = tf.matmul(fc1, weights['wfc1'])
fc1 = tf.nn.softmax(fc1)
fc2 = tf.nn.relu(fc1)
print(fc2)

