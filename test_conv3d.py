import tensorflow as tf
import numpy as np
image_in_man = np.linspace(1, 96, 96).reshape(1, 3, 2, 4, 4)
# [batch, in_depth, in_channels, in_height, in_width]
image_in_tf = image_in_man.transpose(0, 1, 3, 4, 2)
# [batch, in_depth, in_height, in_width, in_channels].
# shape:[1,2,4,4,2]
weight_in_man = np.array(
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]).reshape(1, 2, 2, 3, 3)  # 1,3,4,2,0
weight_in_tf = weight_in_man.transpose(1, 3, 4, 2, 0)
# [filter_depth, filter_height, filter_width, in_channels,out_channels]
# shape: [2,3,3,2,1]
print(image_in_man)
print(weight_in_man)
x = tf.placeholder(dtype=tf.float32, shape=[1, 3, 4, 4, 2], name='x')
w = tf.placeholder(dtype=tf.float32, shape=[2, 3, 3, 2, 1], name='w')
conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
 r_in_tf = sess.run(conv, feed_dict={x: image_in_tf, w: weight_in_tf})
 # [batch, in_depth, in_height, in_width, in_channels].
 print(r_in_tf.shape)
 r_in_man = r_in_tf.transpose(0, 1, 4, 2, 3)
 # [batch, in_depth,in_channels,in_height, in_width].
 print(r_in_man)
