import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
#from tensorflow.python import debug as tf_debug
#import sys
#sys.path.append('/data-xuan/libpython.py')
#import libpython
#import os
#input("pid: " + str(os.getpid()) +", press enter to continue")
import numpy as np
import time
input_size = [1,32,150,150]
filter_size = [3,3,32,1]
output_size = [1,32,150,150]

input0 = np.random.rand(*input_size).astype(np.float32)
output0 = np.random.rand(*output_size).astype(np.float32)

input1 = constant_op.constant(input0,shape=input_size)
filter1 = constant_op.constant(filter_size, shape=[len(filter_size)])
output1 = constant_op.constant(output0,shape=output_size)

#x=tf.nn.conv2d_backprop_filter(inputs,filter_sizes=[3,3,64,1],out_backprop=[1,1,1,64],strides=[2,2],padding='VALID',use_cudnn_on_gpu=False,data_format='NCHW')
x1=tf.nn.depthwise_conv2d_native_backprop_filter(input1,filter1,output1,strides=[1,1,1,1],padding='SAME',data_format='NCHW')

#sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
config = tf.ConfigProto()
config.gpu_options.force_gpu_compatible = True
with tf.Session(config=config) as sess:
    with tf.device('gpu:0'):
        start = time.time()
        print(sess.run(x1))
        end = time.time()
        print("GPU compute dw_conv cost time: ",end-start)

