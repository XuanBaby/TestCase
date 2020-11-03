"""Functional tests for depthwise convolutional operations."""
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function


import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
import time

def ConfigsTest():
  input_sizes = [[1,2,2,64]]
#,[1,3,3,128],[1,38,38,192],[1,38,38,192],[1,75,75,144],[1,75,75,144],[1,150,150,96],[1,150,150,32]]
  filter_sizes = [[3,3,64,1]]
#,[3,3,128,1],[3,3,192,1],[3,3,192,1],[3,3,144,1],[3,3,144,1],[3,3,96,1],[3,3,32,1]]
  out_sizes = [[1,1,1,64]]
#,[1,2,2,128],[1,19,19,192],[1,38,38,192],[1,38,38,144],[1,75,75,144],[1,75,75,96],[1,150,150,32]]
  strides = [2]
#,2,2,1,2,1,2,1]
  VALID = "VALID"
  SAME = "SAME"
  paddings = [VALID]
#,SAME,VALID,SAME,SAME,SAME,VALID,SAME]
 
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes,
                           strides,paddings):
    yield i, f, o, s, p


class DepthwiseConv2DTest(test.TestCase):
  def _BackpropFilter(self,input_sizes, filter_sizes, output_sizes, stride, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)
  
    def _GetVal(use_gpu):
      with self.cached_session(use_gpu=use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes,shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        duration = 0
        start = time.time()
        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = self.evaluate(backprop)
        end = time.time()
        duration += end - start
        print("when use device gpu = %s ,cost time = %fs"%(use_gpu,duration))
        self.assertShapeEqual(ret, backprop)
        return ret
    import pdb
    pdb.set_trace()
    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DFilterGrad(self):
    for index, (input_size, filter_size, output_size, stride, padding) in enumerate(ConfigsTest()):
      tf_logging.info("Testing testDepthwiseConv2DFilterGrad, %dth config: %r * %r, stride: %d, padding: %s",index, input_size, filter_size, stride, padding)
      self._BackpropFilter(input_size, filter_size, output_size, stride, padding)

if __name__ == "__main__":
  test.main()

