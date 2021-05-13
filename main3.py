# Import mlcompute module to use the optional set_mlc_device API for device selection with ML
# Compute.
from tensorflow.python.compiler.mlcompute import mlcompute

# Select Any device.
mlcompute.set_mlc_device(device_name='any')  # Available options are 'cpu', 'gpu', and 'any'.

# mlcompute.set_mlc_device(device_name='gpu')
print("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
print("is_tf_compiled_with_apple_mlc %s" % mlcompute.is_tf_compiled_with_apple_mlc())

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections

from tensorflow import keras
from tensorflow.keras import layers

# tf.compat.v1.disable_eager_execution()

print(f"eagerly? {tf.executing_eagerly()}")
print(tf.config.list_logical_devices())

from datetime import datetime

print(tf.__version__)

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs