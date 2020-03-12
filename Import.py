from pyts.datasets import load_gunpoint
import numpy as np
from numpy import array
from keras.layers import Dropout
from keras import backend as K
from keras.optimizers import Adam
import random
import keras
from collections import deque
import os