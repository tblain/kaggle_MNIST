from __future__ import print_function

import glob
import math
import os

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

df_train = pd.read_csv(
  "/home/tblain/Documents/projet_perso/kaggle_MNIST/train.csv",
  sep=",",
  header=None)

df_train = df_train.reindex(np.random.permutation(df_train.index))

df_test = pd.read_csv(
  "/home/tblain/Documents/projet_perso/kaggle_MNIST/test.csv",
  sep=",",
  header=None)

df_test = df_test.reindex(np.random.permutation(df_test.index))
print(df_train[:,0])

def parse_labels_and_features(dataset):
  """Extracts labels and features.

  This is a good place to scale or transform the features if needed.

  Args:
    dataset: A Pandas `Dataframe`, containing the label on the first column and
      monochrome pixel values on the remaining columns, in row major order.
  Returns:
    A `tuple` `(labels, features)`:
      labels: A Pandas `Series`.
      features: A Pandas `DataFrame`.
  """
  labels = dataset[1:,0]

  # DataFrame.loc index ranges are inclusive at both ends.
  features = dataset.loc[:,1:784]
  # Scale the data to [0, 1] by dividing out the max value, 255.
  features = features / 255

  return labels, features

df_train.loc[:, 72:72]

train_targets, train_examples = parse_labels_and_features(df_train)
train_examples.describe()

test_targets, training_examples = parse_labels_and_features(df_test)
test_examples.describe()
