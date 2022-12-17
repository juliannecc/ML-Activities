from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

class Loader:
  def __init__(self, X, Y, batch_size=64, shuffle=True, polyfeatures=True, degree=4):
    self.X = X
    self.Y = Y
    self.batch_size = batch_size
    self.polyfeatures = polyfeatures
    self.degree = degree
    self.shuffle = shuffle

  def __iter__(self): 
    if self.shuffle: self.shuffle_data()
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    for i in range(0, len(self.X),self.batch_size):
      X_batch = self.X[i:min(i + self.batch_size, len(self.X))]
      Y_batch = self.Y[i:min(i + self.batch_size, len(self.X))]
      if self.polyfeatures: X_batch = self.get_feats(X_batch)
      yield X_batch, Y_batch

  def get_feats(self, x):
    # https://rickwierenga.com/blog/ml-fundamentals/polynomial-regression.html
    b = np.hstack((np.ones((len(x), 1)),x))
    feats = np.hstack([((b[:,1] ** i).reshape((len(x),1))) for i in range(self.degree+1)])
    # feats[:, 1:] = (feats[:, 1:] - np.mean(feats[:, 1:], axis=0)) / np.std(feats[:, 1:], axis=0)

    feats_tensor = Tensor(feats, requires_grad = False)
    return feats_tensor

  # https://stackoverflow.com/questions/32019398/python-sorting-y-value-array-according-to-ascending-x-array
  def unshuffle_data(self):
    L = sorted(zip(self.X,self.Y), key=itemgetter(0))
    self.X, self.Y = zip(*L)

  def shuffle_data(self):
    x_shuffled = []
    y_shuffled = []

    i = np.random.permutation(len(self.X))
    for idx in range(len(self.X)):
        x_shuffled.insert(i[idx],self.X[idx])
        y_shuffled.insert(i[idx],self.Y[idx])
    self.X = x_shuffled
    self.Y = y_shuffled

def plot_loaded(X_train, Y_train, X_validate, Y_validate, X_test, Y_test):
  fig, (ax1,ax2) = plt.subplots(1, 2)
  ax1.scatter(X_train, Y_train)
  ax1.set_title('train data')

  ax2.scatter(X_validate, Y_validate)
  ax2.set_title('validate data')

  plt.savefig("train-validate.png")
  plt.show()

  fig, (ax1,ax2) = plt.subplots(1, 2)
  ax1.scatter(X_test, Y_test)
  ax1.set_title('test data')

  ax2.scatter(X_test, Y_test)
  ax2.scatter(X_train, Y_train)
  ax2.set_title('train & test data')

  plt.savefig("test-train_and_test.png")
  plt.show()