import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


import argparse
from loader import *
from model import *

data_train_default = "https://raw.githubusercontent.com/juliannecc/ML-Activities/main/02_Polynomial_Solver/data_train.csv"
data_test_default = "https://raw.githubusercontent.com/juliannecc/ML-Activities/main/02_Polynomial_Solver/data_test.csv"

parser = argparse.ArgumentParser(description="Settings")
parser.add_argument("-plot", "--plot_data", type=bool, default=True, help="Plots and saves data")
parser.add_argument("-e", "--epochs", type=int, default=200, help="Set Number of Epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Set Batch Size" )
parser.add_argument("-tr", "--train", type=str, default=data_train_default, help="Path to data_train.csv")
parser.add_argument("-te", "--test", type=str, default=data_test_default, help="Path to data_test.csv")
args = parser.parse_args()

def plot_prediction(train, test, X_train,Y_train,X_test, Y_test ):

  X_train_pred = np.linspace(min(X_train), max(X_train), len(X_train))
  Y_train_pred = best_model.forward_pass(train.get_feats(train.X)).reshape(-1, 1)
  plt.scatter(X_train,Y_train)
  plt.plot(X_train_pred, Y_train_pred.data, color='red')
  plt.title("train data vs prediction")
  plt.legend(['train data', 'prediction'])
  plt.savefig("train data vs prediction.png")
  plt.show()

  X_test_pred = np.linspace(min(X_test), max(X_test), len(X_test)).reshape(-1, 1)
  Y_test_pred = best_model.forward_pass(test.get_feats(X_test)).reshape(-1, 1)
  plt.scatter(X_test, Y_test)
  plt.plot(X_test_pred, Y_test_pred.data, color='red')
  plt.title("test data vs prediction")
  plt.legend(['test data', 'prediction'])
  plt.savefig("test data vs prediction.png")
  plt.show()

if __name__ == "__main__":
  data_train = pd.read_csv(args.train)
  data_test = pd.read_csv(args.test)

  X_train, X_validate, Y_train, Y_validate = train_test_split([[i] for i in data_train["x"]], [[i] for i in data_train["y"]], test_size=0.15, random_state=42)
  X_test, Y_test = [[i] for i in data_test["x"]], [[i] for i in data_test["y"]]

  if args.plot_data: plot_loaded(X_train, Y_train, X_validate, Y_validate, X_test, Y_test)

  epochs = args.epochs
  batch_size = args.batch_size
  learning_rate = [0.0003, 0.00003, 0.000003, 0.00000003, 0.0000000003]
  
  best_models = []
  losses = []

  for degree in range (4,0,-1):
    model = Model(degree)
    optimizer = optim.SGD(model.poly, lr=learning_rate[degree])
    train = Loader(X_train, Y_train, batch_size, True, True, degree)
    validate = Loader(X_validate, Y_validate, batch_size, False, True, degree)
    alpha = 0.5

    best = []
    bestloss = None

    for epoch in tqdm(range(epochs)):

      for x, y in train:
        out = model.forward_pass(x).reshape(-1,1)
        # https://www.youtube.com/watch?v=KpBmD3bLfX4
        l2_pen = alpha * model.reshaped_tensor()[1:].mul(model.reshaped_tensor()[1:]).sqrt().sum()
        loss = calc_mse(y, out) + l2_pen
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      validate_loss = 0.0
      for x, y in validate:
        out = model.forward_pass(x).reshape(-1,1)
        vloss = calc_mse(y, out).data[0]
        validate_loss += vloss*len(out.data[0])
      
      validate_loss /= len(train.X)

      if bestloss is None:
        bestloss = validate_loss
      else:
        if bestloss > validate_loss:
          bestloss = validate_loss
          best = [c[0] for c in model.get_polynomial()]

    best_models.append([c[0] for c in model.get_polynomial()])
    losses.append(bestloss)

  best_coeffs = best_models[losses.index(min(losses))]
  print(f"Best Coefficients: {best_coeffs}")

  best_model = Model(len(best_coeffs)-1)
  best_model.load_polynomial(np.array(best_coeffs).reshape(-1,1))

  train = Loader(X_train, Y_train, 64, False, True, len(best_coeffs)-1)
  train.unshuffle_data()

  test = Loader(X_test,Y_test,64, False, True, len(best_coeffs)-1)

  if args.plot_data: plot_prediction(train, test, X_train,Y_train,X_test, Y_test )

  Y_train_pred = best_model.forward_pass(train.get_feats(X_train))
  train_mse = metrics.mean_squared_error(Y_train, Y_train_pred.data)
  train_r2 = metrics.r2_score(Y_train, Y_train_pred.data)

  print(f"train r2: {train_r2}")
  # Testing data
  Y_test_pred = best_model.forward_pass(test.get_feats(X_test)).reshape(-1, 1)
  test_mse = metrics.mean_squared_error(Y_test, Y_test_pred.data)
  test_r2 = metrics.r2_score(Y_test, Y_test_pred.data)

  print(f"test r2: {test_r2}")
