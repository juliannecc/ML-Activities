from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

import numpy as np

class Model:
  def __init__(self, pol_deg):
    self.poly = [Tensor([np.array([np.random.uniform(-1,1)])], requires_grad=True) for i in range(pol_deg+1)]

  def load_polynomial(self, polynomial):
    self.poly = [Tensor([np.array(coeff)], requires_grad = True) for coeff in polynomial]

  def reshaped_tensor(self):
    reshaped_tensor = self.poly[0]
    [reshaped_tensor := reshaped_tensor.cat(self.poly[i]) for i in range(1,len(self.poly))]
    return reshaped_tensor

  def forward_pass(self,x):
    return x.matmul(self.reshaped_tensor()).sum(axis=1)  

  def get_polynomial(self):
    return [i.data[0] for i in self.poly]

def calc_mse(y_pred,y_gt):
  return (y_gt-y_pred).square().mean()