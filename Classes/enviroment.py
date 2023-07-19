import numpy as np
import random

class Environment(): 
  def __init__(self, n_arms, probabilities, id_class): #probabilities is the probability distribution of the arm rewards
    self.n_arms = n_arms
    self.probabilities = probabilities
    self.id_class = id_class

  def n(self, x):
    k=5
    if self.id_class == 0:
      return k * (1-np.exp(-4*x+2*x**3))
    elif self.id_class == 1:
      return k * (1-np.exp(-2*x+2*x**3))
    elif self.id_class == 2:
      return k * (1-np.exp(-3*x+2*x**3))
    else:
      return -1

  def cc(self, x):
    k=2
    if self.id_class == 0:
      return k* (1-np.exp(-3*x+2*x**2))
    elif self.id_class == 1:
      return k* (1-np.exp(-2*x+2*x**2))
    elif self.id_class == 2:
      return k* (1-np.exp(-3*x+2*x**2))
    else:
      return -1

  def draw_n(self, x, noise_std):
    return self.n(x) + np.random.normal(0, noise_std, size = self.n(x).shape)

  def draw_cc(self, x, noise_std):
    return self.cc(x) + np.random.normal(0, noise_std, size = self.cc(x).shape)

  def round(self, pulled_arm): #given an arm, return a reward
    reward = np.random.binomial(1, self.probabilities[pulled_arm])
    return reward
  

  def reward(self, conv_rate, bid, margin):
    return(self.n(bid)*conv_rate*margin - self.cc(bid))
  

class Non_Stationary_Environment(Environment):
  def __init__(self, n_arms, probabilities, id_class, horizon, high_frequency_change):
    super().__init__(n_arms, probabilities, id_class)
    self.time = 0
    self.high_frequency_change = high_frequency_change
    self.n_phases = len(self.probabilities)
    if not(self.high_frequency_change):
      self.phases_size = int(horizon/self.n_phases)
    else:
      self.phases_size = int(horizon/(self.n_phases * 4))


  def round(self, pulled_arm):
    if not (self.high_frequency_change):
      current_phase = int(self.time/self.phases_size)  if int(self.time/self.phases_size) < self.n_phases else self.n_phases - 1
    else:
      current_phase = int(self.time/self.phases_size)%5
    p = self.probabilities[current_phase][pulled_arm]
    reward = np.random.binomial(1, p)
    self.time += 1
    return reward