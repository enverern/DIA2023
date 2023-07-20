import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math

import warnings
from sklearn.exceptions import ConvergenceWarning

from random import choices
warnings.filterwarnings('ignore', category=ConvergenceWarning)

  

class Learner():
  def __init__(self, n_arms):
    self.n_arms = n_arms
    self.t = 0
    self.reward_per_arm = [[] for i in range(n_arms)]
    self.collected_rewards = np.array([])

  def update_observations(self, pulled_arm, reward): #update the observation list once the reward is returned
    self.reward_per_arm[pulled_arm].append(reward)
    self.collected_rewards = np.append(self.collected_rewards, reward)

class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t -1) + reward) / self.t

class TS_Learner(Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
    self.beta_parameters = np.ones((n_arms,2))

  def pull_arm(self):
    idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
    return idx

  def update(self, pulled_arm, reward):
    self.t +=1
    self.update_observations(pulled_arm, reward)
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

class UCB1_Learner(Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
    self.emprical_means = np.zeros(n_arms)
    self.confidence = np.array([np.inf]*n_arms)

  def pull_arm(self):
    upper_conf = self.emprical_means + self.confidence
    return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

  def update(self, pulled_arm, reward):
    self.t += 1
    self.emprical_means[pulled_arm] = (self.emprical_means[pulled_arm]*(self.t-1)+reward)/self.t
    for a in range(self.n_arms):
      n_samples = len(self.reward_per_arm[a])
      self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
    self.update_observations(pulled_arm, reward)
  
class TS_Learner_combined(TS_Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
  
  def pull_arm(self):
    values = np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])
    return np.argmax(values), np.max(values)

class UCB1_Learner_combined(UCB1_Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
  
  def pull_arm(self):
    upper_conf = self.emprical_means + self.confidence
    idxs = np.where(upper_conf == upper_conf.max())[0]
    idx = np.random.choice(idxs)
    return idx, upper_conf[idx]

class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.reward_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.beta_parameters[arm,0] = cum_rew + 1
            self.beta_parameters[arm,1] = n_samples - cum_rew + 1

class SWUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.reward_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.emprical_means[arm] = cum_rew / n_samples if n_samples > 0 else 0
            self.confidence[arm] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf

class GPTS_Learner(Learner):
  def __init__(self, n_arms, arms):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*10
    self.pulled_arms = []
    self.alpha = 1
    self.kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
    self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.iteration = 0
    self.refitFrequency = 13 #13 bc it divides 364

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if self.iteration % self.refitFrequency == 0:
      self.gp.fit(x,y)
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.update_model()

  def pull_arm(self):
    sampled_values = np.random.normal(self.means, self.sigmas)
    return np.argmax(sampled_values), self.means, self.sigmas

class GPTS_Learner_combined(GPTS_Learner):
  def __init__(self, n_arms, arms):
    super().__init__(n_arms, arms)
    self.collected_reward_numc = np.array([])
    self.collected_reward_costc = np.array([])
    self.means_numc = np.zeros(n_arms)
    self.sigmas_numc = np.ones(n_arms)*10
    self.means_costc = np.zeros(n_arms)
    self.sigmas_costc = np.ones(n_arms)*10
    self.gp_numc = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.gp_costc = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.collected_rewards_total = np.array([])

  def update_observations(self, pulled_arm, reward): #update the observation list once the reward is returned
    self.reward_per_arm[pulled_arm].append(reward)
    self.collected_reward_numc = np.append(self.collected_reward_numc, reward[0])
    self.collected_reward_costc = np.append(self.collected_reward_costc, reward[1])
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y_numc = self.collected_reward_numc
    y_costc = self.collected_reward_costc
    if self.iteration % self.refitFrequency == 0:
      self.gp_numc.fit(x,y_numc)
      self.gp_costc.fit(x,y_costc)
    self.means_numc, self.sigmas_numc = self.gp_numc.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.means_costc, self.sigmas_costc = self.gp_costc.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.sigmas_numc = np.maximum(self.sigmas_numc, 1e-2)
    self.sigmas_costc = np.maximum(self.sigmas_costc, 1e-2)
  
  def update(self, pulled_arm, reward, reward_total):
    self.t += 1
    self.collected_rewards_total = np.append(self.collected_rewards_total, reward_total)
    self.update_observations(pulled_arm, reward)
    self.update_model()
  
  def pull_arm(self, predicted_best_conversion_rate):
    pred_y = self.means_numc * predicted_best_conversion_rate - self.means_costc
    pred_sigma = np.sqrt(self.sigmas_numc**2 * predicted_best_conversion_rate**2 + self.sigmas_costc**2)
    sampled_numc = np.random.normal(self.means_numc, self.sigmas_numc)
    sampled_costc = np.random.normal(self.means_costc, self.sigmas_costc)
    sampled_values = sampled_numc * predicted_best_conversion_rate - sampled_costc
    return np.argmax(sampled_values), pred_y, pred_sigma


class GPUCB_Learner(Learner): 
  def __init__(self, n_arms, arms):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*np.inf
    self.pulled_arms = []
    self.alpha = 1
    self.kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
    self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.iteration = 0
    self.refitFrequency = 13 #13 bc it divides 364

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])

  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if self.iteration % self.refitFrequency == 0:
      self.gp.fit(x,y)
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.means = np.array(self.means)
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.update_model()

  def pull_arm(self):
    upper_conf = self.means + 1.96 * self.sigmas
    return np.random.choice(np.where(upper_conf == upper_conf.max())[0]), self.means, self.sigmas


class GPUCB_Learner_combined(GPUCB_Learner):
  def __init__(self, n_arms, arms):
    super().__init__(n_arms, arms)
    self.collected_reward_numc = np.array([])
    self.collected_reward_costc = np.array([])
    self.means_numc = np.zeros(n_arms)
    self.sigmas_numc = np.ones(n_arms)*10
    self.means_costc = np.zeros(n_arms)
    self.sigmas_costc = np.ones(n_arms)*10
    self.gp_numc = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.gp_costc = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2)
    self.collected_rewards_total = np.array([])
  
  def update_observations(self, pulled_arm, reward): #update the observation list once the reward is returned
    self.reward_per_arm[pulled_arm].append(reward)
    self.collected_reward_numc = np.append(self.collected_reward_numc, reward[0])
    self.collected_reward_costc = np.append(self.collected_reward_costc, reward[1])
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y_numc = self.collected_reward_numc
    y_costc = self.collected_reward_costc
    if self.iteration % self.refitFrequency == 0:
      self.gp_numc.fit(x,y_numc)
      self.gp_costc.fit(x,y_costc)
    self.means_numc, self.sigmas_numc = self.gp_numc.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.means_costc, self.sigmas_costc = self.gp_costc.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.sigmas_numc = np.maximum(self.sigmas_numc, 1e-2)
    self.sigmas_costc = np.maximum(self.sigmas_costc, 1e-2)

  def update(self, pulled_arm, reward, reward_total):
    self.t += 1
    self.collected_rewards_total = np.append(self.collected_rewards_total, reward_total)
    self.update_observations(pulled_arm, reward)
    self.update_model()

  def pull_arm(self, predicted_best_conversion_rate):
    pred_y = self.means_numc * predicted_best_conversion_rate - self.means_costc
    pred_sigma = np.sqrt(self.sigmas_numc**2 * predicted_best_conversion_rate**2 + self.sigmas_costc**2)
    upper_conf_numc = self.means_numc + 1.96 * self.sigmas_numc
    lower_conf_costc = self.means_costc - 1.96 * self.sigmas_costc
    upper_conf = upper_conf_numc * predicted_best_conversion_rate - lower_conf_costc
    return np.random.choice(np.where(upper_conf == upper_conf.max())[0]), pred_y, pred_sigma


class CUSUM:
  def __init__(self, M, eps, h):
      self.M = M
      self.eps = eps
      self.h = h
      self.t = 0
      self.reference = 0
      self.g_plus = 0
      self.g_minus = 0

  def update(self, sample):
      self.t += 1
      if self.t <= self.M:
          self.reference += sample/self.M
          return 0
      else:
          s_plus = (sample - self.reference) - self.eps
          s_minus = -(sample - self.reference) - self.eps
          self.g_plus = max(0, self.g_plus + s_plus)
          self.g_minus = max(0, self.g_minus + s_minus)
          return self.g_plus > self.h or self.g_minus > self.h
      
  def reset(self): 
      self.t = 0
      self.g_minus = 0
      self.g_plus = 0
      self.reference = 0


class CUSUM_UCB_Learner(UCB1_Learner):
  def __init__(self, n_arms, M=100, eps=0.05, h=20, alpha=0.01):
    super().__init__(n_arms)
    self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
    self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
    self.detections =  [[] for _ in range(n_arms)]
    self.alpha = alpha
  
  def pull_arm(self):
    if np.random.binomial(1, 1-self.alpha):
        upper_conf = self.emprical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
    else:
        return np.random.randint(0, self.n_arms)
  
  def update(self, pulled_arm, reward):
    self.t += 1
    if self.change_detection[pulled_arm].update(reward):
        self.detections[pulled_arm].append(self.t)
        self.valid_rewards_per_arm[pulled_arm] = []
        self.change_detection[pulled_arm].reset()
    self.update_observations(pulled_arm, reward)
    self.emprical_means[pulled_arm] = np.mean(self.valid_rewards_per_arm[pulled_arm])
    total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arm])
    for a in range(self.n_arms):
        n_samples = len(self.valid_rewards_per_arm[a])
        self.confidence[a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples > 0 else np.inf

  def update_observations(self, pulled_arm, reward):
    self.reward_per_arm[pulled_arm].append(reward)
    self.valid_rewards_per_arm[pulled_arm].append(reward)
    self.collected_rewards = np.append(self.collected_rewards, reward)

class EXP3_Learner(Learner):
  def __init__(self, n_arms, gamma):
    super().__init__(n_arms)
    self.gamma = gamma
    self.weights = [1.0] * n_arms
    self.arm_index = [i for i in range(n_arms)]
    self.probabilityDistribution = []

  def distr(self):
    theSum = float(sum(self.weights))
    return tuple((1.0 - self.gamma) * (w / theSum) + (self.gamma / len(self.weights)) for w in self.weights)

  def pull_arm(self):
    self.probabilityDistribution = self.distr()
    idx = choices(self.arm_index, self.probabilityDistribution)
    return idx[0]   

  def update(self, pulled_arm, reward):
    self.t +=1
    self.update_observations(pulled_arm, reward)
    estimatedReward = 1.0 * reward / self.probabilityDistribution[pulled_arm]
    self.weights[pulled_arm] *= math.exp(estimatedReward * self.gamma / self.n_arms)
