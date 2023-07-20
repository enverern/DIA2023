from Classes.learners import Learner,GPTS_Learner,GPUCB_Learner
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm.autonotebook import tqdm


n_prices = 5
n_bids = 100
cost_of_product = 180
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price*np.array([1,2,3,4,5])
margins = np.array([prices[i]-cost_of_product for i in range(n_prices)])
classes = np.array([0,1,2])
                              #C1   C2   C3
conversion_rate = np.array([[0.19,0.21,0.16], #1*price
                            [0.16,0.17,0.09], #2*price
                            [0.15,0.13,0.06], #3*price
                            [0.03,0.08,0.04], #4*price
                            [0.02,0.04,0.06]  #5*price
                            ])

earnings = np.zeros([5,3]) # conv_rate * margin
for row in range(5):
  earnings[row,:] = conversion_rate[row,:] * margins[row]

normEarnings = earnings.copy()
normEarnings = normEarnings - np.min(normEarnings)
normEarnings = normEarnings / np.max(normEarnings)

env_array = []
for c in classes:
  env_array.append(Environment(n_prices, normEarnings[:,c], c))

#EXPERIMENT BEGIN
T = 365

n_experiments = 100
noise_std = 1

gpts_reward = []
gpucb_reward = []

opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
opt = normEarnings[opt_index][0]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]

opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
opt = normEarnings[opt_index][0] #0 bc only first Class
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]

gpts_y_values = [[] for i in range(T)]
gpts_sigmas = [[] for i in range(T)]

gpucb_y_values = [[] for i in range(T)]
gpucb_sigmas = [[] for i in range(T)]

for e in range(n_experiments):
  print(e)
  env = env_array[0]
  gpts_learner = GPTS_Learner(n_arms = n_bids, arms = bids)
  gpucb_learner = GPUCB_Learner(n_arms = n_bids, arms = bids)

  for t in tqdm(range(T)):
    pulled_arm, y_value, sigmas = gpts_learner.pull_arm()
    gpts_y_values[t].append(y_value)
    gpts_sigmas[t].append(sigmas)
    reward = env.draw_n(bids[pulled_arm],noise_std) * opt - env.draw_cc(bids[pulled_arm],noise_std) # 1 is std
    gpts_learner.update(pulled_arm, reward)

    pulled_arm, y_value, sigmas = gpucb_learner.pull_arm()
    gpucb_y_values[t].append(y_value)
    gpucb_sigmas[t].append(sigmas)
    reward = env.draw_n(bids[pulled_arm],noise_std) * opt - env.draw_cc(bids[pulled_arm],noise_std)# 1 is std
    gpucb_learner.update(pulled_arm, reward)

  gpts_reward.append(gpts_learner.collected_rewards)
  gpucb_reward.append(gpucb_learner.collected_rewards)

gpts_reward = np.array(gpts_reward)
gpucb_reward = np.array(gpucb_reward)

opt_reward = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

for i in range(T):
  if i % 13 == 0:
    gpts_y_values[i] = np.mean(gpts_y_values[i], axis = 0)
    gpts_sigmas[i] = np.mean(gpts_sigmas[i], axis = 0)
    gpucb_y_values[i] = np.mean(gpucb_y_values[i], axis = 0)
    gpucb_sigmas[i] = np.mean(gpucb_sigmas[i], axis = 0)

    x_pred = np.atleast_2d(bids).T
    plt.figure(int(i/13)+1)
    plt.plot(x_pred, opt*env_array[0].n(x_pred)-env_array[0].cc(x_pred), 'r', label=r'$Reward_Curve$')
    plt.plot(x_pred, gpts_y_values[i], 'b', label=u'GPTS predicted curve')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
            np.concatenate([gpts_y_values[i] - 1.9600 * gpts_sigmas[i],
                            (gpts_y_values[i] + 1.9600 * gpts_sigmas[i])[::-1]]),
            alpha=.5, fc='c', ec='None', label='95% confidence interval')
    plt.plot(x_pred, gpucb_y_values[i], 'g', label=u'GP-UCB predicted curve')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
            np.concatenate([gpucb_y_values[i] - 1.9600 * gpucb_sigmas[i],
                            (gpucb_y_values[i] + 1.9600 * gpucb_sigmas[i])[::-1]]),
            alpha=.5, fc='y', ec='None', label='95% confidence interval')
    plt.legend()
    plt.show()
  

fig, axs = plt.subplots(2,2, figsize = (14,7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpts_reward, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpucb_reward, axis = 0)), 'y') 
axs[0][0].plot(np.std(np.cumsum(opt_reward - gpts_reward, axis = 1), axis=0), 'b')   
axs[0][0].plot(np.std(np.cumsum(opt_reward - gpucb_reward, axis = 1), axis=0), 'c')
axs[0][0].legend(["Avg GPTS", "Avg GPUCB1","Std GPTS","Std GPUCB1"])
axs[0][0].set_title("Cumulative Regret")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.cumsum(np.mean(gpts_reward, axis = 0)), 'r')
axs[0][1].plot(np.cumsum(np.mean(gpucb_reward, axis = 0)), 'm')
axs[0][1].plot(np.std(np.cumsum(gpts_reward, axis = 1), axis=0), 'b')
axs[0][1].plot(np.std(np.cumsum(gpucb_reward, axis = 1), axis=0), 'c')
axs[0][1].legend(["Avg GPTS", "Avg GPUCB1","Std GPTS","Std GPUCB1"])
axs[0][1].set_title("Cumulative Reward")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(opt_reward - gpts_reward, axis = 0), 'g')
axs[1][0].plot(np.mean(opt_reward - gpucb_reward, axis = 0), 'y')
axs[1][0].plot(np.std(opt_reward - gpts_reward, axis = 0), 'b')
axs[1][0].plot(np.std(opt_reward - gpucb_reward, axis = 0), 'c')
axs[1][0].legend(["Avg GPTS", "Avg GPUCB1","Std GPTS","Std GPUCB1"])
axs[1][0].set_title("Instantaneous Regret")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.mean(gpts_reward, axis = 0), 'r')
axs[1][1].plot(np.mean(gpucb_reward, axis = 0), 'm')
axs[1][1].plot(np.std(gpts_reward, axis = 0), 'b')
axs[1][1].plot(np.std(gpucb_reward, axis = 0), 'c')
axs[1][1].legend(["Avg GPTS", "Avg GPUCB1","Std GPTS","Std GPUCB1"])
axs[1][1].set_title("Instantaneous Reward")

fig.suptitle('Comparison between GP-TS and GP-UCB1 for learning the optimal advertising strategy(Single Class-Stationary Environment)\n(Conversion rates of prices are known)', fontsize=16)
plt.subplots_adjust(hspace=0.33)
plt.show()
