from Classes.learners import Learner,TS_Learner,UCB1_Learner
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

n_experiments = 1000

ts_rewards_per_experiments = []
ucb1_rewards_per_experiments = []

opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
opt = normEarnings[opt_index][0]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]


for e in tqdm(range(n_experiments)):
  env = env_array[0]
  ts_learner = TS_Learner(n_arms = n_prices)
  ucb1_learner = UCB1_Learner(n_arms = n_prices)
  for t in range(0, T):
    pulled_arm = ts_learner.pull_arm()
    reward = env.round(pulled_arm)
    ts_learner.update(pulled_arm, reward)
    
    pulled_arm = ucb1_learner.pull_arm()
    reward = env.round(pulled_arm) 
    ucb1_learner.update(pulled_arm, reward)

  ts_rewards_per_experiments.append(ts_learner.collected_rewards)
  ucb1_rewards_per_experiments.append(ucb1_learner.collected_rewards)

ts_rewards_per_experiments = np.array(ts_rewards_per_experiments)
ucb1_rewards_per_experiments = np.array(ucb1_rewards_per_experiments)

opt_reward = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

ts_rewards_per_experiments = ts_rewards_per_experiments * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)
ucb1_rewards_per_experiments = ucb1_rewards_per_experiments * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

fig, axs = plt.subplots(2,2,figsize=(14,7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(opt_reward - ts_rewards_per_experiments, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - ucb1_rewards_per_experiments, axis = 0)), 'y') 
axs[0][0].plot(np.std(np.cumsum(opt_reward - ts_rewards_per_experiments, axis = 1), axis=0), 'b')   
axs[0][0].plot(np.std(np.cumsum(opt_reward - ucb1_rewards_per_experiments, axis = 1), axis=0), 'c')
axs[0][0].legend(["Avg TS", "Avg UCB1","Std TS","Std UCB1"])
axs[0][0].set_title("Cumulative Regret")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.cumsum(np.mean(ts_rewards_per_experiments, axis = 0)), 'r')
axs[0][1].plot(np.cumsum(np.mean(ucb1_rewards_per_experiments, axis = 0)), 'm')
axs[0][1].plot(np.std(np.cumsum(ts_rewards_per_experiments, axis = 1), axis = 0), 'b')
axs[0][1].plot(np.std(np.cumsum(ucb1_rewards_per_experiments, axis = 1), axis = 0), 'c')
axs[0][1].legend(["Avg TS", "Avg UCB1","Std TS","Std UCB1"])
axs[0][1].set_title("Cumulative Reward")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(opt_reward - ts_rewards_per_experiments, axis = 0), 'g')
axs[1][0].plot(np.mean(opt_reward - ucb1_rewards_per_experiments, axis = 0), 'y')
axs[1][0].plot(np.std(opt_reward - ts_rewards_per_experiments, axis = 0), 'b')   
axs[1][0].plot(np.std(opt_reward - ucb1_rewards_per_experiments, axis = 0), 'c')
axs[1][0].legend(["Avg TS", "Avg UCB1","Std TS","Std UCB1"])
axs[1][0].set_title("Instantaneous Regret")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.mean(ts_rewards_per_experiments, axis = 0), 'g')
axs[1][1].plot(np.mean(ucb1_rewards_per_experiments, axis = 0), 'y')
axs[1][1].plot(np.std(ts_rewards_per_experiments, axis = 0), 'b')
axs[1][1].plot(np.std(ucb1_rewards_per_experiments, axis = 0), 'c')
axs[1][1].legend(["Avg TS", "Avg UCB1","Std TS","Std UCB1"])
axs[1][1].set_title("Instantaneous Reward")

fig.suptitle('Comparison between TS and UCB1 for learning the optimal price (Single Class-Stationary Environment)\n(Optimal bid for advertising is known)', fontsize=16)
plt.subplots_adjust(hspace=0.33)
plt.show()
