from Classes.learners import Learner,TS_Learner,UCB1_Learner,SWTS_Learner,SWUCB_Learner,CUSUM_UCB_Learner
from Classes.enviroment import Non_Stationary_Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

n_prices = 5
n_bids = 100
cost_of_product = 180
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price*np.array([1,2,3,4,5])
margins = np.array([prices[i]-cost_of_product for i in range(n_prices)])
classes = np.array([0,1,2])
                                    # normal economic conditions
                                      #C1   C2   C3
conversion_rate_phase1 =  np.array([[0.19,0.21,0.16], #1*price
                                    [0.16,0.17,0.09], #2*price
                                    [0.15,0.13,0.06], #3*price
                                    [0.03,0.08,0.04], #4*price
                                    [0.02,0.04,0.06]  #5*price
                                    ])

                                    # bad economic conditions
                                      #C1   C2   C3
conversion_rate_phase2 =  np.array([[0.24,0.20,0.16],  # 1*price
                                    [0.48,0.18,0.10],  # 2*price
                                    [0.03,0.10,0.08],  # 3*price
                                    [0.02,0.03,0.07],  # 4*price
                                    [0.01,0.02,0.05]   # 5*price
                                    ])

                                    # good economic conditions
                                      #C1   C2   C3
conversion_rate_phase3 =  np.array([[0.20,0.17,0.22],  # 1*price
                                    [0.07,0.15,0.15],  # 2*price
                                    [0.09,0.10,0.13],  # 3*price
                                    [0.11,0.11,0.12],  # 4*price
                                    [0.11,0.12,0.10]   # 5*price
                                    ])

earnings_phase1 = np.zeros([5,3]) # conv_rate * margin
earnings_phase2 = np.zeros([5,3]) # conv_rate * margin
earnings_phase3 = np.zeros([5,3]) # conv_rate * margin

for row in range(5):
  earnings_phase1[row,:] = conversion_rate_phase1[row,:] * margins[row]
  earnings_phase2[row,:] = conversion_rate_phase2[row,:] * margins[row]
  earnings_phase3[row,:] = conversion_rate_phase3[row,:] * margins[row]

normEarnings_phase1 = earnings_phase1.copy()
normEarnings_phase1 = normEarnings_phase1 - np.min(normEarnings_phase1)
normEarnings_phase1 = normEarnings_phase1 / np.max(normEarnings_phase1)

normEarnings_phase2 = earnings_phase2.copy()
normEarnings_phase2 = normEarnings_phase2 - np.min(normEarnings_phase2)
normEarnings_phase2 = normEarnings_phase2 / np.max(normEarnings_phase2)

normEarnings_phase3 = earnings_phase3.copy()
normEarnings_phase3 = normEarnings_phase3 - np.min(normEarnings_phase3)
normEarnings_phase3 = normEarnings_phase3 / np.max(normEarnings_phase3)

env_array = []
T = 365
for c in classes:
  env_array.append(Non_Stationary_Environment(n_prices, np.array([normEarnings_phase1[:,c], normEarnings_phase2[:,c], normEarnings_phase3[:,c]]), c, T, 0))

#EXPERIMENT BEGIN


n_experiments = 1000

M = 100 #number of steps to obtain reference point in change detection (for CUSUM)
eps = 0.1 #epsilon for deviation from reference point in change detection (for CUSUM)
h = np.log(T)*2 #threshold for change detection (for CUSUM)

ucb1_rewards_per_experiments = []
swucb_rewards_per_experiments = []
cusum_rewards_per_experiments = []

opt_index_phase1 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[0][0])
opt_phase1 = normEarnings_phase1[opt_index_phase1][0]
optimal_bid_index_phase1 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)]

opt_index_phase2 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)[0][0])
opt_phase2 = normEarnings_phase2[opt_index_phase2][0]
optimal_bid_index_phase2 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)[1][0]
optimal_bid_phase2 = bids[int(optimal_bid_index_phase2)]

opt_index_phase3 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)[0][0])
opt_phase3 = normEarnings_phase3[opt_index_phase3][0]
optimal_bid_index_phase3 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)[1][0]
optimal_bid_phase3 = bids[int(optimal_bid_index_phase3)]

for e in tqdm(range(n_experiments)):
  env_swucb = deepcopy(env_array[0])
  env_cusum = deepcopy(env_array[0])
  env_ucb1 = deepcopy(env_array[0])

  swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = int(T/3))
  cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M, eps = eps, h = h)
  ucb1_learner = UCB1_Learner(n_arms = n_prices)
  for t in range(0, T):

    pulled_arm = swucb_learner.pull_arm()
    reward = env_swucb.round(pulled_arm)
    swucb_learner.update(pulled_arm, reward)

    pulled_arm = cusum_learner.pull_arm()
    reward = env_cusum.round(pulled_arm)
    cusum_learner.update(pulled_arm, reward)

    pulled_arm = ucb1_learner.pull_arm()
    reward = env_ucb1.round(pulled_arm)
    ucb1_learner.update(pulled_arm, reward)

  swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)
  cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)
  ucb1_rewards_per_experiments.append(ucb1_learner.collected_rewards)

swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)
cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)
ucb1_rewards_per_experiments = np.array(ucb1_rewards_per_experiments)


opt_phase1 = opt_phase1 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase2 = opt_phase2 * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
opt_phase3 = opt_phase3 * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)
opt_reward = np.ones([T]) 
opt_reward[:int(T/3)] = opt_reward[:int(T/3)] * opt_phase1 
opt_reward[int(T/3):2*int(T/3)] = opt_reward[int(T/3):2*int(T/3)]* opt_phase2 
opt_reward[2*int(T/3):] = opt_reward[2*int(T/3):] * opt_phase3

swucb_rewards_per_experiments[:int(T/3)] = swucb_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
swucb_rewards_per_experiments[int(T/3):2*int(T/3)] = swucb_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
swucb_rewards_per_experiments[2*int(T/3):] = swucb_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

ucb1_rewards_per_experiments[:int(T/3)] = ucb1_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
ucb1_rewards_per_experiments[int(T/3):2*int(T/3)] = ucb1_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
ucb1_rewards_per_experiments[2*int(T/3):] = ucb1_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

fig, axs = plt.subplots(2,2,figsize=(14,7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(opt_reward - swucb_rewards_per_experiments, axis = 0)), 'tab:blue')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - cusum_rewards_per_experiments, axis = 0)), 'tab:cyan')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - ucb1_rewards_per_experiments, axis = 0)), 'tab:red')
axs[0][0].plot(np.std(np.cumsum(opt_reward - swucb_rewards_per_experiments, axis = 1), axis=0), 'tab:orange')
axs[0][0].plot(np.std(np.cumsum(opt_reward - cusum_rewards_per_experiments, axis = 1), axis=0), 'tab:purple')
axs[0][0].plot(np.std(np.cumsum(opt_reward - ucb1_rewards_per_experiments, axis = 1), axis=0), 'tab:green')
axs[0][0].legend(["Avg SWUCB", "Avg CUSUM", "Avg UCB1", "Std SWUCB", "Std CUSUM", "Std UCB1"])
axs[0][0].set_title("Cumulative Regret")


axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.cumsum(np.mean(swucb_rewards_per_experiments, axis = 0)), 'tab:blue')
axs[0][1].plot(np.cumsum(np.mean(cusum_rewards_per_experiments, axis = 0)), 'tab:cyan')
axs[0][1].plot(np.cumsum(np.mean(ucb1_rewards_per_experiments, axis = 0)), 'tab:red')
axs[0][1].plot(np.std(np.cumsum(swucb_rewards_per_experiments, axis=1), axis=0),'tab:orange')
axs[0][1].plot(np.std(np.cumsum(cusum_rewards_per_experiments, axis=1), axis=0),'tab:purple')
axs[0][1].plot(np.std(np.cumsum(ucb1_rewards_per_experiments, axis=1), axis=0),'tab:green')
axs[0][1].legend(["Avg SWUCB", "Avg CUSUM", "Avg UCB1", "Std SWUCB", "Std CUSUM", "Std UCB1"])
axs[0][1].set_title("Cumulative Reward")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(opt_reward - swucb_rewards_per_experiments, axis = 0), 'tab:blue')
axs[1][0].plot(np.mean(opt_reward - cusum_rewards_per_experiments, axis = 0), 'tab:cyan')
axs[1][0].plot(np.mean(opt_reward - ucb1_rewards_per_experiments, axis = 0), 'tab:red')
axs[1][0].plot(np.std(opt_reward - swucb_rewards_per_experiments, axis=0),'tab:orange')
axs[1][0].plot(np.std(opt_reward - cusum_rewards_per_experiments, axis=0),'tab:purple')
axs[1][0].plot(np.std(opt_reward - ucb1_rewards_per_experiments, axis=0),'tab:green')
axs[1][0].legend(["Avg SWUCB", "Avg CUSUM", "Avg UCB1", "Std SWUCB", "Std CUSUM", "Std UCB1"])
axs[1][0].set_title("Instantaneous Regret")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.mean(swucb_rewards_per_experiments, axis = 0), 'tab:blue')
axs[1][1].plot(np.mean(cusum_rewards_per_experiments, axis = 0), 'tab:cyan')
axs[1][1].plot(np.mean(ucb1_rewards_per_experiments, axis = 0), 'tab:red')
axs[1][1].plot(np.std(swucb_rewards_per_experiments, axis=0),'tab:orange')
axs[1][1].plot(np.std(cusum_rewards_per_experiments, axis=0),'tab:purple')
axs[1][1].plot(np.std(ucb1_rewards_per_experiments, axis=0),'tab:green')
axs[1][1].legend(["Avg SWUCB", "Avg CUSUM", "Avg UCB1", "Std SWUCB", "Std CUSUM", "Std UCB1"])
axs[1][1].set_title("Instantaneous Reward")

fig.suptitle('Comparison between SWUCB and CD-CUSUM and UCB1 for learning the optimal price \n(Single Class - 2 abrupt change in Nonstationary Environment)-(Optimal bid for advertising is known)', fontsize=16)
plt.subplots_adjust(hspace=0.33,top=0.86)
plt.show()
