from Classes.learners import Learner,TS_Learner,UCB1_Learner,GPTS_Learner,GPUCB_Learner,TS_Learner_combined,UCB1_Learner_combined, GPTS_Learner_combined, GPUCB_Learner_combined
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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


opt_index = int(clairvoyant(classes,bids,prices,margins,conversion_rate,env_array)[0][0])
opt = normEarnings[opt_index][0] #0 bc only first Class
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]


#EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE 
T = 365

n_experiments = 100
noise_std = 1

gpts_reward = []
gpucb_reward = []

for e in tqdm(range(n_experiments)):
    env = env_array[0]
    ts_learner = TS_Learner_combined(n_arms = n_prices)
    gpts_learner = GPTS_Learner_combined(n_arms = n_bids, arms = bids)
    ucb1_learner = UCB1_Learner_combined(n_arms = n_prices)
    gpucb_learner = GPUCB_Learner_combined(n_arms = n_bids, arms = bids)

    for t in range(0, T):
      pulled_arm_price_idx, pulled_arm_price_val = ts_learner.pull_arm()
      pulled_arm_bid = gpts_learner.pull_arm(pulled_arm_price_val)
      reward_pricing = env.round(pulled_arm_price_idx)
      reward_numc = env.draw_n(bids[pulled_arm_bid],noise_std)
      reward_costc = env.draw_cc(bids[pulled_arm_bid],noise_std)
      reward_tot = reward_numc * reward_pricing  - reward_costc
      ts_learner.update(pulled_arm_price_idx, reward_pricing)
      gpts_learner.update(pulled_arm_bid, reward = [reward_numc,reward_costc], reward_total = reward_tot)

      pulled_arm_price_idx, pulled_arm_price_val = ucb1_learner.pull_arm()
      pulled_arm_bid = gpucb_learner.pull_arm(pulled_arm_price_val)
      reward_pricing = env.round(pulled_arm_price_idx)
      reward_numc = env.draw_n(bids[pulled_arm_bid],noise_std)
      reward_costc = env.draw_cc(bids[pulled_arm_bid],noise_std)
      reward_tot = reward_numc * reward_pricing  - reward_costc
      ucb1_learner.update(pulled_arm_price_idx, reward_pricing)

    gpts_reward.append(gpts_learner.collected_rewards_total)
    gpucb_reward.append(gpucb_learner.collected_rewards_total)

gpts_reward = np.array(gpts_reward)
gpucb_reward = np.array(gpucb_reward)

opt_reward = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

fig, axs = plt.subplots(2,2,figsize=(14,7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpts_reward, axis = 0)), 'tab:blue')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpucb_reward, axis = 0)), 'tab:cyan')
axs[0][0].plot(np.std(np.cumsum(opt_reward - gpts_reward, axis=1), axis=0),'tab:orange')
axs[0][0].plot(np.std(np.cumsum(opt_reward - gpucb_reward, axis=1), axis=0),'tab:purple')
axs[0][0].legend(["Avg gpts","Avg gpucb","Std gpts","Std gpucb"])
axs[0][0].set_title("Cumulative Regret")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.cumsum(np.mean(gpts_reward, axis = 0)), 'tab:blue')
axs[0][1].plot(np.cumsum(np.mean(gpucb_reward, axis = 0)), 'tab:cyan')
axs[0][1].plot(np.std(np.cumsum(gpts_reward, axis=1), axis=0),'tab:orange')
axs[0][1].plot(np.std(np.cumsum(gpucb_reward, axis=1), axis=0),'tab:purple')
axs[0][1].legend(["Avg gpts","Avg gpucb","Std gpts","Std gpucb"])
axs[0][1].set_title("Cumulative Reward")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(opt_reward - gpts_reward, axis = 0), 'tab:blue')
axs[1][0].plot(np.mean(opt_reward - gpucb_reward, axis = 0), 'tab:cyan')
axs[1][0].plot(np.std(opt_reward - gpts_reward, axis=0),'tab:orange')
axs[1][0].plot(np.std(opt_reward - gpucb_reward, axis=0),'tab:purple')
axs[1][0].legend(["Avg gpts", "Avg gpucb","Std gpts","Std gpucb"])
axs[1][0].set_title("Instantaneous Regret")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.mean(gpts_reward, axis = 0), 'tab:blue')
axs[1][1].plot(np.mean(gpucb_reward, axis = 0), 'tab:cyan')
axs[1][1].plot(np.std(gpts_reward, axis=0),'tab:orange')
axs[1][1].plot(np.std(gpucb_reward, axis=0),'tab:purple')
axs[1][1].legend(["Avg gpts", "Avg gpucb","Std gpts","Std gpucb"])
axs[1][1].set_title("Instantaneous Reward")

fig.suptitle('Comparison between GP-TS and GP-UCB1 for learning the optimal pricing and advertising strategy simultaneously\n(Single Class-Stationary Environment)', fontsize=16)
plt.subplots_adjust(hspace=0.33)
plt.show()
