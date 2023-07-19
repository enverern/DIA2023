import numpy as np

def clairvoyant(classes, bids, prices, margins, conversion_rate, env_array):
    maxPricesIndices = np.array([])
    for c in classes:
        revenue = np.zeros(margins.shape[0])
        for i in range(margins.shape[0]):
            revenue[i] = margins[i] * conversion_rate[i, c]
        maxPricesIndices = np.append(maxPricesIndices, np.argmax(revenue))
    maxPrices = prices[maxPricesIndices.astype(int)]

    bestBidsIndices = np.array([])
    final_rewards = []
    for c in classes:
      rewards = np.array([])
      for bid in bids:
        p = int(maxPricesIndices[c])
        single_reward = env_array[c].n(bid)*conversion_rate[p,c]*margins[p] - env_array[c].cc(bid)
        rewards = np.append(rewards,single_reward)
      bestBidsIndices = np.append(bestBidsIndices, np.argmax(rewards))
      final_rewards.append(np.max(rewards))
    bestBids = bids[bestBidsIndices.astype(int)]
    return (maxPricesIndices,bestBidsIndices,final_rewards)
