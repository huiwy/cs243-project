import numpy as np


class EXP3:
    def __init__(self, eps, C, K, full_info):
        """
        partial informaiton

        C: list of contexts
        K: number of bandits
        """
        self.eps = eps
        self.context2idx = {c: i for c, i in zip(C, range(len(C)))}
        self.weight = np.ones([len(C), K])
        self.K = K
        self.full_info = full_info

    def sample(self, context):
        idx = self.context2idx[context]
        denorm = np.sum(self.weight[idx])
        probs = (1 - self.K * self.eps) * self.weight[idx] / denorm + self.eps
        sampled = np.random.choice(self.K, p=probs)

        self.prev_probs = probs
        self.prev_action = sampled
        self.prev_idx = idx
        return sampled

    def update(self, reward):
        # require the reward of the played bandit
        if not self.full_info:
            u = np.exp(self.eps * reward / self.prev_probs[self.prev_action])
            self.weight[self.prev_idx, self.prev_action] *= u
        else:
            u = np.exp(self.eps * reward / self.prev_probs)
            self.weight[self.prev_idx] *= u


class UCB:
    def __init__(self, c, C, K, full_info) -> None:
        self.c = c
        self.context2idx = {c: i for c, i in zip(C, range(len(C)))}
        self.K = K
        self.Q = np.ones([len(C), K])
        self.rewards = np.zeros([len(C), K])

        self.full_info = full_info
        self.t = 1

    def sample(self, context):
        idx = self.context2idx[context]
        r = self.rewards[idx] / self.Q[idx]
        r += np.sqrt(2 * self.c * np.log(self.t) / (self.Q[idx]))
        sampled = np.argmax(r)

        self.prev_action = sampled
        self.prev_idx = idx
        return sampled

    def update(self, reward):
        if not self.full_info:
            self.rewards[self.prev_idx, self.prev_action] += reward
            self.Q[self.prev_idx, self.prev_action] += 1
            self.t += 1
        else:
            self.rewards[self.prev_idx] += reward
            self.Q[self.prev_idx] += 1
            self.t += self.K


class FTPL:
    # follow the perturbed leader
    def __init__(self, C, K, T):
        """
        full information

        C: list of contexts
        K: number of bandits
        T: time horizon
        """
        self.eps = np.sqrt(np.log(K) / T)
        self.context2idx = {c: i for c, i in zip(C, range(len(C)))}
        self.weight = np.ones([len(C), K])
        self.K = K

    def sample(self, context):
        per = np.random.exponential(self.eps, self.K)
        idx = self.context2idx[context]
        return np.argmax(per + self.weight[idx])

    def update(self, rewards):
        # require the reward of all bandits
        self.weight += rewards


class MultiplicativeWeights:
    def __init__(self, C, K, T):
        """
        full information

        C: list of contexts
        K: number of bandits
        T: time horizon
        """
        self.eps = np.sqrt(np.log(K) / T)
        self.context2idx = {c: i for c, i in zip(C, range(len(C)))}
        self.weight = np.ones([len(C), K])
        self.K = K

    def sample(self, context):
        idx = self.context2idx[context]
        denorm = np.sum(self.weight[idx])
        probs = self.weight[idx] / denorm
        sampled = np.random.choice(self.K, p=probs)

        self.prev_probs = probs
        self.prev_action = sampled
        self.prev_idx = idx
        return sampled

    def update(self, rewards):
        # require the reward of all bandits
        self.weight[self.prev_idx] *= \
            np.exp(self.eps * rewards)
