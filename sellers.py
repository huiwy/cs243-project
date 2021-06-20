import re
import numpy as np

class MultiSellerSimulator:
  def __init__(self, sellers, buyer, full_info, evaluations, probs, T):
    self.sellers = sellers
    self.buyer = buyer
    self.full_info = full_info
    self.evaluations = evaluations
    self.T = T
    self.probs = probs

    self.revenue = 0
    self.othrevenue = 0

  def start(self):
    for t in range(self.T):
      evaluation = np.random.choice(self.evaluations, p=self.probs)
      action = self.buyer.sample(evaluation)

      if action < self.sellers[0].N:
        p, q = self.sellers[0].pull(t, action)
        self.revenue += p*q
      else:
        p, q = self.sellers[1].pull(1)

      if not self.full_info:
        reward = q * (evaluation - p)
        self.buyer.update(reward)
      else:
        p, q = self.sellers[0].full_rewards(t)
        rewards = q * (evaluation - p)
        rewards = np.hstack([rewards, evaluation - self.sellers[1].p])

        self.buyer.update(rewards)        

class Simulator:
  def __init__(self, seller, buyer, full_info, evaluations, probs, T):
    self.seller = seller
    self.buyer = buyer
    self.full_info = full_info
    self.evaluations = evaluations
    self.T = T
    self.probs = probs

    self.revenue = 0

  def start(self):
    for t in range(self.T):
      evaluation = np.random.choice(self.evaluations, p=self.probs)
      action = self.buyer.sample(evaluation)

      p, q = self.seller.pull(t, action)

      self.revenue += q*p
      if not self.full_info:
        reward = q * (evaluation - p)
        self.buyer.update(reward)
      else:
        p, q = self.seller.full_rewards(t)
        rewards = q * (evaluation - p)

        self.buyer.update(rewards)

class PreDemo:
  def __init__(self, T):
    self.T = T
    self.N = 2
  def pull(self, t, b):
    if b == 0:
      return 0, 0
    else:
      if t < self.T/2:
        return 0, 1
      else:
        return 3/4, 1
  
  def full_rewards(self, t):
    if t < self.T/2:
      a, b = 0, 1
    else:
      a, b = 3/4, 1
    return np.array([0, a]), np.array([0, b])

class FixedPrice:
  def __init__(self, p):
    self.p = p

  def pull(self, b):
    return self.p, b

class TheoremB1:
  def __init__(self, C, eps, T):
    self.C = C
    self.eps = eps
    self.T = T
    # self.rho = min(1-eps/2, max(C))
    # self.delta = (1-self.rho)/(1-min(C))
    self.rho = 1-eps/2
    self.delta = eps/2
    
    self.N_emp = 0
    self.N_zero = 0
    self.N_one = 0

    self.N = int(np.log(eps/2)/np.log(1 - self.delta))+1

    self.choices = np.zeros(self.N)

    self.emp = (1 - (1-self.delta)**np.arange(self.N))*self.T
    self.zero = (1-self.delta)**np.arange(self.N)*(1-self.rho)*self.T + self.emp


  def pull(self, t, i):

    self.choices[i] += 1

    emp = (1 - (1 - self.delta)**(i))*self.T
    zero = (1 - self.delta)**(i)*(1-self.rho)*self.T
    if t < emp:
      self.N_emp += 1
      return 0, 0
    elif t < zero + emp:
      self.N_zero += 1
      return 0, 1
    else:
      self.N_one += 1
      return 1, 1

  def full_rewards(self, t):
    q = np.ones(self.N) * t >= self.emp
    p = np.ones(self.N) * t >= self.zero

    return p, q