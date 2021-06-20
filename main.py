from sellers import *
from bidders import *

import numpy as np

# evaluations = np.array([0.25, 0.4, 0.6, 0.98, 1])
# probs = np.array([0.5, 0.2, 0.2, 0.05, 0.05])

evaluations = np.array([0.25, 0.5, 0.99])
probs = np.array([0.5, 0.4, 0.1])

evaluations = np.array([0.25, 0.59])
probs = np.array([0.5, 0.5])

r = 0
t = 1000
for i in range(100):

  seller = TheoremB1(evaluations, 0.2, t)
  # bidder = EXP3(0.001, evaluations, seller.N, True)
  bidder = UCB(1, evaluations, seller.N, True)
  simu = Simulator(seller, bidder, True, evaluations, probs, t)
  simu.start()
  r += simu.revenue

print(r / 100)