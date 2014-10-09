import numpy as np
from numpy.random import random_sample

def softmax(x, t = 1.0):
    ex = np.exp(x/t)
    return ex/np.sum(ex)

class SoftMaxPolicyMixture(object):
    def __init__(self, policies, valuefns, t = 1.0):
        self.policies = policies
        self.valuefns = valuefns
        self.t = t
        self.indicies = np.arange(len(policies))
    def __call__(self, state):
        p = softmax([v(state) for v in self.valuefns], self.t)
        cump = np.cumsum(p)
        return self.policies[np.digitize(random_sample(1), cump)](state)

    
    