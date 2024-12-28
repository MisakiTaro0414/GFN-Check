import random
import numpy as np
from collections import defaultdict

class RLOracle:
    def __init__(self, abstract_state_fn, domains, epsilon=0.25, gamma=1.0, initial_val=0):
        self.abstract_state_fn = abstract_state_fn
        self.learners = {}
        self.choice_sequence = []
        self.epsilon = epsilon
        self.gamma = gamma
        self.initial_val = initial_val

        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = RLLearner(domain)

    def select(self, idx):
        abstract_state = self.abstract_state_fn(self.choice_sequence)
        if idx not in self.learners:
            self.learners[idx] = RLLearner(
                self.epsilon, self.gamma, self.initial_val)
        choice = self.learners[idx].policy(abstract_state)
        self.choice_sequence.append(choice)
        return choice

    # updates upon full episodes
    def reward(self, reward):
        for learner in self.learners.values():
            learner.reward(reward)
        self.choice_sequence = []


class RLLearner:
    def __init__(self, domain, epsilon=0.25, gamma=1.0, initial_val=0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.choice_state_sequence = []
        self.initial_val = initial_val
        self.Q_table = defaultdict(dict)
        self.C_table = defaultdict(dict)
        self.domain = domain

    # updates upon full episodes
    def reward(self, reward):
        # reward the last choice
        if len(self.choice_state_sequence) == 0:
            return
        self.choice_state_sequence[-1][2] = reward
        T = len(self.choice_state_sequence)
        G = 0
        W = 1
        for i in range(T):
            # ranges from T - 1, T - 2, ... 0.
            s, a, r = self.choice_state_sequence[T - i - 1]
            G = self.gamma * G + r
            self.C_table[s][a] = self.C(s, a) + W
            self.Q_table[s][a] = self.Q(
                s, a) + (W / self.C(s, a)) * (G - self.Q(s, a))
        self.choice_state_sequence = []

    def Q(self, s: str, a: str):
        try:
            return self.Q_table[s][a]
        except KeyError:
            return self.initial_val

    def C(self, s: str, a: str):
        try:
            return self.C_table[s][a]
        except KeyError:
            return self.initial_val

    def policy(self, state):
        domain = list(self.domain)
        # Epsilon-greedy strategy
        if np.random.binomial(1, self.epsilon):
            choice = random.choice(domain)
        else:
            self.action_values = np.array([self.Q(state, a) for a in domain])
            action_idx = random.choice(np.flatnonzero(
                self.action_values == self.action_values.max()))  # break ties randomly
            choice = domain[action_idx]
        self.choice_state_sequence.append([state, choice, 0])
        return choice
