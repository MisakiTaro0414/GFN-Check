import random


class RandomOracle:
    def __init__(self, domains):
        self.learners = {}

        idx = 1
        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = domain

    def select(self, idx):
        return random.choice(self.learners[idx])

    def reward(self, r):
        return None
