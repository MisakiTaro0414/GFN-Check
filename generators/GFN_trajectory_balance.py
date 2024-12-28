import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import itertools
import math
import random

losses = []


class GFNOracle_trajectory_balance(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, epsilon):
        super(GFNOracle_trajectory_balance, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        self.hidden_dim = embedding_dim
        hidden_dim = embedding_dim
        # 1 for embedding for empty sequence, and the other is total vocabulary size
        vocab_idx = 1
        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = GFNLearner(hidden_dim, domain, epsilon)
            self.vocab[idx] = dict()
            for x in domain:
                self.vocab[idx][x] = vocab_idx
                vocab_idx += 1
        num_embeddings = 1 + sum(map(lambda d: len(d[0]), domains))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.beta = 1
        self.logZ = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.logZ_lower = 10

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1)

        self.transformer_pf = nn.TransformerEncoder(
            transformer_layer, num_layers=10)

        self.logPf = torch.tensor(0.0)
        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        self.optimizer_policy = torch.optim.Adam(
            [
                # Lower learning rate for embedding layer
                {'params': self.embedding_layer.parameters()},
                # Default learning rate for transformer_pf
                {'params': self.transformer_pf.parameters()},
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},  # Default learning rate for action selectors
            ],
            lr=0.001,  # This will act as the default learning rate if not specified explicitly
        )
        self.optimizer_logZ = torch.optim.Adam(
            [{'params': [self.logZ], 'lr': 1}],
        )

    def encode_choice_sequence(self):
        return [0] + list(map(lambda x: self.vocab[x[0]][x[1]], self.choice_sequence))

    def select(self, idx):
        sequence_embeddings = self.embedding_layer(
            torch.tensor(self.encode_choice_sequence(),
                         dtype=torch.long).unsqueeze(0)
        )
        hidden = self.transformer_pf(sequence_embeddings)
        hidden = hidden[:, 0, :]
        # Select action based on the hidden state
        choice, log_prob, probs = self.learners[idx].policy(hidden)
        self.choice_sequence.append((idx, choice, log_prob))
        self.logPf = self.logPf + log_prob
        return choice

    def reward(self, reward):
        loss = (self.logPf + self.logZ -
                torch.log(torch.Tensor([reward])) * self.beta) ** 2
        losses.append(loss.item())
        self.loss = self.loss + loss
        self.num_generation += 1
        if self.num_generation > 0 and self.num_generation % 1 == 0:
            self.optimizer_policy.zero_grad()
            self.optimizer_logZ.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer_policy.step()
            self.optimizer_logZ.step()
            self.loss = torch.tensor(0.0)

        # Reset choice sequence after updating
        self.choice_sequence = []
        self.logPf = torch.tensor(0.0)


class GFNLearner:
    def __init__(self, hidden_dim, domain, epsilon):
        self.exploration_prob = 1
        self.min_exploration_prob = 0.2
        self.domain = domain
        self.epsilon = epsilon
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        if self.exploration_prob > self.min_exploration_prob:
            self.exploration_prob *= 0.9995
        output = self.action_selector(hidden)
        output = torch.nan_to_num(output)
        probs = F.softmax(output, dim=-1)  # Convert to probabilities
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            sampled_index = random.choice(range(len(self.domain)))
        else:
            sampled_index = torch.multinomial(probs, 1).item()
        return self.domain[sampled_index], torch.log(probs[0][sampled_index]), probs
