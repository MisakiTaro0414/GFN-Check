import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
import math
import random

losses = []

class GFNOracle_flow_matching(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, epsilon):
        super(GFNOracle_flow_matching, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.vocab = dict()
        self.prev_flow = torch.Tensor(0)
        self.prev_curr = []

        # Initialize vocabulary and learners
        vocab_idx = 1
        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = GFNLearner(self.hidden_dim, domain, self.epsilon)
            self.vocab[idx] = dict()
            for x in domain:
                self.vocab[idx][x] = vocab_idx
                vocab_idx += 1

        num_embeddings = 1 + sum(map(lambda d: len(d[0]), domains))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

        # Model parameters
        self.beta = 1
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1)
        self.transformer_pf = nn.TransformerEncoder(transformer_layer, num_layers=10)
        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        self.flow_matching_loss = torch.tensor(0.0)

        # Optimizers
        self.optimizer_policy = torch.optim.Adam(
            [
                {'params': self.embedding_layer.parameters()},
                {'params': self.transformer_pf.parameters()},
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},
            ],
            lr=0.001,  # Default learning rate for other parameters
        )

    def encode_choice_sequence(self):
        """Encodes the current choice sequence into embeddings."""
        return [0] + list(map(lambda x: self.vocab[x[0]][x[1]], self.choice_sequence))

    def select(self, learner_idx):
        """Selects an action for a given domain and index."""
        sequence_embeddings = self.embedding_layer(
            torch.tensor(self.encode_choice_sequence(), dtype=torch.long).unsqueeze(0)
        )
        hidden = self.transformer_pf(sequence_embeddings)
        hidden = hidden[:, 0, :]
        decision_idx, domain, flows = self.learners[learner_idx].policy(hidden)
        self.prev_curr.append((self.prev_flow, flows.sum()))
        self.prev_flow = flows[decision_idx]
        self.choice_sequence.append((learner_idx, domain[decision_idx]))
        return domain[decision_idx]

    def reward(self, reward):
        """Compute the flow matching loss based on rewards."""
        EPS = 1e-8
        reward = math.log(reward + EPS)
        loss = torch.tensor(0.0)
        for idx, (prev, curr) in enumerate(self.prev_curr):
            prev = torch.log(prev + EPS)
            curr = torch.log(curr + EPS)
            if idx == 0: # First step
                continue
            if idx == len(self.prev_curr) - 1: # Last step
                loss += (curr - reward * self.beta) ** 2
            else:
                loss += (prev - curr) ** 2
        self.flow_matching_loss += loss
        self.num_generation += 1
        if self.num_generation > 0 and self.num_generation % 10 == 0:
            self.optimizer_policy.zero_grad()
            self.flow_matching_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer_policy.step()
            self.flow_matching_loss = torch.tensor(0.0)
        # Reset choice sequence after updating
        self.choice_sequence = []
        self.prev_curr = []

class GFNLearner:
    def __init__(self, hidden_dim, domain, epsilon):
        self.epsilon = epsilon
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        output = self.action_selector(hidden)[0]
        output = torch.nan_to_num(output)
        flows = F.softplus(output)
        probs = F.softmax(flows, dim=-1)  # Convert to probabilities
        if random.random() < self.epsilon:
            sampled_index = random.randint(0, len(self.domain) - 1)
        else:
            sampled_index = torch.multinomial(probs, 1).item()
        return sampled_index, self.domain, flows
