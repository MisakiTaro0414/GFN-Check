import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
import math
import random

losses = []


class GFNOracle_detailed_balance(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, epsilon):
        super(GFNOracle_detailed_balance, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        self.curr = []
        self.epsilon = epsilon

        # Initialize vocabulary and learners
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

        # Model parameters
        self.beta = 1
        self.lstm_pf = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.hidden_dim, batch_first=True)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1)
        self.transformer_pf = nn.TransformerEncoder(transformer_layer, num_layers=10)

        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        # Initialize detailed balance loss tensor
        self.detailed_balance_loss = torch.zeros(1, requires_grad=True)

        # Optimizers
        self.optimizer_policy = torch.optim.Adam(
            [
                {'params': self.embedding_layer.parameters()},
                {'params': self.transformer_pf.parameters()},
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},
            ],
            lr=0.001,  # Consider lowering this for stability
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
        # Track flows
        self.p_f= flows[decision_idx] / flows.sum()
        self.curr.append((self.p_f, flows.sum()))
        self.choice_sequence.append((learner_idx, domain[decision_idx]))
        return domain[decision_idx]

    def reward(self, reward):
        EPS = 1e-6
        reward = math.log(reward + EPS)
        # Calculate loss
        loss = torch.tensor(0.0, requires_grad=True)
        for idx, (p_f, f_s) in enumerate(self.curr):
            p_f = torch.log(p_f + EPS)
            f_s = torch.log(f_s + EPS)
            if idx == len(self.curr) - 1:
                continue
            f_s_next = torch.log(self.curr[idx + 1][1])
            if idx == len(self.curr) - 2:
                step_loss = (f_s + p_f - reward) ** 2
            else:
                step_loss = (f_s + p_f - f_s_next) ** 2
            
            loss = loss + step_loss

        self.num_generation += 1
        # Accumulate loss into self.detailed_balance_loss
        # We don't use += to avoid in-place operations on a leaf tensor
        self.detailed_balance_loss = self.detailed_balance_loss + loss

        # Every generation, update parameters
        if self.num_generation > 0 and self.num_generation % 1 == 0:
            self.optimizer_policy.zero_grad()
            self.detailed_balance_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer_policy.step()
            # Reinitialize detailed_balance_loss for next iterations
            self.detailed_balance_loss = torch.zeros(1, requires_grad=True)

        # Reset states after updating
        self.choice_sequence = []
        self.curr = []
        self.p_f= 0.0  # Reset prev_flow to 0

class GFNLearner:
    def __init__(self, hidden_dim, domain, epsilon):
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))
        self.epsilon = epsilon

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

