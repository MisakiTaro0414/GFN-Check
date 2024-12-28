import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
import random

losses = []

class GFNOracle_local_search(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, epsilon):
        super(GFNOracle_local_search, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        self.hidden_dim = embedding_dim
        self.epsilon = epsilon
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
        
        self.transformer_pf = nn.TransformerEncoder(transformer_layer, num_layers=10)

        self.logPf = torch.tensor(0.0)
        self.beta = 10
        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        self.optimizer_policy = torch.optim.Adam(
            [
                {'params': self.embedding_layer.parameters()},  # Lower learning rate for embedding layer
                {'params': self.transformer_pf.parameters()},    # Default learning rate for transformer_pf
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},  # Default learning rate for action selectors
            ],
            lr=0.001,  # This will act as the default learning rate if not specified explicitly
        )
        self.optimizer_logZ = torch.optim.Adam(
            [{'params': [self.logZ], 'lr': 1}],
        )

    def clamp_logZ(self):
        with torch.no_grad():
            self.logZ.copy_(torch.clamp(self.logZ, min=self.logZ_lower))

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

    def calculate_depth(self):
    # Start with depth 0 (root node is at level 0)
        depth = 0
        for a, b, _ in self.choice_sequence:
            if a == 1:
                # Root node (level 0)
                depth = max(depth, 0)  # Root is at level 0
            elif a == 2 and b:
                # Left child exists, increase depth to level 1
                depth = max(depth, 1)
            elif a == 3 and b:
                # Right child exists, increase depth to level 1
                depth = max(depth, 1)
        return depth

    def compute_tree_depth(self, sequence, max_depth):
        def parse_subtree(index, depth):
            idx, choice, _ = sequence[index]
            assert idx == 1
            index += 1
            node_depth = depth
            if depth < max_depth:
                idx, left_bool, _ = sequence[index]
                assert idx == 2
                index += 1
                left_depth = depth
                if left_bool:
                    left_depth, index = parse_subtree(index, depth + 1)
                idx, right_bool, _ = sequence[index]
                assert idx == 3
                index += 1
                right_depth = depth
                if right_bool:
                    right_depth, index = parse_subtree(index, depth + 1)
                node_depth = max(left_depth, right_depth, node_depth)
            return node_depth, index
        tree_depth, _ = parse_subtree(0, 1)
        return tree_depth

class GFNLearner:
    def __init__(self, hidden_dim, domain, epsilon):
        self.exploration_prob = 1
        self.domain = domain
        self.epsilon = epsilon
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        output = self.action_selector(hidden)
        output = torch.nan_to_num(output)
        probs = F.softmax(output, dim=-1)  # Convert to probabilities
        # epsilon greedy
        if random.random() < self.exploration_prob:
            sampled_index = random.choice(range(len(self.domain)))
        else:
            sampled_index = torch.multinomial(probs, 1).item()
        return self.domain[sampled_index], torch.log(probs[0][sampled_index]), probs
