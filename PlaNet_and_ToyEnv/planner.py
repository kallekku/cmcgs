from math import inf
import numpy as np
import torch
from torch import nn
from cmcgs import CMCGS

device = "cuda" if torch.cuda.is_available() else "cpu"
# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(nn.Module):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, state_dim, min_action=-inf, max_action=inf, cmcgs=False, cmcgs_clustering_alg='agglomerative', cmcgs_greedy_action=False, cmcgs_optimal_ratio=0.3, cmcgs_max_clusters=np.inf, cmcgs_interactions=120000, cmcgs_linkage='ward'):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.cmcgs = cmcgs

        if self.cmcgs:
            timesteps = int(cmcgs_interactions / candidates)

            self.cmcgs = CMCGS(candidates, self.min_action, self.max_action, self.action_size, 3, 10, 5, timesteps, cmcgs_clustering_alg, cmcgs_optimal_ratio, top_candidates, 0.1, 0.1, state_dim, cmcgs_greedy_action, clustering_linkage=cmcgs_linkage, max_n_clusters=cmcgs_max_clusters)
    
    def forward(self, belief, state):
        if self.cmcgs:
            with torch.no_grad():
                return self.forward_cmcgs(belief, state)
        else:
            return self.forward_cem(belief, state)

    def forward_cmcgs(self, belief, state):
        return torch.Tensor(self.cmcgs.act(belief, state, self.transition_model, self.reward_model)).to(device)[None]

    def forward_cem(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)    # Sample actions (time x (batch x candidates) x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)    # Clip action range
            # Sample next states
            beliefs, states, _, _ = self.transition_model(state, actions, belief)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)    # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
        # Return first action mean µ_
        return action_mean[0].squeeze(dim=1)
