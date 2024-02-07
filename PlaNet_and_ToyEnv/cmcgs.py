import numpy as np
import scipy.stats
from scipy.stats import norm
import os
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import cycle
import torch
from multiprocessing import Pool
from collections import Counter
import copy

from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import time
import gc


ELITE_RATIO = 0.25
device = "cuda" if torch.cuda.is_available else "cpu"


class CMCGS:
    def __init__(self,
                 N,
                 min_action,
                 max_action,
                 action_size,
                 min_graph_length,
                 max_graph_length,
                 rollout_length,
                 simulation_budget,
                 clustering_alg,
                 optimal_prob,
                 optimal_n_top,
                 optimal_range,
                 elite_ratio,
                 state_dim,
                 greedy_action=False,
                 planet=True,
                 max_n_exps=500,
                 fixed_init_stddev=False,
                 max_n_clusters=np.inf,
                 clustering_linkage='ward',
                 alpha=5,
                 beta=2,
                 ):
        self.planet = planet
        self.N = N

        self.min_action = min_action
        self.max_action = max_action
        self.action_dim = action_size

        # set hyperparams
        self.min_graph_length = min_graph_length
        self.max_graph_length = max_graph_length
        self.rollout_length = rollout_length
        self.max_n_exps = max_n_exps
        self.min_n_exps = max(25, self.max_n_exps // 10)
        self.clustering_alg = clustering_alg
        self.clustering_linkage = clustering_linkage
        self.simulation_budget = simulation_budget
        self.max_n_clusters = max_n_clusters

        self.optimal_prob = optimal_prob
        self.optimal_n_top = optimal_n_top
        self.optimal_range = optimal_range
        self.fixed_init_stddev = fixed_init_stddev

        self.alpha = alpha
        self.beta = beta

        global ELITE_RATIO
        ELITE_RATIO = elite_ratio
        
        self.layers = None
        self.best_action = None
        self.best_rew = None
        self.exp_owner = None
        self.exp_owner = None
        self.states = None
        self.states2 = None
        self.actions = None
        self.rewards = None
        self.n_exps = None
        self.postpone_clustering = None

        self.time_budget_total = 0
        self.time_budget_env_step = 0
        self.time_budget_clustering = 0
        self.time_budget_update_nodes = 0
        self.time_action_bandit = 0
        self.time_q_bandit = 0

        self.stats_n_clustering_tryouts = 0
        self.stats_n_clustering_success = 0

        self.two_d_nav_plot_time_tag = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")[:-3]
        self.two_d_nav_plot_counter = 0

        self.state_dim = state_dim
        self.greedy_action = greedy_action

        self.trajs = []

    def add_node_to_layer(self, l):
        if self.layers is None:
            self.layers = []
        while len(self.layers) < l + 1:
            self.layers.append(
                Layer(
                    nodes=[],
                    exp_owner=-np.ones(self.max_n_exps, dtype=np.int64),
                    states=np.zeros((self.max_n_exps, self.state_dim)),
                    actions=np.zeros((self.max_n_exps, self.action_dim)),
                    next_states=np.zeros((self.max_n_exps, self.state_dim)),
                    rewards=np.zeros(self.max_n_exps),
                    n_exps=0,
                    postpone_clustering=0,
                    fixed_init_stddev=self.fixed_init_stddev,
                )
            )
        self.layers[l].append_node(Q_Node(action_dim=self.action_dim, min_action=self.min_action, max_action=self.max_action, state_dim=self.state_dim, min_n_exps=self.min_n_exps, alpha=self.alpha, beta=self.beta))

    def add_exp_to_replay_memory(self, l, q, exp, rew):
        self.layers[l].add_exp_to_replay_memory(q, exp, rew)

    def act(self, belief=None, state=None, transition_model=None, reward_model=None, env=None, save_all_trajs=False):
        time_total_start = time.perf_counter()

        self.layers = None
        self.add_node_to_layer(0)
            
        if save_all_trajs:
            self.trajs = [] 

        self.best_action = np.zeros(self.action_dim)
        self.best_action[:] = (self.min_action + self.max_action) / 2
        self.best_rew = -np.inf

        N = self.N
        timesteps = 0

        while timesteps < self.simulation_budget:
            # Graph policy
            b, s, exps, done, ep_ret, traj_len = self.graph_policy_parallel(belief, state, transition_model, reward_model, copy.deepcopy(env), N)
            if not done:
                # Default policy
                ep_ret, traj_len = self.default_policy_parallel(b, s, transition_model, reward_model, ep_ret, traj_len, N)
    
            # Backpropagation
            self.backpropagate_parallel(exps, ep_ret)
            timesteps += traj_len

            if save_all_trajs:
                trajs = []
                for i in range(len(exps)):
                    trajs.append(exps[i]['exp'][1])
                trajs = np.array(trajs).transpose((1, 0, 2))
                trajs = list(trajs)
                self.trajs += trajs

        if self.greedy_action:
            action = np.copy(self.best_action)
        else:
            action = self.layers[0].get_optimal_action(self.optimal_n_top)

        self.time_budget_total += time.perf_counter() - time_total_start
        #self.report_time_budget()

        return action

    def graph_policy_parallel(self, belief, state, transition_model, reward_model, env, N):
        q = np.array([0] * N).astype(int)
        exps = []

        if self.planet:
            b1 = belief.repeat(N, 1)
            s1 = state.repeat(N, 1)
        else:
            b1 = None
            s1 = np.array([0])
        sum_r = 0
        traj_len = 0
        done = False
        n_layers = len(self.layers)
        l = 0
        is_optimal = np.random.rand(N) < self.optimal_prob
        while l < n_layers:
            time_action_bandit_start = time.perf_counter()
            a = self.layers[l].get_actions(q, is_optimal, self.optimal_n_top, self.optimal_range)
            self.time_action_bandit += time.perf_counter() - time_action_bandit_start
            time_step_start = time.perf_counter()

            if self.planet:
                b2, s2, _, _ = transition_model(s1, torch.from_numpy(a[None]).to(device).float(), b1)
                b2 = b2.squeeze(0)
                s2 = s2.squeeze(0)
                r = reward_model(b2, s2)
                done = False
                exps.append({'q': q, 'exp': [s1.cpu().numpy(), a, s2.cpu().numpy()]})
            else:
                s2, r, done, info = env.step(a)
                exps.append({'q': q, 'exp': [s1, a, s2]})
            
            self.time_budget_env_step += time.perf_counter() - time_step_start
            sum_r += r
            traj_len += 1
            if not done and l + 1 == n_layers and n_layers < self.max_graph_length and (
                    self.layers[l].n_exps > self.min_n_exps or n_layers < self.min_graph_length
            ):
                self.add_node_to_layer(l + 1)
                n_layers += 1
            if done or l + 1 == n_layers:
                break
            time_q_bandit_start = time.perf_counter()

            if self.planet:
                q = self.q_bandit(l + 1, s2.cpu().numpy())
            else:
                q = self.q_bandit(l+1, s2)
            self.time_q_bandit += time.perf_counter() - time_q_bandit_start
            s1 = s2
            if self.planet:
                b1 = b2
            l += 1
        return b1, s1, exps, done, sum_r, traj_len

    def q_bandit(self, l, s):
        N = s.shape[0]
        n_q = len(self.layers[l].nodes)
        if n_q == 1:
            return np.array([0] * N).astype(int)
        scores = self.layers[l].get_logpdf_parallel(s)
        scores += np.random.random_sample(scores.shape) * 0.01  # Break ties randomly
        max_score_i = np.argmax(scores, axis=1)
        return max_score_i
    
    def default_policy_parallel(self, belief, state, transition_model, reward_model, ep_ret, traj_len, N):
        n_layers = len(self.layers)
        b1 = belief
        s1 = state
        assert traj_len == n_layers
        for t in range(self.rollout_length):
            time_step_start = time.perf_counter()
            a = np.random.uniform(low=self.min_action, high=self.max_action, size=(N, self.action_dim))
            b2, s2, _, _ = transition_model(s1, torch.from_numpy(a[None]).to(device).float(), b1)
            b2 = b2.squeeze(0)
            s2 = s2.squeeze(0)
            r = reward_model(b2, s2)
            done = False
            self.time_budget_env_step += time.perf_counter() - time_step_start
            ep_ret += r
            traj_len += 1
            if done:
                break
            s1 = s2
            b1 = b2
        return ep_ret, traj_len
                    
    def backpropagate_parallel(self, exps, rew):
        # Update the best found action if needed
        if isinstance(rew, torch.Tensor):
            rew = rew.cpu().numpy()

        max_rew = np.amax(rew)
        if max_rew > self.best_rew:
            self.best_rew = max_rew
            max_idx = np.argmax(rew)
            self.best_action[:] = exps[0]['exp'][1][max_idx][:]

        for l in range(len(exps)):
            q = exps[l]['q']
            self.add_exp_to_replay_memory(l, q, exps[l]['exp'], rew)

            if self.layers[l].postpone_clustering > 0:
                self.layers[l].postpone_clustering -= q.shape[0]
            n_exps = min(self.layers[l].n_exps, self.max_n_exps)

            min_n_exps = self.min_n_exps
            desired_n_clusters = n_exps // min_n_exps

            desired_n_clusters = min(len(self.layers[l].nodes) + 1, desired_n_clusters)

            if isinstance(self.max_n_clusters, list):
                desired_n_clusters = min(self.max_n_clusters[l], desired_n_clusters)
            else:
                desired_n_clusters = min(self.max_n_clusters, desired_n_clusters)
            clustered = True
            if l > 0 and len(self.layers[l].nodes) < desired_n_clusters and self.layers[l].postpone_clustering == 0:
                assert len(self.layers[l].nodes) + 1 == desired_n_clusters
                time_cluster_start = time.perf_counter()
                self.stats_n_clustering_tryouts += 1
                # Try to make a new cluster
                cur_layer_data = self.layers[l].states[:n_exps, :]

                if self.clustering_alg == "dbscan":
                    clusters_idx = DBSCAN(eps=1.5, n_jobs = 10).fit(cur_layer_data).labels_
                elif self.clustering_alg == "hdbscan":
                    clusters_idx = HDBSCAN(n_jobs=-1).fit(cur_layer_data).labels_
                elif self.clustering_alg == "kmeans":
                    clusters_idx = MiniBatchKMeans(n_clusters=desired_n_clusters, n_init=3, random_state=int(time.time())).fit(
                        cur_layer_data).labels_
                elif self.clustering_alg == "agglomerative":
                    clusters_idx = AgglomerativeClustering(n_clusters=desired_n_clusters, linkage=self.clustering_linkage).fit(cur_layer_data).labels_
                else:
                    assert self.clustering_alg == "gmm"
                    clusters_idx = GaussianMixture(
                        n_components=desired_n_clusters, covariance_type="full", random_state=int(time.time())).fit\
                            (cur_layer_data).predict(cur_layer_data)
                for c in range(desired_n_clusters):
                    if np.sum(clusters_idx == c) < min_n_exps / 2:
                        clustered = False
                        break
                self.time_budget_clustering += time.perf_counter() - time_cluster_start
                if clustered:
                    self.stats_n_clustering_success += 1
                    self.add_node_to_layer(l)
                    self.layers[l].exp_owner[:n_exps] = clusters_idx
                    assert len(self.layers[l].nodes) == desired_n_clusters
                    time_update_nodes_start = time.perf_counter()
                    self.layers[l].update_q_nodes(np.arange(desired_n_clusters))
                    self.time_budget_update_nodes += time.perf_counter() - time_update_nodes_start
                else:
                    # Postpone the clustering for better performance
                    self.layers[l].postpone_clustering = min_n_exps // 2
            else:
                clustered = False
            if not clustered:
                # No new clustering needed. Update the Q nodes regularly
                time_update_nodes_start = time.perf_counter()

                qs = []
                for q in set(q):
                    indices = self.layers[l].exp_owner == q
                    if np.sum(indices) >= min_n_exps // 2:
                        qs.append(q)
                if len(qs):
                    self.layers[l].update_q_nodes(qs)

                self.time_budget_update_nodes += time.perf_counter() - time_update_nodes_start


    def report_time_budget(self):
        if self.time_budget_total == 0:
            return
        print('MCGS Stats Report:')
        print('\tTotal search time: %.2f' % self.time_budget_total)
        print('\t\tEnv step ratio: %d%%' % (100 * self.time_budget_env_step / self.time_budget_total))
        print('\t\tClustering ratio: %d%%' % (100 * self.time_budget_clustering / self.time_budget_total))
        print('\t\tNode update ratio: %d%%' % (100 * self.time_budget_update_nodes / self.time_budget_total))
        print('\t\tAction bandit ratio: %d%%' % (100 * self.time_action_bandit / self.time_budget_total))
        print('\t\tQ bandit ratio: %d%%' % (100 * self.time_q_bandit / self.time_budget_total))

        print('\tClustering success rate: %d%% (%d out of %d)' % (
                100 * self.stats_n_clustering_success / max(self.stats_n_clustering_tryouts, 1)
                , self.stats_n_clustering_success, self.stats_n_clustering_tryouts))

class Layer:
    def __init__(self, nodes, exp_owner, states, actions, next_states, rewards, n_exps, postpone_clustering, fixed_init_stddev):
        self.nodes = np.array(nodes, dtype=object)
        self.exp_owner = exp_owner
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.n_exps = n_exps
        self.postpone_clustering = postpone_clustering
        self.max_n_exps = rewards.shape[0]
        self.fixed_init_stddev = fixed_init_stddev

    def append_node(self, node):
        self.nodes = np.append(self.nodes, node)

    def get_optimal_action(self, optimal_n_top):
        return self.nodes[0].get_optimal_action(optimal_n_top)

    def get_actions(self, node_idxs, is_optimal, optimal_n_top, optimal_range):
        used_node_idxs = set(node_idxs)
        means = np.zeros((len(self.nodes), *self.nodes[0].action_mean.shape))
        stddevs = np.zeros((len(self.nodes), *self.nodes[0].action_sd.shape))
        use_uniform_action_nodes = np.zeros(len(self.nodes))
        for n in used_node_idxs:
            means[n] = self.nodes[n].action_mean
            stddevs[n] = self.nodes[n].action_sd
            if self.fixed_init_stddev and self.nodes[n].rewards is None:
                stddevs[n] = self.fixed_init_stddev

        means = means[node_idxs]
        stddevs = stddevs[node_idxs]

        means = np.array(means)
        if stddevs.ndim < 2:
            stddevs = np.expand_dims(stddevs, axis=1)
        mean_sampled_actions = np.random.normal(means, stddevs)
        
        optimal_actions = np.zeros_like(means)
        optimal_node_idxs = set(node_idxs[is_optimal])
        
        missing_rewards = [self.nodes[n].rewards is None for n in node_idxs]
        is_optimal[missing_rewards] = False

        if sum(is_optimal) > 0:
            top_indices = []
            for i, n in enumerate(self.nodes):
                if i in optimal_node_idxs:
                    top_indices.append(n.get_top_indices(optimal_n_top))
                else:
                    top_indices.append([])
        
            m = np.zeros_like(means)
            for i, n in enumerate(node_idxs):
                if not is_optimal[i]:
                    continue
                m[i] = self.nodes[n].actions[np.random.choice(top_indices[n]), :]
            optimal_actions = np.random.normal(m, optimal_range * (self.nodes[0].action_max - self.nodes[0].action_min))

        is_optimal = np.expand_dims(is_optimal, axis=1)
        actions = np.where(is_optimal, optimal_actions, mean_sampled_actions)
        return actions

    def my_log_pdf(self, x, loc, scale):
        p1 = np.log(1/(scale * np.sqrt(2 * np.pi)))
        p2 = -0.5*np.square((x-loc)/scale)
        return p1 + p2
    
    def get_logpdf_parallel(self, s):
        means = np.array(list(map(lambda x: x.state_mean, self.nodes)))
        stddevs = np.array(list(map(lambda x: x.state_std, self.nodes)))
        scores = self.my_log_pdf(s[:, None], means, stddevs)

        if scores.ndim == 3:
            scores = scores.mean(axis=2)
        return scores

    def add_exp_to_replay_memory(self, q, exp, rew):
        idxs = np.arange(self.n_exps, self.n_exps + q.shape[0]) % self.max_n_exps
        self.exp_owner[idxs] = q
        self.states[idxs] = exp[0][:]
        self.actions[idxs] = exp[1][:]
        self.next_states[idxs] = exp[2][:]
        self.rewards[idxs] = rew
        self.n_exps += q.shape[0]

    def update_q_nodes(self, qs):
        rews = []
        acts = []
        for q in qs:
            idxs = self.exp_owner == q
            self.nodes[q].set_data(self.states[idxs, :], self.actions[idxs, :], self.rewards[idxs])

            rews.append(self.rewards[idxs])
            acts.append(self.actions[idxs, :])

        max_len = max(list(map(lambda x: len(x), rews)))
        for i, r in enumerate(rews):
            rews[i] = np.concatenate((r, [-np.inf]*(max_len - len(r))), axis=0)
        rewards = np.array(rews)
        weights = np.copy(rewards)
    
        elite_nums = np.asarray(list(map(lambda x: max(5, len(x) * ELITE_RATIO), acts))).astype(int)
        elite_idx = np.argpartition(-rewards, sorted(elite_nums), axis=1)

        for i, q in enumerate(qs):
            self.nodes[q].update(elite_idx[i], elite_nums[i], weights[i])


                 
class Q_Node:
    def __init__(self, action_dim, min_action, max_action, state_dim, min_n_exps, alpha=5, beta=2):
        self.action_dim = action_dim 
        self.action_min = min_action 
        self.action_max = max_action

        self.state_dim = state_dim

        self.state_mean = np.zeros(self.state_dim)
        self.state_std = 0.1 * np.ones(self.state_dim)
        self.pdf = scipy.stats.norm(self.state_mean, self.state_std)

        if isinstance(self.action_min, float):
            self.action_mean = np.array([0.5 * (self.action_min + self.action_max)] * self.action_dim)
            self.action_sd = np.array([0.5 * (self.action_max - self.action_min)] * self.action_dim)
        else:
            self.action_mean = np.array([0.5 * (self.action_min + self.action_max)])
            self.action_sd = np.array([0.5 * (self.action_max - self.action_min)])

        self.n_data = 0

        self.min_n_exps = min_n_exps
        self.alpha = alpha
        self.beta = beta

        self.actions = None
        self.rewards = None
        self.elite_idx = None

        self.gpr = None

    def get_top_indices(self, optimal_n_top):
        if self.elite_idx is not None:
            optimal_n_top = min(len(self.elite_idx)-1, optimal_n_top)
            top_indices = np.argpartition(-self.rewards[self.elite_idx], optimal_n_top)
            top_indices = self.elite_idx[top_indices[:optimal_n_top]]
        else:
            if optimal_n_top == len(self.rewards):
                top_indices = np.arange(len(self.rewards))
            elif optimal_n_top < len(self.rewards):
                top_indices = np.argpartition(-self.rewards, optimal_n_top)
            else:
                raise RuntimeError
        return top_indices

    def get_optimal_action(self, optimal_n_top):
        # This function should be only be called for getting the optimal action at the end of planning
        if self.rewards is None or len(self.rewards) < self.min_n_exps / 2:
            return self.action_min + np.random.rand(self.action_dim) * (self.action_max - self.action_min)

        top_indices = self.get_top_indices(optimal_n_top)
        selected_idx = top_indices[np.random.randint(0, optimal_n_top)]

        actions = self.actions[top_indices[:optimal_n_top]]
        act = actions.mean(axis=0)
        return act

    def set_data(self, states, actions, rewards):
        self.n_data = states.shape[0]

        self.actions = actions
        self.rewards = rewards
        self.elite_idx = None

        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0)
        self.state_std = np.clip(self.state_std, 0.1, 0.5)
            
    def update(self, elite_idxs, elite_num, weights):
        self.action_mean = None
        elite_idx = elite_idxs[:elite_num]
        M, m = np.max(self.rewards), np.min(self.rewards)
        if M - m > 0.01:
            weights = (weights - m) / (M - m)
            if elite_num < weights.shape[0] and np.sum(weights[elite_idx]) > 1e-2:
                self.elite_idx = elite_idx
                assert self.rewards.shape[0] > max(elite_idx)
                self.action_mean = np.average(self.actions[elite_idx, :], axis=0, weights=weights[elite_idx])
        if self.action_mean is None:
            self.action_mean = np.mean(self.actions, axis=0)

        alpha, beta = self.alpha, self.beta
        squared_diff = ((self.actions[elite_idx] - self.action_mean)**2).sum(axis=0)
        self.action_sd = np.clip(np.sqrt((beta + 0.5 * squared_diff)/(alpha + elite_num/2 - 1)), 0.01, 0.5 * (self.action_max - self.action_min))
