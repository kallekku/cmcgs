import math
import numpy as np
import scipy.stats
from scipy.stats import norm
import os
from matplotlib import pyplot as plt
from matplotlib import rc

from datetime import datetime
import sklearn
from sklearn.cluster import MiniBatchKMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time
import gc
            
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy import linalg


ELITE_RATIO = 0.25


class CMCGS:
    def __init__(self, config, model):
        self.config = config

        self.env = model

        self.action_space = model.action_space
        self.action_dim = self.action_space.shape[0]
        self.observation_space = model.observation_space
        self.state_dim = model.observation_space.low.shape[0]

        # set hyperparams
        self.min_graph_length = config.min_graph_length
        self.rollout_length = config.rollout_length
        self.max_n_exps = config.simulation_budget // 5
        self.min_n_data_per_q_node = max(25, self.max_n_exps // 10)
        self.min_update_divisor = config.min_update_divisor

        self.clustering_alg = config.clustering_alg
        self.simulation_budget = config.simulation_budget

        self.optimal_prob = config.optimal_prob
        self.optimal_n_top = config.optimal_n_top
        self.optimal_range = config.optimal_range

        self.max_n_clusters = config.max_n_clusters

        self.alpha = config.alpha
        self.beta = config.beta

        global ELITE_RATIO
        ELITE_RATIO = config.elite_ratio
        
        self.layers = None
        self.best_action = None
        self.best_traj = None
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

        self.save_exploration = False
        self.exploration_trajs = []
        self.rollout_indices = []
        self.reset()

    def set_save_exploration(self, val):
        self.save_exploration = val

    def reset_exploration_trajs(self):
        self.exploration_trajs = []
        self.rollout_indices = []

    def reset(self):
        self.env.reset()
        self.env.save_checkpoint()

    def add_node_to_layer(self, l):
        if self.layers is None:
            self.layers = []
        while len(self.layers) < l + 1:
            self.layers.append({
                'nodes': []
                , 'exp_owner': -np.ones(self.max_n_exps, dtype=np.int64)
                , 'states': np.zeros((self.max_n_exps, self.state_dim))
                , 'actions': np.zeros((self.max_n_exps, self.action_dim))
                , 'states2': np.zeros((self.max_n_exps, self.state_dim))
                , 'rewards': np.zeros(self.max_n_exps)
                , 'n_exps': 0
                , 'postpone_clustering': 0
            })
        self.layers[l]['nodes'].append(Q_Node(action_space=self.env.action_space, state_dim=self.state_dim,
                                              layer=l, min_n_data_per_q_node=self.min_n_data_per_q_node,
                                              alpha=self.alpha,
                                              beta=self.beta,
                                              min_update_divisor=self.min_update_divisor,))

    def add_exp_to_replay_memory(self, l, q, exp, rew):
        idx = self.layers[l]['n_exps'] % self.max_n_exps
        self.layers[l]['exp_owner'][idx] = q
        self.layers[l]['states'][idx, :] = exp[0][:]
        self.layers[l]['actions'][idx, :] = exp[1][:]
        self.layers[l]['states2'][idx, :] = exp[2][:]
        self.layers[l]['rewards'][idx] = rew
        self.layers[l]['n_exps'] += 1

    def act(self, observation=None, skip_step=False):
        time_total_start = time.perf_counter()

        self.env.save_checkpoint()

        self.layers = None
        self.add_node_to_layer(0)

        self.best_action = (self.env.action_space.low + self.env.action_space.high) / 2
        self.best_traj = None
        self.best_rew = -np.inf

        timesteps = 0
        while timesteps < self.simulation_budget:
            rollout_states = []
            # Graph policy
            exps, done, ep_ret, traj_len = self.graph_policy(self.env, observation)
            if not done:
                # Default policy
                ep_ret, traj_len, rollout_states = self.default_policy(self.env, ep_ret, traj_len)
    
            # Backpropagation
            self.backpropagate(exps, ep_ret, rollout_states)
            timesteps += traj_len

            # Load the saved model state
            self.env.load_checkpoint()

            if self.save_exploration:
                state_traj = [e['exp'][2] for e in exps]
                state_traj = [exps[0]['exp'][0]] + state_traj
                self.rollout_indices.append(len(state_traj)) 
                for s in rollout_states:
                    state_traj.append(s)
                state_traj = np.asarray(state_traj)
                self.exploration_trajs.append(state_traj)

        action = np.copy(self.best_action)
        if skip_step:
            return action

        # Important!! update the model to keep it in sync with env.
        s2, _, _, info = self.env.step(action)

        self.time_budget_total += time.perf_counter() - time_total_start

        return action

    def graph_policy(self, env, cur_obs):
        q = 0
        exps = []
        s1 = cur_obs
        sum_r = 0
        traj_len = 0
        done = False
        n_layers = len(self.layers)
        l = 0
        is_optimal = np.random.rand() < self.optimal_prob
        while l < n_layers:
            time_action_bandit_start = time.perf_counter()
            a = self.layers[l]['nodes'][q].action_bandit(is_optimal, self.optimal_n_top, self.optimal_range)
            self.time_action_bandit += time.perf_counter() - time_action_bandit_start
            time_step_start = time.perf_counter()
            s2, r, done, info = env.step(a)
            self.time_budget_env_step += time.perf_counter() - time_step_start
            exps.append({'q': q, 'exp': [s1, a, s2]})
            sum_r += r
            traj_len += 1
            if not done and l + 1 == n_layers and (
                    self.layers[l]['n_exps'] > self.min_n_data_per_q_node or n_layers < self.min_graph_length
            ):
                self.add_node_to_layer(l + 1)
                n_layers += 1
            if done or l + 1 == n_layers:
                break
            time_q_bandit_start = time.perf_counter()
            q = self.q_bandit(l + 1, s2)
            self.time_q_bandit += time.perf_counter() - time_q_bandit_start
            s1 = s2
            l += 1
        return exps, done, sum_r, traj_len

    def q_bandit(self, l, s):
        n_q = len(self.layers[l]['nodes'])
        if n_q == 1:
            return 0
        max_score = -np.inf
        max_score_i = -1
        for q in range(n_q):
            pdf = self.layers[l]['nodes'][q].get_pdf(s)
            score = np.mean(pdf)
            score += np.random.random_sample() * 0.01  # Break ties randomly
            if score > max_score:
                max_score = score
                max_score_i = q
        return max_score_i

    def default_policy(self, env, ep_ret, traj_len):
        n_layers = len(self.layers)
        assert traj_len == n_layers
        states = []
        for t in range(self.rollout_length):
            time_step_start = time.perf_counter()
            ac = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.shape)
            ob, r, done, info = env.step(ac)
            states.append(ob)
            self.time_budget_env_step += time.perf_counter() - time_step_start
            ep_ret += r
            traj_len += 1
            if done:
                break
        return ep_ret, traj_len, states

    @ignore_warnings(category=ConvergenceWarning)
    def backpropagate(self, exps, rew, rollout_states=[]):
        # Update the best found action if needed
        if rew > self.best_rew:
            self.best_rew = rew
            self.best_action[:] = exps[0]['exp'][1][:]
            self.best_traj = [exps[0]['exp'][0]]
            for e in exps:
                self.best_traj.append(e['exp'][2])
            for r in rollout_states:
                self.best_traj.append(r)
            self.best_traj = np.array(self.best_traj)

        for l in range(len(exps)):
            q = exps[l]['q']

            self.add_exp_to_replay_memory(l, q, exps[l]['exp'], rew)

            if self.layers[l]['postpone_clustering'] > 0:
                self.layers[l]['postpone_clustering'] -= 1
            n_exps = min(self.layers[l]['n_exps'], self.max_n_exps)

            min_n_data_per_q_node = self.min_n_data_per_q_node
            desired_n_clusters = n_exps // min_n_data_per_q_node    
            desired_n_clusters = min(len(self.layers[l]['nodes']) + 1, desired_n_clusters)
            if self.max_n_clusters:
                desired_n_clusters = min(self.max_n_clusters, desired_n_clusters)
            clustered = True
            if l > 0 and len(self.layers[l]['nodes']) < desired_n_clusters and self.layers[l]['postpone_clustering'] == 0:
                assert len(self.layers[l]['nodes']) + 1 == desired_n_clusters
                time_cluster_start = time.perf_counter()
                self.stats_n_clustering_tryouts += 1
                # Try to make a new cluster
                cur_layer_data = self.layers[l]['states'][:n_exps, :]
               
                if self.clustering_alg == "kmeans":
                    clusters_idx = MiniBatchKMeans(n_clusters=desired_n_clusters, n_init=3, random_state=int(time.time())).fit(
                        cur_layer_data).labels_
                elif self.clustering_alg == "agglomerative":
                    clusters_idx = AgglomerativeClustering(n_clusters=desired_n_clusters).fit(cur_layer_data).labels_
                elif self.clustering_alg == "gmm":
                    clusters_idx = GaussianMixture(
                        n_components=desired_n_clusters, covariance_type="diag", random_state=int(time.time())).fit\
                            (cur_layer_data).predict(cur_layer_data)
                elif self.clustering_alg == "spectral":
                    clusters_idx = SpectralClustering(n_clusters=desired_n_clusters, n_init=3).fit(cur_layer_data).labels_
                else:
                    print("Unknown clustering algorithm")
                    raise RuntimeError
                for c in range(desired_n_clusters):
                    if np.sum(clusters_idx == c) < min_n_data_per_q_node / 2:
                        clustered = False
                        break
                self.time_budget_clustering += time.perf_counter() - time_cluster_start
                if clustered:
                    self.stats_n_clustering_success += 1
                    self.add_node_to_layer(l)
                    self.layers[l]['exp_owner'][:n_exps] = clusters_idx
                    assert len(self.layers[l]['nodes']) == desired_n_clusters
                    time_update_nodes_start = time.perf_counter()
                    for q in range(desired_n_clusters):
                        indices = self.layers[l]['exp_owner'] == q
                        self.layers[l]['nodes'][q].update(self.layers[l]['states'][indices, :], self.layers[l]['actions'][indices, :], self.layers[l]['rewards'][indices])
                    self.time_budget_update_nodes += time.perf_counter() - time_update_nodes_start
                else:
                    # Postpone the clustering for better performance
                    self.layers[l]['postpone_clustering'] = min_n_data_per_q_node // 2
            else:
                clustered = False
            if not clustered:
                # No new clustering needed. Update the Q node regularly.
                indices = self.layers[l]['exp_owner'] == q
                if np.sum(indices) >= min_n_data_per_q_node // self.min_update_divisor:
                    time_update_nodes_start = time.perf_counter()
                    self.layers[l]['nodes'][q].update(self.layers[l]['states'][indices, :], self.layers[l]['actions'][indices, :], self.layers[l]['rewards'][indices])
                    self.time_budget_update_nodes += time.perf_counter() - time_update_nodes_start

    def report_time_budget(self):
        if self.time_budget_total == 0:
            return
        print('CMCGS Stats Report:')
        print('\tTotal search time: %.2f' % self.time_budget_total)
        print('\t\tEnv step ratio: %d%%' % (100 * self.time_budget_env_step / self.time_budget_total))
        print('\t\tClustering ratio: %d%%' % (100 * self.time_budget_clustering / self.time_budget_total))
        print('\t\tNode update ratio: %d%%' % (100 * self.time_budget_update_nodes / self.time_budget_total))
        print('\t\tAction bandit ratio: %d%%' % (100 * self.time_action_bandit / self.time_budget_total))
        print('\t\tQ bandit ratio: %d%%' % (100 * self.time_q_bandit / self.time_budget_total))

        print('\tClustering success rate: %d%% (%d out of %d)' % (
                100 * self.stats_n_clustering_success / max(self.stats_n_clustering_tryouts, 1)
                , self.stats_n_clustering_success, self.stats_n_clustering_tryouts))
        print('\tClustering time: %.2f' % (self.time_budget_clustering))
        print('\tNode update time: %.2f' % (self.time_budget_update_nodes))


class Q_Node:
    def __init__(self, action_space, state_dim, layer, min_n_data_per_q_node, alpha=5, beta=1, min_update_divisor=2):
        self.action_dim = action_space.low.shape[0]
        self.action_min = action_space.low
        self.action_max = action_space.high
        self.layer = layer

        self.state_dim = state_dim

        self.state_mean = np.zeros(self.state_dim)
        self.state_std = 0.1 * np.ones(self.state_dim)

        self.action_mean = 0.5 * (self.action_min + self.action_max)
        self.action_sd = 0.5 * (self.action_max - self.action_min)

        self.n_data = 0

        self.min_n_data_per_q_node = min_n_data_per_q_node
        self.min_update_divisor = min_update_divisor

        self.actions = None
        self.rewards = None

        self.gpr = None

        self.alpha = alpha
        self.beta = beta
    
    def get_gaussian_pdf(self, x):
        loc = self.state_mean
        scale = self.state_std
        p1 = 1/(scale * np.sqrt(2 * np.pi))
        p2 = -0.5*np.square((x-loc)/scale)
        return p1 * np.exp(p2)

    def get_pdf(self, x):
        return self.get_gaussian_pdf(x)

    def action_bandit(self, is_optimal, optimal_n_top, optimal_range):
        if self.rewards is None or len(self.rewards) < self.min_n_data_per_q_node / self.min_update_divisor:
            return np.random.normal(self.action_mean, self.action_sd)

        if is_optimal:
            top_indices = np.argpartition(-self.rewards, optimal_n_top)
            selected_idx = top_indices[np.random.randint(0, optimal_n_top)]
            action = np.random.normal(self.actions[selected_idx, :], optimal_range * (self.action_max - self.action_min))
            return action

        action = np.random.normal(self.action_mean, self.action_sd)
        return action

    def update(self, states, actions, rewards):

        self.n_data = states.shape[0]

        self.actions = actions
        self.rewards = rewards

        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0)
        self.state_std = np.clip(self.state_std, 0.1, 0.5)

        self.action_mean = None
        weights = np.copy(rewards)
        M, m = np.max(weights), np.min(weights)
        if M - m > 0.01:
            weights = (weights - m) / (M - m)

            elite_num = max(5, int(weights.shape[0] * ELITE_RATIO))
            if elite_num < weights.shape[0]:
                elite_idx = np.argpartition(rewards, elite_num, axis=None)[-elite_num:]
                if np.sum(weights[elite_idx]) > 1e-2:
                    self.action_mean = np.average(actions[elite_idx, :], axis=0, weights=weights[elite_idx])
        else:
            elite_num = len(self.rewards)
            elite_idx = np.arange(elite_num)

        if self.action_mean is None:
            self.action_mean = np.mean(actions, axis=0)
        
        squared_diff = ((actions[elite_idx] - self.action_mean)**2).sum(axis=0)
        self.action_sd = np.clip(np.sqrt((self.beta + 0.5 * squared_diff)/(self.alpha + elite_num/2 - 1)), 0.01, 0.5 * (self.action_max - self.action_min))
