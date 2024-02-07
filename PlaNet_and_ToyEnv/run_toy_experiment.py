import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

from cmcgs import CMCGS

class ObstacleToyEnv:
    def __init__(self, depth, threshold):
        self.threshold = np.abs(threshold)
        self.depth = depth
        self.actions = []
        
        # Variables for visualization

        self.y_divisor = 2
        self.block_size_matplotlib = 1
        self.goals = [(self.depth, i) for i in range(-self.depth, self.depth + 1, 2)]
        self.goal_size = 0.1
        self.player_size = 0.1
        self.box_obstacle_width = 0.25
        self.box_obstacle_height = 1.75
            
        self.obstacles = []
        for i in range(1, self.depth+1):
            for j in range(-i + 1, i, 2):
                self.obstacles.append((i-0.5, j, self.box_obstacle_height))

    def reset(self):
        self.actions = []
        return 0

    def step(self, act, verbose=False):
        if act.ndim == 1:
            act = act[:, None]

        if act.shape[0] != 1 and not len(self.actions):
            act_arr = act
            self.actions.append(act)
        elif act.shape[0] != 1:
            try:
                old_actions = np.array(self.actions, dtype=float)
            except:
                batch_size = max(list(map(lambda x: x.shape[0], self.actions)))
                old_actions = []
                for a in self.actions:
                    if a.shape[0] != batch_size:
                        a = a.repeat(batch_size, axis=0)
                    old_actions.append(a)
                old_actions = np.asarray(old_actions)

            if old_actions.shape[1] != 1:
                old_arr = old_actions.squeeze(axis=2).transpose((1, 0))
            else:
                old_arr = old_actions.squeeze(axis=1).transpose((1, 0)).repeat(act.shape[0], axis=0)
            act_arr = np.concatenate((old_arr, act), axis=1)
            self.actions.append(act)
        else:
            self.actions.append(act)
            act_arr = np.asarray(self.actions).squeeze(axis=1).transpose((1, 0))
        
        obs = act_arr
        if obs.shape[1] < self.depth:
            missing_zeros = np.zeros((obs.shape[0], self.depth-obs.shape[1]))
            obs = np.concatenate((obs, missing_zeros), axis=1)
       
        if act_arr.shape[1] < self.depth:
            r = np.zeros((act_arr.shape[0]))
            d = False
        else:
            precond_1 = np.logical_or((act_arr > self.threshold).all(axis=1), (act_arr < -self.threshold).all(axis=1))
            precond_2 = np.logical_or(act_arr < -self.threshold, act_arr > self.threshold).all(axis=1)

            r = np.where(precond_2, 0.5, 0)
            r = np.where(precond_1, 1, r)
            d = True

        obs = obs.clip(-1, 1).sum(axis=-1, keepdims=True)
        return obs, r, d, {}
    
    def create_map(self):
        n_steps = self.depth

        screen_width = self.block_size_matplotlib * (n_steps + 1)
        screen_height = self.block_size_matplotlib * (2*n_steps + 1) / self.y_divisor

        rc('figure', figsize=(screen_width, screen_height))
        fig, axs = plt.subplots(1, 1)
        ax = axs

        ax.set_xlim([-0.75, n_steps + 0.75])
        ax.set_ylim([(-self.depth - 0.75) / self.y_divisor, (self.depth + 0.75) / self.y_divisor])
        ax.set_xticks([])
        ax.set_yticks([])

        # Goal
        for i, g in enumerate(self.goals):
            if not i or i == len(self.goals) - 1:
                ax.add_patch(plt.Circle((g[0], g[1] / self.y_divisor), self.goal_size*2, color='g', zorder=2))
            else:
                ax.add_patch(plt.Circle((g[0], g[1] / self.y_divisor), self.goal_size, color='g', zorder=2))

        # Obstacles
        w = self.box_obstacle_width
        for [x, y, h] in self.obstacles:
            ax.add_patch(plt.Rectangle((x - w / 2, (y - h / 2) / self.y_divisor), w, h / self.y_divisor, color='dimgray', zorder=2))

        return fig, ax


    def get_player_coordinates(self):
        # Player
        # Check if we're at a wall
        if len(self.actions) and np.any(np.abs(self.actions) < 1):
            # At a wall
            pos_y = 0
            for i, a in enumerate(self.actions):
                if np.abs(a) < 1:
                    pos_x = i + 0.5 - self.box_obstacle_width/2
                    pos_y += a[0, 0]*self.box_obstacle_height/2
                    break
                else:
                    pos_y += np.clip(a, -self.threshold, self.threshold)[0, 0]
        else:
            # Not at a wall
            pos_x = len(self.actions)
            pos_y = np.clip(self.actions, -1, 1).sum() 
        pos_y /= self.y_divisor
        return pos_x, pos_y

    def add_player(self, fig, ax):
        player_pos = self.get_player_coordinates()
        ax.add_patch(plt.Circle(player_pos, self.player_size, color='blue', zorder=2))
        return fig, ax

    def plot_goals(self, fig, ax):
        for i, g in enumerate(self.goals):
            if not i or i == len(self.goals) - 1:
                ax.add_patch(plt.Circle((g[0], g[1] / self.y_divisor), self.goal_size*2, color='g', zorder=2))
            else:
                ax.add_patch(plt.Circle((g[0], g[1] / self.y_divisor), self.goal_size, color='g', zorder=2))
        return fig, ax

    def plot_act(self, act, fig, ax):
        curr_x, curr_y = self.get_player_coordinates()
        if self.collided():
            return fig, ax 
        new_x = curr_x + (0.5 - self.box_obstacle_width/2)*0.75
        if np.abs(act) < 1:
            new_y = curr_y + (act*self.box_obstacle_height/2)*0.75 / self.y_divisor
        else:
            new_y = curr_y + (np.clip(act, -self.threshold, self.threshold))*0.75 / self.y_divisor

        plt.plot([curr_x, new_x], [curr_y, new_y], linestyle='solid', c='red', lw=2)
        ax.add_patch(plt.arrow(new_x, new_y, dx=0.1*(new_x-curr_x), dy=0.1*(new_y-curr_y), width=0.025, color='red', lw=2, zorder=2))
        return fig, ax
        
    def collided(self):
        curr_x, curr_y = self.get_player_coordinates()
        return not np.allclose(curr_x, int(curr_x))
        
    def scatter(self):
        return np.random.uniform(low=-0.05, high=0.05, size=2)

    def plot_trajectory(self, traj, fig=None, ax=None):
        assert len(self.actions) + len(traj) == self.depth
        
        if fig is None:
            fig, ax = self.create_map()
        
        if self.collided():
            return fig, ax 
 
        w = self.box_obstacle_width
        h = self.box_obstacle_height
 
        curr_x, curr_y = self.get_player_coordinates()
        agent_pos = [[curr_x, curr_y]]
        for i, act in enumerate(traj):
            if abs(act) >= 1:
                curr_y += np.sign(act) / self.y_divisor
                s = self.scatter()
                agent_pos.append([curr_x + 0.5 - w/2 + s[0], curr_y + s[1] / self.y_divisor])
                s = self.scatter()
                agent_pos.append([curr_x + 1 + s[0], curr_y + s[1] / self.y_divisor])
            else:
                agent_pos.append([curr_x + 0.49 - w/2, curr_y + act*h/(2 * self.y_divisor)])
                break
            curr_x += 1

        agent_pos = np.array(agent_pos, dtype=float)

        plt.scatter(agent_pos[:, 0], agent_pos[:, 1], c='lightgrey', s=1, zorder=1)
        for i in range(1, len(agent_pos)):
            Xs = [agent_pos[i-1, 0], agent_pos[i, 0]]
            Ys = [agent_pos[i-1, 1], agent_pos[i, 1]]
            plt.plot(Xs, Ys, linestyle='dashed', c='lightgrey', zorder=1, linewidth=1)
            
        return fig, ax

    def plot_trajs(self, trajs, act=None, clusters=None):
        fig, ax = self.create_map()
        for t in trajs:
            if t.ndim == 2:
                t = t.squeeze(1)
            fig, ax = self.plot_trajectory(t, fig, ax)
        if act is not None:
            fig, ax = self.plot_act(act, fig, ax)
        if clusters is not None:
            fig, ax = self.plot_clusters(fig, ax, clusters, self.depth - trajs[0].shape[0])
        fig, ax = self.plot_goals(fig, ax)
        fig, ax = self.add_player(fig, ax)
        plt.show()
        plt.close('all')

    def plot_clusters(self, fig, ax, clusters, init_ctr):
        for i, c in enumerate(clusters):
            if not i:
                continue
            for n in c:
                x = i + init_ctr
                y = n[0] / self.y_divisor
                ax.add_patch(plt.Circle((x, y), 0.066, color='black', zorder=2))
        
                new_x = x + 0.35 - (self.box_obstacle_width/2)*0.75
                new_y = y + np.clip(n[1], -self.threshold, self.threshold)[0]*0.75 / self.y_divisor

                plt.plot([x, new_x], [y, new_y], linestyle='solid', c='black', lw=2)
                ax.add_patch(plt.arrow(new_x, new_y, dx=0.1*(new_x-x), dy=0.1*(new_y-y), width=0.025, color='black', lw=2, zorder=2))
        return fig, ax

def plot(horizon, thr, trajs, env=None, act=None):
    if env is None:
        env = ObstacleToyEnv(horizon, thr)
        env.reset()
    env.plot_trajs(trajs, act)


def cmcgs_plot(horizon, thr, cmcgs, env=None, act=None):
    if env is None:
        env = ObstacleToyEnv(horizon, thr)
        env.reset()
    clusters = []
    for i, layer in enumerate(cmcgs.layers):
        l_nodes = []
        for j, node in enumerate(layer.nodes):
            l_nodes.append((node.state_mean[0], node.action_mean, node.action_bandit(True, 1, greedy=True)))
        clusters.append(l_nodes)

    n_trajs = len(cmcgs.trajs)
    env.plot_trajs(np.array(cmcgs.trajs)[np.random.choice(n_trajs, n_trajs//10, replace=False)], act, clusters)


class CEM:
    def __init__(self, n_iter, horizon, elite_frac):
        self.n_iter = n_iter
        self.batch_size = 2400 // n_iter
        self.elite_frac = elite_frac
        self.horizon = horizon
        self.n_elite = max(1, int(self.batch_size * self.elite_frac))

    def get_act(self, env):
        h = self.horizon
        mean = np.zeros((h))
        std = np.ones((h))
        all_acts = []
        for i in range(self.n_iter):
            acts = np.random.normal(loc=mean, scale=std, size=(self.batch_size, h))
            all_acts.append(acts)
            e = copy.deepcopy(env)
            _, r, _, _ = e.step(acts)
            best = np.argpartition(-r, self.n_elite)[:self.n_elite]
            elites = acts[best]
            mean = elites.mean(axis=0)
            if not elites.shape[0]:
                std = np.ones((h))
            else:
                std = elites.std(axis=0)
        all_acts = np.array(all_acts)
        all_acts = all_acts.reshape(-1, all_acts.shape[-1])
        return np.array([[mean[0]]]), self.n_elite


def test_cem_episode(n_iter, elite_frac):
    horizon = 5
    thr = 1

    env = ObstacleToyEnv(horizon, thr)
    env.reset()

    done = False
    tot_r = 0
    i = 0
    traj = []
    while not done:
        c = CEM(n_iter, horizon-i, elite_frac)
        act, n_elite = c.get_act(copy.deepcopy(env))
        _, r, done, _ = env.step(act)
        traj.append(float(act))
        tot_r += r
        i += 1
    return tot_r[0], n_elite, np.array(traj)

def test_cem(random_shooting=False):
    evals = 1000
    if random_shooting:
        n_iters = [1]
        elite_fracs = [1e-10] 
    else:
        n_iters = [5]
        elite_fracs = [0.01] 
    for n in n_iters:
        for e in elite_fracs:
            sum_r = 0
            rs = []
            trajs = []
            basic = 0
            additional = 0
            for _ in range(evals):
                r, n_elite, traj = test_cem_episode(n, e)
                trajs.append(traj)
                sum_r += r
                if r == 1:
                    additional += 1
                if r >= 0.5:
                    basic += 1
                rs.append(r)
            print("{} {:.2f} {:.3f} {:.3f} {} {} {}".format(n, e, sum_r/evals, np.std(rs)/np.sqrt(evals), n_elite, basic, additional))

def test_cmcgs_episode():
    horizon = 5
    thr = 1
    
    env = ObstacleToyEnv(horizon, thr)
    env.reset()

    done = False
    i = 0
    tot_r = 0
    while not done:
        m = CMCGS(800, min_action=-5, max_action=5, action_size=1, min_graph_length=horizon-i, max_graph_length=horizon-i,
                  rollout_length=0, simulation_budget=3*(horizon-i), clustering_alg='agglomerative', optimal_prob=0.5,
                  optimal_n_top=50, optimal_range=0.1, elite_ratio=0.1, state_dim=horizon,
                  greedy_action=True, planet=False, max_n_exps=1000,
                  max_n_clusters=2, fixed_init_stddev=1, alpha=5, beta=2,
                 )
        act = m.act(env=copy.deepcopy(env))
        _, r, done, _ = env.step(act)
        tot_r += r
        i += 1
    return tot_r[0]

def test_cmcgs():
    evals = 10000
    tot_r = 0
    basic = 0
    additional = 0
    rs = []
    for i in range(1, evals+1):
        r = test_cmcgs_episode()
        tot_r += r
        if r == 1:
            additional += 1
        if r >= 0.5:
            basic += 1
        rs.append(r)
        if not i % 100:
            rs_arr = np.asarray(rs)
            mean = np.mean(rs_arr)
            std = np.std(rs_arr)
            se = std/np.sqrt(i)
            print("{} {:.3f} {:.3f} {:.3f} {} {}".format(i, mean, std, se, basic, additional), flush=True)

if __name__ == "__main__":
    print("Running random shooting experiment")
    test_cem(random_shooting=True)
    print("Running CEM experiment")
    test_cem(random_shooting=False)
    print("Running CMCGS experiment")
    test_cmcgs()
