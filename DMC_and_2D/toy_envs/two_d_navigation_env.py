import time
import numpy as np
from math import cos, sin
from matplotlib import pyplot as plt
from matplotlib import rc
import gym
from gym import spaces


class TwoDimNavigationEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
        , 'video.frames_per_second': 75
    }

    def __init__(self, obstacle_mode="circles"):
        self.obstacle_mode = obstacle_mode
        self._max_episode_steps = 10
        self.action_repeat = 50
        self.block_size = 100
        self.block_size_matplotlib = 2
        self.player_size = 0.05
        self.goal_size = 0.1
        self.box_obstacle_width = 0.15
        self.max_vel = 1.25

        if obstacle_mode == "circles":
            self.obstacles = [
                [3, 2.5, 1]
                , [4, 4, 0.3]
                , [6, 2.25, 0.4]
                , [8, 5, 1.75]
                , [8.5, 1.5, 0.7]
            ]
        else:
            self.obstacles = [
                [2, 1.5, 1]
                , [2, 3, 1]
                , [2, 4.5, 1]
                , [3, 1.6, 1.2]
                , [3, 3, 0.9]
                , [3, 4.4, 1.2]
                , [4, 1.5, 1]
                , [4, 4.5, 1]
                , [5, 3, 3.5]
                , [6, 1.25, 0.5]
                , [6, 3, 1.5]
                , [6, 4.75, 0.5]
                , [7, 2, 2]
                , [7, 4.3, 1.4]
                , [8, 1.5, 1]
                , [8, 3.25, 1]
                , [8, 4.5, 1]
                , [9, 1.5, 1]
                , [9, 4, 2]
            ]
            if True:
                self.obstacles = [
                    [2, 3, 2.25]
                    #, [3, 1.6, 1.2]
                    #, [3, 3, 0.9]
                    #, [3, 4.4, 1.2]
                    , [4, 1.5, 1]
                    , [4, 4.5, 1]
                    , [5, 1.5, 1]
                    , [5, 3, 1]
                    , [5, 4.5, 1]
                    , [6, 1.25, 0.5]
                    , [6, 3, 1.5]
                    , [6, 4.75, 0.5]
                    , [7, 2, 2]
                    , [7, 4.3, 1.4]
                    , [8, 1.5, 1]
                    , [8, 3.25, 1]
                    , [8, 4.5, 1]
                    , [9, 1.5, 1]
                    , [9, 4, 2]
                ]
        self.goal_pos = [self._max_episode_steps, 4]

        self.t = 0
        self.pos = 0  # [-1, 1]
        self.vel = 0  # [-1, 1]
        self.acc = 0  # [-1, 1]  # This is used for rendering purposes only.
        high = np.array([1, 1, 1])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([1])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.viewer = None

        self.spec = {'id': '2d-navigation-%s' % obstacle_mode}

        self.reset()

    @property
    def player_pos(self):
        return [(self.t + 1) * self.block_size, (3 + 2 * self.pos) * self.block_size]

    @property
    def dt(self):
        return 0.2  # This can be set "almost" arbitrarily.

    def step(self, action, render=False):
        info = {'render': []}

        self.acc = action[0]
        acc = np.clip(action[0], -1, 1)

        rew_ac = np.exp(-10 * np.power(acc, 2))  # Encourage lower velocity
        rew_goal = 0
        rew_obs = 0
        rew_alive = 1

        dt = 1 / self.action_repeat
        done = False
        for s in range(self.action_repeat):
            # Update velocity and then the position
            self.vel += acc * dt
            self.vel = np.clip(self.vel, -1, 1)
            self.pos += self.vel * self.max_vel * dt
            if self.pos >= 1 and self.vel > 0:
                self.vel = 0
            if self.pos <= -1 and self.vel < 0:
                self.vel = 0
            self.pos = np.clip(self.pos, -1, 1)
            self.t += dt
            if render:
                frame = self.render('rgb_array')
                info['render'].append(frame)
            # Check collision with obstacles
            pos = np.array(self.player_pos)
            for [x, y, s] in self.obstacles:
                obs_pos = np.array([x * self.block_size, y * self.block_size])
                if self.obstacle_mode == "circles":
                    if np.linalg.norm(pos - obs_pos) < self.block_size * (s + self.player_size):
                        rew_obs = -1
                        rew_alive = 0
                        done = True
                        break
                else:
                    w, h = self.box_obstacle_width * self.block_size / 2, s * self.block_size / 2
                    if np.abs(pos[0] - obs_pos[0]) < w + 0.001 and np.abs(pos[1] - obs_pos[1]) < h + 0.001:
                        rew_obs = -1
                        rew_alive = 0
                        done = True
                        break
            '''
            for [x, y, s] in self.obstacles:
                
                obs = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            '''
            # Reached the goal?
            if not done and np.linalg.norm(pos - self.block_size * np.array(self.goal_pos)) < self.block_size * (self.goal_size + self.player_size):
                rew_goal = 1
                done = True
            if done:
                break
        if not done:
            done = (self.t + dt) >= (self._max_episode_steps - 1)
        reward = 0.01 * rew_alive + 0.02 * rew_ac + 0.97 * rew_goal + rew_obs
        return self._get_obs(), reward, done, info

    def reset(self):
        self.t = 0
        self.pos = 0
        self.vel = 0
        self.acc = 0
        return self._get_obs()

    def _get_obs(self):
        t = self.t / (self._max_episode_steps - 1) * 2 - 1  # Map self.t to [-1, 1]
        return np.array([t, self.pos, self.vel])

    def get_state(self):
        return self._get_obs()

    def set_state(self, state):
        self.t = np.rint((state[0] + 1) / 2 * (self._max_episode_steps - 1))
        self.pos = state[1]
        self.vel = state[2]

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        
        screen_width = self.block_size * (self._max_episode_steps + 1)
        screen_height = self.block_size * 6

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        self.viewer.geoms.clear()

        # Vertical lines
        for s in range(self._max_episode_steps):
            line = rendering.Line([(s + 1) * self.block_size, self.block_size]
                                  , [(s + 1) * self.block_size, screen_height - self.block_size])
            line.set_color(0.75, 0.75, 0.75)
            self.viewer.add_geom(line)

        # Horizontal lines
        for s in range(5):
            line = rendering.Line([self.block_size, (s + 1) * self.block_size]
                                  , [screen_width - self.block_size, (s + 1) * self.block_size])
            line.set_color(0.75, 0.75, 0.75)
            self.viewer.add_geom(line)

        # Goal
        goal = rendering.make_circle(self.goal_size * self.block_size, filled=True)
        goal.add_attr(rendering.Transform(translation=(self.goal_pos[0] * self.block_size, self.goal_pos[1] * self.block_size)))
        goal.set_color(0, 0.75, 0)
        self.viewer.add_geom(goal)

        # Obstacles
        for [x, y, s] in self.obstacles:
            if self.obstacle_mode == "circles":
                obs = rendering.make_circle(s * self.block_size, filled=True)
            else:
                w, h = self.box_obstacle_width * self.block_size, s * self.block_size
                r = w / 2
                l = -r
                t = h / 2
                b = -t
                obs = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            obs.add_attr(rendering.Transform(translation=(x * self.block_size, y * self.block_size)))
            obs.set_color(0.25, 0.25, 0.25)
            self.viewer.add_geom(obs)

            # line = rendering.Line([self.player_pos[0], self.player_pos[1]], [x * self.block_size, y * self.block_size])
            # line.set_color(0.75, 0, 0)
            # self.viewer.add_geom(line)

        # Player
        player = rendering.make_circle(self.player_size * self.block_size, filled=True)
        player.add_attr(rendering.Transform(translation=(self.player_pos[0], self.player_pos[1])))
        player.set_color(0, 0, 1)
        self.viewer.add_geom(player)

        # Acceleration (3 lines are rendered to draw an arrow)
        unit = self.block_size * 0.5
        player_pos = np.array([self.player_pos[0], self.player_pos[1]])
        end_pos = player_pos + np.array([1, self.acc]) * unit
        diff = 0.3 * (player_pos - end_pos)
        line = rendering.Line(player_pos, end_pos)
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)

        theta = np.deg2rad(30)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        edge = np.dot(rot, diff)
        line = rendering.Line(end_pos, end_pos + edge)
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)

        theta = np.deg2rad(-30)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        edge = np.dot(rot, diff)
        line = rendering.Line(end_pos, end_pos + edge)
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def plot(self, fig=None, ax=None):
        n_steps = self._max_episode_steps

        if fig is None and ax is None:
            screen_width = self.block_size_matplotlib * (n_steps + 1)
            screen_height = self.block_size_matplotlib * (5 + 1)

            rc('figure', figsize=(screen_width, screen_height))
            fig, axs = plt.subplots(1, 1)
            # [axi.set_axis_off() for axi in axs.ravel()]
            ax = axs

        ax.set_xlim([-0.75, n_steps - 0.25])
        ax.set_ylim([0.25, 5.75])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xticks(np.arange(n_steps))
        # ax.set_yticks(np.arange(5) + 1)

        # Vertical lines
        for s in range(n_steps):
            ax.plot([s, s], [1, 5], 'silver')

        # Horizontal lines
        for s in range(5):
            ax.plot([0, n_steps - 1], [s + 1, s + 1], 'silver')

        # Goal
        ax.add_patch(plt.Circle((self.goal_pos[0] - 1, self.goal_pos[1]), self.goal_size, color='g', zorder=2))

        # Obstacles
        if self.obstacle_mode == "circles":
            for [x, y, r] in self.obstacles:
                ax.add_patch(plt.Circle((x - 1, y), r, color='dimgray', zorder=2))
        else:
            w = self.box_obstacle_width
            for [x, y, h] in self.obstacles:
                ax.add_patch(plt.Rectangle((x - w / 2 - 1, y - h / 2), w, h, color='dimgray', zorder=2))

        # Player
        player_pos = (self.t, 3 + 2 * self.pos)
        ax.add_patch(plt.Circle(player_pos, self.player_size, color='royalblue', zorder=2))
        # v = np.array([1, self.vel])
        # v = v / np.linalg.norm(v) * 0.25
        # ax.add_patch(plt.arrow(player_pos[0], player_pos[1], v[0], v[1], width=0.025, color='royalblue', zorder=2))
        return fig, ax

    def plot_visited_states(self, name, trajectories, step=0, next_state=None, rollout_indices=None):
        fig, ax = self.plot()
        n_trajs = len(trajectories)
        obstacles = np.asarray(self.obstacles)

        for i, traj in enumerate(trajectories):
            # Drop the velocity, keep t (i.e. x-pos and y-pos)
            if rollout_indices is not None:
                states = traj[rollout_indices[i]-1:, :-1]
                Xs = (states[:, 0] + 1) * 5 * 9/10
                Ys = 3 + 2*states[:, 1]
                plt.plot(Xs, Ys, 'ro', linestyle="--")
                plt.scatter(Xs, Ys, c="red")

            states = traj[:, :-1]
            if rollout_indices is not None:
                states = states[:rollout_indices[i]]    

            Xs = (states[:, 0] + 1) * 5 * 9/10
            Ys = 3 + 2*states[:, 1]
            
            plt.scatter(Xs, Ys)
            plt.plot(Xs, Ys, 'bo', linestyle="--")
       
        if next_state is not None:
            X = [(next_state[0] + 1) * 5 * 9/10]
            Y = [3 + 2*next_state[1]]
            plt.scatter(X, Y, c='purple')
            Xs = [step, X[0]]
            Ys = [3+2*self.pos, Y[0]]
            plt.plot(Xs, Ys, 'kD', linestyle="-")

        # Obstacles
        if self.obstacle_mode == "circles":
            for [x, y, r] in self.obstacles:
                ax.add_patch(plt.Circle((x - 1, y), r+0.01, color='dimgray', zorder=2))
        else:
            w = self.box_obstacle_width
            for [x, y, h] in self.obstacles:
                ax.add_patch(plt.Rectangle((x - w / 2 - 1, y - h / 2), w, h, color='dimgray', zorder=2))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = TwoDimNavigationEnv()
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action, render=True)
        if done:
            env.reset()
    env.close()
