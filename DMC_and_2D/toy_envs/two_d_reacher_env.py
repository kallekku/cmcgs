import numpy as np
import gym
from gym import spaces
from matplotlib import rc
from matplotlib import pyplot as plt

DEBUG_NO_GOAL = False
DEBUG_KEEP_GOAL_FIXED = True
DEBUG_RENDER_EXTRA_STUFF = False
DEBUG_PLOT_VISITED_STATES = False


def get_dist_point_to_line_segment(A, B, E):
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Case 1
    if AB_BE > 0:
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = np.sqrt(x * x + y * y)
    # Case 2
    elif AB_AE < 0:
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = np.sqrt(x * x + y * y)
    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = np.sqrt(x1 * x1 + y1 * y1)
        reqAns = np.abs(x1 * y2 - y1 * x2) / mod

    return reqAns


class TwoDimReacherEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
        , 'video.frames_per_second': 20
    }

    def __init__(self, env_name="2d-reacher", seed=0):
        self.fps = 20
        self.dt = 1 / self.fps
        self.max_ep_len_sec = 10
        self._max_episode_steps = self.max_ep_len_sec * self.fps

        self.screen_width = 960
        self.screen_height = 540
        self.block_size = 100
        self.unit_size = 50
        self.goal_size = 0.15
        self.end_effector_size = 5
        self.max_vel = 0.75
        self.DEBUG_PLOT_VISITED_STATES = False

        if "three-poles" in env_name:
            self.lengths = [2, 2, 1]
        elif "four-poles" in env_name:
            self.lengths = [1.75, 1.5, 1.25, 1]
        elif "five-poles" in env_name:
            self.lengths = [2, 1.75, 1.5, 1.25, 1]
        elif "fifteen-poles" in env_name:
            self.lengths = [0.5] * 15
        elif "thirty-poles" in env_name:
            self.lengths = [0.25] * 30
        self.angles = None
        self.velocities = None

        self.goal_pos = np.array([0.55, 0.7])
        # self.goal_pos = np.array([0.75, 0.55])

        self.obs_pos = []
        if "medium" in env_name:
            self.obs_pos = [
                np.array([0.6, 0.7, 0.02])
            ]
        elif "hard" in env_name:
            self.obs_pos = [
                np.array([0.6, 0.7, 0.02])
                , np.array([0.3, 0.4, 0.025])
                # , np.array([0.6, 0.35, 0.015])
                # , np.array([0.75, 0.3, 0.075])
            ]

        self.timestep = 0

        high = np.ones(len(self.lengths) * 2) * 2
        self.observation_space = spaces.Box(high * 0, high, dtype=np.float32)
        high = np.ones(len(self.lengths))
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.random = np.random.RandomState(seed)
        self.viewer = None

        self.spec = {'id': '2d-reacher'}

        # self.reset()

    def step(self, action):
        action = np.clip(action, -1, 1)

        # Update velocities and then the angles
        self.velocities += action * self.dt
        self.velocities = np.clip(self.velocities, -1, 1)
        for d in range(len(self.lengths)):
            angles = np.copy(self.angles)
            angles[d:] += self.velocities[d:] * self.max_vel * self.dt
            for a in range(len(self.lengths)):
                if angles[a] < 0:
                    angles[a] += 2
                if angles[a] > 2:
                    angles[a] -= 2

            collided = False
            # Check for collisions
            pos_a = np.array([0.5, 0.5])
            fwd = np.array([1, 0])
            for a in range(len(self.lengths)):
                sin, cos = np.sin(angles[a] * np.pi), np.cos(angles[a] * np.pi)
                fwd = np.array([fwd[0] * cos - fwd[1] * sin, fwd[0] * sin + fwd[1] * cos])
                pos_b = pos_a + self.lengths[a] * np.array([fwd[0] * self.unit_size / self.screen_width, fwd[1] * self.unit_size / self.screen_height])
                for obs_i, [obs_x, obs_y, obs_r] in enumerate([np.array([0.5, 0.5, 0.01])] + self.obs_pos):
                    if a == 0 and obs_i == 0:
                        # Ignore the collision of the first arm with robot's base
                        continue
                    if get_dist_point_to_line_segment(pos_a, pos_b, np.array([obs_x, obs_y])) < obs_r:
                        collided = True
                        break
                if collided:
                    break
                pos_a = pos_b

            if collided:
                self.velocities[d] = 0
            else:
                # No collisions detected. Update self.angles
                self.angles = angles
                break

        done = False

        rew_ac = np.exp(-10 * np.power(action, 2))  # Encourage lower acceleration
        rew_ac = np.mean(rew_ac)

        # Reached the goal?
        ee_pos = self.get_end_effector_pos()
        dist2goal = np.linalg.norm(ee_pos - self.goal_pos)
        rew_goal = 0#np.exp(-250 * np.power(dist2goal, 2))

        if not DEBUG_NO_GOAL and round(dist2goal, 3) < 0.015:
            rew_goal = 1
            done = True

        self.timestep += 1
        if self.timestep >= self._max_episode_steps:
            done = True

        # reward = 0.01 * rew_ac + 0.99 * rew_goal
        reward = rew_goal
        return self._get_obs(), reward, done, {'dist2goal':dist2goal, 'ee_pos':ee_pos, "goal_pos":self.goal_pos}

    def reset(self):
        self.angles = np.zeros(len(self.lengths))
        self.velocities = np.zeros(len(self.lengths))
        self.timestep = 0
        if not DEBUG_NO_GOAL and not DEBUG_KEEP_GOAL_FIXED:
            max_radius_h = np.sum(self.lengths) * self.unit_size / self.screen_width
            max_radius_v = np.sum(self.lengths) * self.unit_size / self.screen_height
            while True:
                angle = self.random.random() * 2 * np.pi
                pos_on_circle_area = np.array([np.cos(angle) * max_radius_h, np.sin(angle) * max_radius_v])
                rnd = 0.05 + 0.9 * self.random.random()
                self.goal_pos = pos_on_circle_area * rnd + 0.5
                has_overlap_with_obstacles = False
                for [x, y, r] in self.obs_pos:
                    if np.linalg.norm(self.goal_pos - np.array([x, y])) < r:
                        has_overlap_with_obstacles = True
                        break
                if not has_overlap_with_obstacles:
                    break

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate((self.angles, self.velocities))

    def get_state(self):
        return [self.angles, self.velocities, self.timestep, self.goal_pos]

    def set_state(self, state):
        self.angles = np.array(state[0])
        self.velocities = np.array(state[1])
        self.timestep = state[2]
        self.goal_pos = np.array(state[3])

    def get_end_effector_pos(self):
        pos = np.array([0.5, 0.5])
        fwd = np.array([1, 0])
        for a in range(len(self.lengths)):
            sin, cos = np.sin(self.angles[a] * np.pi), np.cos(self.angles[a] * np.pi)
            fwd = np.array([fwd[0] * cos - fwd[1] * sin, fwd[0] * sin + fwd[1] * cos])
            pos += self.lengths[a] * np.array([fwd[0] * self.unit_size / self.screen_width, fwd[1] * self.unit_size / self.screen_height])
        return pos

    def plot(self, path=None):
        rc('figure', figsize=(self.screen_width // self.block_size, self.screen_height // self.block_size))
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim([0, self.screen_width])
        ax.set_ylim([0, self.screen_height])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Base
        ax.add_patch(plt.Circle((self.screen_width/2, self.screen_height/2), 0.25*self.unit_size, color='dimgray'))

        # Goal
        ax.add_patch(plt.Circle((self.goal_pos[0] * self.screen_width, self.goal_pos[1] * self.screen_height), self.goal_size*self.unit_size, color='green'))

        # Obstacles
        for [x, y, r] in self.obs_pos:
            ax.add_patch(plt.Circle((x * self.screen_width, y * self.screen_height), r * self.screen_width, color='black'))
        
        # Arms
        pos = np.array([0.5 * self.screen_width, 0.5 * self.screen_height])
        fwd = np.array([1, 0])
        up = np.array([0, 1])
        fwd_orig = fwd.copy() 
        h = self.unit_size / 4
        for a in range(len(self.lengths)):
            sin, cos = np.sin(self.angles[a] * np.pi), np.cos(self.angles[a] * np.pi)
            fwd = np.array([fwd[0] * cos - fwd[1] * sin, fwd[0] * sin + fwd[1] * cos])
            up = np.array([up[0] * cos - up[1] * sin, up[0] * sin + up[1] * cos])
            w = self.lengths[a] * self.unit_size
           
            bottom_left = pos - h * up / 2
            #top_left = pos + h * up / 2
            #top_right = pos + h * up / 2 + w * fwd
            bottom_right = pos - h * up / 2 + w * fwd

            #circles = [
            #    bottom_left,
            #    top_left,
            #    top_right,
            #    bottom_right,
            #]
           
            rot_angle = np.arcsin((bottom_right[1] - bottom_left[1]) / (fwd_orig[0] * h * 2)) * 180 / np.pi
            if bottom_right[0] - bottom_left[0] < 0:
                rot_angle = 180 - rot_angle

            ax.add_patch(plt.Rectangle(bottom_left, w, h, color='dimgray', fill=None, angle=rot_angle))
            #for c in circles:
            #    ax.add_patch(plt.Circle(c, 3, color='blue'))
            pos += w * fwd
    
        # End effector
        end_effector_pos = self.get_end_effector_pos()
        ax.add_patch(plt.Circle((end_effector_pos[0] * self.screen_width, end_effector_pos[1] * self.screen_height), self.end_effector_size, color='red'))

        if path is not None:
            plt.savefig(path)
            print("Saved image to path {}".format(path))
            print("Exiting")
            exit(0)
        else:
            plt.show()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        self.viewer.geoms.clear()

        # Base
        base = rendering.make_circle(0.25 * self.unit_size, filled=True)
        base.add_attr(rendering.Transform(translation=(self.screen_width / 2, self.screen_height / 2)))
        base.set_color(0.5, 0.5, 0.5)
        self.viewer.add_geom(base)

        # Goal
        if not DEBUG_NO_GOAL:
            goal = rendering.make_circle(self.goal_size * self.unit_size, filled=True)
            goal.add_attr(rendering.Transform(
                translation=(self.goal_pos[0] * self.screen_width, self.goal_pos[1] * self.screen_height)))
            goal.set_color(0, 0.75, 0)
            self.viewer.add_geom(goal)

        # Obstacles
        for [x, y, r] in self.obs_pos:
            goal = rendering.make_circle(r * self.screen_width, filled=True)
            goal.add_attr(rendering.Transform(translation=(x * self.screen_width, y * self.screen_height)))
            goal.set_color(0, 0, 0)
            self.viewer.add_geom(goal)

        # Arms
        pos = np.array([0.5 * self.screen_width, 0.5 * self.screen_height])
        fwd = np.array([1, 0])
        up = np.array([0, 1])
        h = self.unit_size / 4
        for a in range(len(self.lengths)):
            sin, cos = np.sin(self.angles[a] * np.pi), np.cos(self.angles[a] * np.pi)
            fwd = np.array([fwd[0] * cos - fwd[1] * sin, fwd[0] * sin + fwd[1] * cos])
            up = np.array([up[0] * cos - up[1] * sin, up[0] * sin + up[1] * cos])
            w = self.lengths[a] * self.unit_size
            obs = rendering.make_polygon([pos - h * up / 2, pos + h * up / 2, pos + h * up / 2 + w * fwd, pos - h * up / 2 + w * fwd], filled=False)
            obs.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(obs)
            pos += w * fwd

        # End-effector
        # ee = rendering.make_circle(self.goal_size * self.unit_size, filled=True)
        # ee.add_attr(rendering.Transform(translation=(pos[0], pos[1])))
        # ee.set_color(0, 0, 0.75)
        # self.viewer.add_geom(ee)

        if DEBUG_RENDER_EXTRA_STUFF:
            # Draw a line to the end-effector
            ee = self.get_end_effector_pos()
            line = rendering.Line([0.5 * self.screen_width, 0.5 * self.screen_height]
                                  , [ee[0] * self.screen_width, ee[1] * self.screen_height])
            line.set_color(0, 0.75, 0)
            self.viewer.add_geom(line)

            # Draw the area in which the goal pos is chosen
            max_radius_h = np.sum(self.lengths) * self.unit_size / self.screen_width
            max_radius_v = np.sum(self.lengths) * self.unit_size / self.screen_height
            prev = None
            for a in range(360):
                angle = a / 180 * np.pi
                pos = np.array([np.cos(angle) * max_radius_h, np.sin(angle) * max_radius_v]) + 0.5
                if prev is not None:
                    line = rendering.Line([prev[0] * self.screen_width, prev[1] * self.screen_height]
                                          , [pos[0] * self.screen_width, pos[1] * self.screen_height])
                    line.set_color(0, 0.75, 0)
                    self.viewer.add_geom(line)
                prev = pos

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = TwoDimReacherEnv("thirty-poles-hard")
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        # env.render('human' if render_online else 'rgb_array')
        env.render('human')
        if done:
            env.reset()
    env.close()
