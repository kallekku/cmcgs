#%%
import gym
import dm_control
from dm_control import suite
import time
import copy

from dmc2gym import DMC2GYMWrapper
from simulator import SimulatorWrapper
# %%
env = suite.load('walker', 'walk', {'random': 1})
env = DMC2GYMWrapper(env)
env = SimulatorWrapper(env)
env.reset()
#%%
action = env.action_space.sample()
obs = env.step(action)

physics_state1 = env.physics.get_state()
checkpoint = copy.deepcopy(env)

# rollout 10 times:
for _ in range(10):
    obs, r, d, _ = env.step(action)

#%%
env2 = copy.deepcopy(checkpoint)

physics_state2 = env2.physics.get_state()

for _ in range(10):
    obs2, r2, d2, _ = env2.step(action)

#%%
###### test deepcopy physics ###### 
checkpoint2_physics = copy.deepcopy(env2.physics)
physics_state3 = env2.physics.get_state()
for _ in range(10):
    obs2, r2, d2, _ = env2.step(action)

env2.physics = copy.deepcopy(checkpoint2_physics)
physics_state4 = env2.physics.get_state()

for _ in range(10):
    obs3, r3, d3, _ = env2.step(action)

#%% 
######## test deepcopy both physics and tasks
env3 = copy.deepcopy(checkpoint)
physics_state5 = copy.deepcopy(env3.physics)
physics_state5_np = env3.physics.get_state()
task_state5 = copy.deepcopy(env3.task)

for _ in range(10):
    obs3, r3, d3, _ = env3.step(action)


env3.physics = copy.deepcopy(physics_state5)
env3.task = copy.deepcopy(task_state5)
physics_state5_np_ = env3.physics.get_state()
for _ in range(10):
    obs4, r4, d4, _ = env3.step(action)
