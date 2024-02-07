#%%
from termios import IUCLC
from absl.testing import absltest
from simulator import SimulatorWrapper
from dmc2gym import DMC2GYMWrapper

from dm_control import suite
import numpy as np 

def flat_obs(o):
    return np.concatenate([o[k].flatten() for k in o], dtype=np.float32)

class SimulatorTest(absltest.TestCase):
    
    def _check_equal(self, a, b):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)

        else:
            self.assertEqual(a, b)
        
    def test_simulator_fidelity(self):
        """ Test whether the simulator match the ground truth."""
        env = suite.load('walker','walk', {'random': 0})
        
        simulator = suite.load('walker','walk', {'random': 0})
        simulator = DMC2GYMWrapper(simulator)
        simulator = SimulatorWrapper(simulator)

        for _ in range(10):
            true_timestep = env.reset()
            self.assertTrue(simulator.needs_reset)
            obs = simulator.reset()

            self.assertFalse(simulator.needs_reset)
            self._check_equal(flat_obs(true_timestep.observation), np.float32(obs))
            
            while not true_timestep.last():
                action = np.float32(np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, 
                            size=env.action_spec().shape))
                
                true_timestep = env.step(action)

                obs, rew, done, _ = simulator.step(action)
                
                self._check_equal(flat_obs(true_timestep.observation), np.float32(obs))
                self._check_equal(true_timestep.reward, rew)
                self._check_equal(true_timestep.last(), done)

    def test_checkpoint(self):
        """ Tests whether checkpointing restores the state correctly"""
        simulator = suite.load('walker','walk', {'random': 0})
        simulator = DMC2GYMWrapper(simulator)
        simulator = SimulatorWrapper(simulator) 

        simulator.reset()

        checkpoint = simulator.save_checkpoint()

        action = simulator.action_space.sample() 

        t1_obs, t1_r, t1_d, _ = simulator.step(action)

        # step the model once and load the checkpint.
        for i in range(5):
            t2_obs, t2_r, t2_d, _ = simulator.step(action) # rollout with the same action

        # test checkpoint    
        simulator.load_checkpoint(checkpoint)
        self.assertFalse(simulator.needs_reset)
        # self._check_equal(checkpoint.env_state.physics.get_state(), simulator.physics.get_state())
        
        tt1_obs, tt1_r, tt1_d, _ = simulator.step(action)
        # # test t1
        self._check_equal(tt1_obs, t1_obs)
        self._check_equal(tt1_r, t1_r)
        self._check_equal(tt1_d, t1_d)

        # test t2
        for i in range(5):
            tt2_obs, tt2_r, tt2_d, _ = simulator.step(action)
        
        self._check_equal(tt2_obs, t2_obs)
        self._check_equal(tt2_r, t2_r)
        self._check_equal(tt2_d, t2_d)
            


if __name__ == '__main__':
    absltest.main()
# %%
