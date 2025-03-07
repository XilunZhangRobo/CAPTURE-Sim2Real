from robosuite import load_controller_config
from robosuite.environments.manipulation.scoop import Scooping
from robosuite.controllers import controller_factory

import gym
import numpy as np
import time
import cv2
import imageio
import copy
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
import pickle as pkl
from scipy.spatial.transform import Rotation




class Scoop_Balance_mujoco(gym.Env):
    
    """
    A physics-based simulation environment using MuJoCo for a robotic scooping task.
    
    This environment provides a realistic simulation of a Kinova3 robot attempting to
    place a rod at a desired position. It uses the MuJoCo physics engine for accurate physical interactions.

    Key Features:
        - Full physics simulation with MuJoCo
        - Realistic robot kinematics and dynamics
        - Visual rendering capabilities
        - Detailed state tracking (positions, quaternions, rotations)
        - Models physical properties (mass, friction, gravity)

    Note: While more realistic, this environment is computationally intensive and may
    run slower than its simplified counterpart.
    """
    def __init__(self, render=False):
        self.render = render

        config = load_controller_config(default_controller="OSC_POSITION")

        self.camera_name = "agentview"
        self.env = Scooping(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names=self.camera_name,
            horizon=50,
            control_freq=20,
            initialization_noise=None,
            default_table_size=False,
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float32
        )

        self.observation_space = None

        self._seed = None
        self.context = {}

        self.obs = self.env.reset()

    def step(self, action, writer=None):
        done = False
        pos = []
        front_view = []
        success = 0
        quat = []
        rotation = []
        spatula_pos = []
        spatula_quat = []
        ## flip action to match the location
        # grasp_point = self.get_com(self.obs["rodbalance_pos"])
        grasp_point = action[0]
        self.set_obj_pos([grasp_point])
        for i in range(50):
            # obs, _, _, info = self.env.step(np.zeros(4))
            obs, _, _, info = self.env.step(np.array([0, 0.5, 0, 1]))
            if writer is not None:
                # cv2.imwrite("scoop_sim.png",  cv2.cvtColor(cv2.flip(info[self.camera_name + "_image"], 0), cv2.COLOR_RGB2BGR))
                # input()
                writer.append_data(cv2.flip(info[self.camera_name + "_image"], 0))

        pos.append(obs["rodbalance_pos"])
        quat.append(obs["rodbalance_quat"])
        spatula_pos.append(obs["spatula_pos"])
        spatula_quat.append(obs["spatula_quat"])

        # get the rotation alone the y axis
        euler_angles = Rotation.from_quat(obs["rodbalance_quat"]).as_euler(
            "xyz", degrees=False
        )
        rotation.append(euler_angles[0])

        if self.env._check_success():
            success = 1

        obs = self.get_obs()
        reward = self.get_reward(rotation)
        done = True

        info = {
            "reward": reward,
            "success": success,
            "render_image": front_view,
            "pos": pos,
            "quat": quat,
            "spatula_pos": spatula_pos,
            "spatula_quat": spatula_quat,
            "rotation": rotation,
        }

        return obs, reward, done, info

    def get_com(self, rod_pos):
        ## get mass of the objects
        print ("rod_pos", rod_pos)
        env_context = self.get_context()
        M_rod = env_context["dynamic@rodbalance_main@mass"]  # mass of the rod
        x_rod = rod_pos[1]  # position of the rod
        M_cube = env_context["dynamic@rodbalance_rod_com@mass"]  # mass of the cube
        cube_pose = env_context["init@rodbalance@com"]
        x_cube = cube_pose * self.env.rod_size[1] + rod_pos[1]
        center_of_mass = (M_rod * x_rod + M_cube * x_cube) / (M_rod + M_cube)
        print ('center of mass', center_of_mass)
        grasp_point = (x_rod + center_of_mass) / self.env.rod_size[1]
        print ("grasp point", grasp_point)
        return grasp_point

    def set_obj_pos(self, action):
        rod_balance_size = self.env.rod_size
        rod_left = self.obs["rodbalance_pos"][0] - rod_balance_size[1]
        rod_right = self.obs["rodbalance_pos"][0] + rod_balance_size[1]
        contact_point_dist = rod_left + (rod_right - rod_left) * (action[0] + 1) / 2
        z_offset = self.env.rod_size[2] + 0.01 + self.env.spatula.size[2]
        target_pos = [
            self.obs["spatula_pos"][0],
            self.obs["spatula_pos"][1] + contact_point_dist,
            self.obs["spatula_pos"][2] + z_offset,
        ]
        pose = self.env.sim.data.get_joint_qpos("rodbalance_joint0")
        pose[:3] = target_pos
        self.env.sim.data.set_joint_qpos("rodbalance_joint0", pose)

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)

    def get_obs(self):
        # return self.env.get_obs()
        return None

    def get_reward(self, rotation):
        if np.abs(rotation) < np.radians(5):
            return 1
        return 0

    def get_context(self):
        # print (self.env)
        return self.env.get_context()

    def set_context(self, context):
        self.env.set_context(context)
        self.context = context

    def print_context(self):
        context = self.get_context()
        for k, v in context.items():
            print(f"{k}: {v}")
    
    def set_env_params(self, initial_pos):
        pass

    @staticmethod
    def get_state_traj(info):
        state_traj = info["rotation"]
        modified_state = np.zeros_like(state_traj)
        threahold = np.radians(8)
        modified_state[state_traj < -threahold] = -1
        modified_state[state_traj > threahold] = 1
        # print(modified_state, state_traj)
        return modified_state.reshape([1, 1])

    @staticmethod
    def get_env_params(info):
        return 0

class Scoop_Balance(gym.Env):
    """
    A simplified, abstract environment for the rod balancing task without physics simulation.
    
    This environment provides a fast, stable alternative to the physics-based simulation
    by using a purely mathematical model. It's designed for rapid prototyping and testing
    of balancing strategies.

    Key Features:
        - Fast execution with no physics simulation
        - Simple state representation (-1, 0, 1 for tilt states)
        - Deterministic behavior
        - Easy to debug and modify
        - Perfect for initial algorithm development

    The environment determines success by comparing the action to the center of mass,
    using a threshold of ±0.05 from the optimal point.
    
    Note: The model trained used this environment has been tested in both simulation and real world. 
    (It has similiar/better performance than the mujuco version in both simulation and real world)
    """
    def __init__(self, render=False):
        self.render = render

        self.action_space = gym.spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float32
        )

        self.observation_space = None
        self.context = {"init@rodbalance@com": 0}
        

    def step(self, action, writer=None):
        center_of_mass = self.context["init@rodbalance@com"]
        if np.abs(center_of_mass - action[0]) < 0.05:
            obs = np.array([0])
        elif center_of_mass < action[0]:
            obs = np.array([1])
        else:
            obs = np.array([-1])
        
        reward = 1 if obs == 0 else 0
        done = True
        

        info = {
            "reward": reward,
            "obs": obs,
        }

        return self.get_obs(), reward, done, info


    def reset(self):
        return self.get_obs()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)

    def get_obs(self):
        return None

    def get_reward(self, rotation):
        if np.abs(rotation) < np.radians(8):
            return 1
        return 0

    def get_context(self):
        return copy.deepcopy(self.context)

    def set_context(self, context):
        self.context = copy.deepcopy(context)

    def print_context(self):
        context = self.get_context()
        for k, v in context.items():
            print(f"{k}: {v}")
    
    def set_env_params(self, initial_pos):
        pass

    @staticmethod
    def get_state_traj(info):
        obs = info["obs"]
        return obs.reshape([1, 1])

    @staticmethod
    def get_env_params(info):
        return 0


def main():
    # This is for simulation evaluation purpose 
    env = Scoop_Balance_mujoco(render=True)
    set_random_seed(0)
    env.reset()
    test_context = env.get_context()

    test_context["init@rodbalance@com"] = 1
    env.set_context(test_context)
    env.reset()

    test_context_list = np.linspace(-1, 1, 10)
    test_context_list = [0.33]
    print (test_context_list)
    writer = imageio.get_writer("./dummyDemo_video.mp4", fps=env.env.control_freq)
    action_list = np.linspace(-1, 1, 10)
    action_list = [-0.3]
    for j in range(len(test_context_list)):
        print (f"Test context: {test_context_list[j]}")
        for i in range(len(action_list)):
            test_context["init@rodbalance@com"] = test_context_list[j]
            env.set_context(test_context)
            env.reset()
            action = np.array([action_list[i]])
            obs, reward, done, info = env.step(action, writer)
            print(env.get_state_traj(info))
            break
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
