from gym_env.robosuite import load_controller_config
from gym_env.robosuite.environments.manipulation.push_one_plate import Pusher

# from gym_env.utilities import ENV_CONTEXTS

import gym
import numpy as np
import time
import cv2
import imageio
import copy
import matplotlib.pyplot as plt


USE_BIRDVIEW = True

class PusherOneSingleAction(gym.Env):
    def __init__(self, render=False):
        self.render = render

        config = load_controller_config(
            default_controller="OSC_POSITION",
        )

        self.camera_name = "frontview"
        self.env = Pusher(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=self.render,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names=self.camera_name,
            control_freq=20,
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-0.28, -0.1, -0.01 * np.pi, 0.3]),
            high=np.array([-0.22, 0.1, 0.01 * np.pi, 0.5]),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.context = {}
        self._seed = None

    def step(self, action, writer=None):
        """
        :param action: 4D action vector (x, y location of the pusher, push angle, push velocity)
        """
        act = copy.deepcopy(action)
        rewards = []
        done = False
        state_keys = ["cup_pos"]
        states = {x: [] for x in state_keys}
        horizon = 40
        # print ('horizon:', horizon)
        camera_view = []
        tmp = self.env.sim.data.get_joint_qpos("knob_joint0")
        tmp[:2] = [action[0], action[1]]
        self.env.sim.data.set_joint_qpos("knob_joint0", tmp)

        """ === Modify the meaning of anlge (action[2]) === """
        cup_pos = self.env.sim.data.get_joint_qpos("cup_joint0")
        # Calculate the angle between the knob and the cup COM, as the reference angle
        y_diff = cup_pos[1] - action[1]
        x_diff = cup_pos[0] - action[0]
        reference_angle = np.arctan2(y_diff, x_diff)

        # update the action[2] with the reference angle
        act[2] = reference_angle + action[2]

        x_init, y_init, theta, v = action
        cup_pos = self.env.sim.data.get_joint_qpos("cup_joint0")

        ## Setting where knob needs to stop to match the real kinova pusher action
        move_distance = np.linalg.norm([x_init, y_init] - cup_pos[:2]) - 0.045
        push_time = move_distance / v
        on_edge = False
        wall_contact = False
        for _ in range(horizon):
            self.set_knob_vel(act, push_time)

            obs, reward, done, infos = self.env.step(np.zeros(4))
            
            if self.env.check_contact_with_wall():
                wall_contact = True
            if self.env.check_on_edge():
                on_edge = True
            
            if writer is not None:
                # cv2.imwrite("tmp.png",  cv2.cvtColor(cv2.rotate(infos[self.camera_name + "_image"], cv2.ROTATE_180), cv2.COLOR_RGB2BGR))
                # input()
                writer.append_data(
                    cv2.rotate(infos[self.camera_name + "_image"], cv2.ROTATE_180)
                )

            if self.render:
                time.sleep(0.01)

            # get state of the env
            for key in state_keys:
                states[key].append(obs[key])
            rewards.append(reward)
            if self.render:
                camera_view.append(infos[self.camera_name + "_image"].copy())
        
        if wall_contact:
            # print ("Wall contact")
            rewards[-1] += 0.1
        if on_edge:
            # print ("On edge")
            rewards[-1] -= 0.5
        
        # process observations
        observation = self.get_obs()
        done = True
        info = {
            "rewards": rewards,
            **states,
            "success": self.env._check_success(),
            "dist_to_goal": self.env.get_cup_to_goal(),
            "render_image": camera_view,
            "env_params": np.array(self.env.cup_init_pos[:2]),
            "contact_wall": self.env.check_contact_with_wall(),
            "on_edge": self.env.check_on_edge(),
        }

        return observation, rewards[-1], done, info

    def set_knob_vel(self, action, push_time):
        vel = np.zeros(6)  # x, y, z, around x, around y, around z
        if self.env.sim.data.time < push_time:
            x_init, y_init, theta, v = action
            # self.env.default_knob_pos = np.array([x_init, y_init, 0.8])
            velocity = v
            vel_x = velocity * np.cos(theta)
            vel_y = velocity * np.sin(theta)
            vel[0] = vel_x
            vel[1] = vel_y
        self.env.sim.data.set_joint_qvel("knob_joint0", vel)

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def render(self):
        pass

    def close(self):
        # print("Closing environment")
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)

    @staticmethod
    def normalize_state(pos, default_pos, pos_range):
        lb, ub = pos_range
        assert lb <= ub

        if lb == ub:
            return np.zeros_like(pos)
        else:
            pos_norm = pos - default_pos
            pos_norm = (pos_norm - lb) / (ub - lb)
            pos_norm = pos_norm * 2 - 1
            # print(
            #     "range: ",
            #     pos_range,
            #     "pos",
            #     pos,
            #     "default_pose",
            #     default_pos,
            #     "pos_norm",
            #     pos_norm,
            # )
            return pos_norm

    def get_obs(self):
        # get state and normalize
        cup_pos = self.env.cup_init_pos
        cup_range = np.array(self.env.default_cup_pos_range)
        shift_y = self.env.default_cup_shift_y
        cup_range[0] -= shift_y
        cup_range[1] += shift_y
        cup_pos_norm = self.normalize_state(
            cup_pos[:2], self.env.default_cup_pos[:2], cup_range
        )
        cup_pos_norm[0] *= cup_range[1] / (cup_range[1] - shift_y)
        # print("cup_pos_norm", cup_pos_norm)
        return cup_pos_norm[:2]
        target_pos = self.env.target_pos
        # target_pos_norm = self.normalize_state(
        #     target_pos, self.env.default_target_pos, self.env.default_target_pos_range
        # )
        # return np.concatenate([cup_pos_norm[:2], target_pos_norm[:2]])

    def get_context(self):
        return self.env.get_context()

    def set_context(self, context):
        self.env.set_context(context)
        self.context = context
        
    def set_env_params(self, initial_puck_pos):
        self.env.defined_puck_pos = initial_puck_pos
        # print ("using the defined puck pos: ", initial_puck_pos)

    @staticmethod
    def get_state_traj(info):
        state_traj = []
        for index in range(len(info["cup_pos"])):
            trajectory = info["cup_pos"][index][:2]
            trajectory = np.array(trajectory).flatten()
            state_traj.append(trajectory)
        return np.array(state_traj)
    
    @staticmethod
    def get_env_params(info):
        if "env_params" in info:
            return info["env_params"]
        else:
            return None


def main():
    # This is for simulation evaluation purpose 
    env = PusherOneSingleAction(render=1)

    env.reset()
    e_dr_real = env.get_context()

    e_dr_real["dynamic@floor1_table_collision@friction_sliding"] = 0.02
    e_dr_real["dynamic@floor2_table_collision@friction_sliding"] = 0.02
    e_dr_real["dynamic@knob_g0@damping_ratio"] = -3
    e_dr_real["dynamic@x_left_wall_g0@damping_ratio"] = -3
    e_dr_real["dynamic@x_right_wall_g0@damping_ratio"] = -3
    e_dr_real["dynamic@y_front_wall_g0@damping_ratio"] = -3

    env.set_context(e_dr_real)
    env.reset()
    writer = imageio.get_writer("./dummyDemo_video.mp4", fps=env.env.control_freq)
    damping_range = np.arange(-5, -30, -5)
    for i in range(1):
        e_dr_real["dynamic@knob_g0@damping_ratio"] = damping_range[i]
        env.set_context(e_dr_real)
        env.env.defined_puck_pos = [-0.147, 0.036]
        env.reset()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        action[0] = -0.22
        action[3] = 0.5
        action[1] = 0.0
        action[2] = 0.0
        obs, reward, done, info = env.step(action, writer)

        # plot the trajectory of cup
        cup_traj = env.get_state_traj(info)
        print ("cup_traj", cup_traj[0])
        # print (cup_traj.shape)
        ## save the trajectory
        plt.plot(cup_traj[:, 0], cup_traj[:, 1], label=f"{damping_range[i]}")

    env.close()
    writer.close()
    
    plt.savefig("cup_trajectory.png")

if __name__ == "__main__":
    main()
