from collections import OrderedDict

import numpy as np
import numbers
import copy
import robosuite as suite
import cv2
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import  CylinderObject, Cube, CubeVisual, BoxObject, MilkObject, MilkVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjmod_custom import DynamicsModder, LightingModder, TextureModder
from robosuite.controllers import load_controller_config

""" 
This is Pusher task
"""


# Lift --> Pusher
class Pusher_Box(SingleArmEnv):
    """
    This class corresponds to the pushing task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(0.1, 5e-2, 1e-4),
            # table_friction=(1, 5e-3, 1e-4),
            # table_friction=(1e-2, 1, 1),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=True,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            # frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=368,
            # camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            renderer="mujoco",
            renderer_config=None,
            threshold=0.06,
            default_table_size=False,
            deterministic_reset=False,
            goal_state = None,
            random_goal = False,
    ):
        
        # [May16] camera shift
        """ == [May 16] For camera shift_x, shift_y: self.state_keys_affectedByCameraShift == """
        self.state_keys_affectedByCameraShift = [
                                                'plate1_pos', 'plate1_quat', 
                                                'plate2_pos', 'plate2_quat'
                                            ]   # Extracted from the step() in env_pusher.py

        self.camera_name = camera_names
        # settings for table top
        if default_table_size is True:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (2, 2, 0.05)
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.deterministic_reset = deterministic_reset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        
        # define cube size
        self.cube_size = np.array([0.0508, 0.067, 0.0256])

        # Define default object position and range
        if goal_state is None:
            self.default_target_pos = np.asarray([0.115, 0.044, 0.8255])
        else:
            self.default_target_pos = np.zeros(3)
            self.default_target_pos[:2] = goal_state[:2]
            self.default_target_pos[2] = 0.8255
            
        
        self.default_cube_pos = np.asarray([-0.15, 0.0, 0.8]) # real
        
        self.default_pusher_pos = np.asarray([-0.45, 0.0, 0.8])
        self.default_velocity_delay = 1.0 # 0.85
        
        if goal_state is None:
            self.default_target_rotation = 0.72
        else:
            self.default_target_rotation = goal_state[2]

        self.random_goal = random_goal
        
        self.default_cube_pos_range = np.asarray([0.00, 0.00])
        self.default_velocity_delay_range = np.asarray([0.0, 0.0])
        self.default_pusher_pos_range = np.asarray([0.0, 0.0])
        self.default_target_pos_range = np.asarray([0.0, 0.0])
        self.default_target_rotation_range = np.asarray([0.0, 0.0])
        
        # Define object position and range
        self.cube_init_pos = self.generate_positions(self.default_cube_pos, self.default_cube_pos_range)
        self.velocity_delay = self.generate_positions(self.default_velocity_delay, self.default_velocity_delay_range)
        self.pusher_init_pos = self.generate_positions(self.default_pusher_pos, self.default_pusher_pos_range)
        self.target_rotation = self.generate_positions(self.default_target_rotation, self.default_target_rotation_range)
        self.target_pos = self.generate_positions(self.default_target_pos, self.default_target_pos_range)
        if random_goal:
            self.target_pos, self.target_rotation = self.sample_position()
        
        
        # Define reaching target threshold
        self.threshold = threshold

        self.dynamics_moder = None
        self.lighting_moddder = None
        self.context = {}
        self._seed = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        
    def reset_obj_init_positions(self):
        self.cube_init_pos = self.generate_positions(self.default_cube_pos, self.default_cube_pos_range)
        self.velocity_delay = self.generate_positions(self.default_velocity_delay, self.default_velocity_delay_range)
        self.pusher_init_pos = self.generate_positions(self.default_pusher_pos, self.default_pusher_pos_range)

        if self.random_goal:
            self.target_pos, self.target_rotation = self.sample_position()
        else:
            self.target_rotation = self.generate_positions(self.default_target_rotation, self.default_target_rotation_range)
            self.target_pos = self.generate_positions(self.default_target_pos, self.default_target_pos_range)
    
    @staticmethod
    def generate_positions(pose_mean, pose_range):
        
        assert len(pose_range) == 2
        # print ("pose_mean", pose_mean, "type", type(pose_mean))
        if isinstance(pose_mean, numbers.Number):
            new_pose = pose_mean 
            new_pose += np.random.uniform(pose_range[0], pose_range[1])
            return new_pose
        elif type(pose_mean) is np.ndarray and len(pose_mean) == 3:
            pose = pose_mean.copy()
            pose[:2] += np.random.uniform(pose_range[0], pose_range[1], 2)
            return pose
        else:
            raise ValueError("Pose Mean Needs to be either 1 or 3 dimensional")
         
    def sample_position(self):
        goal = np.array([[0.115, 0.044, 0.825], [0.28, 0, 0.825]])
        rotation = np.array([0.72, 0.0])
        index = np.random.randint(0, 2)
        return goal[index], rotation[index]
        

    def reward(self, action):
        reward = 0.0

        # use a shaping reward
        if self.reward_shaping:
            
            r_goal = 5 * self.get_cube_to_goal()
            r_orient = 2 * self.get_orientation_delta()
            
            reward_goal_orient = 1 / (r_goal+r_orient)
            reward += reward_goal_orient         
            
            if self._check_success():
                reward = 2 * reward
        else:
            if self._check_success():
                reward = 2.25


        return float(reward)
    
        

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        self.robots[0].init_qpos = [0.00662393, 0.97028886,
                                    0.00229584, 1.82422673, -0.0059355, 0.35976223, -1.55589321]
        
        # print ("setting robot initial position:", self.robots[0].init_qpos)
        
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=[0,0,0],
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

           
        self.cube = BoxObject(
                    name="cube",
                    size=(self.cube_size[0], self.cube_size[1], self.cube_size[2]),
                    rgba=[1, 1, 0, 1],
                    obj_type="all",
                    density=1000,
                    solref = [-10000, -40],
                    friction = [0.045, 5e-3, 1e-4],
                    duplicate_collision_geoms=True,
        )
        
        self.pusher = BoxObject(
                        name = "pusher",
                        size = (0.085, 0.0125, 0.0125),
                        rgba = [1, 0, 0, 1],
                        density = 1800,
                        obj_type = "all",
                        solref = [-10000, -10],
                        friction = [0.03, 5e-3, 1e-4],
                        duplicate_collision_geoms=True,
        )
        
        self.goal_visual = BoxObject(
                    name="cube_visual",
                    size=(self.cube_size[0], self.cube_size[1], self.cube_size[2]),
                    rgba=[0, 1, 0, 0.4],
                    obj_type="visual",
                    joints=None,
                    duplicate_collision_geoms=True,
        )
        

        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler",
                mujoco_objects=self.cube,
                x_range=[self.cube_init_pos[0],self.cube_init_pos[0]],
                y_range=[self.cube_init_pos[1],self.cube_init_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0,
                rotation_axis="z",
            ))
        
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler_1",
                mujoco_objects=self.pusher,
                x_range=[self.pusher_init_pos[0],self.pusher_init_pos[0]],
                y_range=[self.pusher_init_pos[1],self.pusher_init_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0256,
                rotation_axis="z",
            ))
        # print ('target_pos_in', self.target_pos)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="VisualSampler",
                mujoco_objects=self.goal_visual,
                x_range=self.target_pos[0] *
                np.ones(2),  # EEF Position x=-0.45
                y_range=self.target_pos[1] *
                np.ones(2),  # EEF position y=-0.04
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.01,
                rotation = self.target_rotation,
                rotation_axis = "z",
            ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube,self.goal_visual, self.pusher],
        )
        

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(
            self.cube.root_body)
        self.goal_visual_body_id = self.sim.model.body_name2id(
            self.goal_visual.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")


            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # print('reserting object positions')
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.goal_visual_body_id] = obj_pos
                    self.sim.model.body_quat[self.goal_visual_body_id] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate(
                        [np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has moved to target position.

        Returns:
            bool: True if cube is at the target place
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        # cube_pos_y = self.sim.data.body_xpos[self.plate1_body_id][1]
        target_pos = copy.deepcopy(self.target_pos)
        threshold = self.threshold
        # check the distance between cube and target position
        # TODO: Need to check the target threshold
        
        orientation_delta = self.get_orientation_delta()
        return np.linalg.norm(cube_pos[:2] - target_pos[:2]) < threshold-0.03 and orientation_delta < threshold
        # return np.abs(cube_pos_x - target_pos_x) + np.abs(cube_pos_y - target_pos_y) < threshold

    def get_cube_to_goal(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        target_pos = copy.deepcopy(self.target_pos)
        return np.linalg.norm(cube_pos - target_pos)
    
    def get_orientation_delta(self):
        # cube_orientation = self.sim.data.body_xquat[self.cube_body_id]
        # z_rotation = self.get_z_rotation(cube_orientation)
        # goal_rotation = self.sim.data.body_xquat[self.goal_visual_body_id]
        z_rotation = self.get_z_rotation(self.get_obs()[3:7])
        # print ('goal_rotation', z_rotation, z_rotation - self.target_rotation)
        return np.linalg.norm(z_rotation - self.target_rotation)
    
    def get_obs(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id].flatten()
        cube_quat = convert_quat(
            self.sim.data.body_xquat[self.cube_body_id], to="xyzw").flatten()
        obs = np.concatenate([
            cube_pos, cube_quat
        ])
        
        return obs.astype(np.float32)
    
    def get_z_rotation(self, quat):
        return np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1-2*(quat[1]**2 + quat[2]**2))
    
    def step(self, action, writer = None):
        obs, reward, done, info = super().step(action)
        info.update(obs)
        info["success"] = 1.0 if self._check_success() else 0.0
        info["cube_dist_to_goal"] = self.get_cube_to_goal()
        
        info["orientation_error"] = self.get_orientation_delta()
        
        info["z_rotation"] = self.get_z_rotation(self.get_obs()[3:7])
        return obs, reward, done, info

    def reset(self):
        if self._seed is not None:
            np.random.seed(self._seed)
        
        # set object positions if they are spcified in the context
        context = copy.deepcopy(self.context)
        # if "init@cube@position" in context:
        #     self.cube_init_pos = context["init@plate1@position"] 
        # if "init@target@position" in context:      
        #     self.target_pos = context["init@target@position"]
        # if "init@knob@velocity_delay" in context:
        #     self.velocity_delay =context["init@knob@velocity_delay"]
        
        # generate init positions
        self.reset_obj_init_positions()

        # restore to default parameter
        obs = super().reset()
        
        """ == [May 16] adjust the observed plate_1 / plate_2 position based on camera shift [1/2] == """
        # state_keys_affectedByCameraShift = ['plate1_pos', 'plate1_quat', 'plate2_pos', 'plate2_quat']
        # for key in self.state_keys_affectedByCameraShift:
        #     obs[key][0] += self.context.get("init@camera@shift_x", 0.0)
        #     obs[key][1] += self.context.get("init@camera@shift_y", 0.0)
            
        
        self.dynamics_modder = DynamicsModder(sim=self.sim)
        self.lighting_modder = LightingModder(sim=self.sim)
        self.texture_modder = TextureModder(sim=self.sim)
        

        self.dynamics_modder.set_context(context)
        self.lighting_modder.set_context(context)
        self.texture_modder.set_context(context)
        
        self.sim.model.opt.integrator = 1
        if self._seed is not None:
            np.random.seed(self._seed)
        # print ('internal reset', self.target_rotation)
        return obs

    def print_context(self):
        # Define function for easy printing
        context = self.get_context()
        for k, v in context.items():
            print(f"{k}: {v}")

    def get_context(self):
        """ === [May 16] [New get_context()] ===
        Include shift_x, shift_y
        """ 
        dynamic_context = self.dynamics_modder.get_context()
        lighting_context = self.lighting_modder.get_context()
        texture_context = self.texture_modder.get_context()
        cube_init_pos = self.cube_init_pos.copy()
        # velocity_delay = self.velocity_delay.copy()
        velocity_delay = self.velocity_delay
        # print("$$$$$$$$$",velocity_delay)
        
        return {
            **dynamic_context, 
            **lighting_context, 
            **texture_context,
            "init@cube@position": cube_init_pos,
            "init@knob@velocity_delay": velocity_delay,
            
            # """=== [May 16] camera shift ==="""
            # "init@camera@shift_x": self.context.get("init@camera@shift_x", 0.0),
            # "init@camera@shift_y": self.context.get("init@camera@shift_y", 0.0),
        }

    def set_context(self, context):
        self.context = copy.deepcopy(context)
        # print (self.target_rotation, 'context')

    def set_seed(self, seed):
        if seed is not None:
            if isinstance(seed, int):
                self._seed = seed
            else:
                self._seed = None
