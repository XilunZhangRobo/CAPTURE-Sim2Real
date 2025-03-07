from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, StepStoneObject, StepStone2Object, CylinderObject, CerealVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import copy
from robosuite.utils.mjmod_custom import DynamicsModder, LightingModder, TextureModder


class Drop(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

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
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="sideview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,# 256 
        camera_widths=512, # 256 
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        default_table_size=False,
        scaled_damping_ratio=None,
        release_height = None,
        threshold = 0.065, 
    ):
        # settings for table top
        if default_table_size is True:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.5, 1.5, 0.05)
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # set damping ratio
        # self.damping_ratio = np.exp(scaled_damping_ratio)
        self.damping_ratio = scaled_damping_ratio
        
        ## set goal range 
        self.threshold = threshold
        
        # set release height
        if release_height is None:
            self.release_height = 0.5
        else:
            self.release_height = release_height
            
        self.target_pos = np.array([0.03, 0, 0.96])
        
        self.dynamics_modder = None
        self.lighting_modder = None
        self.texture_modder = None
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

    def reward(self, action=None):
        
        reward = 0.0
        # Sparse completion reward
        # print ("check success")
        ball_pos = self.sim.data.body_xpos[self.cube_body_id]
        distance = np.linalg.norm(ball_pos - self.target_pos)
        reward -= 8 * distance
        if self._check_success():
            reward = 2.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BallObject(
            name="ball",
            size_min=[0.035, 0.035, 0.035],  # [0.015, 0.015, 0.015],
            size_max=[0.035, 0.035, 0.035],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
            # solref = [0.02, 1.0]
            solref = [-10000, self.damping_ratio], # Using absolute value
            # solref=[0.01, self.damping_ratio],  # 0.001 use exp range(-inf, 0)
            density=500,
        )
        self.goal_visual = CylinderObject(
            name="cube_visual",
            size=(self.threshold, 0.01),
            rgba=[0, 1, 0, 0.4],
            # material=redwood,
            obj_type="visual",
            joints=None,
            duplicate_collision_geoms=True,
        )

        # self.base1 = BoxObject(
        #     name="box1",
        #     # size_min=[0.08, 0.08, 0.08],
        #     # size_max=[0.09, 0.09, 0.09],
        #     size=[0.05, 0.05, 0.01],
        #     rgba=[0, 1, 0, 1],
        #     material=redwood,
        #     # solref=[0.1, 0.4],  # 0.001
        #     # density=0,
        #     joints=[{
        #         # "type": "free",
        #         # "damping": 1,
        #         # "stiffness": 1,
        #         # "limited": True,
        #         # "range": [-0.0, 0.0],
        #     }]
        # )
        self.base1 = StepStoneObject(
            name="step_stone",
        )
        self.base2 = StepStone2Object(
            name="step_stone_2",
        )

        self.placement_initializer = SequentialCompositeSampler(
                name="ObjectSampler")

        self.placement_initializer.append_sampler(UniformRandomSampler(
                name="BallSampler",
                mujoco_objects=self.cube,
                # x_range=[-0.03, 0.03],
                # y_range=[-0.03, 0.03],
                x_range=[-0.3, -0.3],
                y_range=[0, 0],
                rotation=0,
                rotation_axis="y",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                # z_offset=0.01,
                z_offset=self.release_height,  # this is to set the initial height of the object
            ))

            # self.placement_initializer.append_sampler(UniformRandomSampler(
            #     name="CubeSampler",
            #     mujoco_objects=self.base1,
            #     # x_range=[-0.03, 0.03],
            #     x_range=[-0.3, -0.3],
            #     y_range=[0.0, 0.0],
            #     rotation=-1,
            #     rotation_axis="y",
            #     ensure_object_boundary_in_range=False,
            #     ensure_valid_placement=False,
            #     reference_pos=self.table_offset,
            #     z_offset=0.1,
            # ))
            # print (self.target_pos)
        self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                name="VisualSampler",
                mujoco_objects=self.goal_visual,
                x_range=[self.target_pos[0], self.target_pos[0]],   # EEF Position x=-0.45
                y_range=[self.target_pos[1], self.target_pos[1]],   # EEF position y=-0.04
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=self.target_pos[2]-0.8,
            ))
            

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=[self.cube, self.base1],
            mujoco_objects=[self.cube, self.base1, self.base2, self.goal_visual],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
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

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_vel(obs_cache):
                return np.array(self.sim.data.get_body_xvelp(self.cube.root_body))

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

            sensors = [cube_pos, cube_quat, cube_vel, gripper_to_cube_pos]
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
        Check if the ball pass the target region

        Returns:
            bool: True if the ball hit the target region
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]

        ball_pos = self.sim.data.body_xpos[self.cube_body_id]
        distance = np.linalg.norm(ball_pos - self.target_pos)
        ball_vel = self.sim.data.get_body_xvelp(self.cube.root_body)[2]
        # cube is higher than the table top above a margin
        # print ("ball_pos", ball_pos, "distance", distance, "ball_vel", ball_vel)
        return ball_pos[2] >= self.target_pos[2] and distance < self.threshold and ball_vel < 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(obs)
        # print ("success") if self._check_success() else print ("")
        info["success"] = 1 if self._check_success() else 0
        return obs, reward, done, info

    def get_context(self):
        dynamic_context = self.dynamics_modder.get_context()
        lighting_context = self.lighting_modder.get_context()
        Texture_context = self.texture_modder.get_context()
        target_pos = self.target_pos.copy()
        
        """ == [Will include camera shift here]  == """
        
        return {
            **dynamic_context, 
            **lighting_context, 
            **Texture_context,
            "init@target@position": target_pos,
        }

    def set_context(self, context):
        self.context = copy.deepcopy(context)

    def set_seed(self, seed):
        if seed is not None:
            if isinstance(seed, int):
                self._seed = seed
            else:
                self._seed = None
                 
    def reset_obj_init_positions(self):
        self.target_pos = self.target_pos.copy()
        
    def reset(self):
        if self._seed is not None:
            np.random.seed(self._seed)
        # generate init positions
        self.reset_obj_init_positions()
        
        # set object positions if they are spcified in the context
        context = copy.deepcopy(self.context) 
        if "init@target@position" in context:      
            self.target_pos = context["init@target@position"]
        # restore to default parameter
        obs = super().reset()
        
        self.dynamics_modder = DynamicsModder(sim=self.sim)
        self.lighting_modder = LightingModder(sim=self.sim)
        self.texture_modder = TextureModder(sim=self.sim)

        self.dynamics_modder.set_context(context)
        self.lighting_modder.set_context(context)
        self.texture_modder.set_context(context)
        # self.sim.model.opt.integrator = 1
        
        if self._seed is not None:
            np.random.seed(self._seed)
            
        return obs