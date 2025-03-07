from collections import OrderedDict

import numpy as np
import numbers
import copy

import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    CylinderObject,
    BoxObject,
    KnobObject,
    CupObject,
    Floor_1_object,
    Floor_2_object,
)
from robosuite.models.objects import (
    X_Right_Wall_Object,
    X_Left_Wall_Object,
    Y_Front_Wall_Object,
    Y_Back_Wall_Object,
    Block,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    UniformRandomSampler,
    ObjectPositionSampler,
)
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjmod_custom import DynamicsModder, LightingModder
from robosuite.controllers import load_controller_config

""" 
This is Pusher task
"""


# Lift --> Pusher
class Pusher(SingleArmEnv):
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
        camera_heights=720,
        camera_widths=1280,
        # camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        threshold=0.06,
        default_table_size=False,
        deterministic_reset=False,
    ):

        # settings for table top
        if default_table_size is True:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (2, 2, 0.05)
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.6))

        self.deterministic_reset = deterministic_reset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # Define default object position and range
        self.default_target_pos = np.asarray([0.29, 0, 0.8])

        self.default_cup_pos = np.asarray([-0.15, 0.0, 0.8])  # sim

        self.default_knob_pos = np.asarray([-0.35, 0.0, 0.8])

        self.default_target_pos_range = np.asarray([0, 0])
        self.default_cup_pos_range = np.asarray([-0.01, 0.01])
        self.default_cup_shift_y = 0.04
        self.default_knob_pos_range = np.asarray([0.0, 0.0])
        
        ## use this to whether the puck position is defined
        self.defined_puck_pos = None

        # Define object position and range
        self.target_pos = self.generate_positions(
            self.default_target_pos, self.default_target_pos_range
        )
        self.cup_init_pos = self.generate_puck_pos(
            self.default_cup_pos, self.default_cup_pos_range, self.defined_puck_pos
        )
        self.knob_init_pos = self.generate_positions(
            self.default_knob_pos, self.default_knob_pos_range
        )

        # Define reaching target threshold
        self.threshold = threshold

        # modder to modify context
        """
        Previous Version: self.dynamics_moder = None, self.lighting_moder = None
        """
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
        self.target_pos = self.generate_positions(
            self.default_target_pos, self.default_target_pos_range
        )
        self.cup_init_pos = self.generate_puck_pos(
            self.default_cup_pos, self.default_cup_pos_range, self.defined_puck_pos
        )
        self.knob_init_pos = self.generate_positions(
            self.default_knob_pos, self.default_knob_pos_range
        )
        # print("self.cup_init_pos: ", self.cup_init_pos)

    def generate_puck_pos(self, puck_pos, puck_pos_range, defined_puck_pos):
        if defined_puck_pos is not None:
            assert len(defined_puck_pos) == 2 and np.abs(defined_puck_pos[1]) > 0.005, defined_puck_pos
            return defined_puck_pos
        assert len(puck_pos_range) == 2
        shift_y = self.default_cup_shift_y
        puck_left_right = np.random.choice([-shift_y, shift_y])
        puck = puck_pos.copy()
        puck[1] += puck_left_right
        offset = np.random.uniform(puck_pos_range[0], puck_pos_range[1], 2)
        puck[:2] += offset
        return puck
    
    
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
            pose[1] += np.random.uniform(pose_range[0], pose_range[1], 1)
            return pose
        else:
            raise ValueError("Pose Mean Needs to be either 1 or 3 dimensional")

    def reward(self, action):
        """
        Reward function for the task.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # use a shaping reward
        if self.reward_shaping:
            # reward_dist = -self.get_cup_to_goal()
            # reward_nd = 10 * reward_dist
            # reward += reward_nd

            ## shiqi reward: 
            # reaching reward
            # dist = self.get_cup_to_goal_x()
            
            dist = self.get_cup_to_goal()
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += 2 * reaching_reward
            
            ## xilun reward:
            # reward = -self.get_cup_to_goal()
            
            # terminal reward
            if self._check_success():
                # reward += reward_nd * 0.5 + 2.25
                reward += 0.25
        else:
            if self._check_success():
                reward = 2.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return float(reward)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)
        self.robots[0].init_qpos = [
            0.00662393,
            0.97028886,
            0.00229584,
            1.82422673,
            -0.0059355,
            0.35976223,
            -1.55589321,
        ]
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=[0, 0, 0],
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.knob = KnobObject(name="knob")

        self.cup = CupObject(name="cup")

        self.floor1 = Floor_1_object(name="floor1")
        self.floor2 = Floor_2_object(name="floor2")

        self.goal_visual = CylinderObject(
            name="cube_visual",
            size=(self.threshold, 0.01),
            rgba=[0, 1, 0, 0.4],
            # material=redwood,
            obj_type="visual",
            joints=None,
            duplicate_collision_geoms=True,
        )

        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="KnobSampler",
                mujoco_objects=self.knob,
                x_range=[self.knob_init_pos[0], self.knob_init_pos[0]],
                y_range=[self.knob_init_pos[1], self.knob_init_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=[0, 0, 0.8],
                z_offset=-0.05,
            )
        )

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler",
                mujoco_objects=self.cup,
                x_range=[self.cup_init_pos[0], self.cup_init_pos[0]],
                y_range=[self.cup_init_pos[1], self.cup_init_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=[0, 0, 0.8],
                z_offset=-0.055,
            )
        )
        # print ("init_pos", self.cup_init_pos)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="VisualSampler",
                mujoco_objects=self.goal_visual,
                x_range=self.target_pos[0] * np.ones(2),  # EEF Position x=-0.45
                y_range=self.target_pos[1] * np.ones(2),  # EEF position y=-0.04
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=[0, 0, 0.8],
                z_offset=0.01,
            )
        )

        self.xwall_right = X_Left_Wall_Object(
            name="x_left_wall",
        )

        self.xwall_left = X_Right_Wall_Object(
            name="x_right_wall",
        )

        self.ywall_front = Y_Front_Wall_Object(
            name="y_front_wall",
        )

        self.ywall_back = Y_Back_Wall_Object(
            name="y_back_wall",
        )
        self.block = Block(name="block")

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[
                self.knob,
                self.cup,
                self.goal_visual,
                self.xwall_left,
                self.xwall_right,
                self.ywall_front,
                self.ywall_back,
                self.floor1,
                self.floor2,
                self.block,
            ],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        # Additional object references from this env
        self.knob_body_id = self.sim.model.body_name2id(self.knob.root_body)
        self.cup_body_id = self.sim.model.body_name2id(self.cup.root_body)
        self.goal_visual_body_id = self.sim.model.body_name2id(
            self.goal_visual.root_body
        )

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

            # cup observables
            @sensor(modality=modality)
            def cup_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cup_body_id])

            @sensor(modality=modality)
            def cup_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.cup_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def gripper_to_cup_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cup_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cup_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cup_pos, cup_quat, gripper_to_cup_pos]
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
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

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
                gripper=self.robots[0].gripper, target=self.cube
            )

    def _check_success(self):
        """
        Check if cube has moved to target position.

        Returns:
            bool: True if cube is at the target place
        """

        cube_pos = self.sim.data.body_xpos[self.cup_body_id]
        target_pos = self.target_pos
        threshold = self.threshold
        # check the distance between cube and target position
        return np.linalg.norm(cube_pos[:2] - target_pos[:2]) < threshold

    def check_contact_with_wall(self):
        """
        Check if the cup is in contact with the wall

        Returns:
            bool: True if cup is in contact with the wall
        """
        cup_pos = self.sim.data.body_xpos[self.cup_body_id]
        if cup_pos[0] < 0.2 and np.abs(cup_pos[1]) > 0.18:
            return True
        return False
    
    def check_on_edge(self):
        """
        Check if the cup is on the edge of the table

        Returns:
            bool: True if cup is on the edge of the table
        """
        cup_pos = self.sim.data.body_xpos[self.cup_body_id]
        if np.abs(cup_pos[1]) < 0.03:
            return True
        return False
    
    def get_cup_to_goal(self):
        cube_pos = self.sim.data.body_xpos[self.cup_body_id]
        target_pos = self.target_pos
        return np.linalg.norm(cube_pos[:2] - target_pos[:2])
    
    def get_cup_to_goal_x(self):
        cube_pos = self.sim.data.body_xpos[self.cup_body_id]
        target_pos = self.target_pos
        return np.abs(cube_pos[:1] - target_pos[:1])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # print ("knob pos is: ", self.sim.data.body_xpos[self.knob_body_id])
        # print ("cup pos is: ", self.sim.data.body_xpos[self.cup_body_id])
        info.update(obs)
        info["success"] = 1.0 if self._check_success() else 0.0
        info["dist_to_goal"] = self.get_cup_to_goal()
        return obs, reward, done, info

    def reset(self):
        # if self._seed is not None:
        #     np.random.seed(self._seed)

        # generate init positions
        self.reset_obj_init_positions()
        # set object positions if they are spcified in the context
        context = copy.deepcopy(self.context)

        # restore to default parameter
        obs = super().reset()

        self.dynamics_modder = DynamicsModder(sim=self.sim)
        self.lighting_modder = LightingModder(sim=self.sim)

        self.dynamics_modder.set_context(context)
        self.lighting_modder.set_context(context)
        
        # if "position@puck@init_pos" in context:
        #     self.defined_puck_pos = context["position@puck@init_pos"]

        self.sim.model.opt.integrator = 1
        # self.dynamics_modder.mod("knob_main","inertia", 10)
        # self.dynamics_modder.update()
        # if self._seed is not None:
        #     np.random.seed(self._seed)
        # TODO: obs might change after context changed, may need to get the newest obs
        # obs = self._get_observations()

        return obs

    def print_context(self):
        # Define function for easy printing
        context = self.get_context()
        for k, v in context.items():
            print(f"{k}: {v}")

    def get_context(self):
        dynamic_context = self.dynamics_modder.get_context()
        lighting_context = self.lighting_modder.get_context()

        return {
            **dynamic_context,
            **lighting_context,
            # "position@puck@init_pos": self.cup_init_pos[:2],
        }

    def set_context(self, context):
        self.context = copy.deepcopy(context)
        # print ("The context is: ", type(self.context))
        # return self.reset()

    def set_seed(self, seed):
        if seed is not None:
            if isinstance(seed, int):
                self._seed = seed
            else:
                self._seed = None
