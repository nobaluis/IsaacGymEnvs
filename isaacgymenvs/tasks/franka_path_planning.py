import numpy as np
import os
import math
import torch
import torchvision
import random

from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from .utils import *
from torch import Tensor
from typing import Dict, Tuple, List
from torchvision import transforms


class FrankaPathPlanning(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.sim = None

        self.max_episode_length = self.cfg['env']['episodeLength']
        self.termination_height = self.cfg['env']['terminationHeight']
        self.table_dimensions = self.cfg['env']['tableDimensions']
        self.env_spacing = self.cfg['env']['envSpacing']
        self.action_scale = self.cfg['env']['actionScale']
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.data_dir = self.cfg['env']['dataDir']
        self.img_width = self.cfg['env']['imgWidth']
        self.img_height = self.cfg['env']['imgHeight']
        self.gray_scale = self.cfg['env']['grayScale']
        self.path_len = self.cfg['env']['pathLen']
        self.debug_viz = self.cfg['env']['enableDebugViz']
        self.log_metrics = self.cfg['env']['enableLogging']
        self.asset_root = self.cfg['env']['asset']['assetRoot']
        self.asset_file = self.cfg['env']['asset']['assetFileNameFranka']

        # Create dicts to pass to reward function
        self.reward_settings = {
            'scales': [self.cfg['env']['distanceRewardScale'],
                       self.cfg['env']['orientationRewardScale'],
                       self.cfg['env']['controlRewardScale']]
        }

        # Controller type
        self.control_type = self.cfg['env']['controlType']
        assert self.control_type in {'osc', 'ik'}, \
            'Invalid control type specified. Must be one of: {osc, ik}'

        # Dimensions
        self.img_channels = 2 if self.gray_scale else 4
        self.cfg['env']['numObservations'] = self.img_width * self.img_height * self.img_channels
        # actions include: delta EEF if OSC or IK (6)
        self.cfg['env']['numActions'] = 6

        # Init variables
        self.states = dict()  # will be dict filled with relevant states to use for reward calculation
        self.actions = None
        self.j_eef = None  # Jacobian for end effector
        self.mm = None  # Mass matrix
        self.rb_states = None  # State of all rigid bodies (n_envs, n_bodies, 13)
        self.dof_state_tensor = None
        self.actor_state_tensor = None
        self.sphere_actor_id = None
        self.sphere_state = None
        # self.root_states = None
        # self.initial_root_states = None
        self.dof_pos = None
        self.dof_vel = None
        self.pos_action = None
        self.effort_action = None
        self.points_vec3 = None
        self.points_quat = None
        self.targets_tensor = None
        self.offset_rot = None
        self.points_array = None
        self.target_point = None
        self.franka_dof_lower_limits = None
        self.franka_dof_upper_limits = None
        self.franka_effort_limits = None
        self.franka_default_dof_pos = None
        self.texture_dict = dict()  # texture buffer
        self.sphere_pos_list = []  # sphere random positions list
        self.sphere_rot_list = []  # sphere random rotations (z-axis) list
        self.sphere_rad_list = []  # sphere random radius list

        # rotation offset between texture and sphere
        self.sphere_rot_offset = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -math.pi / 2)

        # Torchvision transformations
        self.to_uint8 = transforms.ConvertImageDtype(torch.uint8)
        self.to_float32 = transforms.ConvertImageDtype(torch.float32)
        self.to_gray_scale = transforms.Grayscale(num_output_channels=1)
        self.scale_depth = transforms.Lambda(lambda d: torch.clamp(-d, min=0.0, max=1.0))

        # Init VecTask
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # IK Gains
        self.damping = to_torch([0.05] * 6, device=self.device)

        # OSC Gains
        self.kp = 150.
        self.kd = 2.0 * np.sqrt(self.kp)
        self.kp_null = 10.
        self.kd_null = 2.0 * np.sqrt(self.kp_null)

        # self.kp = to_torch([150.] * 6, device=self.device)
        # self.kd = 2 * torch.sqrt(self.kp)
        # self.kp_null = to_torch([10.] * 7, device=self.device)
        # self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits (OCS)
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        # Load texture buffer
        self.textures_files = get_textures(data_path=self.data_dir)  # get texture files
        for i in range(self.num_envs * 2):
            # pick random texture from files
            _texture_id = np.random.choice(self.textures_files)
            _texture_file = f'{self.data_dir}/textures/{_texture_id}'
            # create texture handle
            self.texture_dict[_texture_id.rstrip('.png')] = self.gym.create_texture_from_file(self.sim, _texture_file)

        # Reset all environments
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self) -> None:
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self) -> None:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_viz_geoms(self) -> None:
        self.axes_geom = gymutil.AxesGeometry(0.5)
        self.point_geom = gymutil.WireframeSphereGeometry(0.005, 24, 24, None, color=(1, 0, 0))

    def _sphere_rand_props(self, sphere_rad: float, table_px=0.425, table_py=0.0) -> Tuple[float, float, float, float]:
        _, _, table_z = self.table_dimensions
        rot = np.random.uniform(0., math.pi/2)  # 0, 2.0 * math.pi
        px = table_px + np.random.uniform(-0.02, 0.01)
        py = table_py + np.random.uniform(-0.03, 0.03)
        pz = table_z + 0.5 * (sphere_rad * 2)
        return rot, px, py, pz

    def _create_envs(self) -> None:
        num_per_row = int(math.sqrt(self.num_envs))
        env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_x, table_y, table_z = self.table_dimensions
        table_asset = self.gym.create_box(self.sim, table_x, table_y, table_z, asset_options)

        # load asset robot
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)
        if franka_asset is None:
            raise Exception('Failed to load the franka asset')

        # create viz geoms
        if self.debug_viz:
            self._create_viz_geoms()

        # config franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        franka_effort_limits = franka_dof_props['effort']
        # franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # send franka limits to torch
        self.franka_dof_lower_limits = to_torch(franka_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(franka_upper_limits, device=self.device)
        self.franka_effort_limits = to_torch(franka_effort_limits, device=self.device)

        # config franka dof mode
        if self.control_type == 'ik':  # position mode
            franka_dof_props['driveMode'][:7].fill(gymapi.DOF_MODE_POS)
            franka_dof_props['stiffness'][:7].fill(400.0)
            franka_dof_props['damping'][:7].fill(40.0)
        elif self.control_type == 'osc':  # force mode
            franka_dof_props['driveMode'][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props['stiffness'][:7].fill(0.0)
            franka_dof_props['damping'][:7].fill(0.0)

        # gripper dof mode
        franka_dof_props['driveMode'][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props['stiffness'][7:].fill(800.0)
        franka_dof_props['damping'][7:].fill(40.0)

        # set default dof states and positions targets
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]  # franka arm
        default_dof_pos[7:] = franka_lower_limits[7:]  # gripper closed
        self.franka_default_dof_pos = torch.from_numpy(default_dof_pos).to(device=self.device)

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state['pos'] = default_dof_pos
        self.default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)

        # get end effector (franka panda_hand) index
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict['panda_hand']

        # start position for franka
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        # start pose for table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.425, 0.0, 0.5 * table_z)

        # start pose for sphere
        sphere_pose = gymapi.Transform()

        # camera sensor properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.img_width
        cam_props.height = self.img_height
        cam_props.enable_tensors = True

        # camera pose
        cam_pose = gymapi.Transform()
        cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -np.pi / 2)  # rotate -90 over y-axis
        cam_pose.p = gymapi.Vec3(0, 0, 0.0584)  # panda_finger length

        # create envs
        self.envs = []
        self.hand_idxs = []
        # self.texture_points_list = []
        # self.sphere_idxs = []
        # self.spheres_rad_list = []
        # self.spheres_pos_list = []
        self.rgb_tensors = []
        self.depth_tensors = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, 'franka', i, 2)

            # add table
            self.gym.create_actor(env, table_asset, table_pose, 'table', i, 0)

            # sphere random props
            sphere_rad = np.random.uniform(0.075, 0.125)
            sphere_rot, sphere_px, sphere_py, sphere_pz = self._sphere_rand_props(sphere_rad)

            # create sphere asset
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            sphere_asset = self.gym.create_sphere(self.sim, sphere_rad, asset_options)

            # sphere pose
            sphere_pose.p = gymapi.Vec3(sphere_px, sphere_py, sphere_pz)
            sphere_pose.r = self.sphere_rot_offset * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), sphere_rot)

            # add sphere
            self.sphere_actor_id = self.gym.create_actor(env, sphere_asset, sphere_pose, 'sphere', i, 0)

            # save sphere props
            self.sphere_rad_list.append(sphere_rad)
            self.sphere_rot_list.append(gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -sphere_rot))
            self.sphere_pos_list.append(gymapi.Vec3(sphere_px, sphere_py, sphere_pz))

            # get global index f sphere in rigid body state tensor
            # sphere_idx = self.gym.get_actor_rigid_body_index(env, sphere_handle, 0, gymapi.DOMAIN_SIM)
            # self.sphere_idxs.append(sphere_idx)

            # add texture to sphere
            # texture_id = np.random.choice(self.textures_files)
            # texture_file = f'{self.data_dir}/textures/{texture_id}'
            # texture_points = get_trajectory(texture_id.rstrip('.png'), data_path=self.data_dir)
            # self.texture_points_list.append(texture_points)
            # texture = self.gym.create_texture_from_file(self.sim, texture_file)
            # self.gym.set_rigid_body_texture(env, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, texture)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, 'panda_hand', gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # add camera sensor
            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            lfinger_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, 'panda_leftfinger')
            self.gym.attach_camera_to_body(cam_handle, env, lfinger_handle, cam_pose, gymapi.FOLLOW_TRANSFORM)

            # obtain camera tensors
            _rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
            _depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_DEPTH)

            # wrap camera tensors in a pytorch tensors
            rgb_tensor = gymtorch.wrap_tensor(_rgb_tensor)  # (W, H, 4)
            depth_tensor = gymtorch.wrap_tensor(_depth_tensor)  # (W, H, 1)
            self.rgb_tensors.append(rgb_tensor)
            self.depth_tensors.append(depth_tensor)
        # setup data
        self.init_data()

    def init_data(self) -> None:
        # Get Jacobian tensor
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'franka')
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # Jacobian's elements corresponding to the end effector
        self.j_eef = jacobian[:, self.franka_hand_index - 1, :, :7]

        # Get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'franka')
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]  # only need elements corresponding to the franka arm

        # Get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # Get actor state tensor
        _actor_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_state_tensor = gymtorch.wrap_tensor(_actor_root_states).view(self.num_envs, -1, 13)  # (n_envs, 13)

        # Get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state_tensor = gymtorch.wrap_tensor(_dof_states).view(self.num_envs, -1, 2)  # (n_envs, n_dofs)

        self.dof_pos = self.dof_state_tensor[..., 0].unsqueeze(-1)
        self.dof_vel = self.dof_state_tensor[..., 1].unsqueeze(-1)

        # Set action tensors
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float, device=self.device)
        self.effort_action = torch.zeros_like(self.pos_action)

        # sphere state
        self.sphere_state = self.actor_state_tensor[:, self.sphere_actor_id, :]

        # target point per env
        self.target_point = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # TODO: move this to another method
        self.points_array = np.empty((self.num_envs, self.path_len, 7), dtype=np.float32)
        self.points_vec3 = np.empty(self.num_envs * self.path_len, dtype=gymapi.Vec3.dtype)
        self.points_quat = np.empty(self.num_envs * self.path_len, dtype=gymapi.Quat.dtype)
        self.targets_tensor = to_torch(self.points_array, device=self.device)

        # # 1-7 map texture points to world coordinates
        # # 1 - get sphere positions
        # sphere_pos_array = np.array(self.sphere_pos_list, dtype=np.float32)  # (num_envs, 3)
        #
        # # 2 - get mapping from texture coord to sphere coord
        # self.points_array = np.empty((self.num_envs, self.path_len, 7), dtype=np.float32)
        # for i in range(self.num_envs):
        #     # For each path convert (s, t) -> (x, y, z, q0...q3)
        #     self.points_array[i, :, :] = trajectory_mapping(self.texture_points_list[i], self.sphere_rad_list[i])
        #
        # # 3 - position offset from world -> sphere -> point in path
        # self.points_array[:, :, :3] += sphere_pos_array[:, np.newaxis, :]  # translation from origin to point
        # self.points_array = self.points_array.reshape((self.num_envs * self.path_len, 7))  # 2D array of points
        #
        # # 4 - convert points pos to Vec3
        # self.points_vec3 = np.empty(self.num_envs * self.path_len, dtype=gymapi.Vec3.dtype)
        # self.points_vec3['x'] = self.points_array[:, 0]
        # self.points_vec3['y'] = self.points_array[:, 1]
        # self.points_vec3['z'] = self.points_array[:, 2]
        #
        # # 5 - convert points orn to Quat
        # self.points_quat = np.empty(self.num_envs * self.path_len, dtype=gymapi.Quat.dtype)
        # self.points_quat['x'] = self.points_array[:, 3]
        # self.points_quat['y'] = self.points_array[:, 4]
        # self.points_quat['z'] = self.points_array[:, 5]
        # self.points_quat['w'] = self.points_array[:, 6]  # Try swap w and z
        #
        # # 6 - transformation offset: rotate to sphere normal + translation over z axis
        # transform = gymapi.Transform()
        # pos_offset = gymapi.Vec3(0, 0, -0.15)  # translation over z
        # for i in range(self.num_envs * self.path_len):
        #     transform.p = self.points_vec3[i]
        #     transform.r = self.points_quat[i]
        #     new_pos = transform.transform_point(pos_offset)
        #     self.points_array[i, :3] = np.array([new_pos.x, new_pos.y, new_pos.z])
        #
        # # 7 - convert to tensor
        # self.targets_tensor = to_torch(self.points_array.reshape((self.num_envs, self.path_len, 7)), device=self.device)
        #
        # # rotation to point inside the sphere (for panda_hand)
        # rot_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi)
        # self.offset_rot = torch.stack(self.num_envs * [torch.tensor([rot_q.x, rot_q.y, rot_q.z, rot_q.w])])\
        #     .to(self.device).view((self.num_envs, 4))
        # # --------------------------------------------------------------------------------------------------------------

    def _update_states(self) -> None:
        # target points
        points_idx = self.target_point.unsqueeze(-1)  # (num_envs, 1)
        index = points_idx.repeat(1, 7)  # (num_envs, 7)
        index = index.unsqueeze(1)  # (num_envs, 1, 7)
        target_poses = torch.gather(self.targets_tensor, 1, index)
        # end-effector state
        hand_pos = self.rb_states[self.hand_idxs, :3]
        hand_rot = self.rb_states[self.hand_idxs, 3:7]
        hand_vel = self.rb_states[self.hand_idxs, 7:]
        self.states.update({
            'hand_pos': hand_pos,
            'hand_rot': hand_rot,
            'hand_vel': hand_vel,
            'target_pos': target_poses[..., :3].view((self.num_envs, 3)),
            'target_rot': target_poses[..., 3:7].view((self.num_envs, 4))
        })

    def _refresh(self) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)  # actor root
        self.gym.refresh_dof_state_tensor(self.sim)  # dof state
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # rigid body state
        self.gym.refresh_jacobian_tensors(self.sim)  # jacobian
        self.gym.refresh_mass_matrix_tensors(self.sim)  # mass matrix
        self._update_states()  # targets

    def compute_reward(self, actions: Tensor) -> None:
        self.rew_buf[:], self.reset_buf[:], self.target_point[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.target_point, actions, self.states,
            self.reward_settings['scales'], self.max_episode_length, self.path_len)

    def _update_sensor_tensor(self) -> None:
        """Update the observation buffer with the image from both rgb and depth sensors

        :return: void
        """
        for i in range(self.num_envs):
            _color_tensor = self.to_float32(self.rgb_tensors[i][..., :3].T)  # (3, W, H) [0,1]
            if self.gray_scale:
                _color_tensor = self.to_gray_scale(_color_tensor)  # (1, W, H)
            _depth_tensor = self.scale_depth(self.depth_tensors[i].T).unsqueeze(0)  # (1, W, H) [0,1]

            # debug
            # to_uint8 = transforms.ConvertImageDtype(torch.uint8)
            # torchvision.io.write_png(to_uint8(_color_tensor).cpu(), f'imgs/{i:02}_{self.progress_buf[i]:03}_rgb.png')
            # torchvision.io.write_png(to_uint8(_depth_tensor).cpu(), f'imgs/{i:02}_{self.progress_buf[i]:03}_depth.png')
            # -----
            # self.obs_buf[i, :] = torch.flatten(torch.cat((_color_tensor, _depth_tensor), dim=0))  # (CH * W * H)
            self.obs_buf[i, ...] = torch.cat((_color_tensor, _depth_tensor), dim=0).T  # (W, H, CH)

    def compute_observations(self) -> Tensor:
        self._refresh()  # update state and target tensors
        # render sensors and refresh camera tensors
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # get observations from sensors
        self._update_sensor_tensor()  # update obs_buf
        # stops accessing the tensors to prevent memory hazards
        self.gym.end_access_image_tensors(self.sim)
        return self.obs_buf

    def _compute_target_points(self, env_id: int, sphere_points: np.ndarray):
        t1 = gymapi.Transform()  # Trans obj for ee offset
        t2 = gymapi.Transform()  # Trans obj for sphere offset
        ee_offset = gymapi.Vec3(0., 0., -0.15)  # translation over z  -0.15

        t2.p = self.sphere_pos_list[env_id]
        t2.r = self.sphere_rot_list[env_id]

        for point_id in range(self.path_len):
            idx = env_id * self.path_len + point_id
            px, py, pz = sphere_points[point_id, :3]
            qx, qy, qz, qw = sphere_points[point_id, 3:]
            point_vec3 = gymapi.Vec3(px, py, pz)
            point_quat = gymapi.Quat(qx, qy, qz, qw)

            # translate offset distance over normal vector
            t1.p = point_vec3
            t1.r = point_quat
            ee_pos = t1.transform_point(ee_offset)

            # translate to sphere position and rotate sphere rotation
            ee_pos = t2.transform_point(ee_pos)
            sp_pos = t2.transform_point(point_vec3)
            sp_rot = t2.r * point_quat

            # update target points
            self.points_vec3[idx] = (sp_pos.x, sp_pos.y, sp_pos.z)
            self.points_quat[idx] = (sp_rot.x, sp_rot.y, sp_rot.z, sp_rot.w)
            self.targets_tensor[env_id, point_id, :] = torch.tensor([ee_pos.x, ee_pos.y, ee_pos.z,
                                                                     sp_rot.x, sp_rot.y, sp_rot.z, sp_rot.w])

    def reset_idx(self, env_ids: Tensor) -> None:
        # Reset aget
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        pos[:, -2:] = self.franka_dof_lower_limits[-2:]  # gripper closed

        # reset dof_state_tensor
        self.dof_pos[env_ids, ...] = pos.unsqueeze(-1)
        self.dof_vel[env_ids, ...] = torch.zeros_like(self.dof_vel[env_ids])

        # set position and effort control to current position
        self.pos_action[env_ids, :] = pos
        self.effort_action[env_ids, :] = torch.zeros_like(pos)

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.pos_action),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.effort_action),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state_tensor),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # reset sphere and target points
        for env_id in env_ids:
            env = self.envs[env_id]
            texture_id, texture = random.choice(list(self.texture_dict.items()))
            # color = np.random.choice(range(256), size=3) / 255
            color = np.array([1.0, 1.0, 1.0])
            color_vec = gymapi.Vec3(color[0], color[1], color[2])
            # set sphere texture and color
            self.gym.set_rigid_body_texture(env, self.sphere_actor_id, 0, gymapi.MESH_VISUAL_AND_COLLISION, texture)
            self.gym.set_rigid_body_color(env, self.sphere_actor_id, 0, gymapi.MESH_VISUAL, color_vec)
            # set new sphere pose
            sphere_rad = self.sphere_rad_list[env_id]  # isn't possible to change the rad after creating the asset
            sphere_rot, sphere_px, sphere_py, sphere_pz = self._sphere_rand_props(sphere_rad)
            self.sphere_rot_list[env_id] = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -sphere_rot)
            self.sphere_pos_list[env_id] = gymapi.Vec3(sphere_px, sphere_py, sphere_pz)
            new_sphere_pose = gymapi.Transform()
            new_sphere_pose.p = gymapi.Vec3(sphere_px, sphere_py, sphere_pz)
            new_sphere_pose.r = self.sphere_rot_offset * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), sphere_rot)
            # set new sphere state
            self.sphere_state[env_id, 0] = new_sphere_pose.p.x
            self.sphere_state[env_id, 1] = new_sphere_pose.p.y
            self.sphere_state[env_id, 2] = new_sphere_pose.p.z
            self.sphere_state[env_id, 3] = new_sphere_pose.r.x
            self.sphere_state[env_id, 4] = new_sphere_pose.r.y
            self.sphere_state[env_id, 5] = new_sphere_pose.r.z
            self.sphere_state[env_id, 6] = new_sphere_pose.r.w
            # compute new target points
            texture_points = get_trajectory(texture_id, data_path=self.data_dir)  # (s, t)
            sphere_points = trajectory_mapping(texture_points, self.sphere_rad_list[env_id])  # (x,y,z,rot)
            self._compute_target_points(env_id, sphere_points)

        # update actor state tensor
        multi_env_spheres_ids_int32 = self.global_indices[env_ids, 2].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.actor_state_tensor),
            gymtorch.unwrap_tensor(multi_env_spheres_ids_int32), len(multi_env_spheres_ids_int32))
        self.target_point[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose: Tensor) -> Tensor:
        """ Compute the force control using the Operational Space Control (OSC)
        :param dpose: error between actual and desired pose
        :return: control signal with the torques values
        """
        # Solve for Operational Space Control
        mm_inv = torch.inverse(self.mm)
        m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states['hand_vel'].unsqueeze(-1))

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        j_eef_inv = m_eef @ self.j_eef @ mm_inv
        u_null = self.kd_null * -self.dof_vel + self.kp_null * (
                (self.default_dof_pos_tensor.view(1, -1, 1) - self.dof_pos + np.pi) % (2 * np.pi) - np.pi)
        u_null = u_null[:, :7]
        u_null = self.mm @ u_null
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self.franka_effort_limits[:7].unsqueeze(0), self.franka_effort_limits[:7].unsqueeze(0))
        return u

    def _compute_ik(self, dpose: Tensor) -> Tensor:
        """Compute the inverse kinematics using the damped least squares method

        :param dpose: error between actual and desired pose
        :return: Control signal with the dof configuration
        """
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def pre_physics_step(self, actions: Tensor):
        # copy actions to device
        self.actions = actions.clone().to(self.device)
        # scale actions
        u_arm = self.actions * self.cmd_limit / self.action_scale
        u_arm = u_arm.unsqueeze(-1)  # (num_envs, 6, 1)

        # compute control
        if self.control_type == 'ik':
            self.pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self._compute_ik(dpose=u_arm)
        elif self.control_type == 'osc':
            self.effort_action[:, :7] = self._compute_osc_torques(dpose=u_arm)
        # deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

    def post_physics_step(self):
        # computing rewards and observations
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # plot viz
        if self.viewer and self.debug_viz:
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.target_point[env_ids] = 0
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # position and orn for the point in the sphere
                pt_idx = i * self.path_len + self.target_point[i]
                sp_x, sp_y, sp_z = self.points_vec3[pt_idx]
                sq_x, sq_y, sq_z, sq_w = self.points_quat[pt_idx]
                sphere_point_pos = gymapi.Transform()
                sphere_point_pos.p = gymapi.Vec3(sp_x, sp_y, sp_z)
                sphere_point_pos.r = gymapi.Quat(sq_x, sq_y, sq_z, sq_w)
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], sphere_point_pos)
                # panda_hand target pos
                hand_target_pos = gymapi.Transform()
                target_point_pos = self.targets_tensor[i, self.target_point[i], :3]
                hand_target_pos.p = gymapi.Vec3(target_point_pos[0], target_point_pos[1], target_point_pos[2])
                gymutil.draw_lines(self.point_geom, self.gym, self.viewer, self.envs[i], hand_target_pos)

        # # set the next target point when ee reaches the threshold
        # pos_err = self.states['target_pos'] - self.states['hand_pos']
        # pos_dis = torch.linalg.norm(pos_err, ord=2, dim=1)
        # self.target_point = torch.where(pos_dis <= 0.03, torch.add(self.target_point, 5), self.target_point)
    # end task definition

# Pytorch JIT scripts

@torch.jit.script
def orientation_error(q_desired: Tensor, q_current: Tensor) -> Tensor:
    cc = quat_conjugate(q_current)
    q_r = quat_mul(q_desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def compute_franka_reward(reset_buf: Tensor, progress_buf: Tensor, target_point: Tensor, actions: Tensor,
                          states: Dict[str, Tensor], reward_scales: List[float], max_episode_length: float,
                          path_len: int) -> Tuple[Tensor, Tensor, Tensor]:
    # compute errors
    pos_err = states['target_pos'] - states['hand_pos']
    rot_err = orientation_error(states['target_rot'], states['hand_rot'])

    # reward scale params
    alpha = reward_scales[0]
    beta = reward_scales[1]
    gamma = reward_scales[2]

    # compute distance reward
    pos_dis = torch.linalg.norm(pos_err, ord=2, dim=1)
    pos_exp = torch.exp(-pos_dis)
    pos_reward = 1.0 / (1.0 + pos_dis ** 2)
    pos_reward *= pos_reward

    # compute orientation reward
    orn_dis = torch.linalg.norm(rot_err, ord=2, dim=1)
    orn_reward = 1.0 / (1.0 + orn_dis ** 2)
    orn_reward *= orn_reward

    # regularization on the actions (summed for each environment)
    ctl_reward = torch.sum(actions ** 2, dim=-1)

    # X1 position reward - control penalty
    reward = pos_reward - 0.01 * ctl_reward
    # + orientation bonus when pos_dis < threshold
    reward = torch.where(pos_dis <= 0.05, reward + 0.85 * orn_reward, reward)
    # + extra bonus for reach target
    reward = torch.where(pos_dis <= 0.03, reward + 10, reward)

    # reward based on position and orientation - control penalty
    # reward = pos_exp * orn_reward - 0.01 * ctl_reward
    # # bonus for reach the target
    # reward = torch.where(pos_dis <= 0.03, reward + 10, reward)

    # set the next target point when ee reaches the threshold
    target_point = torch.where(pos_dis <= 0.03, torch.add(target_point, 5), target_point)

    # reset conditions
    # when episode ends
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    # when path is completed
    reset_buf = torch.where((target_point >= path_len - 1), torch.ones_like(reset_buf), reset_buf)
    # when ee go outside the table dim [0.6, 1.0, 0.3]
    p_x = (states['hand_pos'][:, 0]).squeeze(-1)
    p_y = (states['hand_pos'][:, 1]).squeeze(-1)
    p_z = (states['hand_pos'][:, 2]).squeeze(-1)
    reset_buf = torch.where((p_x < 0.125) | (p_x > 0.725), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((p_y < -0.5) | (p_y > 0.5), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(p_z < 0.3, torch.ones_like(reset_buf), reset_buf)
    return reward, reset_buf, target_point

