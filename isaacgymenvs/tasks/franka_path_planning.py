import numpy as np
import os
import torch

from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from utils import *
from torch import Tensor
from typing import Dict, Tuple, Any
from torchvision import transforms


class FrankaPathPlanning(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.sim = None

        self.max_episode_length = self.cfg['env']['episodeLength']
        self.action_scale = self.cfg['env']['actionScale']
        self.data_dir = self.cfg['env']['dataDir']
        self.img_width = self.cfg['env']['imgWidth']
        self.img_height = self.cfg['env']['imgHeight']
        self.path_len = self.cfg['env']['pathLen']
        self.debug_viz = self.cfg['env']['enableDebugViz']
        self.log_metrics = self.cfg['env']['enableLogging']
        self.asset_root = self.cfg['env']['asset']['assetRoot']
        self.asset_file = self.cfg['env']['asset']['assetFileNameFranka']

        # Create dicts to pass to reward function
        self.reward_settings = {
            'reward_clipped': self.cfg['env']['rewardClipped'],
            'r_dis_scale': self.cfg['env']['distanceRewardScale'],
            'r_orn_scale': self.cfg['env']['orientationRewardScale'],
            'r_ctl_scale': self.cfg['env']['controlRewardScale']
        }

        # Controller type
        self.control_type = self.cfg['env']['controlType']
        assert self.control_type in {'osc', 'ik'}, \
            'Invalid control type specified. Must be one of: {osc, ik}'

        # Dimensions
        # obs include: img_rgb (3, 256, 256) + img_depth (1, 256, 256) = 4 * 256 * 256 = 262144
        self.cfg['env']['numObservations'] = self.img_width * self.img_height * 4
        # actions include: delta EEF if OSC or IK (6)
        self.cfg['env']['numActions'] = 6

        # Init variables
        self.states = dict()  # will be dict filled with relevant states to use for reward calculation
        self.actions = None
        self.j_eef = None  # Jacobian for end effector
        self.mm = None  # Mass matrix
        self.rb_states = None  # State of all rigid bodies (n_envs, n_bodies, 13)
        self.dof_state = None
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

        # Torchvision transformations
        self.process_rgb = transforms.ConvertImageDtype(torch.float32)
        self.process_depth = transforms.Lambda(lambda d: torch.div(torch.clamp(-d, max=1.2), 1.2))

        # Init VecTask
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # IK Gains
        self.damping = to_torch([0.05] * 6, device=self.device)

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits (OCS)
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams) -> None:
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

    def _create_envs(self) -> None:
        spacing = 1.0
        num_per_row = int(math.sqrt(self.num_envs))
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # create table asset
        table_dims = gymapi.Vec3(0.6, 1.0, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # load asset robot
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)
        if franka_asset is None:
            raise Exception('Failed to load the franka asset')

        # config franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        # franka_effort_limits = franka_dof_props['effort']
        # franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

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
        table_pose.p = gymapi.Vec3(0.425, 0.0, 0.5 * table_dims.z)

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

        # get texture files
        self.textures_files = get_textures(data_path=self.data_dir)

        # create envs
        self.envs = []
        self.hand_idxs = []
        self.texture_points_list = []
        self.sphere_idxs = []
        self.spheres_rad_list = []
        self.spheres_pos_list = []
        self.rgb_tensors = []
        self.depth_tensors = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            self.gym.create_actor(env, table_asset, table_pose, 'table', i, 0)

            # create sphere asset
            sphere_rad = np.random.uniform(0.075, 0.1)
            self.spheres_rad_list.append(sphere_rad)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            sphere_asset = self.gym.create_sphere(self.sim, sphere_rad, asset_options)

            # sphere pose
            sphere_pose.p.x = table_pose.p.x  # + np.random.uniform(-0.02, 0.01)
            sphere_pose.p.y = table_pose.p.y  # + np.random.uniform(-0.03, 0.03)
            sphere_pose.p.z = table_dims.z + 0.5 * (sphere_rad * 2)
            sphere_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -np.pi / 2)  # orn offset in world axis
            self.spheres_pos_list.append([sphere_pose.p.x, sphere_pose.p.y, sphere_pose.p.z])

            # add sphere
            sphere_handle = self.gym.create_actor(env, sphere_asset, sphere_pose, 'sphere', i, 0)

            # get global index f sphere in rigid body state tensor
            sphere_idx = self.gym.get_actor_rigid_body_index(env, sphere_handle, 0, gymapi.DOMAIN_SIM)
            self.sphere_idxs.append(sphere_idx)

            # add texture to sphere
            texture_id = np.random.choice(self.textures_files)
            texture_file = f'{self.data_dir}/textures/{texture_id}'
            texture_points = get_trajectory(texture_id.rstrip('.png'), data_path=self.data_dir)
            self.texture_points_list.append(texture_points)
            texture = self.gym.create_texture_from_file(self.sim, texture_file)
            self.gym.set_rigid_body_texture(env, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, texture)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, 'franka', i, 2)

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
            rgb_tensor = gymtorch.wrap_tensor(_rgb_tensor)
            depth_tensor = gymtorch.wrap_tensor(_depth_tensor)
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
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]  # only need elements corresponding to the franka arm

        # Get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # Get dof state tensor
        # _actor_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        # self.root_state = gymtorch.wrap_tensor(_actor_root_states).view(self.num_envs, -1, 13)  # (n_envs, 13)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_states)  # state of all joints (n_envs, n_dofs)
        self.dof_pos = self.dof_state[:, 0].view(self.num_envs, 9, 1)
        self.dof_vel = self.dof_state[:, 1].view(self.num_envs, 9, 1)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

        # target point per env
        self.target_point = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # 1-7 map texture points to world coordinates
        # 1 - get sphere positions
        sphere_pos_array = np.array(self.spheres_pos_list, dtype=np.float32)  # (num_envs, 3)

        # 2 - get mapping from texture coord to sphere coord
        self.points_array = np.empty((self.num_envs, self.path_len, 7), dtype=np.float32)
        for i in range(self.num_envs):
            # For each path convert (s, t) -> (x, y, z, q0...q3)
            self.points_array[i, :, :] = trajectory_mapping(self.texture_points_list[i], self.spheres_rad_list[i])

        # 3 - position offset from world -> sphere -> point in path
        self.points_array[:, :, :3] += sphere_pos_array[:, np.newaxis, :]  # translation from origin to point
        self.points_array = self.points_array.reshape((self.num_envs * self.path_len, 7))  # 2D array of points

        # 4 - convert points pos to Vec3
        self.points_vec3 = np.empty(self.num_envs * self.path_len, dtype=gymapi.Vec3.dtype)
        self.points_vec3['x'] = self.points_array[:, 0]
        self.points_vec3['y'] = self.points_array[:, 1]
        self.points_vec3['z'] = self.points_array[:, 2]

        # 5 - convert points orn to Quat
        self.points_quat = np.empty(self.num_envs * self.path_len, dtype=gymapi.Quat.dtype)
        self.points_quat['x'] = self.points_array[:, 3]
        self.points_quat['y'] = self.points_array[:, 4]
        self.points_quat['z'] = self.points_array[:, 5]
        self.points_quat['w'] = self.points_array[:, 6]  # Try swap w and z

        # 6 - transformation offset: rotate to sphere normal + translation over z axis
        transform = gymapi.Transform()
        pos_offset = gymapi.Vec3(0, 0, -0.15)  # translation over z
        for i in range(self.num_envs * self.path_len):
            transform.p = self.points_vec3[i]
            transform.r = self.points_quat[i]
            new_pos = transform.transform_point(pos_offset)
            self.points_array[i, :3] = np.array([new_pos.x, new_pos.y, new_pos.z])

        # 7 - convert to tensor
        self.targets_tensor = to_torch(self.points_array.reshape((self.num_envs, self.path_len, 7)), device=self.device)

        # rotation to point inside the sphere (for panda_hand)
        rot_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi)
        self.offset_rot = torch.stack(self.num_envs * [torch.tensor([rot_q.x, rot_q.y, rot_q.z, rot_q.w])])\
            .to(self.device).view((self.num_envs, 4))

    def _update_states(self) -> None:
        self.states.update({
            'hand_pos': self.rb_states[self.hand_idxs, :3],
            'hand_rot': self.rb_states[self.hand_idxs, 3:7],
            'hand_vel': self.rb_states[self.hand_idxs, :7],
            'target_pos': self.targets_tensor[:, self.target_point, :3].view((self.num_envs, 3)),
            'target_rot': self.targets_tensor[:, self.target_point, 3:7].view((self.num_envs, 3))
        })

    def _refresh(self) -> None:
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self._update_states()

    def compute_reward(self, actions: Tensor) -> None:
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(self.reset_buf, self.progress_buf, actions,
                                                                   self.states, self.max_episode_length)

    def _update_sensor_tensor(self) -> None:
        """Update the observation buffer with the image from both rgb and depth sensors

        :return: void
        """
        # input:
        #  rgba_obs: [tensor(W, H, 4), ...] with len=num_envs
        #  depth_obs: [tensor(W, H, 1), ...] with len=num_envs
        # output:
        #  rgbd_obs: tensor(num_envs, W * H * 4)
        for i in range(self.num_envs):
            _rgb_norm_vec = self.process_rgb(self.rgb_tensors[i][:3].T)  # (3, W, H)
            _depth_norm_vec = self.process_depth(self.depth_tensors[i].T)  # (W, H)
            self.obs_buf[i, :] = torch.cat([_rgb_norm_vec.view(-1), _depth_norm_vec.view(-1)], dim=0)

    def compute_observations(self) -> Tensor:
        # render sensors and refresh camera tensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # get observations from sensors
        self._update_sensor_tensor()  # update obs_buf
        # stops accessing the tensors to prevent memory hazards
        self.gym.end_access_image_tensors(self.sim)
        return self.obs_buf

    def reset_idx(self, env_ids: Tensor) -> None:
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.pos_action),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.effort_action),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state.view(self.num_envs, -1, 2)),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.target_point[env_ids] = 0

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
        # u = tensor_clamp(u.squeeze(-1),
        #                  -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))
        # return u
        return u.unsqueeze(-1)

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
                target_point_pos = self.points_array[pt_idx, :3]
                hand_target_pos.p = gymapi.Vec3(target_point_pos[0], target_point_pos[1], target_point_pos[2])
                gymutil.draw_lines(self.point_geom, self.gym, self.viewer, self.envs[i], hand_target_pos)

        # set the next target point
        if self.progress_buf % (self.max_episode_length / self.path_len):
            self.target_point += 1
    # end task definition
# Pytorch JIT scripts

@torch.jit.script
def orientation_error(q_desired: Tensor, q_current: Tensor) -> Tensor:
    cc = quat_conjugate(q_current)
    q_r = quat_mul(q_desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def compute_franka_reward(reset_buf: Tensor, progress_buf: Tensor, actions: Tensor, states: Dict[str, Tensor],
                          reward_settings: Dict[str, Any], max_episode_length: float) -> Tuple[Tensor, Tensor]:
    # compute errors
    pos_err = states['target_pos'] - states['hand_pos']
    rot_err = orientation_error(states['target_rot'], states['hand_rot'])
    # reward scale params
    alpha = reward_settings['r_dis_scale']
    beta = reward_settings['r_orn_scale']
    gamma = reward_settings['r_ctl_scale']
    # compute rewards
    dis_reward = -torch.linalg.norm(pos_err, ord=2, dim=1)
    orn_reward = -torch.linalg.norm(rot_err, ord=2, dim=1)
    ctl_reward = -torch.sum(actions ** 2, dim=1)
    reward = alpha * dis_reward + beta * orn_reward + gamma * ctl_reward

    # clipped reward functions
    if reward_settings['reward_clipped']:
        rd_clip = 1 / (10 * torch.clamp(torch.tanh(-dis_reward * math.pi), min=0) + 1)
        ro_clip = 1 / (10 * torch.clamp(torch.tanh(-orn_reward * math.pi), min=0) + 1)
        rc_clip = torch.clamp(torch.tanh(ctl_reward), max=0)
        reward = torch.clamp(alpha * rd_clip + beta * ro_clip + gamma * rc_clip, min=0)

    # compute reset
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    return reward, reset_buf