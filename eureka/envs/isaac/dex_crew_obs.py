import torch
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_apply, quat_from_euler_xyz


class DexCrewBase:
    """Rest of the environment definition omitted."""
    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(self.num_envs, -1)
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(self.num_envs, -1)
        self.fingertip_angvel = self.fingertip_states[:, :, 10:13].reshape(self.num_envs, -1)
        self.nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        self.nut_pos = self.nut_states[:, :3]
        all_contact_forces = torch.norm(self.contact_forces.clone(), dim=-1)
        self.nut_contact = all_contact_forces[:, self.screw_nut_rb_handle].unsqueeze(-1)
        self.nut_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_xhand_hand_dofs:]
        self.nut_dof_vel = self.nut_dof_state[:, 0, 1].unsqueeze(-1)
        self.nut_dof_pos = self.nut_dof_state[:, 0, 0].unsqueeze(-1)



    def compute_observations(self): 
        self._refresh_gym()
        # observation noise
        random_obs_noise_t = torch.normal(0, self.random_obs_noise_t_scale, size=self.xhand_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noisy_joint_pos = random_obs_noise_t + self.random_obs_noise_e + self.xhand_hand_dof_pos 

        t_buf = (self.obs_buf_lag_history[:, -3:,:self.obs_buf.shape[1]//3].reshape(self.num_envs, -1)).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf  # [1, 96]
        
        # deal with normal observation, do sliding windows
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # xhand dim [1, 1, 12]
        cur_tar_buf = self.cur_targets[:, None, :self.num_actions]  # [1, 1, 12]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [1, 1, 24]

        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1) # torch.Size([48, 80, 24])

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:self.numActions] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, self.numActions:self.numActions*2] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)
        
        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]
        self.nut_dof_vel_cf[at_reset_env_ids] = self.nut_dof_vel[at_reset_env_ids]
        
        self.at_reset_buf[at_reset_env_ids] = 0
        rand_rpy = torch.normal(0, self.noisy_rpy_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float)
        rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        _noisy_quat = quat_mul(rand_quat, self.object_rot)
        _noisy_position = torch.normal(0, self.noisy_pos_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float) + self.object_pos
        
        # Update nut history buffers for termination conditions
        prev_nut_dof_pos_history = self.nut_dof_pos_history[:, 1:].clone()
        cur_nut_dof_pos = self.nut_dof_pos.clone().unsqueeze(1)
        self.nut_dof_pos_history[:] = torch.cat([prev_nut_dof_pos_history, cur_nut_dof_pos], dim=1)
            
        prev_nut_contact_history = self.nut_contact_history[:, 1:].clone()
        cur_nut_contact = self.nut_contact.clone().unsqueeze(1)
        self.nut_contact_history[:] = torch.cat([prev_nut_contact_history, cur_nut_contact], dim=1)
        
        if len(at_reset_env_ids) > 0:
            self.nut_dof_pos_history[at_reset_env_ids] = self.nut_dof_pos[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)
            self.nut_contact_history[at_reset_env_ids] = self.nut_contact[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)
        
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :self.numActions*2]  # [1, 30, 32]
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_orientation', value=self.object_rot.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_linvel', value=self.obj_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_orientation', value=self.fingertip_orientation.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_linvel', value=self.ft_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_angvel', value=self.ft_angvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_pos', value=self.nut_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_dof_pos', value=self.nut_dof_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_dof_vel', value=self.nut_dof_vel_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_position', value=self.fingertip_pos.clone())

        if self.point_cloud_sampled_dim > 0:
            # for collecting bc data
            self.point_cloud_buf[:, :self.point_cloud_sampled_dim] = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.obj_point_clouds
            ) + self.object_pos[:, None]  # [1, 100, 3]
    
