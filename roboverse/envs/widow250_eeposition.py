from roboverse.envs.widow250 import Widow250Env
import roboverse
import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.envs import objects
import numpy as np


class Widow250EEPositionEnv(Widow250Env):
    def __init__(self,
                 ee_distance_threshold=0.1,
                 reward_type="ee_position",
                 **kwargs):
        self.arm_min_radius = 0.100
        self.ee_distance_threshold = ee_distance_threshold
        self.ee_target_pose = None
        super(Widow250EEPositionEnv, self).__init__(
            reward_type=reward_type, **kwargs)

    def _load_meshes(self, target_position=None):
        super(Widow250EEPositionEnv, self)._load_meshes(target_position)

    def get_info(self):
        info = {}
        ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        target_coord = np.array(self.ee_target_pose)
        ee_coord = np.array(ee_pos)
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        if euclidean_dist_3d <= self.ee_distance_threshold:
            info['ee_pose_success'] = True
            info['target_coord'] = self.ee_target_pose
        else:
            info['ee_pose_success'] = False

        info['euclidean_distance'] = euclidean_dist_3d
        info['target_coord'] = self.ee_target_pose
        return info

    def get_reward(self, info):
        if self.reward_type == 'ee_position':
            reward = 0
            # Reward weight for reaching the goal position
            g_w = 1000
            # Reward weight according to the distance to the goal
            d_w = 10

            # Reward base
            reward += np.exp(-d_w * info['euclidean_distance'])

            if info['ee_pose_success']:
                reward = g_w * 1
                self.done = True
        else:
            raise NotImplementedError

        return reward

    def reset(self, target=None, seed=None, options=None):
        if target:
            assert len(target) == 6
            self.ee_target_pose = target
        else:
            self.ee_target_pose = self._get_target_pose()

        self.objects[self.target_object] = object_utils.load_object(
            self.target_object,
            self.ee_target_pose,
            object_quat=self.object_orientations[self.target_object],
            scale=self.object_scales[self.target_object])

        bullet.reset()
        bullet.setup_headless()
        self._load_meshes(self.ee_target_pose)
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True  # TODO(avi): Clean this up
        self.done = False
        
        return self.get_observation(), self.get_info()

    def _get_target_pose(self) -> np.ndarray:
        workspace_pose = bullet.get_random_workspace_pose(self.ee_pos_low, self.ee_pos_high, self.arm_min_radius)
        # workspace_pose = [sum(coord) for coord in zip(workspace_pose, self.base_position)]
        return workspace_pose


if __name__ == "__main__":
    env = roboverse.make("Widow250EEPosition-v0",
                         gui=True, transpose_image=False)
    import time
    env.reset()
    # import IPython; IPython.embed()

    for j in range(5):
        for i in range(20):
            obs, rew, done, _, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0., 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()
