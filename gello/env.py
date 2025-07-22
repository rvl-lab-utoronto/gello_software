from pathlib import Path
import time
from typing import Any, Dict, Optional

import h5py
import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        assert "flattened_states" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        observations["flattened_states"] = robot_obs["flattened_states"]
        return observations

    def save_single_timestep(self, save_path: Path, dt, obs: Dict[str, Any], action: np.ndarray):
        """Save a single timestep's actions and observations (no cameras) into an HDF5 file (one file per episode)."""
        # path to the HDF5 file in this episode folder
        h5_path = save_path / "data.h5"

        # open (or create) the HDF5 file in append mode
        with h5py.File(h5_path, "a") as h5f:
            # make a group for this frame
            grp_name = dt.strftime("frame_%Y%m%d_%H%M%S_%f")
            grp = h5f.create_group(grp_name)
            grp.attrs["timestamp"] = dt.strftime("%Y%m%d_%H%M%S_%f")

            # store observations under a subgroup
            obs_grp = grp.create_group("observations")
            for key, val in obs.items():
                # flatten or leave as-is; here we assume val is array-like
                if val.ndim == 0:
                    obs_grp.attrs[key] = val
                else:
                    obs_grp.create_dataset(key, data=val, compression="gzip")

            # store action
            grp.create_dataset("action", data=action, compression="gzip")

            # # store metadata as attributes
            # meta = grp.create_group("metadata")
            # meta.attrs["camera_names"] = np.array(self.camera_names, dtype="S")
            # meta.attrs["camera_width"] = self.camera_width
            # meta.attrs["camera_height"] = self.camera_height


def main() -> None:
    pass


if __name__ == "__main__":
    main()
