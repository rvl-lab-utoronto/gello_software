import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.robot import Robot

assert mujoco.viewer is mujoco.viewer


def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def build_scene(robot_xml_path: str, gripper_xml_path: Optional[str] = None):
    # assert robot_xml_path.endswith(".xml")

    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)
    # arm_copy = mjcf.from_path(xml_path)

    if gripper_xml_path is not None:
        # attach gripper to the robot at "attachment_site"
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    return arena


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "get_jacobian":
                    result = self._robot.get_jacobian()
                elif method == "get_camera_names":
                    result = self._robot.get_camera_names()
                elif method == "render_camera":
                    result = self._robot.render_camera(**args)
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MujocoRobotServer:
    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        ee_body_id: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        self._has_gripper = gripper_xml_path is not None
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu
        self._joint_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._joint_cmd = self._joint_state
        self._ee_body_id = ee_body_id

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

        # Add thread lock for camera rendering
        self._render_lock = threading.Lock()

        # Initialize renderer for camera functionality and pre-allocate to avoid issues
        self._renderer = mujoco.Renderer(self._model, height=128, width=128)

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            # print("yes gripper")
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            # print("no gripper")
            # self._joint_cmd = joint_state.copy()
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1
        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }
    
    def get_jacobian(self):
        # Compute Jacobian
        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))
        current_ee_pos = self.get_observations()["ee_pos_quat"]
        mujoco.mj_jac(self._model, self._data, jac_pos, jac_rot, current_ee_pos[:3], self._ee_body_id)
        
        # Combine position and rotation Jacobians
        jacobian = np.vstack([jac_pos, jac_rot])
        return jacobian
    
    def get_camera_names(self) -> list:
        """Get list of available camera names."""
        camera_names = []
        for i in range(self._model.ncam):
            try:
                cam_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    camera_names.append(cam_name)
                else:
                    camera_names.append(f"camera_{i}")  # fallback name
            except Exception as e:
                print(f"Error getting camera {i} name: {e}")
                camera_names.append(f"camera_{i}")  # fallback name
        return camera_names

    def render_camera(self, camera_name: str, width: int = 128, height: int = 128) -> np.ndarray:
        """Render image from specified camera."""
        try:
            # Initialize renderer if not already done
            if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = mujoco.Renderer(self._model, height=height, width=width)
            
            # Get camera ID
            try:
                cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id < 0:
                    raise ValueError(f"Camera '{camera_name}' not found")
            except:
                try:
                    cam_id = int(camera_name.split('_')[-1]) if 'camera_' in camera_name else int(camera_name)
                    if cam_id >= self._model.ncam:
                        raise ValueError(f"Camera index {cam_id} out of range")
                except:
                    print(f"Error: Invalid camera name '{camera_name}'")
                    return np.zeros((height, width, 3), dtype=np.uint8)
            
            # CRITICAL FIX: Create a separate data copy for rendering to avoid conflicts
            with threading.Lock():  # Thread safety
                data_copy = mujoco.MjData(self._model)
                data_copy.qpos[:] = self._data.qpos[:]
                data_copy.qvel[:] = self._data.qvel[:]
                data_copy.ctrl[:] = self._data.ctrl[:]
                data_copy.time = self._data.time
                
                # Forward the copied data
                mujoco.mj_forward(self._model, data_copy)
                
                # Render using the copied data
                self._renderer.update_scene(data_copy, camera=cam_id)
                rgb_array = self._renderer.render()
            
            return rgb_array
            
        except Exception as e:
            print(f"Error rendering camera '{camera_name}': {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            # cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, 'default_view')
            # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            # viewer.cam.fixedcamid = cam_id
            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                self._data.ctrl[:] = self._joint_cmd
                # self._data.qpos[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos.copy()[: self._num_joints]

                if self._print_joints:
                    print(self._joint_state)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    # TODO remove?
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        self._data.time % 2
                    )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()
