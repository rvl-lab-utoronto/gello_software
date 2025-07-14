import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import zmq
from dm_control import mjcf
import cv2
import sys
import os

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
                elif method == "get_camera_names":
                    result = self._robot.get_camera_names()
                elif method == "render_camera":
                    result = self._robot.render_camera(**args)
                elif method == "reset_simulation":
                    result = self._robot.reset_simulation()
                elif method == "save_model_xml":
                    result = self._robot.save_model_xml(**args)
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
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        show_camera_window: bool = False,
        camera_window_name: str = "default_camera",
        camera_window_size: tuple = (640, 480),
        task: str = None,
        randomize_list: list = None,
        background_images_dir: str = None,
    ):
        self.randomize_func = randomize_list[0] if randomize_list else None
        self.original_xml = randomize_list[1] if randomize_list else None
        self.xml_path = randomize_list[2] if randomize_list else xml_path

        self._has_gripper = gripper_xml_path is not None

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

        # Add thread lock for camera rendering
        self._render_lock = threading.Lock()

        # Camera window settings
        self._show_camera_window = show_camera_window
        self._camera_window_name = camera_window_name
        self._camera_window_size = camera_window_size
        
        self._initialize_simulation()

        self._viewer_ptrs_update_requested = False
        self._cam_window_reset_requested = False

        self._simulation_lock = threading.RLock()  # Use RLock for reentrant locking
        self._reset_requested = False
        self._reset_completed = True

        REPO_ROOT: Path = Path(__file__).parent.parent.parent.parent.parent
        sys.path.insert(0, os.path.abspath(REPO_ROOT))

    def _initialize_simulation(self):
        """Initialize model, data, renderers, and task-specific components."""
        self._model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self._data = mujoco.MjData(self._model)

        # Joint state
        self._num_joints = self._model.nu
        self._joint_state = np.zeros(self._num_joints, dtype=np.float64)
        self._joint_cmd = self._joint_state.copy()

        # Renderer
        self._renderer = mujoco.Renderer(self._model, height=128, width=128)
        self._depth_renderer = mujoco.Renderer(self._model, height=128, width=128); self._depth_renderer.enable_depth_rendering()
        self._sgmnt_renderer = mujoco.Renderer(self._model, height=128, width=128); self._sgmnt_renderer.enable_segmentation_rendering()


        # Optional camera renderer
        self._camera_renderer = None
        if self._show_camera_window:
            self._camera_renderer = mujoco.Renderer(
                self._model,
                height=self._camera_window_size[1],
                width=self._camera_window_size[0],
            )

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

        # record the everything in data as states for replay
        time = self._data.time
        qpos = np.copy(self._data.qpos)
        qvel = np.copy(self._data.qvel)
        flattened_states = np.concatenate([[time],qpos,qvel],axis=0)

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
            "flattened_states": flattened_states,
        }
    
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
                self._depth_renderer = mujoco.Renderer(self._model, height=height, width=width); self._depth_renderer.enable_depth_rendering()
                self._sgmnt_renderer = mujoco.Renderer(self._model, height=height, width=width); self._sgmnt_renderer.enable_segmentation_rendering()
            
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
                    return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8), np.zeros((height, width, 2), dtype=np.uint8)
            
            # CRITICAL FIX: Create a separate data copy for rendering to avoid conflicts
            with self._simulation_lock:  # Thread safety
                data_copy = mujoco.MjData(self._model)
                data_copy.qpos[:] = self._data.qpos[:]
                data_copy.qvel[:] = self._data.qvel[:]
                data_copy.ctrl[:] = self._data.ctrl[:]
                data_copy.time = self._data.time
                
                # Forward the copied data
                mujoco.mj_forward(self._model, data_copy)
                
                # Render using the copied data
                self._renderer.update_scene(data_copy, camera=cam_id)
                self._depth_renderer.update_scene(data_copy, camera=cam_id)
                self._sgmnt_renderer.update_scene(data_copy, camera=cam_id)
                rgb_array = self._renderer.render()
                depth_array = self._depth_renderer.render()
                segmentation_array = self._sgmnt_renderer.render()
            
            return rgb_array, depth_array, segmentation_array
            
        except Exception as e:
            print(f"Error rendering camera '{camera_name}': {e}")
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8), np.zeros((height, width, 2), dtype=np.uint8)

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data, show_left_ui=True, show_right_ui=False) as viewer:
            # Set the viewer to use a specific camera from your XML
            try:
                cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, 'leftshoulder')
                if cam_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
                    print(f"Set GUI view to camera: leftshoulder (ID: {cam_id})")
                else:
                    print("Camera 'leftshoulder' not found, using default view")
            except Exception as e:
                print(f"Error setting camera view: {e}")
            
            # Create cv2 window AFTER MuJoCo viewer is launched
            if self._show_camera_window:
                cv2.namedWindow(self._camera_window_name, cv2.WINDOW_AUTOSIZE)
                # Position the cv2 window to the left to make room for MuJoCo
                cv2.moveWindow(self._camera_window_name, 50, 100)  # x=50, y=100
                print(f"Created camera window: {self._camera_window_name}")
            
            while viewer.is_running():
                step_start = time.time()

                if self._reset_completed == False:
                    continue

                if self._viewer_ptrs_update_requested:
                    if hasattr(viewer, "update_mjptrs"):
                        with viewer.lock():
                            viewer.update_mjptrs(self._model, self._data)
                        print("[serve] viewer pointers updated")
                    else:
                        # --- older MuJoCo: rebuild the viewer completely ---
                        viewer.close()                 # 1) close old window
                        viewer = mujoco.viewer.launch_passive(
                            self._model, self._data,
                            show_left_ui=True, show_right_ui=False
                        )                                  # 2) open a new one
                        try:
                            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, 'leftshoulder')
                            if cam_id >= 0:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                                viewer.cam.fixedcamid = cam_id
                                print(f"Set GUI view to camera: leftshoulder (ID: {cam_id})")
                            else:
                                print("Camera 'leftshoulder' not found, using default view")
                        except Exception as e:
                            print(f"Error setting camera view: {e}")
                        print("[serve] viewer relaunched")
                    self._viewer_ptrs_update_requested = False
                
                if self._cam_window_reset_requested and self._show_camera_window:
                    if self._viewer_ptrs_update_requested:
                        continue # ensure viewer pointers are updated before resetting camera window
                    try:
                        cv2.destroyWindow(self._camera_window_name)
                    except cv2.error:
                        pass  # window may already be gone
                    cv2.namedWindow(self._camera_window_name, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(self._camera_window_name, 50, 100)
                    self._cam_window_reset_requested = False
                    print("[serve] Camera window recreated")

                # Use lock to prevent conflicts during reset
                with self._simulation_lock:
                    # mj_step can be replaced with code that also evaluates
                    # a policy and applies a control signal before stepping the physics.
                    self._data.ctrl[:] = self._joint_cmd
                    # self._data.qpos[:] = self._joint_cmd
                    mujoco.mj_step(self._model, self._data)
                    self._joint_state = self._data.qpos.copy()[: self._num_joints]

                if self._print_joints:
                    print(self._joint_state)

                # Update camera window if enabled
                if self._show_camera_window:
                    if self._cam_window_reset_requested:
                        continue
                    self._update_camera_window()

                # # Example modification of a viewer option: toggle contact points every two seconds.
                # with viewer.lock():
                #     # TODO remove?
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                #         self._data.time % 2
                #     )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # Check if camera window was closed
                if self._show_camera_window:
                    try:
                        if cv2.getWindowProperty(self._camera_window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("Camera window closed, stopping viewer...")
                            break
                    except cv2.error:
                        # Window was closed
                        print("Camera window closed (cv2.error), stopping viewer...")
                        break

        # Cleanup camera window
        if self._show_camera_window:
            try:
                cv2.destroyWindow(self._camera_window_name)
                cv2.waitKey(1)  # Ensure cleanup
                print("Camera window cleaned up")
            except:
                pass

    def _update_camera_window(self):
        """Update the camera window with current view."""
        try:
            # Get camera ID
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_window_name)
            if cam_id < 0:
                # Fallback to first camera if named camera not found
                cam_id = 0
            
            # Render the camera view
            self._camera_renderer.update_scene(self._data, camera=cam_id)
            rgb_array = self._camera_renderer.render()
            
            # Convert RGB to BGR for OpenCV
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow(self._camera_window_name, bgr_array)
            cv2.waitKey(1)  # Non-blocking wait
            
        except Exception as e:
            print(f"Error updating camera window: {e}")

    def reset_simulation(self):
        with self._simulation_lock:
            self._reset_completed = False
            # close all mujoco viewer windows and renderers
            if self._show_camera_window:
                self._cam_window_reset_requested = True

            # randomize the scene if a randomization function is provided
            if self.randomize_func:
                print("Randomizing scene...")
                self.randomize_func(self.original_xml, self.xml_path)
                print("Scene randomized")
            else:
                print("No randomization function provided, using original XML")

            self._initialize_simulation()

            self._viewer_ptrs_update_requested = True
            print("Simulation reset requested")

            self._reset_completed = True

            return {"status": "success"}
            # if self._show_camera_window:
            #         cv2.namedWindow(self._camera_window_name, cv2.WINDOW_AUTOSIZE)
            #         # Position the cv2 window to the left to make room for MuJoCo
            #         cv2.moveWindow(self._camera_window_name, 50, 100)  # x=50, y=100
            #         print(f"Created camera window: {self._camera_window_name}")

            # if self._show_camera_window:
            #     self._update_camera_window()
            
            # mujoco.viewer.launch_passive(self._model, self._data)

    def save_model_xml(self, xml_path: str) -> dict:
        """Save the current model XML to the specified path."""
        try:
            mujoco.mj_saveLastXML(xml_path, self._model)
            return {"status": "success", "message": f"Model saved to {xml_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()


# Example usage
if __name__ == "__main__":
    server = MujocoRobotServer(
        xml_path="/home/sebastiana/Granular_material_benchmark/envs/franka_scooping_env/scooping.xml",
        show_camera_window=True,
        camera_window_name="cam1",  # Name of camera in your XML, or use "default_camera"
        camera_window_size=(640, 480)
    )
    
    try:
        server.serve()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.stop()