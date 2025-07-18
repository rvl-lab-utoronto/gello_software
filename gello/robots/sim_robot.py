import pickle
import threading
import time
from typing import Any, Dict, Optional
import numpy as np

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
                elif method == "iterate_ik":
                    result = self._robot.iterate_ik(args)
                elif method == "solve_ik_analytical":
                    result = self._robot.solve_ik_analytical(args)
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

        self.jac_pos = np.zeros((3, self._model.nv), dtype=np.float64)
        self.jac_rot = np.zeros((3, self._model.nv), dtype=np.float64)

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
        point = self.get_observations()["ee_pos_quat"][:3]
        mujoco.mj_jac(self._model, self._data, self.jac_pos, self.jac_rot, point, self._model.body("spoon").id)
        
        # Combine position and rotation Jacobians
        jacobian = np.vstack([self.jac_pos, self.jac_rot])
        return jacobian[:, :8]

    def _quat_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _calculate_pose_error(self, pos_goal, current_pos, quat_goal, current_quat):
        pos_error = np.subtract(pos_goal, current_pos)

        axis_angle_error = np.zeros(3)
        quat_error = self._quat_multiply(quat_goal, self._quat_conjugate(current_quat))
        mujoco.mju_quat2Vel(axis_angle_error, quat_error, 1)

        return pos_error, axis_angle_error

    def check_joint_limits(self, q):
        for i in range(self._num_joints):
            lower, upper = self._model.jnt_range[i]
            original = q[i]
            q[i] = np.clip(q[i], lower, upper)
            if q[i] != original:
                print(f"CLAMPING joint {i}: {original:.3f} -> {q[i]:.3f}")

    def compute_geometric_constants(self, model, data):
        def get_pos(model, data, name):
            return data.body(name).xpos

        pos = {link: get_pos(model, data, link) for link in [
            "link1", "link2", "link3", "link4",
            "link5", "link6", "link7", "hand"
        ]}

        d1 = pos["link1"][2]
        d3 = pos["link3"][2] - pos["link2"][2]
        d5 = pos["link5"][2] - pos["link4"][2]
        d7e = np.linalg.norm(pos["hand"] - pos["link7"])

        a4 = pos["link4"][0] - pos["link3"][0]
        a7 = pos["link7"][0] - pos["link6"][0]

        vec_24 = pos["link4"] - pos["link2"]
        vec_46 = pos["link6"] - pos["link4"]

        L24 = np.linalg.norm(vec_24)
        L46 = np.linalg.norm(vec_46)
        LL24 = np.linalg.norm(vec_24[[0, 1]])
        LL46 = np.linalg.norm(vec_46[[0, 1]])

        def angle_between(v1, v2):
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        thetaH46 = angle_between(pos["hand"] - pos["link6"], -vec_46)
        theta342 = angle_between(pos["link3"] - pos["link4"],
                                pos["link2"] - pos["link4"])
        theta46H = angle_between(vec_46, pos["hand"] - pos["link6"])

        return {
            "d1": d1, "d3": d3, "d5": d5, "d7e": d7e,
            "a4": a4, "a7": a7,
            "L24": L24, "L46": L46,
            "LL24": LL24, "LL46": LL46,
            "thetaH46": thetaH46,
            "theta342": theta342,
            "theta46H": theta46H,
        }
    
    def quat_to_mat(self, quat):
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        return R

    def solve_ik_analytical(self, goal):
        def norm(v):
            return v / np.linalg.norm(v)

        q = np.zeros(7)

        d1 = 0.3330
        d3 = 0.3160
        d5 = 0.3840
        d7e = 0.2104
        a4 = 0.0825
        a7 = 0.0880

        LL24 = 0.10666225
        LL46 = 0.15426225
        L24 = 0.326591870689
        L46 = 0.392762332715

        thetaH46 = 1.35916951803
        theta342 = 1.31542071191
        theta46H = 0.211626808766

        q7 = 0.0

        pos_goal = goal[:3]
        quat_goal = goal[3:]
        rot_goal = self.quat_to_mat(quat_goal)

        t_ee_0 = np.eye(4)
        t_ee_0[:3, :3] = rot_goal
        t_ee_0[:3, 3] = pos_goal

        z_ee = t_ee_0[:3, 2]
        p_ee = t_ee_0[:3, 3]
        p7 = p_ee - d7e * z_ee

        x_EE_6 = np.array([np.cos(q7 - np.pi/4), -np.sin(q7 - np.pi/4), 0])
        x6 = norm(rot_goal @ x_EE_6)
        p6 = p7 - a7 * x6

        p2 = np.array([0, 0, d1])
        V26 = p6 - p2
        LL26 = np.dot(V26, V26)
        L26 = np.sqrt(LL26)

        theta246 = np.arccos((LL24 + LL46 - LL26) / (2 * L24 * L46))
        q[3] = theta246 + thetaH46 + theta342 - 2 * np.pi

        theta462 = np.arccos((LL26 + LL46 - LL24) / (2 * L26 * L46))
        theta26H = theta46H + theta462
        D26 = -L26 * np.cos(theta26H)

        Z6 = np.cross(z_ee, x6)
        Y6 = np.cross(Z6, x6)
        R6 = np.column_stack((x6, norm(Y6), norm(Z6)))
        V_6_62 = R6.T @ -V26

        phi6 = np.arctan2(V_6_62[1], V_6_62[0])
        theta6 = np.arcsin(D26 / np.linalg.norm(V_6_62[:2]))

        q[5] = theta6 - phi6
        q[5] = (q[5] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        thetaP26 = 3 * np.pi / 2 - theta462 - theta246 - theta342
        thetaP = np.pi - thetaP26 - theta26H
        LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

        z5 = R6 @ np.array([np.sin(q[5]), np.cos(q[5]), 0])
        V2P = p6 - LP6 * z5 - p2

        L2P = np.linalg.norm(V2P)
        if abs(V2P[2] / L2P) > 0.999:
            q[0] = 0
            q[1] = 0
        else:
            q[0] = np.arctan2(V2P[1], V2P[0])
            q[1] = np.arccos(V2P[2] / L2P)

        z3 = V2P / L2P
        y3 = -np.cross(V26, V2P)
        y3 = norm(y3)
        x3 = np.cross(y3, z3)

        R1 = np.array([
            [np.cos(q[0]), -np.sin(q[0]), 0],
            [np.sin(q[0]), np.cos(q[0]), 0],
            [0, 0, 1]
        ])
        R1_2 = np.array([
            [np.cos(q[1]), -np.sin(q[1]), 0],
            [0, 0, 1],
            [-np.sin(q[1]), -np.cos(q[1]), 0]
        ])
        R2 = R1 @ R1_2
        x2_3 = R2.T @ x3
        q[2] = np.arctan2(x2_3[2], x2_3[0])

        VH4 = p2 + d3 * z3 + a4 * x3 - p6 + d5 * z5
        R5_6 = np.array([
            [np.cos(q[5]), -np.sin(q[5]), 0],
            [0, 0, -1],
            [np.sin(q[5]), np.cos(q[5]), 0]
        ])
        R5 = R6 @ R5_6.T
        V_5_H4 = R5.T @ VH4
        q[4] = -np.arctan2(V_5_H4[1], V_5_H4[0])

        q[6] = q7
        q = np.concatenate([q, np.zeros(1)])
        return q


    def iterate_ik(self, goal, mode="sgd"):
        self.tol = 0.3
        self.step_size = 0.05
        self.jacp = np.zeros((3, self._model.nv))
        self.jacr = np.zeros((3, self._model.nv))

        self._data.qpos[:self._num_joints] = self.get_observations()["joint_positions"]

        ik_data = mujoco.MjData(self._model)
        ik_data.qpos[:] = self._data.qpos[:]

        pos_goal = goal[:3]
        quat_goal = goal[3:]

        mujoco.mj_forward(self._model, ik_data)
        current_pos = ik_data.body(self._model.body("hand").id).xpos
        current_quat = ik_data.body(self._model.body("hand").id).xquat

        pos_error, axis_angle_error = self._calculate_pose_error(
            pos_goal, current_pos, quat_goal, current_quat
        )

        while np.linalg.norm(np.concatenate((pos_error, axis_angle_error))) >= self.tol:
            mujoco.mj_jac(
                self._model, ik_data, self.jacp,
                self.jacr, current_pos, self._model.body("hand").id
            )
            if mode == "sgd":
                pos_grad = self.jacp.T @ pos_error
                axis_angle_grad = self.jacr.T @ axis_angle_error
                if pos_grad.shape[0] < ik_data.qpos.shape[0]:
                    padded_grad = np.zeros_like(ik_data.qpos)
                    padded_grad[:pos_grad.shape[0]] = pos_grad
                    pos_grad = padded_grad
                if axis_angle_grad.shape[0] < ik_data.qpos.shape[0]:
                    padded_grad = np.zeros_like(ik_data.qpos)
                    padded_grad[:axis_angle_grad.shape[0]] = axis_angle_grad
                    axis_angle_grad = padded_grad

                ik_data.qpos[:] += self.step_size * (pos_grad + axis_angle_grad)

            elif mode == "pinv":
                jacpose = np.vstack([self.jacp, self.jacr])
                pose_error = np.concatenate([pos_error, axis_angle_error])
                dqpos = np.linalg.pinv(jacpose) @ pose_error
                if dqpos.shape[0] < ik_data.qpos.shape[0]:
                    padded_dqpos = np.zeros_like(ik_data.qpos)
                    padded_dqpos[:dqpos.shape[0]] = dqpos
                    dqpos = padded_dqpos
                
                ik_data.qpos[:] += self.step_size * dqpos

            self.check_joint_limits(ik_data.qpos)
            mujoco.mj_forward(self._model, ik_data)

            current_pos = ik_data.body(self._model.body("hand").id).xpos
            current_quat = ik_data.body(self._model.body("hand").id).xquat
            pos_error, axis_angle_error = self._calculate_pose_error(
                pos_goal, current_pos, quat_goal, current_quat
            )
            print(f"IK Error: {np.linalg.norm(np.concatenate((pos_error, axis_angle_error)))}")

        return ik_data.qpos[:self._num_joints].copy()

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
