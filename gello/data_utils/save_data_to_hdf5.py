import h5py
import os 
from glob import glob
import numpy as np
import pickle
import datetime

def gather_demonstrations_as_hdf5(in_dir, out_dir, env_info=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point
    print(os.listdir(in_dir))
    ip = input("press enter")
    for ep_directory in os.listdir(in_dir):
        print(ep_directory)
        state_paths = os.path.join(in_dir, ep_directory, "*.pkl")
        joint_positions = []
        joint_velocities = []
        ee_pose_quats = []
        gripper_positions = []
        actions = []
        wrist_images = []
        base_images = []
        success = False

        for i, state_file in enumerate(sorted(glob(state_paths))):
            # print(state_file, ep_directory)
            try:
                with open(state_file, "rb") as f:
                    demo = pickle.load(f)
            except Exception as e:
                print(f"Skipping {state_file} because it is corrupted.")
                print(f"Error: {e}")
                raise Exception("Corrupted pkl")
            # breakpoint()

            joint_positions.append(demo["joint_positions"])
            joint_velocities.append(demo["joint_velocities"])
            ee_pose_quats.append(demo["ee_pos_quat"])
            gripper_positions.append(demo["gripper_position"])
            actions.append(demo["control"])
            wrist_images.append(demo["wrist_rgb"])
            base_images.append(demo["base_rgb"])
        

        if len(joint_positions) == 0:
            continue

        print("Demonstration is successful and has been saved")
        assert len(joint_positions) == len(actions) == \
                    len(joint_velocities) == len(ee_pose_quats) == len(wrist_images)

        num_eps += 1 
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))
        ep_data_grp.create_dataset("joint_positions", data=np.array(joint_positions))
        ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
        ep_data_grp.create_dataset("ee_pose_quats", data=np.array(ee_pose_quats))
        ep_data_grp.create_dataset("gripper_positions", data=np.array(gripper_positions))
        ep_data_grp.create_dataset("wrist_images", data=np.array(wrist_images))
        ep_data_grp.create_dataset("base_images", data=np.array(base_images))
        ep_data_grp.create_dataset("actions", data=np.array(actions))        

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env_info"] = str(env_info)
    f.close()




    #         dic = np.load(state_file, allow_pickle=True)
    #         env_name = str(dic["env"])

    #         states.extend(dic["states"])
    #         for ai in dic["action_infos"]:
    #             actions.append(ai["actions"])
    #         success = success or dic["successful"]

    #     if len(states) == 0:
    #         continue

    #     # Add only the successful demonstration to dataset
    #     if success:
    #         print("Demonstration is successful and has been saved")
    #         # Delete the last state. This is because when the DataCollector wrapper
    #         # recorded the states and actions, the states were recorded AFTER playing that action,
    #         # so we end up with an extra state at the end.
    #         del states[-1]
    #         assert len(states) == len(actions)

    #         num_eps += 1
    #         ep_data_grp = grp.create_group("demo_{}".format(num_eps))

    #         # store model xml as an attribute
    #         xml_path = os.path.join(directory, ep_directory, "model.xml")
    #         with open(xml_path, "r") as f:
    #             xml_str = f.read()
    #         ep_data_grp.attrs["model_file"] = xml_str

    #         # write datasets for states and actions
    #         ep_data_grp.create_dataset("states", data=np.array(states))
    #         ep_data_grp.create_dataset("actions", data=np.array(actions))
    #     else:
    #         print("Demonstration is unsuccessful and has NOT been saved")

    # # write dataset attributes (metadata)
    # now = datetime.datetime.now()
    # grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    # grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    # grp.attrs["repository_version"] = suite.__version__
    # grp.attrs["env"] = env_name
    # grp.attrs["env_info"] = env_info

