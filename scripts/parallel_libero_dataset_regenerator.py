import os
os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import h5py
import json
import multiprocessing
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
import uuid
from libero.libero import benchmark

import sys
sys.path.append('.')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from experiments.robot.libero.regenerate_libero_dataset import is_noop


IMAGE_RESOLUTION = 256


class ParallelRegenerator:
    def __init__(self, 
                 num_gpus, 
                 max_processes,
                 task_suite, 
                 data_dir, 
                 target_dir):
        self.num_gpus = num_gpus
        self.max_processes = max_processes
        self.task_suite = task_suite
        self.data_dir = data_dir
        self.target_dir = target_dir
    
    def run(self):
        os.makedirs(self.target_dir, exist_ok=True)
        
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite_dict = benchmark_dict[self.task_suite]()
        num_tasks_in_suite = self.task_suite_dict.n_tasks

        task_ids = []
        processes = []
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # task_name = self.task_suite_dict.get_task(task_id).name
            gpu_id = task_id % self.num_gpus
            p = multiprocessing.Process(target=self._run_single_task, args=(gpu_id, task_id))
            processes.append(p)
            task_ids.append(task_id)
        
        task_ids_splits = [task_ids[i:i + self.max_processes] for i in range(0, len(task_ids), self.max_processes)]
        processes_splits = [processes[i:i + self.max_processes] for i in range(0, len(processes), self.max_processes)]
        for processes, task_ids in zip(processes_splits, task_ids_splits):
            for task_id, p in zip(task_ids, processes):
                task_name = self.task_suite_dict.get_task(task_id).name
                print(f"Starting {p} for task {task_name} on GPU {task_id % self.num_gpus}.")

            print(f"Starting {len(processes)} processes to regenerate data for {num_tasks_in_suite} tasks.")
            
            for p in processes:
                p.start()
                
            for p in processes:
                p.join()
        
        metainfo_json_dict = {}
        for task_id in task_ids:
            metainfo_json_out_path = f"./experiments/robot/libero/{self.task_suite}_{task_id}_metainfo.json"
            with open(metainfo_json_out_path, "r") as f:
                task_metainfo_json_dict = json.load(f)
                metainfo_json_dict.update(task_metainfo_json_dict)
            
        metainfo_json_out_path = f"./experiments/robot/libero/{self.task_suite}_metainfo.json"
        with open(metainfo_json_out_path, "w") as f:
            json.dump(metainfo_json_dict, f, indent=2)
    
    def _run_single_task(self, gpu_id, task_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Get task in suite
        task = self.task_suite_dict.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(self.data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(self.target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        num_replays = 0
        num_success = 0
        num_noops = 0
        metainfo_json_dict = {}

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            
            agentview_segs = []
            eye_in_hand_segs = []
            # object_infos_strs = []
            object_info_ids = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                agentview_segs.append(obs['agentview_segmentation_instance'])
                eye_in_hand_segs.append(obs['robot0_eye_in_hand_segmentation_instance'])
                
                obj_name_to_seg_id = env.instance_to_id
                
                pos_keys = [key for key in obs.keys() if key.endswith('pos')]
                positions = {key: list(obs[key]) for key in pos_keys}

                objects_dict = {**env.env.objects_dict, **env.env.fixtures_dict}
                objects = env.env.sim.data.model.body_names
                
                is_grasps = {}
                for obj_name, obj in objects_dict.items():
                    is_grasps[obj_name] = env.env._check_grasp(env.env.robots[0].gripper, obj)
                
                gripper_to_objs = {}
                # for obj_name, obj in objects_dict.items():
                #     gripper_to_objs[obj_name] = list(env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=False))
                for obj in objects:
                    gripper_to_objs[obj] = list(env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=False))

                gripper_to_obj_distances = {}
                # for obj_name, obj in objects_dict.items():
                #     gripper_to_obj_distances[obj_name] = env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=True)
                for obj in objects:
                    gripper_to_obj_distances[obj] = env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=True)
                
                object_infos = {
                    'position': positions,
                    'is_grasp': is_grasps,
                    'gripper_to_obj': gripper_to_objs,
                    'gripper_to_obj_distance': gripper_to_obj_distances,
                    'name_to_seg_id': obj_name_to_seg_id
                }
                
                # object_infos_str = json.dumps(object_infos)
                # object_infos_strs.append(object_infos_str)
                object_info_id = str(uuid.uuid4())
                object_info_ids.append(object_info_id)

                os.makedirs(os.path.join(self.target_dir, 'object_infos'), exist_ok=True)
                with open(os.path.join(self.target_dir, 'object_infos', f"{object_info_id}.json"), "w") as f:
                    json.dump(object_infos, f, indent=2)
                
                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                num_success += 1

                assert len(actions) == len(agentview_images) == len(agentview_segs) == len(object_info_ids)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                # ep_data_grp.create_dataset("agentview_seg", data=np.stack(agentview_segs, axis=0))
                # ep_data_grp.create_dataset("eye_in_hand_seg", data=np.stack(eye_in_hand_segs, axis=0))
                ep_data_grp.create_dataset("object_infos", data=object_info_ids)

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"

            metainfo_json_dict[task_key] = {}

            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Count total number of successful replays so far
            print(f"[{task_description}] Successes: {num_success} Totals: {num_replays} ({num_success / num_replays * 100:.1f} %)")

            metainfo_json_out_path = f"./experiments/robot/libero/{self.task_suite}_{task_id}_metainfo.json"
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)
            
        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()

        print(f"[{task_description}] Saved regenerated demos for task at: {new_data_path}")


def main(args):
    regenerator = ParallelRegenerator(args.num_gpus, args.max_processes, args.libero_task_suite, 
                                      args.libero_raw_data_dir, args.libero_target_dir)
    regenerator.run()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=6,
        help="Number of GPUs to use for parallel data regeneration.",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=100,
        help="Maximum number of processes to run in parallel.",
    )
    parser.add_argument(
        "--libero-task-suite",
        type=str,
        default='libero_90',
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
    )
    parser.add_argument(
        "--libero-raw-data-dir",
        type=str,
        default='data/libero_90',
        help=("Path to directory containing raw HDF5 dataset. " "Example: ./LIBERO/libero/datasets/libero_spatial"),
    )
    parser.add_argument(
        "--libero-target-dir",
        type=str,
        default='data/libero_90_no_noops',
        help=("Path to regenerated dataset directory. " "Example: ./LIBERO/libero/datasets/libero_spatial_no_noops"),
    )
    args = parser.parse_args()

    # Start data regeneration
    main(args)
