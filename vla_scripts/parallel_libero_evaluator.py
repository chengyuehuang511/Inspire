import os
os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import math
import multiprocessing
import numpy as np
import traceback
from PIL import Image
from pathlib import Path

import sys
sys.path.append('.')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from utils.logger import Logger, reset_logging
from utils.visualize import write_video


def get_image_resize_size(cfg):
    if cfg.model_family == "prismatic":
        resize_size = 224
    elif cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def apply_center_crop(im, t_h, t_w):
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def get_prismatic_vla_action(vla, obs, task_label, unnorm_key, center_crop=False, **kwargs):
    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []

    for img in obs["full_image"]:
        image = Image.fromarray(img)
        image = image.convert("RGB")

        if center_crop:
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(
                image.size, Image.Resampling.BILINEAR
            )  # IMPORTANT: dlimp uses BILINEAR resize
            image = temp_image

        processed_images.append(image)

    if len(processed_images) == 1:
        processed_images = processed_images[0]

    outputs = vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)
    if isinstance(outputs, tuple):
        action, text = outputs
    else:
        action, text = outputs, None
    return action, text


def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


class GenerateConfig:
    def __init__(self,
                 model_family="prismatic",
                 hf_token=Path(".hf_token"),
                 pretrained_checkpoint="runs/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n0+b16+x7/",
                 load_step=None,
                 load_in_8bit=False,
                 load_in_4bit=False,
                 center_crop=True,
                 obs_history=1,
                 use_wrist_image=False,
                 seed=7,
                 task_suite_name="libero_90",
                 num_steps_wait=10,
                 num_trials_per_task=10,
                 num_gpus=8,
                 num_processes=32,
                 save_root="./results",
                 fps=30,
                 with_vqa=False,
                 check_catch=True,
                 check_close=True,
                 vqa_mode='coarse_direction',
                 ):
        self.model_family = model_family
        self.hf_token = hf_token
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_step = load_step
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.center_crop = center_crop
        self.obs_history = obs_history
        self.use_wrist_image = use_wrist_image
        self.seed = seed
        self.task_suite_name = task_suite_name
        self.num_steps_wait = num_steps_wait
        self.num_trials_per_task = num_trials_per_task
        self.num_gpus = num_gpus
        self.num_processes = num_processes
        self.save_root = save_root
        self.fps = fps
        self.with_vqa = with_vqa
        self.check_catch = check_catch
        self.check_close = check_close
        self.vqa_mode = vqa_mode

        self.image_sequence_len = 1
        if self.obs_history == 2 or self.use_wrist_image:
            self.image_sequence_len = 2


class ParallelLiberoEvaluator:
    def __init__(self, cfg, opts=None):
        # [Note] Data root is not used for evaluation
        os.environ["PRISMATIC_DATA_ROOT"] = 'data/prismatic'
        # [Note] Tokenizers parallelism is set to true for faster tokenization
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
        if "image_aug" in cfg.pretrained_checkpoint:
            assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

        self.cfg = cfg
        # self.cfg.unnorm_key = self.cfg.task_suite_name
        self.cfg.unnorm_key = 'libero_90'

        if cfg.task_suite_name == "libero_spatial":
            self.cfg.max_steps = 220  # longest training demo has 193 steps
        elif cfg.task_suite_name == "libero_object":
            self.cfg.max_steps = 280  # longest training demo has 254 steps
        elif cfg.task_suite_name == "libero_goal":
            self.cfg.max_steps = 300  # longest training demo has 270 steps
        elif cfg.task_suite_name == "libero_10":
            self.cfg.max_steps = 520  # longest training demo has 505 steps
        elif cfg.task_suite_name == "libero_90":
            self.cfg.max_steps = 400  # longest training demo has 373 steps

        if opts is not None:
            for key, value in opts.items():
                setattr(self.cfg, key, value)

        if self.cfg.load_step is None:
            checkpoint_files = os.listdir(os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints'))
            steps = [int(file.split('-')[1]) for file in checkpoint_files]
            self.cfg.load_step = max(steps)

        # name like step-000000-epoch-00-loss=0.0000.pt
        checkpoint_files = os.listdir(os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints'))
        load_step = '0' + str(self.cfg.load_step) if self.cfg.load_step < 100000 else str(self.cfg.load_step)
        checkpoint_file = [file for file in checkpoint_files if f"step-{load_step}" in file][0]
        self.cfg.pretrained_checkpoint = os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints', checkpoint_file)
        
    def evaluate(self):
        from libero.libero import benchmark

        self._set_results()
        self._build_logger()
        self.logger.infos('Config', vars(self.cfg))

        self.resize_size = get_image_resize_size(self.cfg)

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.cfg.task_suite_name]()
        num_tasks_in_suite = self.task_suite.n_tasks

        gpus = self._check_free_gpus()
        if self.cfg.num_gpus < len(gpus):
            gpus = gpus[:self.cfg.num_gpus]
        
        task_ids_and_episodes_all_processes = [[] for _ in range(self.cfg.num_processes)]
        idx = 0
        for task_id in range(num_tasks_in_suite):
            # task = self.task_suite.get_task(task_id).language
            for episode in range(self.cfg.num_trials_per_task):
                task_ids_and_episodes_all_processes[idx % self.cfg.num_processes].append((task_id, episode))
                idx += 1

        processes = []
        manager = multiprocessing.Manager()
        summaries = manager.list()
        
        for idx, task_ids_and_episodes in enumerate(task_ids_and_episodes_all_processes):
            gpu = gpus[idx % len(gpus)]
            self.logger.info(f'GPU {gpu}: {task_ids_and_episodes}')
            process = multiprocessing.Process(target=self.evaluate_episodes,
                                              args=(gpu, task_ids_and_episodes, idx == 0, summaries))
            processes.append(process)
            
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self._build_logger(mode='a')
        task_ids = set([summary["task_id"] for summary in summaries])
        for task_id in task_ids:
            task_summaries = [summary for summary in summaries if summary["task_id"] == task_id]
            success_rate = sum([summary["success"] for summary in task_summaries]) / len(task_summaries)
            task_description = task_summaries[0]['task']
            self.logger.info(f"Task {task_id} {task_description} success rate: {success_rate:.2f}")
        
        success_rate = sum([summary["success"] for summary in summaries]) / len(summaries)
        self.logger.info(f"Overall success rate: {success_rate:.2f}")
        self.logger.info("Evaluation finished.")
    
    def evaluate_episodes(self, gpu, task_ids_and_episodes, show_detail, summaries):
        try:
            model, processor = self._build_policy(gpu)
            reset_logging()
            self._build_logger(mode='a')

            for task_id, episode in task_ids_and_episodes:
                self.logger.info(f"GPU {gpu}: task {task_id} episode {episode}")
                summary = self.evalute_single(model, processor, task_id, episode, show_detail)
                summaries.append(summary)
        
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            with open(os.path.join(self.save_dir, f'error_gpu{gpu}.log'), 'w') as f:
                f.write(str(e) + '\n')
                f.write(traceback.format_exc())
            
    def evalute_single(self, model, processor, task_id, episode, show_detail):
        task = self.task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, self.cfg.model_family, resolution=self.resize_size)
        env.seed(episode)
        env.reset()

        # for libero object, we reset the environment
        # so the initial state is not the same as the training data
        if not self.cfg.task_suite_name == 'libero_object':
            initial_states = self.task_suite.get_task_init_states(task_id)
            obs = env.set_init_state(initial_states[episode])
        
        replay_images, replay_wrist_images = [], []
        texts = []
        timestep = 0
        success = False

        while timestep < self.cfg.max_steps + self.cfg.num_steps_wait:
            if timestep < self.cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(self.cfg.model_family))
                self._add_observation(obs, replay_images, replay_wrist_images)
                timestep += 1
                continue

            observation = self._prepare_inputs(obs, replay_images, replay_wrist_images)
            action, text = get_prismatic_vla_action(model, observation, task_description, 
                                                    self.cfg.unnorm_key, center_crop=self.cfg.center_crop)
            texts.append(text)
            if isinstance(action, list):
                action = [normalize_gripper_action(a, binarize=True) for a in action]
            else:
                action = normalize_gripper_action(action, binarize=True)
            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if self.cfg.model_family in ["openvla", "prismatic"]:
                if isinstance(action, list):
                    action = [invert_gripper_action(a) for a in action]
                else:
                    action = invert_gripper_action(action)

            if isinstance(action, list):
                for a in action:
                    obs, reward, done, info = env.step(a.tolist())
                    self._add_observation(obs, replay_images, replay_wrist_images)

                    timestep += 1
                    if show_detail:
                        self.logger.info(f"Step {timestep}: done {done}, {info}")
                    if done:
                        success = True
                        break
                if success:
                    break
            else:
                obs, reward, done, info = env.step(action.tolist())
                self._add_observation(obs, replay_images, replay_wrist_images)

                timestep += 1
                if show_detail:
                    self.logger.info(f"Step {timestep}: done {done}, {info}")
                if done:
                    success = True
                    break
        
        video_save_dir = os.path.join(self.save_dir, f'{task_id}_{task_description}')
        os.makedirs(video_save_dir, exist_ok=True)
        write_video(replay_images, os.path.join(video_save_dir, f'episode{episode}_success={success}.gif'), 
                    texts=None, fps=self.cfg.fps)
        
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": success}
            
    def _set_results(self):
        self.save_dir = os.path.join(self.cfg.save_root, 
                                     f'{self.cfg.task_suite_name}-{self.cfg.model_family}', 
                                     f'step_{self.cfg.load_step}-vqa_{self.cfg.with_vqa}_center_crop_{self.cfg.center_crop}_num-trials-per-task_{self.cfg.num_trials_per_task}')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _build_logger(self, mode='w'):
        self.logger = Logger(os.path.join(self.save_dir, '000.log'), mode=mode)

    def _check_free_gpus(self):
        """ Check free GPUs. """
        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        return [i for i, memory in enumerate(used_memorys) if memory < 1000]

    def _set_gpu(self, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # list_physical devices can avoid cuda error, don't know why
        import tensorflow as tf
        tf.config.list_physical_devices("GPU")
    
    def _build_policy(self, gpu):
        self._set_gpu(gpu)

        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import get_model, set_seed_everywhere

        set_seed_everywhere(self.cfg.seed)
        model = get_model(self.cfg)
        if self.cfg.with_vqa:
            from vla_scripts.openvla_with_vqa import OpenVLAWithVQA
            model = OpenVLAWithVQA(model, self.cfg.check_catch, self.cfg.check_close, self.cfg.vqa_mode)

        # [OpenVLA] Check that the model contains the action un-normalization key
        if self.cfg.model_family in ["openvla", "prismatic"]:
            if self.cfg.unnorm_key not in model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in model.norm_stats:
                self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
            assert self.cfg.unnorm_key in model.norm_stats, f"Action un-norm key {self.cfg.unnorm_key} not found in VLA `norm_stats`!"

        processor = None
        if self.cfg.model_family == "openvla":
            processor = get_processor(self.cfg)
        
        return model, processor
    
    def _add_observation(self, obs, replay_images, replay_wrist_images):
        image = get_libero_image(obs, self.resize_size)
        # Image.fromarray(image).save('test.png')
        replay_images.append(image)

        # use_wrist_image
        if self.cfg.use_wrist_image:
            wrist_img = get_libero_image(obs, self.resize_size, key="robot0_eye_in_hand_image")
            replay_wrist_images.append(wrist_img)

    def _prepare_inputs(self, obs, replay_images, replay_wrist_images):
        # buffering #obs_history images, optionally
        image_history = replay_images[-self.cfg.obs_history :]
        if len(image_history) < self.cfg.obs_history:
            image_history.extend([replay_images[-1]] * (self.cfg.obs_history - len(image_history)))

        # same but for optional wrist images
        if self.cfg.use_wrist_image:
            wrist_image_history = replay_wrist_images[-self.cfg.obs_history :]
            if len(wrist_image_history) < self.cfg.obs_history:
                wrist_image_history.extend(
                    [replay_wrist_images[-1]] * (self.cfg.obs_history - len(wrist_image_history))
                )
            # interleaved images [... image_t, wrist_t ...]
            image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

        # Prepare observations dict
        # Note: OpenVLA does not take proprio state as input
        return {
            "full_image": image_history,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    else:
        raise ValueError(f"Cannot convert {v} to boolean.")


def main(args):
    for step in args.steps:
        cfg = GenerateConfig(
            load_step=step, 
            pretrained_checkpoint=args.pretrained_checkpoint,
            num_trials_per_task=args.num_trials_per_task,
            num_gpus=args.num_gpus,
            num_processes=args.num_processes,
            task_suite_name=args.task_suite_name,
            save_root=args.save_root,
            with_vqa=str_to_bool(args.with_vqa),
            check_catch=str_to_bool(args.check_catch),
            check_close=str_to_bool(args.check_close),
            center_crop=str_to_bool(args.center_crop),
            vqa_mode=args.vqa_mode,
            obs_history=args.obs_history,
            use_wrist_image=args.use_wrist_image,
        )
        evaluator = ParallelLiberoEvaluator(cfg)
        evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--num-processes', type=int, default=32)
    parser.add_argument('--task-suite-name', default='libero_90')
    parser.add_argument('--num-trials-per-task', type=int, default=10)
    parser.add_argument('--pretrained-checkpoint', default='')
    parser.add_argument('--save-root', default='./results')
    parser.add_argument('--with-vqa', type=str, default='False')
    parser.add_argument('--check-catch', type=str, default='True')
    parser.add_argument('--check-close', type=str, default='False')
    parser.add_argument('--vqa-mode', default='coarse_direction')
    parser.add_argument('--steps', nargs='+', type=int)
    parser.add_argument('--obs-history', type=int, default=1)
    parser.add_argument('--use-wrist-image', action='store_true')
    parser.add_argument('--center-crop', type=str, default='True')
    args = parser.parse_args()
    main(args)
