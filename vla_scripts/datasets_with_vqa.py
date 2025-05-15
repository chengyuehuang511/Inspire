import json
import numpy as np
import os
import torch
from collections import OrderedDict
from PIL import Image

from prismatic.models.backbones.vision.base_vision import WrapSequenceImageTransform
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


_OBJECT_INFO_ROOT = '/path/to/data_root/object_infos'
IGNORE_INDEX = -100

_OBJECT_NAME_MAP = OrderedDict()

_OBJECT_NAME_MAP['the black bowl between the plate and the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl from table center'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl in the top drawer of the wooden cabinet'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the cookie box'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the plate'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the cookie box'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the stove'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the wooden cabinet'] = 'akita_black_bowl_1_main'

_OBJECT_NAME_MAP['the front black bowl'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the middle black bowl'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the back black bowl'] = 'akita_black_bowl_3_main'
_OBJECT_NAME_MAP['the black bowl at the front'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl in the middle'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the black bowl at the back'] = 'akita_black_bowl_3_main'
_OBJECT_NAME_MAP['the black bowl on the left'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl'] = 'akita_black_bowl_1_main'


_OBJECT_NAME_MAP['the left bowl'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the right bowl'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the bowl'] = 'akita_black_bowl_1_main'

_OBJECT_NAME_MAP['the wine_bottle'] = 'wine_bottle_1_main'
_OBJECT_NAME_MAP['the ketchup'] = 'ketchup_1_main'
_OBJECT_NAME_MAP['the frying pan'] = 'chefmate_8_frypan_1_main'

_OBJECT_NAME_MAP['the left moka pot'] = 'moka_pot_2_main'
_OBJECT_NAME_MAP['the right moka pot'] = 'moka_pot_1_main'
_OBJECT_NAME_MAP['the moka pot'] = 'moka_pot_1_main'

_OBJECT_NAME_MAP['the yellow and white mug'] = 'white_yellow_mug_1_main'
_OBJECT_NAME_MAP['the white mug'] = 'porcelain_mug_1_main'
_OBJECT_NAME_MAP['the red mug'] = 'red_coffee_mug_1_main'

_OBJECT_NAME_MAP['the white bowl'] = 'white_bowl_1_main'

_OBJECT_NAME_MAP['the butter at the back'] = 'butter_2_main'
_OBJECT_NAME_MAP['the butter at the front'] = 'butter_1_main'
_OBJECT_NAME_MAP['the butter'] = 'butter_1_main'

_OBJECT_NAME_MAP['the chocolate pudding'] = 'chocolate_pudding_1_main'

_OBJECT_NAME_MAP['the alphabet soup'] = 'alphabet_soup_1_main'
_OBJECT_NAME_MAP['the cream cheese'] = 'cream_cheese_1_main'
_OBJECT_NAME_MAP['the cream cheese box'] = 'cream_cheese_1_main'
_OBJECT_NAME_MAP['the tomato sauce'] = 'tomato_sauce_1_main'
_OBJECT_NAME_MAP['the milk'] = 'milk_1_main'
_OBJECT_NAME_MAP['the orange juice'] = 'orange_juice_1_main'
_OBJECT_NAME_MAP['the salad dressing'] = ['new_salad_dressing_1_main', 'salad_dressing_1_main']
_OBJECT_NAME_MAP['the bbq sauce'] = 'bbq_sauce_1_main'

_OBJECT_NAME_MAP['the book on the left'] = 'yellow_book_2_main'
_OBJECT_NAME_MAP['the book on the right'] = 'yellow_book_1_main'
_OBJECT_NAME_MAP['the book'] = 'black_book_1_main'

_OBJECT_NAME_MAP['the left plate'] = 'plate_1_main'
_OBJECT_NAME_MAP['the right plate'] = 'plate_2_main'
_OBJECT_NAME_MAP['the plate'] = 'plate_1_main'

_OBJECT_NAME_MAP['the top drawer of the cabinet'] = ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top']
_OBJECT_NAME_MAP['the middle drawer of the cabinet'] = ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top']
_OBJECT_NAME_MAP['the bottom drawer of the cabinet'] = ['white_cabinet_1_cabinet_bottom', 'wooden_cabinet_1_cabinet_bottom']

_OBJECT_NAME_MAP['top of the cabinet'] = ['white_cabinet_1_main', 'wooden_cabinet_1_main', 'wooden_two_layer_shelf_1_main']

_OBJECT_NAME_MAP['on top of the shelf'] = 'wooden_two_layer_shelf_1_main'
_OBJECT_NAME_MAP['on the cabinet shelf'] = 'wooden_two_layer_shelf_1_main'
_OBJECT_NAME_MAP['under the cabinet shelf'] = 'wooden_two_layer_shelf_1_main'

_OBJECT_NAME_MAP['turn on the stove'] = 'flat_stove_1_button'
_OBJECT_NAME_MAP['turn off the stove'] = 'flat_stove_1_button'
_OBJECT_NAME_MAP['the stove'] = 'flat_stove_1_burner'

_OBJECT_NAME_MAP['the tray'] = 'wooden_tray_1_main'
_OBJECT_NAME_MAP['the wine rack'] = 'wine_rack_1_main'
_OBJECT_NAME_MAP['the rack'] = 'wine_rack_1_main'
_OBJECT_NAME_MAP['the microwave'] = 'microwave_1_main'
_OBJECT_NAME_MAP['the basket'] = 'basket_1_main'
_OBJECT_NAME_MAP['the caddy'] = 'desk_caddy_1_main'

_OBJECT_NAME_MAP['on it'] = 'flat_stove_1_burner'


def find_target_objects(lang):
    if lang == 'put both moka pots on the stove':
        return ['the left moka pot', 'the right moka pot', 'the stove']
    target_objects = []
    for obj in _OBJECT_NAME_MAP:
        if obj in lang:
            target_objects.append(obj)
            lang = lang.replace(obj, '')
    return target_objects


def post_process_object(obj):
    return obj.replace('turn on the stove', 'the stove button') \
              .replace('turn off the stove', 'the stove button') \
              .replace('on top of the shelf', 'top of the shelf') \
              .replace('on the cabinet shelf', 'top of the cabinet shelf') \
              .replace('under the cabinet shelf', 'bottom of the cabinet self') \
              .replace('on it', 'the stove')


def get_relation_to_robot(obj, obj_info, thresh=0.06, check_catch=True, check_close=True, mode='coarse_direction'):
    if mode in obj_info:
        return obj_info[mode][obj]
    
    # print(obj_info['gripper_to_obj'])
    aliases = _OBJECT_NAME_MAP[obj]
    if isinstance(aliases, str):
        alias = aliases
    else:
        alias = None
        for a in aliases:
            if a in obj_info['gripper_to_obj']:
                alias = a
                break
        if alias is None:
            raise ValueError(f"Cannot find alias for {obj} in object info, aliases: {aliases}, available objects: {list(obj_info.get('gripper_to_obj').keys())}.")
    
    gripper_to_obj = obj_info['gripper_to_obj'][alias]
    
    if alias in ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top', 'white_cabinet_1_main', 'wooden_cabinet_1_main']:
        gripper_to_obj[2] = gripper_to_obj[2] + 0.22152
    
    if alias == 'basket_1_main':
        gripper_to_obj[2] = gripper_to_obj[2] + 0.07185
    
    if alias == 'wine_rack_1_main':
        gripper_to_obj[2] = gripper_to_obj[2] + 0.05903
        
    if alias == 'wooden_two_layer_shelf_1_main' and obj in ['on top of the shelf', 'on the cabinet shelf', 'top of the cabinet']:
        gripper_to_obj[2] = gripper_to_obj[2] + 0.22152
    
    if alias.endswith('_main'):
        is_grasp = obj_info['is_grasp'].get(alias[:-5], False)
    elif alias in obj_info['is_grasp']:
        is_grasp = obj_info['is_grasp'][alias]
    else:
        is_grasp = False
    
    if check_catch and is_grasp:
        return 'catch'

    if not check_close:
        thresh = 0

    if mode == 'coarse_direction':
        # return the direction with the largest absolute distance of each axis
        max_idx = np.argmax(np.abs(gripper_to_obj))
        if max_idx == 0:
            if gripper_to_obj[0] < -thresh:
                return 'back'
            if gripper_to_obj[0] > thresh:
                return 'front'
            return 'close'

        if max_idx == 1:
            if gripper_to_obj[1] < -thresh:
                return 'left'
            if gripper_to_obj[1] > thresh:
                return 'right'

        if gripper_to_obj[2] < -thresh:
            return 'down'
        if gripper_to_obj[2] > thresh:
            return 'up'
        return 'close'
    
    elif mode == 'coarse_direction_3d':
        # return the direction of x, y, z
        outputs = []
        if gripper_to_obj[0] < -thresh:
            outputs.append('back')
        elif gripper_to_obj[0] > thresh:
            outputs.append('front')
        else:
            outputs.append('close')

        if gripper_to_obj[1] < -thresh:
            outputs.append('left')
        elif gripper_to_obj[1] > thresh:
            outputs.append('right')
        else:
            outputs.append('close')
            
        if gripper_to_obj[2] < -thresh:
            outputs.append('down')
        elif gripper_to_obj[2] > thresh:
            outputs.append('up')
        else:
            outputs.append('close')
        return ' '.join(outputs)
    
    elif mode == 'fine_direction_3d':
        # return the accurate direction of x, y, z
        outputs = [int(gripper_to_obj[0] * 10), int(gripper_to_obj[1] * 10), int(gripper_to_obj[2] * 10)]
        outputs_clipped = []
        for output in outputs:
            if output > 9:
                output = 9
            elif output < -9:
                output = -9
            outputs_clipped.append(str(output))
        return ' '.join(outputs_clipped)
    
    elif mode == 'coarse_distance':
        # return near, middle or far
        distance = np.linalg.norm(gripper_to_obj)
        if distance < thresh:
            return 'close'
        elif distance < 0.2:
            return 'near'
        elif distance < 0.3:
            return 'middle'
        return 'far'
    
    elif mode == 'fine_distance':
        # return accurate distance
        distance = np.linalg.norm(gripper_to_obj)
        distance *= 10
        if distance > 9:
            distance = 9
        return str(int(distance))
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_question_template(mode):
    if mode == 'coarse_direction':
        return 'In which direction is {} relative to the robot?'
    if mode == 'coarse_direction_3d':
        return 'In which direction is {} relative to the robot? x, y, z:'
    if mode == 'fine_direction_3d':
        return 'What is the accurate position of {} relative to the robot? x, y, z:'
    if mode == 'coarse_distance':
        return 'What is the distance between the robot and {}?'
    if mode == 'fine_distance':
        return 'What is the accurate distance between the robot and {}?'
    raise ValueError(f"Unknown mode: {mode}")


def mask_labels(labels):
    def indexof_sublist(sublst, lst):
        for i in range(len(lst) - len(sublst) + 1):
            if (lst[i:i + len(sublst)] == sublst).all():
                return i
        return -1
    
    # <start> system
    system_start = torch.tensor([151644, 8948])
    # <end>
    system_end = torch.tensor([151645])
    # \n <start> user
    user_start = torch.tensor([198, 151644, 872])
    # <start> assistant \n
    user_end = torch.tensor([151644, 77091, 198])

    # mask system prompt
    start_idx = indexof_sublist(system_start, labels)
    end_idx = indexof_sublist(system_end, labels) + len(system_end)
    labels[start_idx:end_idx] = IGNORE_INDEX
    
    # mask user prompt
    start_idx = indexof_sublist(user_start, labels)
    end_idx = indexof_sublist(user_end, labels) + len(user_end)
    while start_idx != -1:
        labels[start_idx:end_idx] = IGNORE_INDEX
        start_idx = indexof_sublist(user_start, labels)
        end_idx = indexof_sublist(user_end, labels) + len(user_end)
    
    return labels


class RLDSBatchTransformWithVQA(RLDSBatchTransform):
    def __init__(self, name, check_catch=True, check_close=True, mode='coarse_direction', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.check_catch = check_catch
        self.check_close = check_close
        self.mode = mode
    
    def __call__(self, rlds_batch):
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # either a single or multi image, depending on image_window_size
        if self.image_window_size == 1:
            img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            if self.use_wrist_image:
                img = [img, Image.fromarray(rlds_batch["observation"]["image_wrist"][0])]
        else:
            img = [Image.fromarray(rlds_batch["observation"]["image_primary"][t]) for t in range(self.image_window_size)]
            if self.use_wrist_image:
                # wrist images are interleaved
                wrist_img = [
                    Image.fromarray(rlds_batch["observation"]["image_wrist"][t]) for t in range(self.image_window_size)
                ]
                img = [val for tup in zip(img, wrist_img) for val in tup]

        conversation = []

        # if there is no action horizon, remove it here.
        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            # get the last FH + 1 actions (current action + future ones) if required
            action = action[-self.action_tokenizer.required_future_horizon - 1 :]

        tokenized_action = self.action_tokenizer(action)
        raw_action_tokens = self.base_tokenizer(tokenized_action)["input_ids"]
        
        obj_info_id = rlds_batch["object_info"].decode('utf-8')

        if 'union' not in self.name:
            object_info_dir = f'{_OBJECT_INFO_ROOT}/{self.name}_no_noops/object_infos'
        else:
            object_info_dir = _OBJECT_INFO_ROOT.replace('object_infos', 'merged_object_infos')
            
        with open(os.path.join(object_info_dir, f'{obj_info_id}.json'), 'r') as f:
            obj_info = json.load(f)
        
        for obj in find_target_objects(lang):
            conversation.extend(
                [
                    {"from": "human", "value": get_question_template(self.mode).format(post_process_object(obj))},
                    {"from": "gpt", "value": get_relation_to_robot(obj, obj_info, check_catch=self.check_catch, check_close=self.check_close, mode=self.mode)},
                ]
            )

        conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": tokenized_action},
            ]
        )
        # num_answer_tokens = len(raw_action_tokens)

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        # print(prompt_builder.get_prompt())
        # print('catch<|im_end|>' in prompt_builder.get_prompt(), 'close<|im_end|>' in prompt_builder.get_prompt())
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # critical, some tokenizers have different numbers of "end tokens".
        num_end_tokens = 1
        if isinstance(self.base_tokenizer, Qwen2TokenizerFast):
            # Qwen has <|im_end|><|endoftext|> for example
            num_end_tokens = 2

        # labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        labels = mask_labels(labels)
        if not self.predict_stop_token:
            labels[-num_end_tokens:] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


def get_vla_dataset_and_collator(
    data_root_dir,
    data_mix,
    image_transform,
    tokenizer,
    prompt_builder_fn,
    default_image_resolution,
    padding_side="right",
    predict_stop_token=True,
    shuffle_buffer_size=100_000,
    train=True,
    episodic=False,
    image_aug=False,
    action_tokenizer="action_tokenizer",
    future_action_window_size=0,
    image_window_size=1,
    use_wrist_image=False,
    check_catch=True,
    check_close=True,
    mode="coarse_direction",
):
    action_tokenizer = ACTION_TOKENIZERS[action_tokenizer](tokenizer)

    # get the future action window needed from the tokenizer
    future_action_window_size = max(action_tokenizer.required_future_horizon, future_action_window_size)

    load_camera_views = ("primary", "wrist") if use_wrist_image else ("primary",)

    # get the observation history from the image_transform (only needed if its a WrapSequence transform)
    if isinstance(image_transform, WrapSequenceImageTransform):
        if use_wrist_image:
            # expects groupings of two in image sequence len
            assert image_transform.sequence_len % 2 == 0, "With wrist images, image transform must expect 2N images!"
            image_window_size = max(image_transform.sequence_len // 2, image_window_size)
        else:
            image_window_size = max(image_transform.sequence_len, image_window_size)

    batch_transform = RLDSBatchTransformWithVQA(
        name=data_mix,
        check_catch=check_catch,
        check_close=check_close,
        mode=mode,
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        image_window_size=image_window_size,
        use_wrist_image=use_wrist_image,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    if episodic:
        raise NotImplementedError("EpisodicRLDSDataset not yet implemented.")

    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        future_action_window_size=future_action_window_size,
        image_window_size=image_window_size,
        load_camera_views=load_camera_views,
    )

    return dataset, action_tokenizer, collator
