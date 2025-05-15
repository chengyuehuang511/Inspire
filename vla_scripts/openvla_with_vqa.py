import numpy as np
import torch
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM

from .datasets_with_vqa import find_target_objects, get_question_template, post_process_object


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_token_ids(mode='coarse_direction', catch=True, close=True, end=False):
    if mode == 'coarse_direction':
        ids = [454, 1291, 1419, 2359, 2923, 6951]
    elif mode == 'coarse_direction_3d':
        ids = [705, 1290, 1419, 1495, 2115, 6951]
    elif mode == 'fine_direction_3d':
        ids = [12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 220, 481]
    elif mode == 'coarse_distance':
        ids = [19656, 23559, 51659]
    elif mode == 'fine_distance':
        ids = [12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 220]
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")
    
    if catch:
        ids += [7173]
    if close:
        ids += [5552]
        if mode == 'coarse_direction_3d':
            ids += [3265]
    if end:
        ids += [151645]
    
    return ids


def get_max_tokens(mode='coarse_direction'):
    if mode == 'coarse_direction':
        return 1
    if mode == 'coarse_distance':
        return 1
    if mode == 'coarse_direction_3d':
        return 3
    raise NotImplementedError(f"Unsupported mode: {mode}")


class OpenVLAWithVQA:
    def __init__(self, vla, check_catch, check_close, mode):
        self.vla = vla
        self.check_catch = check_catch
        self.check_close = check_close
        self.mode = mode
        self.device = vla.device
        self.norm_stats = vla.norm_stats
        self.prompt_builder = vla.get_prompt_builder()
    
    @torch.inference_mode()
    def predict_action(self, image, instruction, unnorm_key=None, **kwargs):
        self.prompt_builder = self.vla.get_prompt_builder()
        target_objects = find_target_objects(instruction)
        for obj in target_objects:
            self._predict_vqa(image, obj, **kwargs)
        return self._predict_action(image, instruction, unnorm_key, **kwargs)

    @torch.inference_mode()
    def predict_features(self, image, instruction, unnorm_key=None, **kwargs):
        self.prompt_builder = self.vla.get_prompt_builder()
        target_objects = find_target_objects(instruction)
        for obj in target_objects:
            self._predict_vqa(image, obj, **kwargs)
        return self._predict_features(image, instruction, unnorm_key, **kwargs)
    
    def _predict_vqa(self, image, obj, **kwargs):
        vla = self.vla

        image_transform, tokenizer = vla.vision_backbone.get_image_transform(), vla.llm_backbone.tokenizer

        # Build VLA Prompt
        self.prompt_builder.add_turn(role="human", message=get_question_template(self.mode).format(post_process_object(obj)))
        prompt_text = self.prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(vla.device)

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(vla.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(vla.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = vla.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=vla.enable_mixed_precision_training):
            # fmt: off
            outputs = super(PrismaticVLM, vla).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=get_max_tokens(self.mode),
                output_scores=True, 
                return_dict_in_generate=True,
                **kwargs
            )
        # Shape: [seq, vocab_size]
        logits = torch.concat(outputs["scores"])
        
        # mask if the token is not a direction
        mask = torch.zeros_like(logits)
        for i, token_id in enumerate(get_token_ids(self.mode, self.check_catch, self.check_close)):
            mask[:, token_id] = 1
        logits = logits * mask
        generated_ids = torch.argmax(logits, dim=-1)
        
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if 'catch' in answer:
            answer = 'catch'
        
        self.prompt_builder.add_turn(role="gpt", message=answer)
        return answer
        
    def _predict_action(self, image, instruction, unnorm_key = None, **kwargs):
        vla = self.vla
        
        image_transform, tokenizer = vla.vision_backbone.get_image_transform(), vla.llm_backbone.tokenizer

        # Build VLA Prompt
        self.prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = self.prompt_builder.get_prompt()
        # print(prompt_text)

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(vla.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(vla.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(vla.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = vla.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=vla.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, vla).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=vla.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -vla.get_action_dim(unnorm_key) :]
        normalized_actions = vla.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
        actions = self._unnormalize_action(normalized_actions, unnorm_key)
        # print(actions)
        return actions, prompt_text

    def _predict_features(self, image, instruction, unnorm_key = None, **kwargs):
        vla = self.vla
        
        image_transform, tokenizer = vla.vision_backbone.get_image_transform(), vla.llm_backbone.tokenizer

        # Build VLA Prompt
        self.prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = self.prompt_builder.get_prompt()
        # print(prompt_text)

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(vla.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(vla.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(vla.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = vla.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=vla.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, vla).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=vla.get_action_dim(unnorm_key),
                output_scores=True, 
                output_hidden_states=True, 
                output_attentions=True,
                return_dict_in_generate=True,
                **kwargs
            )
            # fmt: on
        return output
    
    def _unnormalize_action(self, normalized_actions, unnorm_key):
        # Un-normalize Actions
        vla = self.vla
        action_norm_stats = vla.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        if isinstance(normalized_actions, list):
            actions = [np.where(mask, 0.5 * (action + 1) * (action_high - action_low) + action_low, action) for action in normalized_actions]
        else:
            actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
        return actions