from typing import Any, List, Dict, Optional
import numpy.typing as npt
import itertools
import random
import json
import torch
import numpy as np
from PIL import Image
import os
import argparse
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration

from tqdm import tqdm

from data import QualcommInteractiveCookingDatasetVideos
from utils import load_frames_into_array


@dataclass
class PromptConfig:
    """Holds a system prompt and a user text template for a given inference mode."""

    system_prompt: str
    user_text_template: str

    def format_user_text(self, **kwargs) -> str:
        """Format the user text template with the provided keyword arguments."""
        return self.user_text_template.format(**kwargs)


PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    "instruction_end": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant who is observing a person cook. "
        ),
        user_text_template=(
            "The person is currently at the following recipe step: {instruction} "
            "Has the person already completed the recipe step?  "
            "If the person has completed the recipe step answer 'YES' else answer 'NO'."
            "If you answer 'YES' describe why you think the person already completed the recipe step. "
        ),
    ),
    "mistake_inference": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant who is observing a person cook. "
            "You should look out for mistakes made by the person. "
        ),
        user_text_template=(
            "The person is trying to complete the following recipe step: {instruction}  "
            "Your task is to check if the person is about to make or has already made a mistake. "
            "Mistakes occur when the person performs actions that deviates from the instruction and DIRECTLY INTERFERES WITH SUCCESSFUL INSTRUCTION COMPLETION. "
            "Do not penalize actions that do not directly interfere with instruction completion (e.g. washing broccoli before cutting, if the instruction just says 'cut broccoli'). "
            "Here are some common types of mistakes that you should look out for: \n\n"
            "1. Technique Error: A mistake in how a step is physically performed. "
            "Examples include chopping with the wrong motion, stirring when folding is required, or spilling during transfer, "
            "producing uneven cuts or texture issues even when tools and amounts are right.\n"
            "Ignore minor technique errors such as not holding or gripping objects properly, holding with a risk of dropping etc. "
            "that do not interfere directly with recipe completion.\n "
            "2. Preparation Error: A setup mistake before executing the step . "
            "Using the wrong or dirty utensil, not washing/peeling/draining ingredients, insufficient draining of fluid, cutting/ chopping without peeling which makes correct execution difficult or unsafe.\n"
            "3. Measurement Error: An error in quantity — wrong counts, volumes, weights, or units. Mixing up teaspoons and tablespoons, "
            "misreading a scale, or miscounting items leads to off ratios and predictable taste or texture problems."
            "4. Temperature Error: A mistake in heat level or thermal state — the applied temperature, starting temperature, or thermal transition is wrong. "
            "Not preheating, using the wrong microwave power, overheating oil, or adding cold liquid when warm is required often causes burning, undercooking, or split emulsions."
            "5. Timing Error: A mistake in duration -- over- or under-doing a step or skipping required rests, proofs, or cooling periods. "
            "Overcooking, underblending, or cutting resting time short typically yields incorrect doneness or unstable textures.\n\n"
            "Assume the recipe step is still in progress. Your task is to identify any mistake that's already visible in the partially completed step. "
            "Do no penalize partially competed recipe steps. "
            "If you observe a mistake answer 'YES', else 'No'. "
            "Your response MUST BEGIN WITH 'YES' or 'NO'. In case you answer 'YES', please follow with a concise feedback to the user describing the mistake (i.e. YES. <feedback>.). Directly address the person."
        ),
    ),
}
    

def build_messages(mode: str, **kwargs) -> List[Dict]:
    """Build the chat message list for the given prompt mode and template variables."""
    if mode not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt mode: {mode}. Available modes: {list(PROMPT_CONFIGS.keys())}")

    config = PROMPT_CONFIGS[mode]
    user_text = config.format_user_text(**kwargs)

    messages = [
        {
            "role": "system",
            "content": config.system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]
    return messages


def load_prompt_configs_from_file(path: str) -> None:
    """Load and register prompt configs from a JSON file, overriding any existing entries."""
    with open(path, "r") as f:
        raw = json.load(f)
    for mode, cfg in raw.items():
        PROMPT_CONFIGS[mode] = PromptConfig(
            system_prompt=cfg["system_prompt"],
            user_text_template=cfg["user_text_template"],
    )


def load_model_and_processor(model_id, cache_dir=None):
    """Load the Qwen3-VL model and its processor from HuggingFace."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    print("Model device map:", model.hf_device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def get_qwen_vl_output(
    mode: str,
    model,
    processor,
    video=None,
    instruction: str = None,
    max_new_tokens: int = 128,
) -> str:
    """
    Run a single inference pass with Qwen3-VL.

    Args:
        mode: Prompt mode key (must exist in PROMPT_CONFIGS).
        model: The loaded Qwen3-VL model.
        processor: The associated processor.
        video: Sequence of video frames to pass as input.
        instruction: Recipe instruction text to embed in the prompt.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Decoded output string from the model.
    """
    assert mode in PROMPT_CONFIGS, f"Unknown mode: {mode}"

    template_kwargs: Dict[str, Any] = {"instruction": instruction}
    messages = build_messages(mode, **template_kwargs)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    video_metadata = {"fps": 2, "total_num_frames": len(video)}
    inputs = processor(
        text=[text],
        images=None,
        videos=video,
        padding=True,
        video_metadata=[video_metadata],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda").to(torch.bfloat16)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]


def run(args):
    """
    Main evaluation loop.

    Iterates over the dataset, streams video frames through a sliding window,
    queries the model to detect instruction completion and cooking mistakes,
    and saves predictions to disk after each video.
    """
    if args.prompt_config_file is not None:
        load_prompt_configs_from_file(args.prompt_config_file)

    video_input_resolution = (args.video_input_width, args.video_input_height)

    save_file_path = os.path.join(args.save_root, args.save_file)
    os.makedirs(args.save_root, exist_ok=True)
    dataset = QualcommInteractiveCookingDatasetVideos(
        captaincook4d_root = args.captaincook4d_root, 
        plan_set = args.plan_set,
        split = args.split, 
        model_fps = args.model_fps
    )

    model, processor = load_model_and_processor(args.model_id, args.cache_dir)

    predictions_to_save = []

    eval_idxs = list(range(len(dataset)))
    for eval_idx in tqdm(eval_idxs):
        print("=" * 80)
        curr_video_predictions_to_save = {}
        data = dataset[eval_idx]

        video_id = data["video_id"]
        curr_video_predictions_to_save["video_id"] = video_id
        curr_video_predictions_to_save["pred_texts"] = []
        curr_video_predictions_to_save["pred_timestamps"] = []

        # load entire video
        video = load_frames_into_array(
            data['video_frame_paths'],
            video_input_resolution=video_input_resolution
        )
        
        # initialize video buffer
        action_idx = 0
        init_video_seek = int((data["global_start_timestamp"] - data["video_frame_timestamps"][0]) * args.video_fps)
        video_buffer_start_index, video_buffer_seek_index = init_video_seek, init_video_seek + args.video_seek_amount

        pbar = tqdm(total=len(data["video_frame_paths"]) - init_video_seek)
        
        while action_idx < data["num_of_instruction_segments"] and \
            video_buffer_seek_index < len(data['video_frame_paths']):
            print("-*" * 40)
            gt_instruction = data["gt_texts"][action_idx][0]
            gt_is_mistake = data["gt_has_mistake"][action_idx]
            gt_mistakes_compiled = [x for x in data["gt_texts"][action_idx] if "Feedback" in x]
            
            gt_instruction_start_time = data["gt_text_timestamps"][action_idx][0]
            gt_instruction_end_time = data["gt_text_timestamps"][action_idx][-1]
            
            if args.turn_based:
                curr_video_predictions_to_save["pred_texts"].append("Instruction: " + gt_instruction)
                curr_video_predictions_to_save["pred_timestamps"].append(gt_instruction_start_time)

                # restart video stream from last instruction start
                init_video_seek = int(gt_instruction_start_time * args.video_fps)
                video_buffer_start_index = init_video_seek
                video_buffer_seek_index = init_video_seek + args.video_seek_amount
            else:
                curr_video_predictions_to_save["pred_texts"].append("Instruction: " + gt_instruction)
                curr_video_predictions_to_save["pred_timestamps"].append(
                    data["video_frame_timestamps"][video_buffer_seek_index]
                )

                # continue to stream video
                init_video_seek = video_buffer_seek_index - args.video_seek_amount
                video_buffer_start_index = init_video_seek
                video_buffer_seek_index = init_video_seek + args.video_seek_amount
            
            while video_buffer_seek_index < len(data['video_frame_paths']):
                print("-" * 40)
                # cap buffer length to avoid exceeding max_buffer_size
                video_buffer_start_index = max(
                    video_buffer_seek_index - args.max_buffer_size,
                    init_video_seek
                )
                
                print(
                    f"{gt_instruction}; "
                    f"GT segment end time: {gt_instruction_end_time:.1f}; "
                    f"GT segment has mistake:{gt_is_mistake}; "
                    f"GT_mistakes: {gt_mistakes_compiled}"
                )
                print(
                    f"Curr input clip start to end: {(video_buffer_start_index / 2.)} sec - {(video_buffer_seek_index / 2.)} sec -- "
                    f"Curr input clip len: {(video_buffer_seek_index - video_buffer_start_index) / 2.} sec / total len: {(len(data['video_frame_paths']) / 2.)} sec ."
                )

                # check if the person has completed current instruction
                pred_instruction_end = get_qwen_vl_output(
                    "instruction_end",
                    model,
                    processor,
                    video=video[video_buffer_start_index:video_buffer_seek_index],
                    instruction=gt_instruction,
                )

                print(f"Has the instruction been completed: {pred_instruction_end}")
                
                if "yes" in pred_instruction_end.lower():
                    # move on to next instruction
                    pred_instruction_end_time = data["video_frame_timestamps"][video_buffer_seek_index] - data["video_frame_timestamps"][0]

                    print(f"gt_instruction_end_time:{gt_instruction_end_time:.1f}, pred_instruction_end_time:{pred_instruction_end_time:.1f}")

                    curr_video_predictions_to_save["pred_texts"].append("Success: Good job!")
                    curr_video_predictions_to_save["pred_timestamps"].append(
                        data["video_frame_timestamps"][video_buffer_seek_index]
                    )

                    action_idx += 1
                    video_buffer_seek_index += args.video_seek_amount
                    pbar.update(args.video_seek_amount)
                    break
                else:
                    # check if person has made a mistake
                    mistake_inference = get_qwen_vl_output(
                        "mistake_inference",
                        model,
                        processor,
                        video=video[video_buffer_start_index:video_buffer_seek_index],
                        instruction=gt_instruction,
                        max_new_tokens=1024,
                    )

                    print(f"mistake_inference: {mistake_inference}")
                    
                    if "yes" in mistake_inference.lower():
                        predicted_feedback = mistake_inference.lower().replace("yes.", "").replace("yes", "").strip()
                        print(f"Saving feedback: --  {predicted_feedback}")
                        curr_video_predictions_to_save["pred_texts"].append(f"Feedback: {predicted_feedback}")
                        curr_video_predictions_to_save["pred_timestamps"].append(
                            data["video_frame_timestamps"][video_buffer_seek_index]
                        )

                    video_buffer_seek_index += args.video_seek_amount
                    pbar.update(args.video_seek_amount)

        print(f"Saving predictions to: {save_file_path}")
        predictions_to_save.append(curr_video_predictions_to_save)
        with open(save_file_path, 'w') as f:
            json.dump(predictions_to_save, f)
        
        print(">>>\n")


def parse_args():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Run cooking video evaluation with Qwen3-VL")

    parser.add_argument("--turn_based", action="store_true", default=False,
                        help="Whether to use turn-based mode")
    parser.add_argument("--dataset_name", type=str, default="eccv",
                        help="Dataset name")
    parser.add_argument("--save_root", type=str, default="./predictions",
                        help="Root directory to save predictions")
    parser.add_argument("--save_file", type=str, default="predictions.json",
                        help="Filename to save predictions")
    parser.add_argument("--captaincook4d_root", type=str,
                        required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--plan_set", type=str, default="main",
                        help="Annotation type")
    parser.add_argument("--model_fps", type=int, default=2,
                        help="Model FPS")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model ID to load from HuggingFace")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for model weights")
    parser.add_argument("--llm_max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for LLM generation")
    parser.add_argument("--max_buffer_size", type=int, default=240,
                        help="Maximum video buffer size in frames (2 minutes at 2fps)")
    parser.add_argument("--video_fps", type=int, default=2,
                        help="Video frames per second")
    parser.add_argument("--detection_time_diff_threshold", type=float, default=15.0,
                        help="Detection time difference threshold in frames (30 sec / 2 fps)")
    parser.add_argument("--video_seek_amount", type=int, default=10,
                        help="Number of frames to seek forward in video")
    parser.add_argument("--prompt_config_file", type=str, default=None,
                        help=(
                            "Optional path to a JSON file that overrides the default prompt configs. "
                            "Expected format: {\"<mode>\": {\"system_prompt\": \"...\", \"user_text_template\": \"...\"}, ...}"
                        ))
    parser.add_argument("--video_input_width", type=int, default=640,
                        help="Width of the video input resolution")
    parser.add_argument("--video_input_height", type=int, default=360,
                        help="Height of the video input resolution")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
