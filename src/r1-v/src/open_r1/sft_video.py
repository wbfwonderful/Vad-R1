# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

import swanlab
swanlab.sync_wandb()

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""


    SYSTEM_PROMPT = "You are a multimodal reasoning assistant tasked with analyzing videos."
    
    
    USER_PROMPT = """Your task is to analyze whether the given video is abnormal or normal. Think before answering, and generate:
1. A structured reasoning process enclosed in <think></think> tags
2. A final explanation enclosed in <answer></answer> tags

For abnormal videos, the reasoning should be based on a structured 4-step process:
<think> must include the following four steps enclosed in corresponding tags:
<step1>: Scene Description — Provide an objective overview of the environment and normal behaviors, without mentioning any abnormal activity or speculation.
<step2>: Abnormal Event Description — Describe the abnormal event and its approximate spatial location (e.g., bottom left of the frame), without explaining why it is abnormal.
<step3>: Abnormal Event Recognition — Explain why this event is considered abnormal compared to normal patterns or expectations.
<step4>: Causal Reasoning and Social Norms — Analyze potential negative consequences and explain how this behavior violates social norms or expectations.

<answer> must be a single, coherent paragraph in natural language, which includes exactly the following five tags:
<which>: Which type of anomaly occurred
<what>: What happened (describe the anomalous event)
<when>: When it happened, in normalized frame indices (e.g., <when>[0.25, 0.45]</when>)
<where>: Where it happened (use approximate spatial descriptions)
<why>: Why it is considered abnormal
<how>: How this behavior could cause harm or violate norms

Example Output for Abnormal Videos:
<think> 
<step1>The video shows ...</step1>
<step2>Next, we observe an abnormal event ...</step2> 
<step3>Based on these observations ...</step3> 
<step4>As a result, this behavior ...</step4> 
</think> 
<answer>
The anomaly category of the video is <which>...</which>. In this video, <what>a pedestrian ...</what>, occurring during the time range <when>[0.121, 0.826]</when>. The event takes place approximately in the <where>lower-left area of the frame</where>. This is considered abnormal because <why>pedestrians are expected to ...</why>. As a result, <how>such behavior could ...</how>.
</answer>

For normal videos, the reasoning should be simplified to just two steps:
<think> must include only the following two steps:
<step1>: Scene and Object Description — Provide a concise and objective overview of the environment and typical behaviors, without mentioning anomalies.
<step2>: Normal Event Explanation — Explain why the video is considered normal.

<answer> must be a single, coherent paragraph in natural language, which includes the following three tags:
<which>: Define the video as \"Normal.\"
<what>: A concise description of the event in the video.
<why>: Why it is considered normal.

Example Output for Normal Videos:
<think> 
<step1>The video shows scenes ...</step1> 
<step2>Based on the described scenes ...</step2> 
</think> 
<answer>
The video is classified as <which>Normal</which>. In this video, <what>people are ...</what>. This is considered normal because <why>...</why>
</answer>
"""


    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": example['path']
                },
                {
                    "type": "text",
                    "text": USER_PROMPT
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": example['think'] + "\n" + example['answer']}]
        }

    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}


    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )


    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    if "Qwen2_VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2_5_VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="VideoAnomaly-Reasoning-SFT")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
