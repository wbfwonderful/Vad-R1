# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified, Qwen2VLGRPOAnomalyTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import swanlab
swanlab.sync_wandb()

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    anomaly_reward: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using anomaly reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<which>\s*(.*?)\s*</which>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "" 

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = sol
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

        except Exception as e:
            print(f"Error in reward_fn for question_type")
            reward = 0.0
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    

    pattern_normal = r"<think>\s*<step1>(.*?)</step1>\s*<step2>(.*?)</step2>\s*</think>\s*<answer>\s*(.*?)<which>(.*?)</which>(.*?)<what>(.*?)</what>(.*?)<why>(.*?)</why>(.*?)\s*</answer>"
    

    pattern_abnormal = r"<think>\s*<step1>(.*?)</step1>\s*<step2>(.*?)</step2>\s*<step3>(.*?)</step3>\s*<step4>(.*?)</step4>\s*</think>\s*<answer>\s*(.*?)<which>(.*?)</which>(.*?)<what>(.*?)</what>(.*?)<when>(.*?)</when>(.*?)<where>(.*?)</where>(.*?)<why>(.*?)</why>(.*?)<how>(.*?)</how>(.*?)\s*</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]
    

    matches_normal = [re.fullmatch(pattern_normal, content, re.DOTALL) for content in completion_contents]
    

    matches_abnormal = [re.fullmatch(pattern_abnormal, content, re.DOTALL) for content in completion_contents]
    

    results = []
    for match_normal, match_abnormal in zip(matches_normal, matches_abnormal):
        if match_normal:
            results.append(1.0)  
        elif match_abnormal:
            results.append(1.0)  
        else:
            results.append(0.0)  
    
    return results


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


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
<which>: Define the video as \"Abnormal\"
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
The video is classified as <which>Abnormal</which>. In this video, <what>a pedestrian ...</what>, occurring during the time range <when>[0.121, 0.826]</when>. The event takes place approximately in the <where>lower-left area of the frame</where>. This is considered abnormal because <why>pedestrians are expected to ...</why>. As a result, <how>such behavior could ...</how>.
</answer>

For normal videos, the reasoning should be simplified to just two steps:
<think> must include only the following two steps:
<step1>: Scene and Object Description — Provide a concise and objective overview of the environment and typical behaviors, without mentioning anomalies.
<step2>: Normal Event Explanation — Explain why the video is considered normal.

<answer> must be a single, coherent paragraph in natural language, which includes the following three tags:
<which>: Define the video as \"Normal\"
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

def main(script_args, training_args, model_args):

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def make_conversation_image_and_video(example):

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT
                        }
                        ]
                }]
            }
        
        return msg


    dataset = dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOAnomalyTrainer

    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
