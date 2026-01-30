from openai import OpenAI

import os
import base64
import csv
import json
import numpy as np
from PIL import Image
from io import BytesIO
import re

PROMPT = """Provide your evaluation on the three dimensions below.  
For each dimension output a FLOAT score from 0.0 to 1.0 — higher means better (1 = fully satisfactory).

◆ Reasonability  
  • Is the causal chain complete and consistent with common sense / social norms?  
  • Is the reasoning self-consistent, with no contradictions or hallucinations?  
  • Higher score ⇒ more logical and coherent explanation.

◆ Detail  
  • Does the answer cover the key elements?  
  • Does it mention key anomaly details: actor, action, location, time span, outcome, etc.?  
  • Higher score ⇒ richer and more comprehensive details.

◆ Consistency  
  • Is the content factually aligned with the provided ground-truth scene, anomaly class, and time span?  
  • Does it avoid introducing objects/actions/scenes absent from the ground truth?  
  • Higher score ⇒ greater factual alignment and fewer hallucinations.

Output format: 
Respond with one line only: a Python dictionary string whose keys are  
reasonability, detail, and consistency, each mapped to a FLOAT.  
Do not include any explanations or extra text.

Example:  
{"reasonability": 0.83, "detail": 0.71, "consistency": 0.92}
"""


# Set your API key and base URL from Qwen, OpenAI, or other compatible services
client = OpenAI(
    api_key="",
    base_url="" 
)

test_model = "qwen-flash" # or other model
 

def remove_tags(text):
    text_wo_tags = re.sub(r'<[^>]+>', '', text).replace('\n', '')
    return text_wo_tags


def load_jsonl_as_dict_gt(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data[entry['video']] = entry['think'] + entry['answer']

    return data

def load_jsonl_as_dict_output(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            output = entry['output']

            data[entry['video']] = output
    return data



pred_list = [
    "baseline1.jsonl",
    "baseline2.jsonl",
    "our_model.jsonl",
]

# gt
gt_file_path = './Vad-Reasoning-SFT-test.jsonl' 

gt_data = load_jsonl_as_dict_gt(gt_file_path)  


i = 0

for pred_file_path in pred_list:

    model = pred_file_path.split("/")[-1].replace(".jsonl", "").replace("test_", "")
    pred_data = load_jsonl_as_dict_output(pred_file_path)  

    with open(f"./llm_{model}.jsonl", "w") as jsonl_file:
        for video_name, gt_item in gt_data.items():

            pred_item = pred_data.get(video_name)
            i += 1
            print(i, model, video_name)
            if pred_item:
                gt_think = gt_item
                pred_think = pred_item

                try:
                    response = client.chat.completions.create(
                        model=test_model,
                        messages=[
                            {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": "You are a intelligent assistant designed for evaluating the generative outputs for video-based pairs. You will be given two answers, one reference ground truth and one generated answer. Your task is to give the score of the predicted answers. Note: the reference ground truth is just one example of a correct answer. The generated answer can still be acceptable if it offers a reasonable and relevant alternative, even with different wording, structure, or focus."},
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": PROMPT
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Reference GT: {gt_think}"
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Generated answer: {pred_think}"
                                    },
                                ]
                            }
                        ]
                    )
                    output = response.choices[0].message.content
                except Exception as e:
                    print("error")
                    output = ""
                    
                output_data = {
                    "video": video_name,
                    "output": output 
                }

                json.dump(output_data, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")



