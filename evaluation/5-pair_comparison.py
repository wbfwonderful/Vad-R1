from openai import OpenAI

import os
import base64
import csv
import json
import numpy as np
from PIL import Image
from io import BytesIO
import re


PROMPT = """Please act as an impartial and objective judge to evaluate which of two AI assistant responses better explains a given video event.

You will be given:
- A brief summary of the video, including whether the situation is **normal** or **abnormal**;
- Reasoning outputs from two AI assistants.

Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Answer as concisely as possible.

Note: The video may show either a normal or abnormal situation. If the response interprets a normal video as abnormal, or an abnormal video as normal, then the response is considered a failure. 

After your analysis, provide your final verdict in this format:
[[A]] — Assistant A is better
[[B]] — Assistant B is better
[[C]] — Both are equally good"""

# Set your API key and base URL from Qwen, OpenAI, or other compatible services
client = OpenAI(
  base_url="XX",
  api_key="XX",
)


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


# other baselines
pred_list = [
    "baseline1.jsonl",
    "baseline2.jsonl",

]
# Our model
base_pred = "results.jsonl"

gt_file_path = './Vad-Reasoning-SFT-test.jsonl'  

gt_data = load_jsonl_as_dict_gt(gt_file_path) 
base_data = load_jsonl_as_dict_output(base_pred)  



for pred_file_path in pred_list:
    i = 0
    model = pred_file_path.split("/")[-1].replace(".jsonl", "").replace("test_", "")
    pred_data = load_jsonl_as_dict_output(pred_file_path)  

    # ========================
    # pair_A or pair_B
    with open(f"./pair_A/{model}.jsonl", "w") as jsonl_file:
    # with open(f"./pair_B/{model}.jsonl", "w") as jsonl_file:
    # ========================

        for video_name, gt_item in gt_data.items():

            pred_item = pred_data.get(video_name)
            pred_base = base_data.get(video_name)
            i += 1
            print(i, model, video_name)
            if pred_item and pred_base:
                gt_think = remove_tags(gt_item)
                pred_think = pred_item

                try:
                    completion = client.chat.completions.create(
                        model="qwen-flash", # or other model
                        messages=[
                            {
                            "role": "user",
                            "content": [
                                {
                                "type": "text",
                                "text": PROMPT
                                },
                                {
                                # ========================

                                "type": "text",
                                # pair_A or pair_B
                                "text": f"Video Summary: {gt_think}; Assistant A: {pred_base}; Assistant B: {pred_item}"
                                # "text": f"Video Summary: {gt_think}; Assistant A: {pred_item}; Assistant B: {pred_base}"
                                # ========================
                                
                                }
                            ]
                            }
                        ]
                    )
                    output = (completion.choices[0].message.content).replace('\n', '')
                except Exception as e:
                    print("error")
                    output = ""

                output_data = {
                    "video": video_name,
                    "output": output  
                }

                json.dump(output_data, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")



