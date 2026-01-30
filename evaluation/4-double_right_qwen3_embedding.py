from sentence_transformers import SentenceTransformer
import time
import json
import re
# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def extract_which(text):
    which_match = re.search(r'<which>(.*?)</which>', text, re.DOTALL)

    if which_match:
        which_part = which_match.group(1) 
        return which_part
    else:
        print("error")
        return None

def extract_think_content(text):
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[0]
    else:
        print("No <think> content found in the text.")

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
    "our_model.jsonl"
]

gt_file_path = 'Vad-Reasoning-SFT-test.jsonl'  

gt_data = load_jsonl_as_dict_gt(gt_file_path) 


for pred_file_path in pred_list:
    i = 0
    model_name = pred_file_path.split("/")[-1].replace(".jsonl", "").replace("test_", "")
    pred_data = load_jsonl_as_dict_output(pred_file_path)  

    with open(f"./qwen0_6B/double_right_{model_name}.jsonl", "w") as jsonl_file:
        for video_name, gt_item in gt_data.items():
            pred_item = pred_data.get(video_name)
            i += 1
            
            if pred_item:
                gt_think = gt_item
                pred_think = pred_item

                pred_type = extract_which(pred_think)
                gt_type = extract_which(gt_think)

                pred = model.encode([remove_tags(pred_think)], prompt="Represent this reasoning text for semantic similarity.")
                gt = model.encode([remove_tags(gt_think)], prompt="Represent this reasoning text for semantic similarity.")

                similarity = model.similarity(pred, gt)

                output_data = {
                    "video": video_name,
                    "similarity": similarity.item(),
                    "pred_type": pred_type,
                    "anomaly_type": gt_type
                }
                print(i, model_name, video_name, similarity.item())
                json.dump(output_data, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")

                