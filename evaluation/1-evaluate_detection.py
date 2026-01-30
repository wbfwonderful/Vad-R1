
import re
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def remove_tags(text):
    text_wo_tags = re.sub(r'<[^>]+>', '', text).replace('\n', '')
    return text_wo_tags


def extract_when(text):
    when_match = re.search(r'<when>\[([0-9.]+)\s*,\s*([0-9.]+)\]</when>', text, re.DOTALL)
    if when_match:
        when_part = [float(when_match.group(1).strip()), float(when_match.group(2).strip())]  
        return when_part
    else:
        return [0.0, 0.0]


def extract_which(text):
    which_match = re.search(r'<which>(.*?)</which>', text, re.DOTALL)

    if which_match:
        which_part = which_match.group(1)  
        return which_part
    else:
        print("error")
        return None


def load_jsonl_as_dict_gt(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry['anomaly_type'] != 'Normal':
                gt = "Abnormal"
                time = [round(entry['start'] / entry['total_frames'], 3), round(entry['end'] / entry['total_frames'], 3)]
            else:
                gt = "Normal"
                time = [0, 1]

            
            data[entry['video']] = {
                'type': gt,
                'time': time
            }

    return data

def load_jsonl_as_dict_output(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            output = entry['output']
            which = extract_which(output)
            if which == "Normal":
                gt = "Normal"
                time = [0.0, 1.0]
            else:
                time = extract_when(output)
                gt = "Abnormal"

            data[entry['video']] = {
                'type': gt,
                'time': time
            }
    return data


def calculate_miou(gt_time, pred_time):
    intersection = max(0, min(gt_time[1], pred_time[1]) - max(gt_time[0], pred_time[0]))  
    union = max(gt_time[1], pred_time[1]) - min(gt_time[0], pred_time[0])  
    return intersection / union if union > 0 else 0  


def calculate_recall_at_threshold(gt_time, pred_time, threshold=0.5):
    intersection = max(0, min(gt_time[1], pred_time[1]) - max(gt_time[0], pred_time[0])) 
    union = max(gt_time[1], pred_time[1]) - min(gt_time[0], pred_time[0])  
    return 1 if intersection / union >= threshold else 0  

def calculate_metrics(gt_file_path, pred_file_path):
    gt_data = load_jsonl_as_dict_gt(gt_file_path)  
    pred_data = load_jsonl_as_dict_output(pred_file_path)  

    all_gt = []
    all_pred = []

    miou_scores = []
    r03_scores = []
    r05_scores = []
    r07_scores = []

    i = 0
    for video_name, gt_item in gt_data.items():

        pred_item = pred_data.get(video_name)
        
        if pred_item:
            all_gt.append(gt_item['type'])
            all_pred.append(pred_item['type'])

            if gt_item['type'] == pred_item['type']:

                miou = calculate_miou(gt_item['time'], pred_item['time'])
                miou_scores.append(miou)

                r03 = calculate_recall_at_threshold(gt_item['time'], pred_item['time'], threshold=0.3)
                r05 = calculate_recall_at_threshold(gt_item['time'], pred_item['time'], threshold=0.5)
                r07 = calculate_recall_at_threshold(gt_item['time'], pred_item['time'], threshold=0.7)
                
                r03_scores.append(r03)
                r05_scores.append(r05)
                r07_scores.append(r07)
            else:
                r03_scores.append(0.0)
                r05_scores.append(0.0)
                r07_scores.append(0.0)    
                miou_scores.append(0.0)            


    accuracy = accuracy_score(all_gt, all_pred)
    precision = precision_score(all_gt, all_pred, pos_label='Abnormal', average='binary')
    recall = recall_score(all_gt, all_pred, pos_label='Abnormal', average='binary')
    f1 = f1_score(all_gt, all_pred, pos_label='Abnormal', average='binary')

    miou_avg = np.mean(miou_scores)
    r03_avg = np.mean(r03_scores)
    r05_avg = np.mean(r05_scores)
    r07_avg = np.mean(r07_scores)


    results_cls = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou_avg,
        'r03': r03_avg,
        'r05': r05_avg,
        'r07': r07_avg
    }

    return results_cls

gt_file_path = '/XX/Vad-Reasoning-SFT-test.jsonl'  

pred_file_path = '/XX/results.jsonl' 


metrics = calculate_metrics(gt_file_path, pred_file_path)


for key, value in metrics.items():
    print(f"{key}: {value:.3f}")
