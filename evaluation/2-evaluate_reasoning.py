
import re
import json

import numpy as np
import sys
sys.setrecursionlimit(5000) 

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import re

def calculate_bleu(reference_sentence, candidate_sentence):


    reference = [word_tokenize(reference_sentence.lower())]  
    candidate = word_tokenize(candidate_sentence.lower())  
  
    bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))  
    bleu_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)) 
    bleu_3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))  
    bleu_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)) 

    return bleu_1, bleu_2, bleu_3, bleu_4


def cal_meteor(reference_sentence, candidate_sentence):
    
    reference = [word_tokenize(reference_sentence.lower())] 
    candidate = word_tokenize(candidate_sentence.lower())  

    score = meteor_score(reference, candidate)

    return score

def cal_rouge(reference_sentence, candidate_sentence):
    rouge = Rouge()
    scores = rouge.get_scores(candidate_sentence, reference_sentence)

    rouge_1_r = scores[0]['rouge-1']['r']
    rouge_1_p = scores[0]['rouge-1']['p']
    rouge_1_f = scores[0]['rouge-1']['f']

    rouge_2_r = scores[0]['rouge-2']['r']
    rouge_2_p = scores[0]['rouge-2']['p']
    rouge_2_f = scores[0]['rouge-2']['f']
    
    rouge_l_r = scores[0]['rouge-l']['r']
    rouge_l_p = scores[0]['rouge-l']['p']
    rouge_l_f = scores[0]['rouge-l']['f']
    return rouge_1_r, rouge_1_p, rouge_1_f, rouge_2_r, rouge_2_p, rouge_2_f, rouge_l_r, rouge_l_p, rouge_l_f

def remove_tags(text):

    text_wo_tags = re.sub(r'<[^>]+>', '', text).replace('\n', '')
    return text_wo_tags


def load_jsonl_as_dict_gt(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data[entry['video']] = remove_tags(entry['think']) + remove_tags(entry['answer'])
            

    return data


def load_jsonl_as_dict_output(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            output = entry['output']

            data[entry['video']] = remove_tags(output)
    return data



def calculate_metrics(gt_file_path, pred_file_path):
    gt_data = load_jsonl_as_dict_gt(gt_file_path) 
    pred_data = load_jsonl_as_dict_output(pred_file_path) 

    results = {
        'bleu_1': [],
        'bleu_2': [],
        'bleu_3': [],
        'bleu_4': [],
        'meteor': [],
        'rouge_1_r': [],
        'rouge_1_p': [],
        'rouge_1_f': [],
        'rouge_2_r': [],
        'rouge_2_p': [],
        'rouge_2_f': [],
        'rouge_l_r': [],
        'rouge_l_p': [],
        'rouge_l_f': [],
    }

    i = 0
    for video_name, gt_item in gt_data.items():

        pred_item = pred_data.get(video_name)
        i += 1

        if pred_item:
            gt_think = gt_item
            pred_think = pred_item

            if len(pred_think) >= 5:
                bleu_scores = calculate_bleu(gt_think, pred_think)


                meteor_score_val = cal_meteor(gt_think, pred_think)

                rouge_scores = cal_rouge(gt_think, pred_think)

                results['bleu_1'].append(bleu_scores[0])
                results['bleu_2'].append(bleu_scores[1])
                results['bleu_3'].append(bleu_scores[2])
                results['bleu_4'].append(bleu_scores[3])
                results['meteor'].append(meteor_score_val)
                results['rouge_1_r'].append(rouge_scores[0])
                results['rouge_1_p'].append(rouge_scores[1])
                results['rouge_1_f'].append(rouge_scores[2])
                results['rouge_2_r'].append(rouge_scores[3])
                results['rouge_2_p'].append(rouge_scores[4])
                results['rouge_2_f'].append(rouge_scores[5])
                results['rouge_l_r'].append(rouge_scores[6])
                results['rouge_l_p'].append(rouge_scores[7])
                results['rouge_l_f'].append(rouge_scores[8])
            else:
                results['bleu_1'].append(0.0)
                results['bleu_2'].append(0.0)
                results['bleu_3'].append(0.0)
                results['bleu_4'].append(0.0)
                results['meteor'].append(0.0)
                results['rouge_1_r'].append(0.0)
                results['rouge_1_p'].append(0.0)
                results['rouge_1_f'].append(0.0)
                results['rouge_2_r'].append(0.0)
                results['rouge_2_p'].append(0.0)
                results['rouge_2_f'].append(0.0)
                results['rouge_l_r'].append(0.0)
                results['rouge_l_p'].append(0.0)
                results['rouge_l_f'].append(0.0)


    return results


gt_file_path = '/XX/Vad-Reasoning-SFT-test.jsonl'  

pred_file_path = '/XX/results.jsonl' 


metrics = calculate_metrics(gt_file_path, pred_file_path)


for key, value in metrics.items():
    print(f"{key}: {np.mean(value):.3f}")