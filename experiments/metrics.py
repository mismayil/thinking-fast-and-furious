import re
import numpy as np
import evaluate 

bleu = evaluate.load("bleu")

def accuracy(preds, gts):
    accuracy_full = np.array(preds) == np.array(gts)
    accuracy_agg = accuracy_full.mean()
    return accuracy_agg, accuracy_full

def match_result(answer, GT):
    answer_nums = re.findall(r'\d+\.\d+', answer)
    GT_nums = re.findall(r'\d+\.\d+', GT)
    if len(answer_nums) % 2 != 0:
        print()
        print('STRIPPING THE LAST ONE:', answer_nums)
        print()
        answer_nums = answer_nums[:-1]
    # transform string into float
    answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
    GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)

    length = len(GT_nums)

    matched = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred in answer_nums:
        closest_distance = float('inf')
        closest_gt = None
        closest_id = None
        
        for i, gt in enumerate(GT_nums):
            distance = np.sum(np.abs(pred - gt))
            if distance < closest_distance:
                closest_distance = distance
                closest_gt = gt
                closest_id = i

        if closest_distance < 16:
            true_positives += 1
            matched.append(closest_gt)  
            GT_nums = np.delete(GT_nums, closest_id, axis=0) 
        else:
            false_positives += 1
        
    false_negatives = length - true_positives
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    return matched, F1

def distance_f1(preds, gts):
    f1_full = np.zeros(len(preds))
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        _, sample_f1 = match_result(pred, gt)
        f1_full[i] = sample_f1
    f1_agg = f1_full.mean()
    return f1_agg, f1_full


def calculate_bleu(preds, gts):
    bleus_full = np.zeros(len(preds))
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        bleus_full[i] = bleu.compute(predictions=[pred], references=[gt])['bleu']
    bleu_agg = bleus_full.mean()
    return bleu_agg, bleus_full