import re
import numpy as np
import evaluate 
import argparse
import json
from tqdm import tqdm

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

def process_scene(scene_id, scene):
    samples = {}
    for frame_id, frame in scene['key_frames'].items():
        question_id = 0
        for question_type, questions in frame['QA'].items():
            for question_info in questions:
                question = question_info['Q']
                answer = question_info['A'] if "A" in question_info else ""
                sample_id = f"{scene_id}_{frame_id}_{question_id}"
                question_id += 1
                samples[sample_id] = {
                    "question_type": question_type,
                    "question_text": question,
                    "answer": answer,
                    "tag": question_info['tag']
                }
    return samples

def process_dataset(data_path, output_path=None):
    with open(data_path, "r") as f:
        dataset = json.load(f)
    samples = {}
    for scene_id, scene in tqdm(dataset.items()):
        samples.update(process_scene(scene_id, scene))
    if output_path:
        with open(output_path, "w") as f:
            json.dump(samples, f)
    return samples

def load_prediction(json_path):
    with open(json_path, 'r') as f:
       prediction = json.load(f)
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-path', type=str, default="data/test/test_eval.json")
    parser.add_argument('--pred-path', type=str, default="outputs/fine-tuned/test-eval-idefics2-8b-fine-tuned-chain-gt.json")
    
    args = parser.parse_args()

    acc_preds, acc_gts, acc_ids = [], [], []

    bleu_preds, bleu_gts, bleu_ids = [], [], []

    distance_preds, distance_gts, distance_ids  = [], [], []

    predictions = load_prediction(args.pred_path)

    references = process_dataset(args.ref_path)

    for i, prediction in enumerate(predictions):
        question_type = prediction['question_type']
        tag = references[prediction['id']]['tag'][0]

        answer = prediction['answer']
        gt = prediction['gt']
        if tag == 0:
            acc_preds.append(answer)
            acc_gts.append(gt)
            acc_ids.append(i)
        if tag == 1:
            bleu_preds.append(answer)
            bleu_gts.append(gt)
            bleu_ids.append(i)
        if tag == 2:
            distance_preds.append(answer)
            distance_gts.append(gt)
            distance_ids.append(i)
    
    print('Accuracy:', accuracy(acc_preds, acc_gts)[0])
    print('BLEU:', calculate_bleu(bleu_preds, bleu_gts)[0])
    print('Distance avg F1:', distance_f1(distance_preds, distance_gts)[0])