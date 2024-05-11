import argparse
import re
import json
import numpy as np

def get_metrics(input_path):
    with open(input_path) as f:
        content = f.readlines()
    
    scores = {}

    for line in content:
        match = re.fullmatch(r"(?P<score_name>accuracy|chatgpt|match|final) score:\s*(?P<score>[0-9.]+)\s*", line)
        if match:
            scores[match.group("score_name")] = float(match.group("score"))
        elif line.startswith("language score:"):
            score_json = line.strip("language score:")
            scores["language"] = json.loads(score_json.replace("'", '"'))
    
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-paths", type=str, nargs="+", help="Input paths to metric files")
    parser.add_argument("-o", "--output-path", type=str, help="Output path for agg metrics")

    args = parser.parse_args()

    accuracy_scores = []
    chatgpt_scores = []
    match_scores = []
    language_scores = []
    final_scores = []

    for input_path in args.input_paths:
        scores = get_metrics(input_path)
        accuracy_scores.append(scores["accuracy"])
        chatgpt_scores.append(scores["chatgpt"])
        match_scores.append(scores["match"])
        language_scores.append(scores["language"])
        final_scores.append(scores["final"])
    
    agg_scores = {
        "accuracy": np.mean(accuracy_scores),
        "chatgpt": np.mean(chatgpt_scores),
        "match": np.mean(match_scores),
        "language": {key: np.mean([lscore[key] for lscore in language_scores]) for key in language_scores[0].keys()},
        "final": np.mean(final_scores)
    }

    with open(args.output_path, "w") as f:
        json.dump(agg_scores, f, indent=4)

if __name__ == "__main__":
    main()