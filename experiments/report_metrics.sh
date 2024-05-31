#!/bin/bash
prediction_path=${1}
test_path=${2:-"data/test/test_eval.json"}

echo "Challenge evaluation"
python ../drivelm/challenge/evaluation.py --root_path1 ${prediction_path} --root_path2 ${test_path}

echo "Our evaluation"
python metrics.py --ref-path ${test_path} --pred-path ${prediction_path}