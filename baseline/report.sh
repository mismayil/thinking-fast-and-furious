#!/bin/bash
prediction_path=$1
python drivelm/challenge/evaluation.py --root_path1 $prediction_path --root_path2 /home/rak/thinking-fast-and-furious/drivelm/challenge/test.json