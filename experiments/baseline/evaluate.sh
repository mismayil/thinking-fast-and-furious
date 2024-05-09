#!/bin/bash

python ../drivelm/challenge/llama_adapter_v2_multimodal7b/demo.py --llama_dir /mnt/nlpdata1/share/models/llama2-7B --checkpoint /mnt/nlpdata1/home/ismayilz/cs503-project/llama_adapter_v2_finetuned/checkpoint-3.pth --data /mnt/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/test_eval_llama.json  --output ../test-output-tff.json --batch_size 4 --num_processes 1