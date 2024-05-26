#!/bin/bash

cd /mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb
${CONDA} run -n ${CONDA_ENV} python finetune.py > output.log 2>&1

#python demo.py --llama_dir /mnt/nlpdata1/share/models/llama2-7B --checkpoint ../llama_adapter_v2_ckpt/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth --data ../test_llama.json  --output ../output.json --batch_size 2 --num_processes 1