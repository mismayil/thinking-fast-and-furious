#!/usr/bin/bash

cd ../../drivelm/challenge/llama_adapter_v2_multimodal7b/
bash exps/finetune.sh \
        /mnt/nlpdata1/share/models/llama2-7B \
        /mnt/nlpdata1/home/ismayilz/cs503-project/models/llama_adapter_v2_pretrained/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth \
        /mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/baseline/finetune_data_config.yaml \
        /mnt/nlpdata1/home/ismayilz/cs503-project/models/llama_adapter_v2_finetuned