import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import os, json

from modeling import process_dataset, produce_idefics_dataset, eval_model

os.environ["HF_HOME"] = "/mnt/nlpdata1/home/ismayilz/.cache/huggingface"

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True
IMAGE_DIR = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/samples'

if __name__ == "__main__":
    test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/v1_1_val_nus_q_only.json'
    idefics_test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/v1_1_val_nus_q_only_idefics2.json'
    checkpoint_dir = "/mnt/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-vb/checkpoint-1000"
    model_dir = "HuggingFaceM4/idefics2-8b"
    
    processor = AutoProcessor.from_pretrained(
        model_dir,
        do_image_splitting=False
    )

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
        )

        model.enable_adapters()
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        ).to(DEVICE)
    
    test_dataset = process_dataset(test_data_path, image_dir=IMAGE_DIR)

    test_idefics_dataset = produce_idefics_dataset(test_dataset, output_path=idefics_test_data_path)

    predictions = eval_model(model, test_idefics_dataset, processor, batch_size=16)

    path = "/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/outputs/v1_1-val-idefics2-8b-fine-tuned-redcircle-vb-1000step.json"
    with open(path, "w") as f:
        json.dump(predictions, f, indent=4)