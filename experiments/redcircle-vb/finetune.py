import wandb
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import os
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

from modeling import process_dataset, produce_idefics_dataset, GVQADataCollator

os.environ["HF_HOME"] = "/mnt/nlpdata1/home/ismayilz/.cache/huggingface"

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

with open("/mnt/nlpdata1/home/ismayilz/.wandb.key", "r") as f:
    wandb_key = f.read().strip()
    
wandb.login(key=wandb_key)
wandb.init(
    project="thinking-fast-and-furious",
)
wandb_name=wandb.run.name

if __name__ == "__main__":
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    train_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/v1_1_train_nus_ext_converted.json'
    test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/drivelm/challenge/test_eval.json'
    idefics_train_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/train_idefics_redcircle_vb.json'
    idefics_test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/test_idefics_redcircle_vb.json'
    # checkpoint_dir = f"/home/cchang/CS503_VisualIntelligence/thinking-fast-and-furious/baseline/experiments/eea/models/idefics_redcircle/{wandb_name}"
    # checkpoint_dir = "/mnt/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-prime-music-8-500step/checkpoint-500"
    checkpoint_dir = "/mnt/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-vb"
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
            model_dir,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        ).to(DEVICE)
    
    train_dataset = process_dataset(train_data_path)
    test_dataset = process_dataset(test_data_path)

    train_idefics_dataset = produce_idefics_dataset(train_dataset, output_path=idefics_train_data_path)
    test_idefics_dataset = produce_idefics_dataset(test_dataset, output_path=idefics_test_data_path)

    data = load_dataset('json', data_files=idefics_train_data_path, split=None)
    split_dataset = data['train'].train_test_split(test_size=0.025)

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        output_dir=checkpoint_dir,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="idefics-8b-redcircle-vb"
    )

    data_collator = GVQADataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"], # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
    )

    trainer.train()