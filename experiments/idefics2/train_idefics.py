import argparse
import wandb
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import train_test_split 
from data_utils import GVQADataCollator
import os

CHECKPOINT_DIR = "/home/rak/thinking-fast-and-furious/experiments/idefics2/models"


def parse_args():
    parser = argparse.ArgumentParser(description='Train IDEFICS model')
    parser.add_argument(
        "--steps",
        type=int,
        help="number of training steps",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="experiment name (for logging and checkpointing)",
    )
    
    parser.add_argument(
        "--train-data-path",
        type=str,
        help="train data (after mcq conversion) path",
    )
    return parser.parse_args()
    

def main():
    args = parse_args()
    with open("/home/rak/wandb.key", "r") as f:
        wandb_key = f.read().strip()
        
    wandb.login(key=wandb_key)
    wandb.init(
        project="DriveLM",
    )
    
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.experiment_name)
    
    processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False,
        init_lora_weights="gaussian"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.add_adapter(lora_config)
    model.enable_adapters()

    data = load_dataset('json', data_files=args.train_data_path, split=None)
    scene_ids = list(set(data['train']['scene_id']))
    # splitting by scene
    train_ids, val_ids = train_test_split(scene_ids, test_size=0.025)
    train_dataset = data['train'].filter(lambda x: x['scene_id'] in train_ids)
    val_dataset = data['train'].filter(lambda x: x['scene_id'] in val_ids)
    print(f'Train set: {len(train_dataset)}, Val set: {len(val_dataset)}')
    
    
    training_args = TrainingArguments(
        max_steps=args.steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=25,
        output_dir=checkpoint_dir,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.experiment_name,
        load_best_model_at_end=True,
    )
    data_collator = GVQADataCollator(processor, chat_template='tagged')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
