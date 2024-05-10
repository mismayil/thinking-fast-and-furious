import argparse
import wandb
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from transformers.image_utils import load_image
from datasets import load_dataset
from sklearn.model_selection import train_test_split 
import os

CHECKPOINT_DIR = "experiments/eea/models/"
IMAGE_SRC_X, IMAGE_SRC_Y = 1600, 900
IMAGE_TGT_X, IMAGE_TGT_Y = int(IMAGE_SRC_X / 2.5), int(IMAGE_SRC_Y / 2.5)


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
        default='experiments/eea/data/nuscenes/train_v1_ext_idefics.json'
    )
    return parser.parse_args()
    


class GVQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            sample_images = [load_image(image_path).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_path in example['images'].values()]
            answer_text = example["answer"]
            answer_message = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_text}
                    ]
            }
            user_message = example['user_message'][0]
            messages = [user_message, answer_message]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(sample_images)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch


def main():
    args = parse_args()
    with open("/home/rak/wandb.key", "r") as f:
        wandb_key = f.read().strip()
        
    wandb.login(key=wandb_key)
    wandb.init(
        project="DriveLM",
    )
    
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        warmup_steps=20,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        output_dir=checkpoint_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.experiment_name,
    )
    data_collator = GVQADataCollator(processor)

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