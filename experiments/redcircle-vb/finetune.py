import pathlib
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import os
from accelerate import Accelerator
import argparse

from modeling import process_dataset, produce_idefics_dataset, GVQADataCollator, load_model, prepare_object_detection_dataset, load_processor

MNT_POINT = "/mnt/u14157_ic_nlp_001_files_nfs"

if not pathlib.Path(MNT_POINT).exists():
    MNT_POINT = "/mnt"

DEVICE = "cuda"
USE_LORA = False
USE_QLORA = True

with open(f"{MNT_POINT}/nlpdata1/home/ismayilz/.wandb.key", "r") as f:
    wandb_key = f.read().strip()
os.environ["WANDB_API_KEY"] = wandb_key
os.environ["WANDB_PROJECT"] = "thinking-fast-and-furious"
# HOME = "/home/azureuser"

if __name__ == "__main__":
    # raw_train_data_path = f"{HOME}/tff-data/nuscenes/v1_1_train_nus.json"
    # train_data_path = f"{HOME}/tff-data/nuscenes/v1_1_train_nus_ext_converted.json"
    train_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/v1_1_train_nus_ext_converted.json"
    # train_od_data_path = f"{HOME}/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/v1_1_train_nus_od.json"
    # idefics_train_data_path = f"{HOME}/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/train_idefics_redcircle_vb_chain_od.json"
    idefics_train_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/train_idefics_chain.json"
    # checkpoint_dir = f"{HOME}/tff-models/idefics2-redcircle-vb-chain-od"
    checkpoint_dir = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/models/idefics2-8b-chain-no-mask"
    model_dir = "HuggingFaceM4/idefics2-8b"
    
    processor = load_processor(model_dir)
    model = load_model(model_dir, eval_mode=False, use_lora=USE_LORA, use_qlora=USE_QLORA, device=DEVICE)
    train_dataset = process_dataset(train_data_path)
    # train_od_dataset = prepare_object_detection_dataset(raw_train_data_path, output_path=train_od_data_path)
    # train_dataset = train_dataset + train_od_dataset
    train_idefics_dataset = produce_idefics_dataset(train_dataset, output_path=idefics_train_data_path, apply_context="chain")
    idefics_dataset = load_dataset('json', data_files=idefics_train_data_path, split=None)
    idefics_dataset = idefics_dataset["train"].train_test_split(test_size=0.025)

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
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
        save_total_limit=2,
        run_name="idefics2-8b-chain-no-mask"
    )

    data_collator = GVQADataCollator(processor, apply_redcircle=False, apply_input_masking=False)

    accelerator = Accelerator()
    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=idefics_dataset["train"],
        eval_dataset=idefics_dataset["test"], # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
    ))

    trainer.train()