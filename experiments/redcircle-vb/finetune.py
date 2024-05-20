import wandb
from transformers import AutoProcessor
import pathlib
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from modeling import process_dataset, produce_idefics_dataset, GVQADataCollator, load_model

MNT_POINT = "/mnt/u14157_ic_nlp_001_files_nfs"

if not pathlib.Path(MNT_POINT).exists():
    MNT_POINT = "/mnt"

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

with open(f"{MNT_POINT}/nlpdata1/home/ismayilz/.wandb.key", "r") as f:
    wandb_key = f.read().strip()
    
wandb.login(key=wandb_key)
wandb.init(
    project="thinking-fast-and-furious",
)
wandb_name=wandb.run.name

if __name__ == "__main__":
    train_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/v1_1_train_nus_ext_converted.json"
    test_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/drivelm/challenge/test_eval.json"
    idefics_train_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/data/nuscenes/train_idefics_redcircle_vb_chain.json"
    checkpoint_dir = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-vb-chain"
    model_dir = "HuggingFaceM4/idefics2-8b"
    
    processor = AutoProcessor.from_pretrained(
        model_dir,
        do_image_splitting=False
    )

    model = load_model(model_dir, eval_mode=False, use_lora=USE_LORA, use_qlora=USE_QLORA, device=DEVICE)
    train_dataset = process_dataset(train_data_path, apply_context="chain")
    train_idefics_dataset = produce_idefics_dataset(train_dataset, output_path=idefics_train_data_path)
    idefics_dataset = load_dataset('json', data_files=idefics_train_data_path, split=None)
    idefics_dataset = idefics_dataset["train"].train_test_split(test_size=0.025)

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
        report_to="wandb"
    )

    data_collator = GVQADataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=idefics_dataset["train"],
        eval_dataset=idefics_dataset["test"], # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
    )

    trainer.train()