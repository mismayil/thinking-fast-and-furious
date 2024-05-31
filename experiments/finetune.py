import pathlib
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import os
from accelerate import Accelerator
import random
import argparse

from modeling import process_dataset, produce_idefics_dataset, GVQADataCollator, load_model, prepare_object_detection_dataset, load_processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="HuggingFaceM4/idefics2-8b", help="Path to the model directory")
    parser.add_argument("--raw-train-data-path", type=str, default="data/train/nuscenes/v1_1_train_nus.json", help="Path to the raw training data")
    parser.add_argument("--train-data-path", type=str, default="data/train/nuscenes/v1_1_train_nus_ext_converted.json", help="Path to the preprocessed training data")
    parser.add_argument("--image-dir", type=str, default="data/train/nuscenes/samples", help="Path to the image directory")
    parser.add_argument("--cache-dir", type=str, default=".cache/huggingface", help="Path to the cache directory")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size per device")
    parser.add_argument("--grad-acc-steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging steps")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--save-strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--save-steps", type=int, default=100, help="Save steps")
    parser.add_argument("--eval-strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--report-to", type=str, default="wandb", help="Report to")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Save total limit")
    parser.add_argument("--run-name", type=str, default="idefics2-8b", help="Run name")
    parser.add_argument("--wandb-project", type=str, default="thinking-fast-and-furious", help="Wandb project name")

    parser.add_argument("--apply-redcircle", action="store_true", help="Apply redcircle")
    parser.add_argument("--apply-input-masking", action="store_true", help="Apply input masking")
    parser.add_argument("--apply-context", type=str, default="chain", help="Apply context")
    parser.add_argument("--apply-object-detection", action="store_true", help="Apply object detection")
    parser.add_argument("--apply-verbalization", action="store_true", help="Apply redcircle verbalization")
    parser.add_argument("--apply-image-tagging", action="store_true", help="Apply image tagging")
    
    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = args.wandb_project

    train_data_path = pathlib.Path(args.train_data_path)
    idefics_train_data_path = pathlib.Path(args.train_data_path).parent / f"{train_data_path.stem}_idefics.json"
    
    processor = load_processor(args.model_path)
    model = load_model(args.model_path, eval_mode=False, use_lora=args.use_lora, use_qlora=args.use_qlora, device=args.device, cache_dir=args.cache_dir)
    train_dataset = process_dataset(train_data_path, image_dir=args.image_dir)
    
    if args.apply_object_detection:
        train_od_dataset = prepare_object_detection_dataset(args.raw_train_data_path, image_dir=args.image_dir)
        train_dataset = train_od_dataset + train_dataset
        train_dataset = random.sample(train_dataset, len(train_dataset))

    train_idefics_dataset = produce_idefics_dataset(train_dataset, output_path=idefics_train_data_path, apply_context=args.apply_context)
    idefics_dataset = load_dataset('json', data_files=idefics_train_data_path, split=None)
    idefics_dataset = idefics_dataset["train"].train_test_split(test_size=0.025)

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=args.report_to,
        save_total_limit=args.save_total_limit,
        run_name=args.run_name
    )

    data_collator = GVQADataCollator(processor, chat_template="tagged" if args.apply_image_tagging else None,
                                     apply_redcircle=args.apply_redcircle,
                                     apply_input_masking=args.apply_input_masking,
                                     verbalize_refs=args.apply_verbalization)

    accelerator = Accelerator()
    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=idefics_dataset["train"],
        eval_dataset=idefics_dataset["test"]
    ))

    trainer.train()