import json
import argparse

from modeling import process_dataset, produce_idefics_dataset, eval_model, load_model, load_processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt-path", type=str, default="HuggingFaceM4/idefics2-8b", help="Path to the model directory")
    parser.add_argument("--test-data-path", type=str, default="data/test/test_eval.json", help="Path to the test data")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--image-dir", type=str, default="data/train/nuscenes/samples", help="Path to the image directory")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
    parser.add_argument("--output-path", type=str, default="outputs/test-eval-idefics2", help="Output path for predictions")

    parser.add_argument("--apply-redcircle", action="store_true", help="Apply redcircle")
    parser.add_argument("--apply-context", type=str, default="chain", help="Apply context")
    parser.add_argument("--apply-verbalization", action="store_true", help="Apply redcircle verbalization")
    parser.add_argument("--apply-image-tagging", action="store_true", help="Apply image tagging")
    parser.add_argument("--context-from-gt", action="store_true", help="Use ground truth context")

    args = parser.parse_args()

    processor = load_processor(args.ckpt_path)
    model = load_model(args.ckpt_path, eval_mode=True, use_lora=args.use_lora, use_qlora=args.use_qlora, device=args.device)
    
    test_dataset = process_dataset(args.test_data_path, image_dir=args.image_dir)
    test_idefics_dataset = produce_idefics_dataset(test_dataset, apply_context=args.apply_context if args.context_from_gt else None)

    predictions = eval_model(model, test_idefics_dataset, processor, 
                             batch_size=args.batch_size, 
                             apply_context=args.apply_context if not args.context_from_gt else None, 
                             apply_redcircle=args.apply_redcircle, 
                             verbalize_refs=args.apply_verbalization,
                             chat_template="tagged" if args.apply_image_tagging else None)

    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=4)