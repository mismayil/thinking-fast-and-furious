from transformers import AutoProcessor
import os, json, pathlib

from modeling import process_dataset, produce_idefics_dataset, eval_model, load_model

MNT_POINT = "/mnt/u14157_ic_nlp_001_files_nfs"

if not pathlib.Path(MNT_POINT).exists():
    MNT_POINT = "/mnt"

os.environ["HF_HOME"] = f"{MNT_POINT}/nlpdata1/home/ismayilz/.cache/huggingface"

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True
# IMAGE_DIR = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/samples"
IMAGE_DIR = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/samples"

if __name__ == "__main__":
    # test_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/v1_1_val_nus_q_only.json"
    test_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/drivelm/challenge/test_eval.json"
    # idefics_test_data_path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/val/nuscenes/v1_1_val_nus_q_only_idefics2.json"
    # checkpoint_dir = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-vb/checkpoint-1000"
    checkpoint_dir = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/models/idefics2-8b-chain-no-mask-1000"
    model_dir = "HuggingFaceM4/idefics2-8b"
    
    processor = AutoProcessor.from_pretrained(
        model_dir,
        do_image_splitting=False
    )
    
    model = load_model(checkpoint_dir, eval_mode=True, use_lora=USE_LORA, use_qlora=USE_QLORA, device=DEVICE)
    test_dataset = process_dataset(test_data_path, image_dir=IMAGE_DIR)

    test_idefics_dataset = produce_idefics_dataset(test_dataset)

    predictions = eval_model(model, test_idefics_dataset, processor, batch_size=2, apply_context="chain", apply_redcircle=False)

    # path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/outputs/v1_1-val-idefics2-8b-fine-tuned-redcircle-vb-1000step.json"
    path = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-vb/outputs/test-eval-idefics2-8b-fine-tuned-chain-no-mask-inf-pred-1000steps.json"
    with open(path, "w") as f:
        json.dump(predictions, f, indent=4)