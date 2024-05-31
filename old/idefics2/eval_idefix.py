from data_utils import load_and_resize_images, vizualize_frames, TAGGED_CHAT_TEMPLATE
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from tqdm import tqdm
import torch
import copy
import json
import argparse
from termcolor import colored
from datasets import load_dataset

def eval_model(model, processor, test_set, verbose=False):
    model.eval()
    predictions = []
    device = model.device
    for idefics_sample in tqdm(test_set):
        image_paths = idefics_sample['images']
        images = load_and_resize_images(idefics_sample)
        if verbose:
            vizualize_frames(image_paths)
    
        prompt = processor.apply_chat_template(idefics_sample['user_message'], add_generation_prompt=True, chat_template=TAGGED_CHAT_TEMPLATE)
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
        predicted_text = generated_texts[0].split('\n')[-1][len("Assistant: "):]
        prediction = copy.deepcopy(idefics_sample)
        prediction['gt'] = prediction['answer']
        prediction['answer'] = predicted_text
        predictions.append(prediction)
        if verbose:
            print(colored(idefics_sample['question_text'], 'blue'))
            print(colored('Predicted:', 'blue'), predicted_text)
            print(colored('GT:', 'blue'), prediction['gt'])
    return predictions



def parse_args():
    parser = argparse.ArgumentParser(description='Eval IDEFICS model')
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="checkpoint path of the model to be evaluated",
    )
    
    parser.add_argument(
        "--test-data-path",
        type=str,
        default='/home/rak/thinking-fast-and-furious/baseline/experiments/eea/data/nuscenes/test_idefics.json',
        help="test data (after mcq conversion) path",
    )
    
    parser.add_argument(
        "--prediction-data-path",
        type=str,
        help="prediction data path (json)",
    )
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    # Create inputs
    finetuned_model = Idefics2ForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    
    test_idefics_dataset  = load_dataset('json', data_files=args.test_data_path, split='train')

    predictions = eval_model(finetuned_model, processor, test_idefics_dataset)
    with open(args.prediction_data_path, "w") as f:
        json.dump(predictions, f)
        
if __name__ == "__main__":
    main()