import wandb
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers.image_utils import load_image
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from typing import Dict
from tqdm import tqdm
import copy

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

IMAGE_DIR = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/samples'
IMAGE_PATH_PREFIX = '../nuscenes/samples'
IMAGE_SRC_X, IMAGE_SRC_Y = 1600, 900
IMAGE_TGT_X, IMAGE_TGT_Y = int(IMAGE_SRC_X / 2.5), int(IMAGE_SRC_Y / 2.5)

train_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/v1_1_train_nus_ext_converted.json'
test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/drivelm/challenge/test_eval.json'
idefics_train_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-v3/data/nuscenes/train_idefics_redcircle_v3.json'
idefics_test_data_path = '/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-v3/data/nuscenes/test_idefics_redcircle_v3.json'
zero_shot_predictions_path = "/mnt/nlpdata1/home/ismayilz/cs503-project/thinking-fast-and-furious/experiments/redcircle-v3/outputs/test-eval-idefics2-8b-zero-shot_redcirecle_v3.json"
# checkpoint_dir = f"/home/cchang/CS503_VisualIntelligence/thinking-fast-and-furious/baseline/experiments/eea/models/idefics_redcircle/{wandb_name}"
# checkpoint_dir = "/mnt/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-prime-music-8-500step/checkpoint-500"
checkpoint_dir = "/mnt/nlpdata1/home/ismayilz/cs503-project/models/idefics2-redcircle-v3"
model_dir = "HuggingFaceM4/idefics2-8b"

import os 

os.environ["HF_HOME"] = "/mnt/nlpdata1/home/ismayilz/.cache/huggingface"

processor = AutoProcessor.from_pretrained(
    model_dir,
    do_image_splitting=False
)


import pathlib

def vizualize_frames(image_paths):
    y_view_mapping = {"MIDDLE": 1, "LEFT": 0, "RIGHT": 2}
    fig, axes = plt.subplots(2, 3, figsize=(48, 18))
    for i, (image_view, image_path) in enumerate(image_paths.items()):
        # image = Image.open(image_path)
        image=copy.deepcopy(image_path)
        _, x, y = f"{image_view}_MIDDLE".split("_")[:3]
        x_id = int(x == 'BACK')
        axes[x_id][y_view_mapping[y]].imshow(image)
        axes[x_id][y_view_mapping[y]].set_title(image_view)
        axes[x_id][y_view_mapping[y]].axis('off')
    plt.show()
    
def process_scene(scene_id, scene):
    samples = []
    for frame_id, frame in scene['key_frames'].items():
        image_paths = {view_name: view_path.replace(IMAGE_PATH_PREFIX, IMAGE_DIR) for view_name, view_path in frame['image_paths'].items()}
        assert len(image_paths) == 6, "not all views provided"
        question_id = 0
        for question_type, questions in frame['QA'].items():
            for question_info in questions:
                question = question_info['Q']
                answer = question_info['A'] if "A" in question_info else ""
                sample_id = f"{scene_id}_{frame_id}_{question_id}"
                question_id += 1
                question_text = question
                samples.append({
                    "id": sample_id, #change key here from sample_id to id
                    "question_type": question_type,
                    "question_text": question_text,
                    "images": image_paths,
                    "answer": answer,
                    "tag": question_info["tag"]
                })
    return samples


def process_dataset(data_path, output_path=None):
    with open(data_path, "r") as f:
        dataset: Dict[str, str] = json.load(f)
    samples = []
    for scene_id, scene in tqdm(dataset.items()):
        samples.extend(process_scene(scene_id, scene))
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
    return samples


def convert_sample_to_idefics(sample):
    idefics_sample = copy.deepcopy(sample)
    question = sample["question_text"]
    user_message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": "CAM_BACK"},
                {"type": "image", "text": "CAM_BACK_LEFT"},
                {"type": "image", "text": "CAM_BACK_RIGHT"},
                {"type": "image", "text": "CAM_FRONT"},
                {"type": "image", "text": "CAM_FRONT_LEFT"},
                {"type": "image", "text": "CAM_FRONT_RIGHT"},
                {"type": "text", "text": question},
            ]
        }
    ]
    idefics_sample["user_message"] = user_message
    return idefics_sample


def produce_idefics_dataset(samples, output_path=None):
    idefics_samples = []
    for sample in samples:
        idefics_samples.append(convert_sample_to_idefics(sample))
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(idefics_samples, f, indent=4)
            
    return idefics_samples

import re
def objects_to_dict(question):
    # get all objects in the question
    objects = re.findall(r'<[^>]*>', question)
    unique_objects = list(set(objects))
    result = {}
    for obj in unique_objects:
        # Remove '<' and '>' and split by comma
        parts = obj.strip('<>').split(',')
        # The identifier seems to be the second element based on your example
        identifier = parts[1]
        # Coordinates are the last two elements
        coordinates = [float(parts[2]), float(parts[3])]
        # Check if the identifier already exists in the dictionary
        if identifier in result:
            # Append the new coordinates to the existing list
            result[identifier].append(coordinates)
        else:
            # Otherwise, create a new list with the coordinates
            result[identifier] = [coordinates]
    # result will look like {'CAM_BACK': [[1088.3, 497.5]], 'CAM_FRONT': [[1043.2, 82.2]]}
    return result


def draw_circle(image_path,image_key, objects, colors=["red"]):
    image = load_image(image_path)
    assert len(objects) <= len(colors)

    if image_key in objects.keys() and bool(objects):
        for coordinate, color in zip(objects[image_key], colors):
            draw = ImageDraw.Draw(image)
            # Define the radius of the circle and the color
            # Base on paper: we draw red circles over the images, with radius r = 0.06H and thickness t = 0.01H, where H is the shorter side of the image.
            H= min(image.size)
            radius = 0.06 * H
            thickness = 0.01 * H
            x = float(coordinate[0])
            y = float(coordinate[1])
            # Calculate the bounding box of the circle to be drawn
            left_up_point = (int(x - radius), int(y - radius))
            right_down_point = (int(x + radius), int(y + radius))
            draw.ellipse([left_up_point, right_down_point], outline=color, fill=None, width=int(thickness))
            #for checking center
            # radius_center=10
            # left_up_point = (int(x - radius_center), int(y - radius_center))
            # right_down_point = (int(x + radius_center), int(y + radius_center))
            # draw.ellipse([left_up_point, right_down_point],fill='blue')

    return image

def construct_for_viz(image_paths,images):
    for i,key in enumerate(image_paths.keys()):
        image_paths[key]=images[i]
    return image_paths

def eval_model(model, test_set, verbose=False):
    predictions = []
    for idefics_sample in tqdm(test_set):
        image_paths = idefics_sample['images']
        objects = objects_to_dict(idefics_sample['question_text'])
        colors = ["red", "blue", "black", "white"]
        images = [draw_circle(image_paths[image_key], image_key, objects, colors=colors).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_key in image_paths.keys()]
        
        if verbose:
            image_viz=construct_for_viz(copy.deepcopy(image_paths),images)
            vizualize_frames(image_viz)
            print('objects:',objects)
        
        prompt = processor.apply_chat_template(idefics_sample['user_message'], add_generation_prompt=True)
        
        raw_objects = re.findall(r'<[^>]*>', idefics_sample['question_text'])

        for object, color in zip(raw_objects, colors[:len(objects)]):
            prompt = prompt.replace(object, f"the object marked with {color} circle")
            prompt = prompt.replace("object the object", "the object")
        
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
        predicted_text = generated_texts[0].split('\n')[-1][len("Assistant: "):]
        prediction = copy.deepcopy(idefics_sample)
        prediction['gt'] = prediction['answer']
        prediction['answer'] = predicted_text
        predictions.append(prediction)
        if verbose:
            print(idefics_sample['question_text'])
            print('Predicted:', predicted_text)
            print('GT:', prediction['gt'])
    return predictions

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
            objects = objects_to_dict(example['question_text'])
            colors = ["red", "blue", "black", "white"]
            sample_images = [draw_circle(example['images'][image_key], image_key, objects, colors=colors).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_key in example['images'].keys()]

            #for make sure red circle is there
            # image_viz=construct_for_viz(copy.deepcopy(example['images']),sample_images)
            # vizualize_frames(image_viz)
            # print('Question:',)
            # print('objects:',objects)
            
            answer_text = example["answer"]
            answer_message = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_text}
                    ]
            }
            user_message = example['user_message'][0]
            question = user_message["content"][-1]["text"]
            raw_objects = re.findall(r'<[^>]*>', question)

            for object, color in zip(raw_objects, colors[:len(objects)]):
                question = question.replace(object, f"the object marked with {color} circle")
                question = question.replace("object the object", "the object")

            user_message["content"][-1]["text"] = question
            messages = [user_message, answer_message]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            # print(text)
            images.append(sample_images)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch
    
if __name__ == "__main__":
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
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

    from datasets import load_dataset

    data = load_dataset('json', data_files=idefics_train_data_path, split=None)
    split_dataset = data['train'].train_test_split(test_size=0.025)

    from transformers import TrainingArguments, Trainer

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
        run_name="idefics-8b-redcircle-v3"
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