from transformers.image_utils import load_image
import json
import matplotlib.pyplot as plt
from PIL import ImageDraw
from typing import Dict
from tqdm import tqdm
import copy, pathlib, re, os
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import torch
from peft import LoraConfig

MNT_POINT = "/mnt/u14157_ic_nlp_001_files_nfs"

if not pathlib.Path(MNT_POINT).exists():
    MNT_POINT = "/mnt"

CACHE_DIR = f"{MNT_POINT}/nlpdata1/home/ismayilz/.cache/huggingface"
os.environ["HF_HOME"] = CACHE_DIR

DEVICE = "cuda:0"
IMAGE_DIR = f"{MNT_POINT}/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/samples"
IMAGE_PATH_PREFIX = '../nuscenes/samples'
IMAGE_SRC_X, IMAGE_SRC_Y = 1600, 900
IMAGE_TGT_X, IMAGE_TGT_Y = int(IMAGE_SRC_X / 2.5), int(IMAGE_SRC_Y / 2.5)

TAGGED_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{line['text']}}{{': <image> ' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

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
    
def process_scene(scene_id, scene, image_dir=IMAGE_DIR, apply_context=None):
    samples = []
    for frame_id, frame in scene['key_frames'].items():
        image_paths = {view_name: view_path.replace(IMAGE_PATH_PREFIX, image_dir) for view_name, view_path in frame['image_paths'].items()}
        assert len(image_paths) == 6, "not all views provided"
        question_id = 0
        context = "Context:"
        last_context = "Context: None"
        for question_type, questions in frame['QA'].items():
            current_context = ""
            
            for question_info in questions:
                question = question_info['Q']
                answer = question_info['A'] if "A" in question_info else ""
                sample_id = f"{scene_id}_{frame_id}_{question_id}"
                question_id += 1

                if apply_context == "graph":
                    question_text = f"{context}\nTask:\n{question}"
                elif apply_context == "chain":
                    question_text = f"{last_context}\nTask:\n{question}"
                else:
                    question_text = question

                samples.append({
                    "id": sample_id, #change key here from sample_id to id
                    "question_type": question_type,
                    "question_text": question_text.strip(),
                    "images": image_paths,
                    "answer": answer,
                    "tag": question_info["tag"]
                })

                current_context = f"{current_context}\nQ:{question}\nA:{answer}"

            last_context = f"Context:\n{current_context.strip()}"
            context = f"{context}\n{current_context.strip()}"

    return samples


def process_dataset(data_path, output_path=None, image_dir=IMAGE_DIR, apply_context=None):
    with open(data_path, "r") as f:
        dataset: Dict[str, str] = json.load(f)
    samples = []
    for scene_id, scene in tqdm(dataset.items()):
        samples.extend(process_scene(scene_id, scene, image_dir=image_dir, apply_context=apply_context))
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

def get_objects(text):
    return re.findall(r'<[^>]*>', text)

def objects_to_dict(objects):
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


def draw_circle(image_path, image_key, objects, colors=["red"]):
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

def prepare_prompt(sample, verbose=False):
    image_paths = sample['images']
    question_text = sample['question_text']
    question = question_text
    context = None

    if "Task:" in question_text:
        parts = question_text.split("Task:")
        context = parts[0].strip()
        question = parts[1].strip()

    context_objects = get_objects(context)
    question_objects = get_objects(question)
    dict_objects = objects_to_dict(question_objects)
    colors = ["red", "blue", "black", "white"]
    images = [draw_circle(image_paths[image_key], image_key, dict_objects, colors=colors).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_key in image_paths.keys()]
    
    if verbose:
        image_viz = construct_for_viz(copy.deepcopy(image_paths), images)
        vizualize_frames(image_viz)
        print('objects:', dict_objects)

    for object, color in zip(question_objects, colors[:len(dict_objects)]):
        question = question.replace(object, f"the object marked with {color} circle")
        question = question.replace("object the object", "the object")

        if object in context_objects:
            context = context.replace(object, f"the object marked with {color} circle")
            context = context.replace("object the object", "the object")
    
    prompt = question

    if context:
        prompt = f"{context}\nTask:\n{question}"
    else:
        prompt = f"Context: None\nTask:\n{question}"

    sample["user_message"][0]["content"][-1]["text"] = prompt
    sample["prompt"] = prompt

    return prompt, images

def batched(lst, size=4):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def eval_model(model, test_set, processor, batch_size=4, verbose=False, chat_template=TAGGED_CHAT_TEMPLATE, apply_context=None):
    def _eval_on_dataset(dataset):
        predictions = []
        
        for idefics_batch in tqdm(batched(dataset, batch_size), total=len(test_set)//batch_size):
            eval_batch = [prepare_prompt(sample, verbose=verbose) for sample in idefics_batch]
            batch_messages = [sample["user_message"] for sample in idefics_batch]
            batch_images = [b[1] for b in eval_batch]
            batch_texts = processor.apply_chat_template(batch_messages, add_generation_prompt=False, chat_template=chat_template)
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idefics_sample, generated_text in zip(idefics_batch, generated_texts):
                predicted_text = generated_text.split('\n')[-1][len("Assistant: "):]
                prediction = copy.deepcopy(idefics_sample)
                prediction['gt'] = prediction['answer']
                prediction['answer'] = predicted_text
                predictions.append(prediction)
                if verbose:
                    print(idefics_sample["user_message"][0]["content"][-1]["text"])
                    print('Predicted:', predicted_text)
                    print('GT:', prediction['gt'])

        return predictions

    if apply_context == "chain":
        perception_set = [sample for sample in test_set if sample["question_type"] == "perception"]
        prediction_set = [sample for sample in test_set if sample["question_type"] == "prediction"]
        planning_set = [sample for sample in test_set if sample["question_type"] == "planning"]
        behavior_set = [sample for sample in test_set if sample["question_type"] == "behavior"]

        previous_predictions = None

        for dataset in [perception_set, prediction_set, planning_set, behavior_set]:
            for sample in dataset:
                if previous_predictions:
                    scene, frame, _ = sample["id"].split("_")
                    previous_samples = [s for s in previous_predictions if s["id"].startswith(f"{scene}_{frame}")]
                    context = "\n".join([f"Q:{s['question_text']}\nA:{s['answer']}" for s in previous_samples])
                    sample["user_message"][0]["content"][-1]["text"] = f"Context:\n{context}\nTask:\n{sample['question_text']}"
                else:
                    sample["user_message"][0]["content"][-1]["text"] = f"Context: None\nTask:\n{sample['question_text']}"
            previous_predictions = _eval_on_dataset(dataset)
    else:
        return _eval_on_dataset(test_set)

class GVQADataCollator:
    def __init__(self, processor, chat_template="tagged"):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.chat_template = processor.chat_template
        if chat_template == 'tagged':
            self.chat_template = TAGGED_CHAT_TEMPLATE

    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            prompt, sample_images = prepare_prompt(example)
            answer_text = example["answer"]
            answer_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text}
                ]
            }
            messages = [example["user_message"][0], answer_message]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, chat_template=self.chat_template)
            texts.append(text.strip())
            images.append(sample_images)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

def load_model(model_path, eval_mode=False, use_lora=False, use_qlora=False, device="cuda:0"):
    model = None
    
    if use_qlora or use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            use_dora=False if use_qlora else True,
            init_lora_weights="gaussian"
        )
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if use_qlora else None,
            cache_dir=CACHE_DIR
        )
        if not eval_mode:
            model.add_adapter(lora_config)
        model.enable_adapters()
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
            cache_dir=CACHE_DIR 
        ).to(device)
    
    return model