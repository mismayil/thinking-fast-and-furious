from transformers.image_utils import load_image
import json
import matplotlib.pyplot as plt
from PIL import ImageDraw
from typing import Dict
from tqdm import tqdm
import copy, pathlib, re, os
from transformers import BitsAndBytesConfig, Idefics2ForConditionalGeneration
import torch
from peft import LoraConfig
import random 
from collections import defaultdict
from transformers import AutoProcessor

MNT_POINT = "/mnt/u14157_ic_nlp_001_files_nfs"

if not pathlib.Path(MNT_POINT).exists():
    MNT_POINT = "/mnt"

CACHE_DIR = f"{MNT_POINT}/nlpdata1/home/ismayilz/.cache/huggingface"
DEVICE = "cuda"
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
    
def process_scene(scene_id, scene, image_dir=IMAGE_DIR):
    samples = []
    for frame_id, frame in scene['key_frames'].items():
        image_paths = {view_name: view_path.replace(IMAGE_PATH_PREFIX, image_dir) for view_name, view_path in frame['image_paths'].items()}
        assert len(image_paths) == 6, "not all views provided"
        question_id = 0

        for question_type, questions in frame['QA'].items():            
            for question_info in questions:
                question = question_info['Q']
                answer = question_info['A'] if "A" in question_info else ""
                sample_id = f"{scene_id}_{frame_id}_{question_id}"
                question_id += 1

                samples.append({
                    "id": sample_id, #change key here from sample_id to id
                    "question_type": question_type,
                    "question_text": question.strip(),
                    "images": image_paths,
                    "answer": answer,
                    "tag": question_info["tag"]
                })

    return samples


def process_dataset(data_path, output_path=None, image_dir=IMAGE_DIR):
    with open(data_path, "r") as f:
        dataset: Dict[str, str] = json.load(f)
    samples = []
    for scene_id, scene in tqdm(dataset.items(), desc="Processing dataset"):
        samples.extend(process_scene(scene_id, scene, image_dir=image_dir))
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

object_detection_questions = [
    "What is the object {}?",
    "What is the bounding box and the category of the {} in {} of the ego vehicle?",
    "What is the center coordinate of the {} in {} of the ego vehicle?", 
    "What is the status of the {} ({})?"
]

def create_object_detection_questions(ko_id, key_object):
    _, view, center_coords = ko_id[1:-1].split(',', 2)
    category = key_object['Category']
    status =  key_object['Status']
    description = key_object['Visual_description'].rstrip('.')
    bbox = ",".join(map(str, key_object['2d_bbox']))
    view_verbalized = " ".join(view.split('_')[1:]).lower()

    n_question_options = 3 + (status is not None)
    qa_id = random.randint(1, n_question_options) - 1
    obj_questions = []

    for qa_id in range(n_question_options):
        question_args = [[ko_id], [description, view_verbalized],  [description, view_verbalized], [description, ko_id]]
        answer_args =  [[description], [bbox, category], [center_coords], [status]]
        question = object_detection_questions[qa_id].format(*question_args[qa_id])
        answer = " ".join(answer_args[qa_id])
        obj_questions.append({
            "qa_id": qa_id,
            "question": question,
            "answer": answer
        })

    return obj_questions
    
def prepare_object_detection_dataset(data_path, output_path=None, question_type="auxiliary", image_dir=IMAGE_DIR):
    with open(data_path, "r") as f:
        dataset = json.load(f)
    samples = []
    
    for scene_id, scene in tqdm(dataset.items(), desc="Preparing object detection dataset"):
        for frame_id, frame in scene['key_frames'].items():
            image_paths = {view_name: view_path.replace(IMAGE_PATH_PREFIX, image_dir) for view_name, view_path in frame['image_paths'].items()}
            for object_id, (ko_id, key_object) in enumerate(frame['key_object_infos'].items()):
                obj_questions = create_object_detection_questions(ko_id, key_object)
                for obj_q in obj_questions:
                    sample_id = f"{scene_id}_{frame_id}_{object_id}_{obj_q['qa_id']}"
                    sample = {
                        "id": sample_id,
                        "question_type": question_type,
                        "question_text": obj_q["question"],
                        "images": image_paths,
                        "answer": obj_q["answer"]
                    }
                    samples.append(sample)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)

    return samples

def parse_sample_id(sample_id):
    id_parts = sample_id.split("_")
    scene_id, frame_id, question_id = id_parts[0], id_parts[1], id_parts[2:]
    return scene_id, frame_id, question_id

def add_context_to_idefics_dataset(idefics_dataset, apply_context=None, apply_on_stages="all"):
    if not apply_context:
        return idefics_dataset
    
    previous_stage_map = {
        "auxiliary": None,
        "perception": None,
        "prediction": "perception",
        "planning": "prediction",
        "behavior": "planning"
    }
    
    if apply_context == "chain":
        idefics_dataset_map = defaultdict(list)

        for sample in tqdm(idefics_dataset, desc="Creating dataset map"):
            scene_id, frame_id, _ = parse_sample_id(sample["id"])
            idefics_dataset_map[f"{scene_id}_{frame_id}"].append(sample)

        for sample in tqdm(idefics_dataset, desc="Adding context"):
            stage = sample["question_type"]

            if apply_on_stages == "all" or stage == apply_on_stages or stage in apply_on_stages:
                scene_id, frame_id, _ = parse_sample_id(sample["id"])
                previous_stage = previous_stage_map[stage]
                context = None
                
                if previous_stage:
                    previous_stage_samples = [s for s in idefics_dataset_map[f"{scene_id}_{frame_id}"] if s["question_type"] == previous_stage]
                    context = "\n".join([f"Q:{s['question_text']}\nA:{s['answer']}" for s in previous_stage_samples])
                
                if context:
                    sample["user_message"][0]["content"][-1]["text"] = f"Context:\n{context}\nTask:\n{sample['question_text']}"
                else:
                    sample["user_message"][0]["content"][-1]["text"] = f"Context: None\nTask:\n{sample['question_text']}"
    else:
        raise NotImplementedError(f"{apply_context} not supported")

    return idefics_dataset

def produce_idefics_dataset(samples, output_path=None, apply_context=None, apply_context_on_stages="all"):
    idefics_samples = []
    
    for sample in samples:
        idefics_samples.append(convert_sample_to_idefics(sample))
    
    add_context_to_idefics_dataset(idefics_samples, apply_context=apply_context, apply_on_stages=apply_context_on_stages)
    
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(idefics_samples, f, indent=4)
            
    return idefics_samples

def get_objects(text):
    return list(set(re.findall(r'<[^>]+,[^>]+,[^>]+,[^>]+>', text)))

def parse_object_ref(object_ref):
    refs = object_ref.strip('<>').split(',')
    object_id = refs[0]
    image_id = refs[1]
    coordinates = [float(refs[2]), float(refs[3])]
    return object_id, image_id, coordinates

def draw_circle(image_path, image_key, circles):
    image = load_image(image_path)

    for object_ref, color in circles:
        object_id, image_id, coordinates = parse_object_ref(object_ref)
        
        if image_id == image_key:
            draw = ImageDraw.Draw(image)
            # Define the radius of the circle and the color
            # Base on paper: we draw red circles over the images, with radius r = 0.06H and thickness t = 0.01H, where H is the shorter side of the image.
            H= min(image.size)
            radius = 0.06 * H
            thickness = 0.01 * H
            x = float(coordinates[0])
            y = float(coordinates[1])
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

def verbalize_obj_ref(text, object, color):
    text = text.replace(object, f"the object marked with {color} circle")
    text = text.replace("object the object", "the object")
    text = text.replace("the the", "the")
    return text

def load_and_resize_images(example):
    sample_images = [load_image(image_path).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_path in example['images'].values()]
    return sample_images

def prepare_prompt(sample, verbose=False, verbalize_refs=True, apply_redcircle=True, apply_redcircle_only_to_question=False):
    image_paths = sample['images']
    question_text = sample["user_message"][0]["content"][-1]["text"]
    question = question_text
    context = None

    if apply_redcircle and apply_redcircle_only_to_question:
        if "Task:" in question_text:
            parts = question_text.split("Task:")
            context = parts[0].strip()
            question = parts[1].strip()

    if apply_redcircle:
        question_objects = get_objects(question)
        colors = ["red", "blue", "black", "white", "green", "yellow", "grey", "orange"]
        assert len(question_objects) <= len(colors)
        circles = list(zip(question_objects, colors))
        images = [draw_circle(image_paths[image_key], image_key, circles).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_key in image_paths.keys()]
    else:
        images = [load_image(image_path).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_path in image_paths.values()]
    
    if verbose:
        image_viz = construct_for_viz(copy.deepcopy(image_paths), images)
        vizualize_frames(image_viz)

    if apply_redcircle and verbalize_refs:
        for object, color in circles:
            question = verbalize_obj_ref(question, object, color)

        if context:
            context = verbalize_obj_ref(context, object, color)
    
    prompt = question

    if context:
        prompt = f"{context}\nTask:\n{question}"

    sample["user_message"][0]["content"][-1]["text"] = prompt
    sample["prompt"] = prompt

    return prompt, images

def batched(lst, size=4):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def apply_perception_trick(predictions):
    perception_predictions = [pred for pred in predictions if pred["question_type"] == "perception" and "what are the important objects in the current scene?" in pred["question_text"].lower()]
    prediction_map = defaultdict(list)

    for pred in tqdm(predictions, desc="Creating prediction map"):
        scene_id, frame_id, _ = parse_sample_id(pred["id"])
        prediction_map[f"{scene_id}_{frame_id}"].append(pred)
    
    for pred in tqdm(perception_predictions, desc="Applying perception trick"):
        scene_id, frame_id, _ = parse_sample_id(pred["id"])
        relevant_predictions = prediction_map[f"{scene_id}_{frame_id}"]
        ref_objects = set()
        answer = pred["answer"]
        id_prefix = "The IDs of these objects are"
        
        if id_prefix in answer:
            for relevant_pred in relevant_predictions:
                objects = get_objects(relevant_pred["question_text"])
                ref_objects.update(objects)
            
            ref_objects = sorted(ref_objects, key=lambda o: parse_object_ref(o)[0])
            answer_prefix, _ = answer.split(id_prefix)
            ref_objects[-1] = "and " + ref_objects[-1]
            ref_objects_txt = ", ".join(ref_objects)
            pred["answer"] = f"{answer_prefix}{id_prefix} {ref_objects_txt}."    
    return predictions

def eval_model(model, test_set, processor, batch_size=4, verbose=False, chat_template=TAGGED_CHAT_TEMPLATE, 
               apply_context=None, verbalize_refs=True, apply_redcircle=True, apply_redcircle_only_to_question=False):
    def _eval_on_dataset(dataset):
        predictions = []
        
        for idefics_batch in tqdm(batched(dataset, batch_size), total=len(dataset)//batch_size+1):
            eval_batch = [prepare_prompt(sample, verbose=verbose, verbalize_refs=verbalize_refs, apply_redcircle=apply_redcircle, apply_redcircle_only_to_question=apply_redcircle_only_to_question) for sample in idefics_batch]
            batch_messages = [sample["user_message"] for sample in idefics_batch]
            batch_images = [b[1] for b in eval_batch]
            batch_texts = processor.apply_chat_template(batch_messages, add_generation_prompt=True, chat_template=chat_template)
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idefics_sample, generated_text in zip(idefics_batch, generated_texts):
                predicted_text = generated_text.split('\n')[-1][len("Assistant: "):]
                prediction = copy.deepcopy(idefics_sample)
                prediction['gt'] = prediction['answer']
                prediction['answer'] = predicted_text
                predictions.append(prediction)
                if verbose:
                    print()
                    print(idefics_sample["user_message"][0]["content"][-1]["text"])
                    print('Predicted:', predicted_text)
                    print('GT:', prediction['gt'])
                    print()

        return predictions

    if apply_context == "chain":
        auxiliary_set = [sample for sample in test_set if sample["question_type"] == "auxiliary"]
        perception_set = [sample for sample in test_set if sample["question_type"] == "perception"]
        prediction_set = [sample for sample in test_set if sample["question_type"] == "prediction"]
        planning_set = [sample for sample in test_set if sample["question_type"] == "planning"]
        behavior_set = [sample for sample in test_set if sample["question_type"] == "behavior"]

        previous_predictions = []
        predictions = []

        for stage, dataset in [("auxiliary", auxiliary_set), ("perception", perception_set), ("prediction", prediction_set), ("planning", planning_set), ("behavior", behavior_set)]:
            add_context_to_idefics_dataset(previous_predictions + dataset, apply_context=apply_context, apply_on_stages=stage)
            previous_predictions = _eval_on_dataset(dataset)
            predictions.extend(previous_predictions)
        
        return predictions
    else:
        return _eval_on_dataset(test_set)

N_TOKENS_TO_MASK_AFTER_EOU = 6

class GVQADataCollator:
    def __init__(self, processor, chat_template="tagged", verbose=False, verbalize_refs=True, 
                 apply_redcircle=True, apply_redcircle_only_to_question=False, apply_input_masking=False):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.end_of_utterance_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<end_of_utterance>")
        ]
        self.chat_template = processor.chat_template
        if chat_template == 'tagged':
            self.chat_template = TAGGED_CHAT_TEMPLATE
        self.verbose = verbose
        self.verbalize_refs = verbalize_refs
        self.apply_redcircle = apply_redcircle
        self.apply_redcircle_only_to_question = apply_redcircle_only_to_question
        self.apply_input_masking = apply_input_masking

    def _build_chat_template_mask(self, input_ids):
        batch_size, seq_length = input_ids.size()
        first_end_of_utterance_ids = (input_ids == self.end_of_utterance_id).nonzero()[::2, 1] +  N_TOKENS_TO_MASK_AFTER_EOU
        mask_range = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        mask = mask_range < first_end_of_utterance_ids.unsqueeze(-1)
        return mask

    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            prompt, sample_images = prepare_prompt(example, verbose=self.verbose, verbalize_refs=self.verbalize_refs, apply_redcircle=self.apply_redcircle, apply_redcircle_only_to_question=self.apply_redcircle_only_to_question)
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

        if self.apply_input_masking:
            labels_mask = self._build_chat_template_mask(batch["input_ids"])
            labels[labels_mask] = self.image_token_id
        else:
            labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id

        batch["labels"] = labels

        return batch

def load_processor(model_dir):
    return AutoProcessor.from_pretrained(
        model_dir,
        do_image_splitting=False
    )

def load_model(model_path, eval_mode=False, use_lora=False, use_qlora=False, device="cuda", cache_dir=CACHE_DIR):
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
            cache_dir=cache_dir,
            device_map="auto"
        )
        if not eval_mode:
            model.add_adapter(lora_config)
        model.enable_adapters()
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
            cache_dir=cache_dir 
        ).to(device)
    
    return model