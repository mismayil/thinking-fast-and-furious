from transformers.image_utils import load_image
import matplotlib.pyplot as plt
from PIL import Image
import torch

# tokens corresponding to '\nAssistant:' that we want to mask out when calculating the loss
N_TOKENS_TO_MASK_AFTER_EOU = 5 
IMAGE_SRC_X, IMAGE_SRC_Y = 1600, 900
IMAGE_TGT_X, IMAGE_TGT_Y = int(IMAGE_SRC_X / 2.5), int(IMAGE_SRC_Y / 2.5)
TAGGED_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{line['text']}}{{': <image> ' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def vizualize_frames(image_paths):
    y_view_mapping = {"MIDDLE": 1, "LEFT": 0, "RIGHT": 2}
    fig, axes = plt.subplots(2, 3, figsize=(48, 18))
    for i, (image_view, image_path) in enumerate(image_paths.items()):
        image = Image.open(image_path)
        _, x, y = f"{image_view}_MIDDLE".split("_")[:3]
        x_id = int(x == 'BACK')
        axes[x_id][y_view_mapping[y]].imshow(image)
        axes[x_id][y_view_mapping[y]].set_title(image_view)
        axes[x_id][y_view_mapping[y]].axis('off')
    plt.show()
    
    
def load_and_resize_images(example):
    sample_images = [load_image(image_path).resize((IMAGE_TGT_X, IMAGE_TGT_Y)) for image_path in example['images'].values()]
    return sample_images

class GVQADataCollator:
    def __init__(self, processor, chat_template=None):
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


    def _build_chat_template_mask(self, input_ids):
        batch_size, seq_length = input_ids.size()
        first_end_of_utterance_ids = (input_ids == self.end_of_utterance_id).nonzero()[::2, 1] +  N_TOKENS_TO_MASK_AFTER_EOU
        mask_range = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        mask = mask_range < first_end_of_utterance_ids.unsqueeze(-1)
        # masking the last <EOU> token
        mask[:, -1] = True
        return mask

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            sample_images = load_and_resize_images(example)
            answer_text = example["answer"]
            answer_message = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_text}
                    ]
            }
            user_message = example['user_message'][0]
            messages = [user_message, answer_message]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, chat_template=self.chat_template)
            texts.append(text.strip())
            images.append(sample_images)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        # added user message masking 
        labels_mask = self._build_chat_template_mask(batch["input_ids"])
        labels[labels_mask] = self.image_token_id
        batch["labels"] = labels

        return batch
