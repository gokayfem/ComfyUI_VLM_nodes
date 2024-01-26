from .moondream import VisionEncoder, TextModel
from huggingface_hub import snapshot_download
import torch
import os
import hashlib
import numpy as np
from torchvision import transforms
from threading import Thread
from transformers import TextIteratorStreamer
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions, add_model_folder_path

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
image_encoder_cache_path = os.path.join(output_directory, "image_encoder_cache")
class VisionTextQuestionNode:
    def __init__(self):
        self.model_path = snapshot_download("vikhyatk/moondream1")
        self.vision_encoder = VisionEncoder(self.model_path)
        self.text_model = TextModel(self.model_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "answer_questions"

    CATEGORY = "VLM Nodes/Visual Question Answering"

    def process_image(self, image):
        # Calculate checksum of the image

        image_array = image.numpy()  # Convert Tensor to NumPy array
        image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
        image = transforms.ToPILImage()(image[0].permute(2, 0, 1))
        # Check if `image_encoder_cache/{image_hash}.pt` exists, if so load and return it.
        # Otherwise, save the encoded image to `image_encoder_cache/{image_hash}.pt` and return it.
        cache_path = f"{image_encoder_cache_path}/{image_hash}.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path).to(DEVICE, dtype=DTYPE)
        else:
            image_vec = self.vision_encoder(image)
            os.makedirs(image_encoder_cache_path, exist_ok=True)
            torch.save(image_vec, cache_path)
            return image_vec.to(DEVICE, dtype=DTYPE)

    def answer_questions(self, image, question):
        image_embeds = self.process_image(image)
        streamer = TextIteratorStreamer(self.text_model.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            image_embeds=image_embeds, question=question, streamer=streamer
        )
        thread = Thread(target=self.text_model.answer_question, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        full_sentence = ""
        for new_text in streamer:
            buffer += new_text
            if not new_text.endswith("<") and not new_text.endswith("END"):
                print(buffer, end="", flush=True)
                full_sentence += buffer
                buffer = ""
        return (full_sentence,)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"VisionTextQuestion": VisionTextQuestionNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"VisionTextQuestion": "VisionQuestionAnswering Node"}


