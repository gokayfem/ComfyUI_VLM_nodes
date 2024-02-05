from .moondream import VisionEncoder, TextModel
from huggingface_hub import snapshot_download
import torch
import os
import hashlib
from torchvision import transforms

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
image_encoder_cache_path = os.path.join(output_directory, "image_encoder_cache")
class MoonDream:
    def __init__(self):
        self.model_path = snapshot_download("vikhyatk/moondream1", revision="5cd8d1ecd7e0d8d95222543e1960d340ddffbfef")
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

    CATEGORY = "VLM Nodes/MoonDream"

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
        full_sentence = self.text_model.answer_question(image_embeds, question)
        return (full_sentence,)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"MoonDream": MoonDream}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"MoonDream": "MoonDream Node"}


