from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from torchvision.transforms import ToPILImage
import io
from PIL import Image

# Define the directory for saving files related to the MCLLaVA model
files_for_mcllava_model = Path(__file__).resolve().parent / "files_for_mcllava"
files_for_mcllava_model.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class MCLLaVAModelPredictor:
    def __init__(self):
        self.model_path = snapshot_download("visheratin/MC-LLaVA-3b",
                                            local_dir=files_for_mcllava_model,
                                            force_download=False,  # Set to True if you always want to download, regardless of local copy
                                            local_files_only=False,  # Set to False to allow downloading if not available locally
                                            local_dir_use_symlinks="auto",  # or set to True/False based on your symlink preference
                                            ignore_patterns=["*.bin", "*.jpg", "*.png"])  # Exclude certain file types
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def generate_predictions(self, pil_image, prompt, temperature, top_p, max_crops, num_tokens):
        # Load the image
        # Save the PIL image to a bytes buffer instead of a file on disk.
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')

        # Move to the beginning of the buffer so Image.open can read from it.
        buffer.seek(0)

        # Open the image as if it was a 'raw' image from an HTTP response.
        image_input = Image.open(buffer)

        final_prompt = f"""<|im_start|>user
<image>
{prompt}<|im_end|>
<|im_start|>assistant
"""

        with torch.inference_mode():
            inputs = self.processor(final_prompt, [image_input], self.model, max_crops=max_crops, num_tokens=num_tokens)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, use_cache=False, top_p=top_p, temperature=temperature, eos_token_id=self.processor.tokenizer.eos_token_id)

        generated_text = self.processor.tokenizer.decode(output[0]).replace(final_prompt, "").replace("<|im_end|>", "")
        return generated_text

# Example of integrating MCLLaVAModelPredictor into a node-like structure
class MCLLaVAModel:
    def __init__(self):
        self.predictor = MCLLaVAModelPredictor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ( "STRING",{"multiline": True, "default": "", },),
                "temperature": ( "FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},),
                "top_p": ( "FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},),
                "max_crops": ( "INT", {"default": 100, "min": 1, "max": 300, "step": 1},),
                "num_tokens": ( "INT", {"default": 728, "min": 1, "max": 2048, "step": 1},),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "generate_image_description"

    CATEGORY = "VLM Nodes/MC-LLaVA"

    def generate_image_description(self, image, prompt, temperature, top_p, max_crops, num_tokens):
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        
        response = self.predictor.generate_predictions(pil_image, prompt, temperature, top_p, max_crops, num_tokens)
        return (response, )

NODE_CLASS_MAPPINGS = {"MCLLaVAModel": MCLLaVAModel}
NODE_DISPLAY_NAME_MAPPINGS = {"MCLLaVAModel": "MC-LLaVA Node"}



        
