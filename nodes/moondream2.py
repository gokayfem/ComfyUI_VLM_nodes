from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
import folder_paths


# Define the directory for saving files related to your new model
files_for_moondream2 = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_moondream2"
files_for_moondream2.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class Moondream2Predictor:
    def __init__(self):
        self.model_path = snapshot_download("vikhyatk/moondream2", 
                                            local_dir=files_for_moondream2,
                                            force_download=False,  # Set to True if you always want to download, regardless of local copy
                                            local_files_only=False,  # Set to False to allow downloading if not available locally
                                            revision="2024-04-02",  # Specify the revision date for version control
                                            local_dir_use_symlinks="auto",  # or set to True/False based on your symlink preference
                                            ignore_patterns=["*.bin", "*.jpg", "*.png"])  # Customize based on need
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def generate_predictions(self, image_path, question):
        # Load and process the image
        image_input = Image.open(image_path).convert("RGB")
        enc_image = self.model.encode_image(image_input)

        # Generate predictions
        generated_text = self.model.answer_question(enc_image, question, self.tokenizer)

        return generated_text

class Moondream2model:
    def __init__(self):
        self.predictor = Moondream2Predictor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_input": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "moondream2_generate_predictions"

    CATEGORY = "VLM Nodes/Moondream2"

    def moondream2_generate_predictions(self, image, text_input):
        # Convert tensor image to PIL Image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_moondream2 / "temp_image.png"
        pil_image.save(temp_path)      
        
        response = self.predictor.generate_predictions(temp_path, text_input)
        return (response, )

NODE_CLASS_MAPPINGS = {"Moondream2model": Moondream2model}
NODE_DISPLAY_NAME_MAPPINGS = {"Moondream2model": "Moondream-2 Node"}
