from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
import folder_paths
# Define the directory for saving files related to your new model
files_for_new_model = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_kosmos2"
files_for_new_model.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class KosmosModelPredictor:
    def __init__(self):
        self.model_path = snapshot_download("microsoft/kosmos-2-patch14-224", 
                                            local_dir=files_for_new_model,
                                            force_download=False,  # Set to True if you always want to download, regardless of local copy
                                            local_files_only=False,  # Set to False to allow downloading if not available locally
                                            local_dir_use_symlinks="auto",
                                            ignore_patterns=["*.bin", "*.jpg", "*.png"])  # or set to True/False based on your symlink preference
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_predictions(self, image_path, main_text):
        # Load the image
        image_input = Image.open(image_path).convert("RGB")
        
        text_input = f"<grounding>{main_text}: "

        # Process the inputs
        inputs = self.processor(text=text_input, images=image_input, return_tensors="pt").to(self.device)

        # Generate predictions
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )

        # Decode the generated IDs
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # By default, the generated text is cleanup and the entities are extracted.
        processed_text, entities = self.processor.post_process_generation(generated_text)

        return processed_text[len(main_text)+2:]

# Example of integrating NewModelPredictor into a node-like structure
class Kosmos2model:
    def __init__(self):
        self.predictor = KosmosModelPredictor()

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

    FUNCTION = "new_model_generate_predictions"

    CATEGORY = "VLM Nodes/Kosmos-2"

    def new_model_generate_predictions(self, image, text_input):
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_new_model / "temp_image.png"
        pil_image.save(temp_path)      
        
        response = self.predictor.generate_predictions(temp_path, text_input)
        return (response, )

NODE_CLASS_MAPPINGS = {"Kosmos2model": Kosmos2model}
NODE_DISPLAY_NAME_MAPPINGS = {"Kosmos2model": "Kosmos-2 Node"}