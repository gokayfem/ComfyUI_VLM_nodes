import os
import subprocess
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
import folder_paths
from transformers import AutoModel, AutoTokenizer

# Define the directory for saving MiniCPM files
MINICPM_PATH = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "minicpm_files"
MINICPM_PATH.mkdir(parents=True, exist_ok=True)

# Available GGUF model variants and their file sizes (in GB)
GGUF_MODELS = {
    "Q2_K (3GB)": "ggml-model-Q2_K.gguf",
    "Q3_K (3.8GB)": "ggml-model-Q3_K.gguf",
    "Q4_K_M (4.7GB)": "ggml-model-Q4_K_M.gguf",
    "Q5_K_M (5.4GB)": "ggml-model-Q5_K_M.gguf",
    "Q8_0 (8.1GB)": "ggml-model-Q8_0.gguf",
    "F16 (15.2GB)": "ggml-model-f16.gguf"
}

class MiniCPMPredictor:
    def __init__(self, model_name='openbmb/MiniCPM-V-2_6', context_length=4096, temp=0.7, 
                 top_p=0.8, top_k=100, repeat_penalty=1.05):
        self.context_length = context_length
        self.temp = temp
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                               attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Model loaded successfully.")
    
    def generate(self, image_path, prompt):
        """Generate response using the model"""
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        
        try:
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

class MiniCPMNode:
    def __init__(self):
        self.predictor = None
        self.current_model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to be analyzed by MiniCPM-V"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                    "tooltip": "Instructions for the model. Be specific about what aspects of the image you want analyzed."
                }),
                "model_variant": (list(GGUF_MODELS.keys()), {
                    "tooltip": "Model size/quality tradeoff. Smaller models (Q2-Q4) are faster but less accurate. Larger models (Q8, F16) provide better quality but require more VRAM."
                }),
                "context_length": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 8192,
                    "tooltip": "Maximum length of text context. Larger values allow longer conversations but use more memory. Default 4096 works well for most cases."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness in generation. Lower values (0.1-0.5) are more focused and deterministic. Higher values (0.8-2.0) increase creativity and variance."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Nucleus sampling threshold. Lower values make responses more focused. Higher values allow more diverse word choices."
                }),
                "top_k": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Limits the number of tokens considered for each generation step. Lower values increase focus, higher values allow more variety."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.05,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalizes word repetition. Values above 1.0 discourage repeated phrases. Higher values (>1.3) may affect fluency."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/MiniCPM-V"

    def download_model(self, model_filename):
        """Download model files from Huggingface"""
        try:
            print(f"Downloading model: {model_filename}...")
            model_path = hf_hub_download(
                repo_id="openbmb/MiniCPM-V-2_6-gguf",
                filename=model_filename,
                local_dir=MINICPM_PATH,
                local_dir_use_symlinks=False
            )
            
            print("Downloading mmproj model if not exists...")
            mmproj_path = hf_hub_download(
                repo_id="openbmb/MiniCPM-V-2_6-gguf",
                filename="mmproj-model-f16.gguf",
                local_dir=MINICPM_PATH,
                local_dir_use_symlinks=False
            )
            
            print("Download complete.")
            return Path(model_path), Path(mmproj_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {str(e)}")

    def generate(self, image, prompt, model_variant, context_length=4096,
                temperature=0.7, top_p=0.8, top_k=100, repeat_penalty=1.05):
        
        # Get model filename from variant name
        model_filename = GGUF_MODELS[model_variant]
        
        # Initialize or update predictor if needed
        if (self.predictor is None or 
            self.current_model != model_filename):
            
            # Download model if needed
            model_path, mmproj_path = self.download_model(model_filename)
            
            # Initialize predictor
            try:
                self.predictor = MiniCPMPredictor(
                    model_name='openbmb/MiniCPM-V-2_6',
                    context_length=context_length,
                    temp=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty
                )
                self.current_model = model_filename
            except Exception as e:
                return (f"Error initializing model: {str(e)}",)
        
        # Save input image temporarily
        temp_image = MINICPM_PATH / "temp_input.png"
        Image.fromarray(np.uint8(image[0] * 255)).save(temp_image)
        
        try:
            # Generate response
            response = self.predictor.generate(temp_image, prompt)
            
            # Clean up
            temp_image.unlink(missing_ok=True)
            
            return (response,)
            
        except Exception as e:
            return (f"Error during generation: {str(e)}",)

# Register the node
NODE_CLASS_MAPPINGS = {
    "MiniCPMNode": MiniCPMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniCPMNode": "MiniCPM-V Model"
}
