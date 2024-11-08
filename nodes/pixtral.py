import logging
import torch
import os
from PIL import Image
from pathlib import Path
import warnings
from huggingface_hub import snapshot_download
import psutil
from typing import Optional, Dict, Any, List, Union, Tuple

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import TextChunk, ImageURLChunk

import folder_paths

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PixtralMistralNode')

# Filter specific warnings
warnings.filterwarnings('ignore', message='.*The model weights are not tied.*')
warnings.filterwarnings('ignore', message='.*You should use.*max_memory.*')
warnings.filterwarnings('ignore', message='.*`local_dir_use_symlinks` parameter is deprecated.*')

# Check CUDA availability at module load
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"CUDA enabled: {device_name} with {device_memory:.1f}GB VRAM")
else:
    logger.warning("CUDA not available. Running on CPU will be extremely slow and is not recommended.")

# Define the directory for saving Pixtral files
PIXTRAL_PATH = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_pixtral"
PIXTRAL_PATH.mkdir(parents=True, exist_ok=True)

# Available Pixtral models with descriptions
PIXTRAL_MODELS = {
    "Pixtral-12B (Instruct Tuned)": {
        "repo": "mistralai/Pixtral-12B-2409",
        "description": "12B parameter model optimized for instruction following"
    },
    "Pixtral-12B-Base": {
        "repo": "mistralai/Pixtral-12B-Base-2409",
        "description": "Base 12B parameter model for custom fine-tuning"
    }
}

class PixtralMistralPredictor:
    def __init__(self, model_name: str):
        self.model_name = PIXTRAL_MODELS[model_name]["repo"]
        
        # Create a safe directory name
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        model_dir = PIXTRAL_PATH / safe_name
        
        # Download model if needed
        logger.info(f"Downloading/loading {model_name}...")
        try:
            self.model_path = snapshot_download(
                repo_id=self.model_name,
                local_dir=model_dir,
                allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded to: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {str(e)}")
        
        try:
            self._initialize_model()
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Out of memory while loading model. Try:\n"
                    "1. Closing other applications\n"
                    "2. Restarting ComfyUI"
                ) from e
            raise

    def _initialize_model(self):
        """Initialize model using mistral-inference"""
        try:
            logger.info("Loading tokenizer...")
            tokenizer_path = os.path.join(self.model_path, "tekken.json")
            if not os.path.exists(tokenizer_path):
                raise RuntimeError(f"tekken.json not found at {tokenizer_path}")
            
            self.tokenizer = MistralTokenizer.from_file(tokenizer_path)
            
            logger.info("Loading model...")
            self.model = Transformer.from_folder(self.model_path)
            logger.info("Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256, temperature: float = 0.35) -> str:
        """Generate response using mistral-inference"""
        try:
            # Convert PIL Image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create base64 string (mistral-inference expects this format)
            import base64
            img_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Format input for Pixtral using data URL scheme
            img_url = f"data:image/png;base64,{img_b64}"
            user_content = [
                ImageURLChunk(image_url=img_url),
                TextChunk(text=prompt)
            ]
            
            # Encode the input
            try:
                tokens, images = self.tokenizer.instruct_tokenizer.encode_user_content(user_content, False)
                logger.info(f"Input encoded successfully - token length: {len(tokens)}")
            except Exception as e:
                logger.error(f"Error encoding input: {str(e)}")
                raise RuntimeError(f"Failed to encode input: {str(e)}")
            
            # Generate response with safety checks
            try:
                out_tokens, _ = generate(
                    [tokens],
                    self.model,
                    images=[images],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
                
                if not out_tokens or len(out_tokens) == 0:
                    logger.error("No output tokens generated")
                    return "Error: Model did not generate any output"
                    
                logger.info(f"Generated {len(out_tokens[0])} tokens")
                
                # Decode response with safety check
                if len(out_tokens) > 0 and len(out_tokens[0]) > 0:
                    result = self.tokenizer.decode(out_tokens[0])
                    return result.strip()
                else:
                    return "Error: No valid output generated"
                
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                raise RuntimeError(f"Generation failed: {str(e)}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "Out of memory during generation. Try:\n"
                    "1. Reducing max_new_tokens\n"
                    "2. Clearing ComfyUI cache\n"
                    "3. Restarting ComfyUI"
                ) from e
            raise

class PixtralMistralNode:
    def __init__(self):
        """Initialize PixtralMistralNode attributes"""
        self.predictor: Optional[PixtralMistralPredictor] = None
        self.current_model: Optional[str] = None
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to be analyzed by Pixtral"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                    "tooltip": "Instructions for the model. Be specific about what aspects of the image you want analyzed."
                }),
                "model_name": (list(PIXTRAL_MODELS.keys()), {
                    "default": "Pixtral-12B (Instruct Tuned)",
                    "tooltip": "⚠️ WARNING: This model requires ~25GB VRAM!"
                }),
                "max_new_tokens": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Maximum tokens to generate. Higher values need more VRAM."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower = more focused, higher = more creative."
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model loaded in VRAM after execution. Saves time for multiple runs but uses more memory."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Pixtral"

    def generate(self, image: torch.Tensor, prompt: str, model_name: str,
                max_new_tokens: int = 256, temperature: float = 0.35,
                keep_loaded: bool = False) -> tuple[str]:
        try:
            # Initialize or update predictor if needed
            if (self.predictor is None or
                self.current_model != model_name):

                # Clean up old model
                self._cleanup_predictor()

                # Initialize new predictor
                self.predictor = PixtralMistralPredictor(model_name)
                self.current_model = model_name

            # Convert tensor to PIL Image
            # Ensure image is properly scaled from 0-1 to 0-255 range
            pil_image = Image.fromarray(
                (image[0] * 255).numpy().astype('uint8')
            )

            # Generate response
            response = self.predictor.generate(
                pil_image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # Clean up if keep_loaded is False
            if not keep_loaded:
                logger.info("Unloading model as keep_loaded is False")
                self._cleanup_predictor()

            return (response,)

        except torch.cuda.OutOfMemoryError:
            self._cleanup_predictor()
            error_msg = (
                "Out of memory during generation. Try:\n"
                "1. Reducing max_new_tokens\n"
                "2. Clearing ComfyUI cache\n"
                "3. Restarting ComfyUI"
            )
            return (error_msg,)
        except Exception as e:
            self._cleanup_predictor()
            return (f"Error during generation: {str(e)}",)

    def _cleanup_predictor(self):
        """Helper method to clean up predictor resources"""
        if self.predictor is not None:
            if hasattr(self.predictor, 'model'):
                del self.predictor.model
            if hasattr(self.predictor, 'tokenizer'):
                del self.predictor.tokenizer
            self.predictor = None
            torch.cuda.empty_cache()

    def __del__(self):
        """Ensure cleanup when node is destroyed"""
        self._cleanup_predictor()

# Register the node
NODE_CLASS_MAPPINGS = {
    "PixtralMistralNode": PixtralMistralNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixtralMistralNode": "Pixtral Vision-Language Model (Mistral)"
}