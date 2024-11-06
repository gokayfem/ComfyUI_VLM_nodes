import torch
import os
from PIL import Image
from pathlib import Path
import folder_paths
import logging
import warnings
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import torch.amp.autocast_mode
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MolmoNode')

# Filter specific warnings
warnings.filterwarnings('ignore', message='.*The model weights are not tied.*')
warnings.filterwarnings('ignore', message='.*You should use.*max_memory.*')

# Define the directory for saving Molmo files
MOLMO_PATH = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_molmo"
MOLMO_PATH.mkdir(parents=True, exist_ok=True)

# Memory configurations with detailed descriptions
MEMORY_MODES = {
    "Full Precision (45GB+ Required)": {
        "description": "Uses full FP16 precision. Requires ~45GB total system RAM, including 24GB+ VRAM.",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "double_quant": False,
        "cpu_offload": False
    },
    "8-bit Quantized (25GB+ Required)": {
        "description": "Uses 8-bit quantization. Requires ~25GB total system RAM. Good balance of quality and memory usage.",
        "load_in_8bit": True,
        "load_in_4bit": False,
        "double_quant": False,
        "cpu_offload": False
    },
    "4-bit Quantized (15GB+ Required)": {
        "description": "Uses 4-bit quantization. Requires ~15GB total system RAM. Lowest memory usage, slight quality impact.",
        "load_in_8bit": False,
        "load_in_4bit": True,
        "double_quant": True,
        "cpu_offload": False
    },
    "4-bit + CPU Offload (12GB+ Required)": {
        "description": "Uses 4-bit quantization with CPU offloading. Slowest but lowest VRAM usage (~12GB).",
        "load_in_8bit": False,
        "load_in_4bit": True,
        "double_quant": True,
        "cpu_offload": True
    }
}

# Available Molmo models
MOLMO_MODELS = {
    "MolmoE-1B (Efficient)": {
        "repo": "allenai/MolmoE-1B-0924",
        "description": "Mixture-of-Experts model, smallest option (still requires significant RAM)"
    },
    "Molmo-7B-D (Best 7B)": {
        "repo": "allenai/Molmo-7B-D-0924",
        "description": "⚠️ Very large model, requires more RAM than MolmoE-1B"
    },
    "Molmo-7B-O (Alternative 7B)": {
        "repo": "allenai/Molmo-7B-O-0924",
        "description": "⚠️ Very large model, requires more RAM than MolmoE-1B"
    }
}

class SystemResources:
    @staticmethod
    def get_system_memory():
        return psutil.virtual_memory().total / (1024 ** 3)  # GB
        
    @staticmethod
    def get_available_vram():
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB

    @staticmethod
    def check_memory_requirements(memory_mode):
        config = MEMORY_MODES[memory_mode]
        required_ram = 15 if config["load_in_4bit"] else (25 if config["load_in_8bit"] else 45)
        available_ram = SystemResources.get_system_memory()
        available_vram = SystemResources.get_available_vram()
        
        warnings = []
        if available_ram < required_ram:
            warnings.append(f"WARNING: This memory mode requires {required_ram}GB total RAM, but only {available_ram:.1f}GB available")
        
        min_vram = 12 if config["cpu_offload"] else 24
        if available_vram < min_vram:
            warnings.append(f"WARNING: Recommended minimum {min_vram}GB VRAM, but only {available_vram:.1f}GB available")
            
        return warnings

class MolmoPredictor:
    def __init__(self, model_name, memory_mode="4-bit Quantized (15GB+ Required)", use_autocast=True):
        self.model_name = MOLMO_MODELS[model_name]["repo"]
        self.memory_config = MEMORY_MODES[memory_mode]
        self.use_autocast = use_autocast and torch.cuda.is_available()
        
        # Check system resources
        warnings = SystemResources.check_memory_requirements(memory_mode)
        for warning in warnings:
            logger.warning(warning)
        
        # Download model if needed
        logger.info(f"Downloading/loading {model_name} in {memory_mode} mode...")
        self.model_path = snapshot_download(
            self.model_name,
            local_dir=MOLMO_PATH / model_name,
            local_dir_use_symlinks="auto"
        )
        
        try:
            # Configure quantization
            compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            quant_config = None
            
            if self.memory_config["load_in_4bit"] or self.memory_config["load_in_8bit"]:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=self.memory_config["load_in_8bit"],
                    load_in_4bit=self.memory_config["load_in_4bit"],
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self.memory_config["double_quant"],
                    bnb_4bit_quant_type="nf4"  # More accurate than fp4
                )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            device_map = "auto" if self.memory_config["cpu_offload"] else None
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=compute_dtype
            )
            
            logger.info(f"Successfully loaded {model_name}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Out of memory while loading model. Current mode: {memory_mode}\n"
                    "Try:\n"
                    "1. Using a more aggressive memory saving mode\n"
                    "2. Closing other applications\n"
                    "3. Restarting ComfyUI"
                ) from e
            raise

    def generate(self, image, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, top_k=50):
        try:
            # Process inputs
            inputs = self.processor.process(
                images=[image],
                text=prompt
            )
            
            # Move inputs to device and create batch
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
            
            # Configure generation
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_strings="<|endoftext|>",
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # Generate with autocast if enabled
            if self.use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.model.generate_from_batch(
                        inputs,
                        generation_config,
                        tokenizer=self.processor.tokenizer
                    )
            else:
                output = self.model.generate_from_batch(
                    inputs,
                    generation_config,
                    tokenizer=self.processor.tokenizer
                )
            
            # Get input size before cleanup
            input_size = inputs['input_ids'].size(1)
            
            # Clean up
            del inputs
            torch.cuda.empty_cache()
            
            # Extract and decode generated tokens using saved size
            generated_tokens = output[0, input_size:]
            return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "Out of memory during generation. Try:\n"
                    "1. Using a more aggressive memory saving mode\n"
                    "2. Reducing max_new_tokens\n"
                    "3. Clearing ComfyUI cache\n"
                    "4. Restarting ComfyUI"
                ) from e
            raise

class MolmoNode:
    def __init__(self):
        self.predictor = None
        self.current_model = None
        self.current_memory_mode = None
        self.current_autocast = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to be analyzed by Molmo"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                    "tooltip": "Instructions for the model. Be specific about what aspects of the image you want analyzed."
                }),
                "model_name": (list(MOLMO_MODELS.keys()), {
                    "tooltip": "⚠️ WARNING: These are very large models requiring significant RAM/VRAM. Start with MolmoE-1B."
                }),
                "memory_mode": (list(MEMORY_MODES.keys()), {
                    "default": "4-bit Quantized (15GB+ Required)",
                    "tooltip": "Controls RAM/VRAM usage. Use most aggressive option that works on your system."
                }),
                "max_new_tokens": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 2048,
                    "tooltip": "Maximum tokens to generate. Higher values need more VRAM. Start small (200) and increase if needed."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness. Lower (0.1-0.5) = more focused, higher (0.8-2.0) = more creative."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Nucleus sampling. Lower = more focused on likely tokens, higher = more diverse vocabulary."
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Limits token choices to top K most likely. Lower = more focused, higher = more variety."
                }),
                "use_autocast": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enables mixed precision. Keeps quality while reducing VRAM usage. Recommended ON."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Molmo"

    def generate(self, image, prompt, model_name, memory_mode="4-bit Quantized (15GB+ Required)", 
                max_new_tokens=200, temperature=0.7, top_p=0.9, top_k=50, use_autocast=True):
        
        try:
            # Initialize or update predictor if needed
            if (self.predictor is None or 
                self.current_model != model_name or 
                self.current_memory_mode != memory_mode or
                self.current_autocast != use_autocast):
                
                # Clean up old model if it exists
                if self.predictor is not None:
                    del self.predictor.model
                    del self.predictor.processor
                    torch.cuda.empty_cache()
                
                self.predictor = MolmoPredictor(
                    model_name,
                    memory_mode=memory_mode,
                    use_autocast=use_autocast
                )
                self.current_model = model_name
                self.current_memory_mode = memory_mode
                self.current_autocast = use_autocast
            
            # Convert tensor to PIL Image
            pil_image = Image.fromarray((image[0] * 255).numpy().astype('uint8'))
            
            # Generate response
            response = self.predictor.generate(
                pil_image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            return (response,)
            
        except Exception as e:
            # Clean up on error
            if hasattr(self, 'predictor') and self.predictor is not None:
                del self.predictor.model
                del self.predictor.processor
                self.predictor = None
            torch.cuda.empty_cache()
            return (f"Error: {str(e)}",)

# Register the node
NODE_CLASS_MAPPINGS = {
    "MolmoNode": MolmoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MolmoNode": "Molmo Vision-Language Model"
}