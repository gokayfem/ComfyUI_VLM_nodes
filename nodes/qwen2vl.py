import torch
import psutil
import os
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToPILImage
from huggingface_hub import snapshot_download
import folder_paths
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

def check_flash_attention():
    """Check if flash attention 2 is available"""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False

FLASH_ATTENTION_AVAILABLE = check_flash_attention()

# Define the directory for saving Qwen2-VL files
files_for_qwen2vl = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_qwen2vl"
files_for_qwen2vl.mkdir(parents=True, exist_ok=True)

# Model VRAM requirements (approximate, in GB)
MODEL_VRAM_REQUIREMENTS = {
    "Qwen2-VL-2B": 4,
    "Qwen2-VL-7B": 14,
    "Qwen2-VL-72B": 40,
    "Qwen2-VL-2B-AWQ": 2,
    "Qwen2-VL-2B-GPTQ-Int4": 2,
    "Qwen2-VL-2B-GPTQ-Int8": 3,
    "Qwen2-VL-7B-AWQ": 5,
    "Qwen2-VL-7B-GPTQ-Int4": 5,
    "Qwen2-VL-7B-GPTQ-Int8": 8,
    "Qwen2-VL-72B-AWQ": 20,
    "Qwen2-VL-72B-GPTQ-Int4": 20,
    "Qwen2-VL-72B-GPTQ-Int8": 25,
}

QWEN2_VL_MODELS = {
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen2-VL-2B-AWQ": "Qwen/Qwen2-VL-2B-Instruct-AWQ",
    "Qwen2-VL-2B-GPTQ-Int4": "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
    "Qwen2-VL-2B-GPTQ-Int8": "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
    "Qwen2-VL-7B-AWQ": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    "Qwen2-VL-7B-GPTQ-Int4": "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
    "Qwen2-VL-7B-GPTQ-Int8": "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
    "Qwen2-VL-72B-AWQ": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
    "Qwen2-VL-72B-GPTQ-Int4": "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
    "Qwen2-VL-72B-GPTQ-Int8": "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
}

MEMORY_EFFICIENT_CONFIGS = {
    "Balanced (8-bit)": {
        "load_in_8bit": True,
        "load_in_4bit": False,
        "cpu_offload": False,
        "attention_mode": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else None,
    },
    "Maximum Savings (4-bit)": {
        "load_in_8bit": False,
        "load_in_4bit": True,
        "cpu_offload": True,
        "attention_mode": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else None,
    },
    "CPU Offload": {
        "load_in_8bit": False,
        "load_in_4bit": False,
        "cpu_offload": True,
        "attention_mode": None,
    },
    "Default": {
        "load_in_8bit": False,
        "load_in_4bit": False,
        "cpu_offload": False,
        "attention_mode": None,
    }
}

class SystemResources:
    @staticmethod
    def get_available_memory():
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024 * 1024 * 1024)

    @staticmethod
    def get_available_vram():
        """Get available VRAM in GB"""
        if not torch.cuda.is_available():
            return 0
        
        try:
            torch.cuda.empty_cache()  # Clear unused cached memory
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        except:
            return 0

    @staticmethod
    def check_resources(model_name, memory_mode):
        """Check if system has enough resources for the model"""
        required_vram = MODEL_VRAM_REQUIREMENTS.get(model_name, 0)
        config = MEMORY_EFFICIENT_CONFIGS[memory_mode]
        
        # Adjust VRAM requirements based on memory mode
        if config["load_in_8bit"]:
            required_vram = required_vram * 0.5  # Approximately half VRAM usage
        elif config["load_in_4bit"]:
            required_vram = required_vram * 0.25  # Approximately quarter VRAM usage
        elif config["cpu_offload"]:
            required_vram = required_vram * 0.7  # Rough estimate for CPU offloading
            
        available_vram = SystemResources.get_available_vram()
        available_memory = SystemResources.get_available_memory()
        
        # Need at least 2GB system memory buffer
        required_system_memory = required_vram + 2
        
        error_messages = []
        if available_vram < required_vram:
            error_messages.append(
                f"Insufficient VRAM: Model {model_name} requires {required_vram:.1f}GB VRAM, "
                f"but only {available_vram:.1f}GB available. "
                "Try using a more aggressive memory saving mode."
            )
        
        if available_memory < required_system_memory:
            error_messages.append(
                f"Insufficient system memory: Need at least {required_system_memory:.1f}GB, "
                f"but only {available_memory:.1f}GB available"
            )
            
        return error_messages

class Qwen2VLPredictor:
    def __init__(self, model_name, memory_mode="Balanced (8-bit)"):
        # Check system resources
        error_messages = SystemResources.check_resources(model_name, memory_mode)
        if error_messages:
            raise RuntimeError("\n".join(error_messages))
            
        self.model_path = snapshot_download(
            QWEN2_VL_MODELS[model_name],
            local_dir=files_for_qwen2vl / model_name,
            force_download=False,
            local_files_only=False,
            revision="main"
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Get memory configuration
            config = MEMORY_EFFICIENT_CONFIGS[memory_mode]
            
            # Setup quantization config if needed
            quantization_config = None
            if config["load_in_8bit"] or config["load_in_4bit"]:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=config["load_in_8bit"],
                    load_in_4bit=config["load_in_4bit"],
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Setup device map for CPU offloading
            device_map = "auto" if config["cpu_offload"] else None
            
            # Base model kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": device_map,
                "quantization_config": quantization_config,
            }
            
            # Add attention optimization if specified and available
            if config["attention_mode"]:
                try:
                    model_kwargs["attn_implementation"] = config["attention_mode"]
                except Exception as e:
                    print(f"Warning: Flash Attention 2 requested but not available: {str(e)}")
            
            # Load model with appropriate settings based on model type
            if "GPTQ" in model_name or "AWQ" in model_name:
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                **model_kwargs
            )
                
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                process = psutil.Process()
                mem_info = process.memory_info()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Out of VRAM while loading {model_name}. Try:\n"
                    "1. Using a more aggressive memory saving mode\n"
                    "2. Using a smaller model (e.g., 2B instead of 7B)\n"
                    "3. Using a quantized version (AWQ/GPTQ)\n"
                    "4. Clearing other models from memory\n"
                    "5. Restarting ComfyUI\n"
                    f"Process memory: {mem_info.rss / 1024**3:.1f}GB"
                ) from e
            raise
        
    def process_video(self, video_frames, fps=1.0):
        """Process video frames for video understanding"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames,
                        "fps": fps
                    }
                ]
            }
        ]
        return messages
        
    def generate_predictions(self, image_path, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, video_frames=None, fps=1.0):
        try:
            # Handle video input if provided
            if video_frames:
                messages = self.process_video(video_frames, fps)
                messages[0]["content"].append({"type": "text", "text": prompt})
            else:
                # Standard image processing
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            
            # Process the inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            )
            
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and return the response
                generated_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return generated_text.strip()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "Out of VRAM during generation. Try:\n"
                        "1. Using a more aggressive memory saving mode\n"
                        "2. Reducing max_new_tokens\n"
                        "3. Using a smaller model\n"
                        "4. Using a quantized version (AWQ/GPTQ)"
                    ) from e
                raise
                
        except Exception as e:
            return f"Error during generation: {str(e)}"

class Qwen2VLNode:
    def __init__(self):
        self.predictor = None
        self.current_model = None
        self.current_memory_mode = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "model_name": (list(QWEN2_VL_MODELS.keys()),),
                "memory_mode": (list(MEMORY_EFFICIENT_CONFIGS.keys()),),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                })
            },
            "optional": {
                "video_frames": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Qwen2-VL"

    def generate(self, image, text_input, model_name, memory_mode="Balanced (8-bit)", 
                max_new_tokens=512, temperature=0.7, top_p=0.9, video_frames=None, fps=1.0):
        # Initialize or update predictor if model or memory mode changed
        if (self.predictor is None or self.current_model != model_name or 
            self.current_memory_mode != memory_mode):
            
            # Clean up old model
            if self.predictor is not None:
                del self.predictor.model
                del self.predictor.processor
                del self.predictor.tokenizer
                torch.cuda.empty_cache()
            
            try:
                self.predictor = Qwen2VLPredictor(model_name, memory_mode)
                self.current_model = model_name
                self.current_memory_mode = memory_mode
            except Exception as e:
                return (f"Error initializing model: {str(e)}",)
            
        # Convert tensor image to PIL Image and save temporarily
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_qwen2vl / "temp_image.png"
        pil_image.save(temp_path)
                        
        video_frame_list = None
        if video_frames is not None:
            video_frame_list = [str(temp_path)]  # Use current image as first frame
            # Add additional video frames if provided
            for frame in video_frames[1:]:
                frame_path = files_for_qwen2vl / f"temp_frame_{len(video_frame_list)}.png"
                ToPILImage()(frame.permute(2, 0, 1)).save(frame_path)
                video_frame_list.append(str(frame_path))
        
        try:
            # Generate response
            response = self.predictor.generate_predictions(
                temp_path,
                text_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                video_frames=video_frame_list,
                fps=fps
            )
            
            # Clean up all temporary files
            try:
                os.remove(temp_path)
                if video_frame_list:
                    for frame_path in video_frame_list[1:]:
                        try:
                            os.remove(frame_path)
                        except:
                            pass
            except:
                pass
            
            return (response,)
            
        except Exception as e:
            return (f"Error during generation: {str(e)}",)

# Register the node
NODE_CLASS_MAPPINGS = {
    "Qwen2VLNode": Qwen2VLNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2VLNode": "Qwen2-VL Model"
}
