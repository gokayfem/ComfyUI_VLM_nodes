import os
import torch
import time  # Add this
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, logging, AutoConfig
from PIL import Image, ImageDraw, ImageColor, ImageFilter
from pathlib import Path
import folder_paths
import numpy as np
import shutil
import json
import random
import torchvision.transforms.functional as F
import re
import psutil
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

logging.set_verbosity_error()  # Only show errors, not info/warnings

MODELS_DIR = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0])

class Paligemma:
    def __init__(self):
        print("Initializing Paligemma")
        self.default_model_id = "gokaygokay/sd3-long-captioner-v2"
        self.model = None
        self.processor = None
        self.current_model_id = None
        
        # Ensure the models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Move these to be class variables instead of instance variables
        self.available_tasks = [
            "Captioning",
            "Segmentation",
            "Question Answering"
        ]
        
        # Full list of available models
        self.available_models = [
            "Custom",
            "gokaygokay/sd3-long-captioner-v2",
            "google/paligemma-3b-ft-cococap-448",
            "google/paligemma-3b-ft-refcoco-seg-896",
            "google/paligemma-3b-ft-rsvqa-hr-224",
            "google/paligemma-3b-ft-science-qa-448",
            "google/paligemma-3b-ft-vqav2-448",
            "google/paligemma-3b-mix-224",
            "google/paligemma-3b-mix-224-jax",
            "google/paligemma-3b-mix-224-keras",
            "google/paligemma-3b-mix-448",
            "google/paligemma-3b-mix-448-jax", 
            "google/paligemma-3b-mix-448-keras",
            "google/paligemma-3b-pt-224",
            "google/paligemma-3b-pt-224-jax",
            "google/paligemma-3b-pt-224-keras",
            "google/paligemma-3b-pt-448",
            "google/paligemma-3b-pt-448-jax",
            "google/paligemma-3b-pt-448-keras",
            "google/paligemma-3b-pt-896",
            "google/paligemma-3b-pt-896-jax",
            "google/paligemma-3b-pt-896-keras"
        ]


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to process with Paligemma"
                }),
                "model_id": (
                    ["gokaygokay/sd3-long-captioner-v2"] + [  # Default first
                        "Custom",  # Moved up for better UX
                        "google/paligemma-3b-ft-refcoco-seg-896",
                        "google/paligemma-3b-ft-cococap-448",
                        "google/paligemma-3b-ft-rsvqa-hr-224",
                        "google/paligemma-3b-ft-science-qa-448",
                        "google/paligemma-3b-ft-vqav2-448",
                        "google/paligemma-3b-mix-224",
                        "google/paligemma-3b-mix-224-jax",
                        "google/paligemma-3b-mix-224-keras",
                        "google/paligemma-3b-mix-448",
                        "google/paligemma-3b-mix-448-jax",
                        "google/paligemma-3b-mix-448-keras",
                        "google/paligemma-3b-pt-224",
                        "google/paligemma-3b-pt-224-jax",
                        "google/paligemma-3b-pt-224-keras",
                        "google/paligemma-3b-pt-448",
                        "google/paligemma-3b-pt-448-jax",
                        "google/paligemma-3b-pt-448-keras",
                        "google/paligemma-3b-pt-896",
                        "google/paligemma-3b-pt-896-jax",
                        "google/paligemma-3b-pt-896-keras",
                    ],
                    {
                        "default": "gokaygokay/sd3-long-captioner-v2",
                        "tooltip": "Select the Paligemma model to use"
                    }
                ),
                "custom_model_id": (  # Moved from optional to required, right after model_id
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Enter Hugging Face model repo (e.g., 'gokaygokay/sd3-long-captioner-v2') or full URL (e.g., 'https://huggingface.co/gokaygokay/sd3-long-captioner-v2')"
                    }
                ),
                "task_type": (
                    ["Captioning", "Segmentation", "Question Answering"],
                    {
                        "default": "Captioning",
                        "tooltip": (
                            "Select task type:\n"
                            "- Captioning: Generates detailed image descriptions\n"
                            "- Segmentation: Creates masks for specified objects/regions (use ft-refcoco-seg model)\n"
                            "- Question Answering: Answers questions about the image"
                        )
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "tooltip": (
                            "Input text based on task type:\n"
                            "- For Captioning: Describe what aspects to focus on\n"
                            "- For Segmentation: Describe what to segment (e.g., 'segment the dog', 'find the red car')\n"
                            "- For Q&A: Enter your question about the image"
                        )
                    }
                ),
            },
            "optional": {
                # Model Configuration
                "precision": (
                    ["bfloat16", "float32"],  # Reordered with bfloat16 as default
                    {
                        "default": "bfloat16",
                        "tooltip": "Model precision - affects VRAM usage and processing speed"
                    }
                ),
                "device": (
                    ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    {
                        "default": "cuda",
                        "tooltip": "Device to run model on. 'cuda' for best performance if available"
                    }
                ),
                "quantization": (
                    ["None", "8bit", "4bit"],
                    {
                        "default": "None",
                        "tooltip": "Model quantization for reduced VRAM usage"
                    }
                ),

                # Generation Parameters (for Captioning and Q&A)
                "max_tokens": (
                    "INT",
                    {
                        "default": 256,
                        "min": 1,
                        "max": 1024,
                        "tooltip": "[Captioning/Q&A] Maximum number of tokens to generate"
                    }
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 512,
                        "tooltip": "[Captioning/Q&A] Minimum number of tokens to generate"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "[Captioning/Q&A] Temperature for text generation"
                    }
                ),

                # Segmentation-specific Parameters
                "fill_mask": (
                    ["True", "False"],
                    {
                        "default": "True",
                        "tooltip": "[Segmentation Only] Fill the segmented regions with semi-transparent color overlays"
                    }
                ),
                "mask_color": (
                    "STRING",
                    {
                        "default": "white",
                        "tooltip": "[Segmentation Only] Color for segmentation mask (e.g., 'white', 'black', '#FF0000')"
                    }
                ),
                "mask_opacity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "[Segmentation Only] Opacity of the segmentation overlay (0.0 = transparent, 1.0 = solid)"
                    }
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "[Segmentation Only] Threshold for binary mask creation"
                    }
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "[Segmentation Only] Blur radius for mask edges (0 for no blur)"
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("description", "mask", "visualization")
    FUNCTION = "process_task"
    CATEGORY = "VLM Nodes/Paligemma"
    OUTPUT_NODE = True

    @classmethod
    def DISPLAY_SIZE(cls):
        return 600, 400  # width, height in pixels

    def tensor2pil(self, image):
        """Convert tensor to PIL Image with proper memory handling."""
        try:
            # Check input validity
            if image is None:
                raise ValueError("Input image is None")
            
            if isinstance(image, torch.Tensor):
                # Check tensor size to prevent OOM
                total_elements = image.numel() * image.element_size()
                available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) \
                    if torch.cuda.is_available() else psutil.virtual_memory().available
                
                if total_elements > available_memory * 0.8:  # Use only 80% of available memory
                    raise RuntimeError("Image tensor too large for available memory")
                
                # Handle different tensor formats
                try:
                    if len(image.shape) == 4:  # (B, C, H, W)
                        image = image.squeeze(0)
                    if image.shape[0] == 1:  # Grayscale
                        image = image.repeat(3, 1, 1)
                    elif image.shape[0] == 4:  # RGBA
                        image = image[:3]  # Take only RGB channels
                    
                    # Ensure proper value range
                    if image.max() > 1.0 or image.min() < 0:
                        image = image / 255.0 if image.max() > 1.0 else image
                    
                    image = (image.clamp(0, 1) * 255).round().to(torch.uint8)
                    image = image.cpu().numpy()
                    if image.shape[0] == 3:  # Convert from CHW to HWC format
                        image = image.transpose(1, 2, 0)
                        
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError("GPU out of memory during tensor processing")
                except Exception as e:
                    raise RuntimeError(f"Failed to process tensor: {str(e)}")
                    
                try:
                    return Image.fromarray(image)
                except Exception as e:
                    raise RuntimeError(f"Failed to create PIL image from array: {str(e)}")
                    
            elif isinstance(image, np.ndarray):
                # Check array size
                if image.size * image.itemsize > psutil.virtual_memory().available * 0.8:
                    raise RuntimeError("Image array too large for available memory")
                    
                try:
                    if len(image.shape) == 4:  # (B, H, W, C)
                        image = image.squeeze(0)
                    if image.shape[-1] == 1:  # Grayscale
                        image = np.repeat(image, 3, axis=-1)
                    elif image.shape[-1] == 4:  # RGBA
                        image = image[..., :3]  # Take only RGB channels
                        
                    return Image.fromarray(image.astype(np.uint8))
                except Exception as e:
                    raise RuntimeError(f"Failed to process numpy array: {str(e)}")
                    
            elif isinstance(image, Image.Image):
                return image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            print(f"Error in tensor2pil conversion: {str(e)}")
            raise

    def pil2tensor(self, image):
        """Convert PIL Image to tensor with memory handling."""
        try:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
                
            # Check image size
            w, h = image.size
            channels = len(image.getbands())
            required_memory = w * h * channels * 4  # 4 bytes per float32
            
            if required_memory > psutil.virtual_memory().available * 0.8:
                raise RuntimeError("Image too large for available memory")
                
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            try:
                # Convert to numpy array
                np_image = np.array(image)
                
                # Convert to float32 and normalize to [0, 1]
                tensor = torch.from_numpy(np_image).float() / 255.0
                
                # Add batch dimension if needed and ensure BHWC format
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                    
                return tensor
                
            except np.core._exceptions.MemoryError:
                raise RuntimeError("Not enough memory to convert image to array")
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError("Not enough GPU memory to create tensor")
            except Exception as e:
                raise RuntimeError(f"Failed to convert image: {str(e)}")
                
        except Exception as e:
            print(f"Error in pil2tensor conversion: {str(e)}")
            raise

    def get_model_path(self, model_id):
        """Get the full path for a model, checking both standard and HF cache structures."""
        if model_id == "Custom":
            return None
            
        # Convert model ID to HuggingFace cache format
        hf_format = model_id.replace('/', '--')
        
        # Define possible paths in order of preference
        paths = [
            Path(MODELS_DIR) / f"models--{hf_format}" / "snapshots/refs/main",
            Path(MODELS_DIR) / f"models--{hf_format}",
            Path(MODELS_DIR) / hf_format,
            Path(MODELS_DIR) / model_id.split('/')[-1]
        ]
        
        # Check each path silently for valid model files
        for path in paths:
            if path.exists():
                model_files = list(path.glob('*.bin')) + list(path.glob('*.safetensors'))
                config_file = path / 'config.json'
                if model_files and config_file.exists():
                    return path

        # If no valid path found, use first path for download location
        return paths[0]

    def _download_model(self, model_id, dtype, device_map, quantization_config, max_retries=3, retry_delay=5):
        """Helper method to download model and processor."""
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries}...")
                
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    cache_dir=str(MODELS_DIR),
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    cache_dir=str(MODELS_DIR),
                    trust_remote_code=True
                )
                
                print("Download successful!")
                return

            except Exception as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError("GPU out of memory error. Try using CPU or reducing model precision.")
                
                if attempt < max_retries - 1:
                    print(f"Download failed: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to download model after {max_retries} attempts: {str(e)}")

    def load_model(self, model_id=None, precision='float32', device='cpu', quantization=None, custom_model_id=None):
        try:
            # Unload current model if it exists
            if self.model is not None:
                del self.model
                del self.processor
                torch.cuda.empty_cache()

            # Determine the effective model ID
            if model_id == "Custom" and custom_model_id:
                effective_model_id = custom_model_id
            elif model_id == "Custom" and not custom_model_id:
                raise ValueError("Custom model selected but no custom model ID provided")
            else:
                effective_model_id = model_id or self.default_model_id

            if self.current_model_id == effective_model_id:
                return

            # Set up model loading configurations
            dtype = torch.float32 if precision == 'float32' else torch.bfloat16
            device_map = 'auto' if device.startswith('cuda') else None
            
            # Configure quantization
            quantization_config = None
            if quantization == '8bit':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == '4bit':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)

            # Load model from local path or download if needed
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                effective_model_id,
                torch_dtype=dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                cache_dir=str(MODELS_DIR),
                trust_remote_code=True,
                use_safetensors=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                effective_model_id,
                cache_dir=str(MODELS_DIR),
                trust_remote_code=True
            )

            self.current_model_id = effective_model_id

        except Exception as e:
            print(f"\nError in load_model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_task(self, image, prompt, task_type, model_id=None, precision='float32', device='cpu', 
                    quantization=None, custom_model_id=None, mask_threshold=0.5, mask_blur=0, 
                    max_tokens=256, min_tokens=10,
                    temperature=0.7, num_beams=5, do_sample="True", early_stopping="True",
                    fill_mask="True", mask_color="white", mask_opacity=0.5):
        """Process various vision-language tasks."""
        try:
            start_time = time.time()
            
            # Add visual separator and status
            print(f"\n{'='*50}")
            print(f"Processing {task_type} task:")
            print(f"{'='*50}")
            print(f"Prompt: {prompt}")
            print(f"Model: {model_id if model_id != 'Custom' else custom_model_id}\n")

            # Validate task/model combination
            if task_type == "Segmentation":
                if "refcoco-seg" not in (model_id or ''):
                    print("\nWarning: Using non-segmentation model for segmentation task!")
                    print("Recommended model: google/paligemma-3b-ft-refcoco-seg-896")
                    print("Results may be suboptimal.\n")

            # Ensure the model is loaded for the correct task
            if self.model is None or self.current_model_id != model_id:
                self.load_model(model_id, precision, device, quantization, custom_model_id)

            # Convert input tensor to PIL Image
            pil_image = self.tensor2pil(image)
            W, H = pil_image.size

            # Initialize default return values
            description = ""
            mask_tensor = torch.zeros((1, H, W), dtype=torch.float32)
            visualization = image.clone()  # Default to input image
            
            if task_type == "Captioning":
                description = self.process_image(pil_image, prompt, max_tokens, min_tokens, 
                                              temperature, num_beams, do_sample, early_stopping)
                visualization = image.clone()
                    
            elif task_type == "Segmentation":
                fill_mask = fill_mask.lower() == "true"
                description, mask_pil, vis_pil = self.process_segmentation(
                    pil_image, 
                    prompt, 
                    fill_mask=fill_mask,
                    mask_color=mask_color, 
                    mask_opacity=float(mask_opacity),
                    mask_threshold=mask_threshold,
                    mask_blur=mask_blur
                )
                
                if mask_pil is not None:
                    mask_tensor = self.pil2tensor(mask_pil)
                if vis_pil is not None:
                    visualization = self.pil2tensor(vis_pil)
                    
            elif task_type == "Question Answering":
                description = self.process_question_answering(
                    pil_image, prompt, max_tokens, min_tokens,
                    temperature, num_beams, do_sample, early_stopping
                )
                visualization = image.clone()

            # Ensure mask is properly formatted
            if isinstance(mask_tensor, torch.Tensor):
                if len(mask_tensor.shape) == 3:  # If RGB, take first channel
                    mask_tensor = mask_tensor[:, :, 0]
                mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension if needed
            
            # Ensure visualization is properly formatted
            if isinstance(visualization, Image.Image):
                visualization = self.pil2tensor(visualization)

            # Add completion timing and status
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n{'='*50}")
            print(f"Task completed in {duration:.2f} seconds")
            print(f"{'='*50}\n")
            
            return (description, mask_tensor, visualization)

        except Exception as e:
            print(f"\nError in process_task: {str(e)}")
            H, W = image.shape[1:3] if len(image.shape) == 4 else image.shape[:2]
            
            print(f"\n{'='*50}")
            print(f"Task failed after {time.time() - start_time:.2f} seconds")
            print(f"{'='*50}\n")
            
            return (
                f"Error: {str(e)}", 
                torch.zeros((1, H, W), dtype=torch.float32),
                image.clone()
            )

    def process_image(self, image, prompt, max_tokens=256, min_tokens=10, 
                     temperature=0.7, num_beams=5, do_sample="True", 
                     early_stopping="True"):
        """Process image captioning task."""
        try:
            print(f"Generating caption with prompt: {prompt}")
            
            # Ensure image is PIL
            if not isinstance(image, Image.Image):
                image = self.tensor2pil(image)
                    
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Move inputs to device and correct dtype
            inputs = {
                "input_ids": inputs["input_ids"].to(device=self.model.device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(device=self.model.device, dtype=torch.long),
                "pixel_values": inputs["pixel_values"].to(device=self.model.device, dtype=self.model.dtype)
            }
            
            input_len = inputs["input_ids"].shape[-1]

            # Convert string bools to actual bools
            do_sample = do_sample.lower() == "true"
            early_stopping = early_stopping.lower() == "true"

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams,
                    early_stopping=early_stopping
                )
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                print(f"Generated caption: {decoded}")
                return decoded
        except Exception as e:
            print(f"Error processing image: {e}")
            return f"Error processing image: {str(e)}"

    def process_segmentation(self, image, prompt, fill_mask=True, mask_color="white", 
                            mask_opacity=0.5, mask_threshold=0.5, mask_blur=0):
        """Process segmentation task with proper Paligemma format handling."""
        try:
            formatted_prompt = "<REFERRING_EXPRESSION_SEGMENTATION> Find and segment: " + prompt
            
            # Ensure image is PIL
            if not isinstance(image, Image.Image):
                image = self.tensor2pil(image)
            
            W, H = image.size
            
            # Process inputs
            inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device=self.model.device, dtype=torch.long if k != "pixel_values" else self.model.dtype) 
                     for k, v in inputs.items()}

            # Generate segmentation
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=5
                )

            # Process output
            decoded_text = self.processor.decode(outputs[0], skip_special_tokens=False)
            print(f"Raw model output: {decoded_text}")
            
            # Extract coordinates and segments
            coords = []
            segments = []
            current_segment = []
            
            # Parse all tokens in sequence
            tokens = re.findall(r'<(loc|seg)(\d+)>', decoded_text)
            
            for token_type, value in tokens:
                if token_type == 'loc':
                    # Convert flat index to x,y coordinates
                    idx = int(value)
                    x = idx % W
                    y = idx // W
                    current_segment.append((x, y))
                elif token_type == 'seg' and current_segment:
                    # Start new segment
                    if len(current_segment) >= 3:  # Need at least 3 points for a polygon
                        segments.append(current_segment)
                    current_segment = []
            
            # Add final segment if exists
            if len(current_segment) >= 3:
                segments.append(current_segment)

            if not segments:
                return "", None, image.convert('RGB')

            # Create mask
            mask_image = Image.new('L', (W, H), 0)
            mask_draw = ImageDraw.Draw(mask_image)
            
            # Draw all segments
            for points in segments:
                mask_draw.polygon(points, fill=255, outline=255)

            # Apply optional blur
            if mask_blur > 0:
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            
            # Apply threshold
            if mask_threshold != 0.5:
                mask_array = np.array(mask_image)
                mask_array = (mask_array > (mask_threshold * 255)).astype(np.uint8) * 255
                mask_image = Image.fromarray(mask_array)
            
            # Create visualization if requested
            if fill_mask:
                vis_image = image.copy().convert('RGBA')
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Convert color name to RGB
                if isinstance(mask_color, str):
                    try:
                        mask_color = ImageColor.getrgb(mask_color)
                    except ValueError:
                        print(f"Invalid color '{mask_color}', defaulting to white")
                        mask_color = (255, 255, 255)
                
                # Draw all segments in overlay
                mask_color_rgba = (*mask_color, int(255 * mask_opacity))
                for points in segments:
                    draw.polygon(points, fill=mask_color_rgba)
                
                vis_image = Image.alpha_composite(vis_image, overlay)
                vis_image = vis_image.convert('RGB')
            else:
                vis_image = image.convert('RGB')

            return "", mask_image, vis_image

        except Exception as e:
            print(f"Error in segmentation processing: {e}")
            return f"Error: {str(e)}", None, image.convert('RGB')

    def process_question_answering(self, image, prompt, max_tokens=256, min_tokens=10,
                                   temperature=0.7, num_beams=5, do_sample="True", 
                                   early_stopping="True"):
            """Process question answering task."""
            try:
                print(f"Processing question with prompt: {prompt}")
                
                # Ensure image is PIL
                if not isinstance(image, Image.Image):
                    image = self.tensor2pil(image)
                        
                # Process inputs
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                
                # Move inputs to device and correct dtype
                inputs = {
                    "input_ids": inputs["input_ids"].to(device=self.model.device, dtype=torch.long),
                    "attention_mask": inputs["attention_mask"].to(device=self.model.device, dtype=torch.long),
                    "pixel_values": inputs["pixel_values"].to(device=self.model.device, dtype=self.model.dtype)
                }
                
                input_len = inputs["input_ids"].shape[-1]

                # Convert string bools to actual bools
                do_sample = do_sample.lower() == "true"
                early_stopping = early_stopping.lower() == "true"

                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        min_new_tokens=min_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        num_beams=num_beams,
                        early_stopping=early_stopping
                    )
                    generation = generation[0][input_len:]
                    decoded = self.processor.decode(generation, skip_special_tokens=True)
                    print(f"Generated answer: {decoded}")
                    return decoded
                    
            except Exception as e:
                print(f"Error processing question: {e}")
                return f"Error processing question: {str(e)}"

    def __del__(self):
        """Cleanup method to ensure model is properly unloaded"""
        if hasattr(self, 'model') and self.model is not None:
            try:
                print("Cleaning up PaLI-Gemma resources...")
                del self.model
                del self.processor
                torch.cuda.empty_cache()
                print("Cleanup completed successfully")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "Paligemma": Paligemma
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Paligemma": "Paligemma"
}