import folder_paths
import os
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
from torchvision.transforms import ToPILImage
import gc
import torch


supported_LLava_extensions = set(['.gguf'])

try:
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = (folder_paths.folder_names_and_paths["LLavacheckpoints"][0], supported_LLava_extensions)
except:
    # check if LLavacheckpoints exists otherwise create
    if not os.path.isdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints")):
        os.mkdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints"))
        
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = ([os.path.join(folder_paths.models_dir, "LLavacheckpoints")], supported_LLava_extensions)
    
class LLavaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),   
              "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 8192, "step": 64}),
              "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
              "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
              "clip": ("CUSTOM", {"default": ""}),
                             }}
                
    
    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llava_checkpoint"

    CATEGORY = "VLM Nodes/LLava"
    def load_llava_checkpoint(self, ckpt_name, max_ctx, gpu_layers, n_threads, clip ):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        llm = Llama(model_path = ckpt_path, chat_handler=clip,offload_kqv=True, f16_kv=True, use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, echo=False) 
        return (llm, ) 
    
class LlavaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {               
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"), ), 
                             }}
    
    RETURN_TYPES = ("CUSTOM", )
    RETURN_NAMES = ("clip", )
    FUNCTION = "load_clip_checkpoint"

    CATEGORY = "VLM Nodes/LLava"
    def load_clip_checkpoint(self, clip_name):
        clip_path = folder_paths.get_full_path("LLavacheckpoints", clip_name)
        clip = Llava15ChatHandler(clip_model_path = clip_path, verbose=False)        
        return (clip, ) 

class LLavaSamplerSimple:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",{"forceInput": True} ),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),              
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text(self, image, prompt, model, temperature):
        

        # Assuming 'image' is a PyTorch tensor of shape [C, H, W]
        # Convert the PyTorch tensor to a PIL image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))

        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # You can change the format if needed

        # Get the bytes from the buffer
        image_bytes = buffer.getvalue()

        # Encode the bytes to base64
        base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        # Now, `base64_string` contains the base64-encoded string of the image

        llm = model
        response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": "You are an assistant who perfectly describes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url" : base64_string}},
                        {"type" : "text", "text": f"{prompt}"}
                    ]
                }
                
            ],
            temperature = temperature,
        )

        return (f"{response['choices'][0]['message']['content']}", )
    
class LLavaSamplerAdvanced:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_msg": ("STRING",{"default" : "You are an assistant who perfectly describes images."}),
                "prompt": ("STRING",{"forceInput": True, "default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}), 
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step":1})                             
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text_advanced(self, image, system_msg, prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, repeat_penalty, top_k,seed):
        
        # Assuming 'image' is a PyTorch tensor of shape [C, H, W]
        # Convert the PyTorch tensor to a PIL image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))

        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # You can change the format if needed

        # Get the bytes from the buffer
        image_bytes = buffer.getvalue()

        # Encode the bytes to base64
        base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        # Now, `base64_string` contains the base64-encoded string of the image

        llm = model
        response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url" : base64_string}},
                        {"type" : "text", "text": f"{prompt}"}
                    ]
                }

            ],
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            repeat_penalty = repeat_penalty,
            seed=seed

        )


        return (f"{response['choices'][0]['message']['content']}", )
    
class LLavaOptionalMemoryFreeSimple:
    def __init__(self):
        self.llm = None  # Store the model instance
        self.clip = None  # Store the clip instance

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 128000, "step": 64}),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "unload": ("BOOLEAN", {"default": False}),  # Add unload parameter
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text(self, ckpt_name, clip_name, max_ctx, gpu_layers, n_threads, image, prompt, temperature, unload):
        # Load the model
        

        # Load the clip
        clip_path = folder_paths.get_full_path("LLavacheckpoints", clip_name)
        self.clip = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)

        # Load model
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        self.llm = Llama(model_path = ckpt_path, chat_handler=self.clip, offload_kqv=True, f16_kv=True, use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, echo=False)

        # Assuming 'image' is a PyTorch tensor of shape [C, H, W]
        # Convert the PyTorch tensor to a PIL image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))

        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # You can change the format if needed

        # Get the bytes from the buffer
        image_bytes = buffer.getvalue()

        # Encode the bytes to base64
        base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        # Now, `base64_string` contains the base64-encoded string of the image

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_string}},
                        {"type": "text", "text": f"{prompt}"}
                    ]
                }
            ],
            temperature=temperature,
        )

        if unload and self.llm is not None:
            del self.llm  # Unload the model
            self.llm = None  # Remove reference to the model
            gc.collect()
            torch.cuda.empty_cache()

        if unload and self.clip is not None:
            del self.clip  # Unload the clip
            self.clip = None  # Remove reference to the clip
            gc.collect()
            torch.cuda.empty_cache()

        return (f"{response['choices'][0]['message']['content']}", )

class LLavaOptionalMemoryFreeAdvanced:
    def __init__(self):
        self.llm = None  # Store the model instance
        self.clip = None  # Store the clip instance

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 128000, "step": 64}),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "image": ("IMAGE",),
                "system_msg": ("STRING", {"default": "You are an assistant who perfectly describes images."}),
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "unload": ("BOOLEAN", {"default": False}),  # Add unload parameter
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text_advanced(self, ckpt_name, clip_name, max_ctx, gpu_layers, n_threads, image, system_msg, prompt, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, seed, unload):

        # Load the clip
        clip_path = folder_paths.get_full_path("LLavacheckpoints", clip_name)
        self.clip = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)

        # Load model
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        self.llm = Llama(model_path = ckpt_path, chat_handler=self.clip, offload_kqv=True, f16_kv=True, use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, echo=False)

        # Assuming 'image' is a PyTorch tensor of shape [C, H, W]
        # Convert the PyTorch tensor to a PIL image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))

        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # You can change the format if needed

        # Get the bytes from the buffer
        image_bytes = buffer.getvalue()

        # Encode the bytes to base64
        base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        # Now, `base64_string` contains the base64-encoded string of the image

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_string}},
                        {"type": "text", "text": f"{prompt}"}
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        if unload and self.llm is not None:
            del self.llm  # Unload the model
            self.llm = None  # Remove reference to the model
            gc.collect()
            torch.cuda.empty_cache()

        if unload and self.clip is not None:
            del self.clip  # Unload the clip
            self.clip = None  # Remove reference to the clip
            gc.collect()
            torch.cuda.empty_cache()

        return (f"{response['choices'][0]['message']['content']}", )

NODE_CLASS_MAPPINGS = {
    "LLava Loader Simple": LLavaLoader,
    "LLavaSamplerSimple": LLavaSamplerSimple,
    "LlavaClipLoader": LlavaClipLoader,
    "LLavaSamplerAdvanced": LLavaSamplerAdvanced,
    "LLavaOptionalMemoryFreeSimple": LLavaOptionalMemoryFreeSimple,
    "LLavaOptionalMemoryFreeAdvanced": LLavaOptionalMemoryFreeAdvanced,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLava Loader Simple": "LLava Loader Simple",
    "LLavaSamplerSimple": "LLava Sampler Simple",
    "LlavaClipLoader": "Llava Clip Loader",
    "LLavaSamplerAdvanced": "LLava Sampler Advanced",
    "LLavaOptionalMemoryFreeSimple": "LLava Optional Memory Free Simple",
    "LLavaOptionalMemoryFreeAdvanced": "LLava Optional Memory Free Advanced",
}
