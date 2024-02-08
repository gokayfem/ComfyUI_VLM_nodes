import folder_paths
import os
from llama_cpp import Llama
from .prompts import system_msg_prompts


supported_LLava_extensions = set(['.gguf'])

try:
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = (folder_paths.folder_names_and_paths["LLavacheckpoints"][0], supported_LLava_extensions)
except:
    # check if LLavacheckpoints exists otherwise create
    if not os.path.isdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints")):
        os.mkdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints"))
        
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = ([os.path.join(folder_paths.models_dir, "LLavacheckpoints")], supported_LLava_extensions)
    
class LLMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),   
              "max_ctx": ("INT", {"default": 2048, "min": 300, "max": 100000, "step": 64}),
              "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
              "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                            }
                }
                
    
    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llm_checkpoint"

    CATEGORY = "VLM Nodes/LLM"
    def load_llm_checkpoint(self, ckpt_name, max_ctx, gpu_layers, n_threads):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        llm = Llama(model_path = ckpt_path, chat_format="chatml", n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, verbose=False, echo=False) 
        return (llm, ) 
    
class LLMSamplerAdvanced:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_msg": ("STRING",{"default" : "You are an assistant who perfectly describes images."}),
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}), 
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),                             
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(self, system_msg, prompt, model, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty):
        llm = model
        response = llm.create_chat_completion(messages=[
            {"role": "system", "content": system_msg_prompts},
            {"role": "user", "content": prompt + " Assistant:"},
        ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            
        )
        return (f"{response['choices'][0]['message']['content']}", )

NODE_CLASS_MAPPINGS = {
    "LLMLoader": LLMLoader,
    "LLMSamplerAdvanced": LLMSamplerAdvanced,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMLoader": "LLMLoader",
    "LLMSamplerAdvanced": "LLMSamplerAdvanced",
}