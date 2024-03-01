from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
from huggingface_hub import snapshot_download
from torchvision import transforms
import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer 

from io import BytesIO
from torchvision.transforms import ToPILImage

# Define your local directory where you want to save the files
files_for_internlm = Path(__file__).resolve().parent / "files_for_internlm"

# Check if the directory exists, create if it doesn't (optional)
files_for_internlm.mkdir(parents=True, exist_ok=True)


def download_internlm():
    # Ensure the correct behavior based on the existence of the local directory
    print(f"Target directory for download: {files_for_internlm}")
    
    # Call snapshot_download with specified parameters
    path = snapshot_download(
        "internlm/internlm-xcomposer2-vl-7b-4bit",  # Example repo_id
        local_dir=files_for_internlm,
        force_download=False,  # Set to True if you always want to download, regardless of local copy
        local_files_only=False,  # Set to False to allow downloading if not available locally
        local_dir_use_symlinks="auto"  # or set to True/False based on your symlink preference
    )
    print(f"Model path: {path}")
    return path

class Internlm:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)    

    FUNCTION = "internlm_chat"

    CATEGORY = "VLM Nodes/Internlm"

    def internlm_chat(self, image, question):
        from auto_gptq.modeling._base import BaseGPTQForCausalLM
        model_path = download_internlm()
        print(f"Model path: {model_path}")
        class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
            layers_block_name = "model.layers"
            outside_layer_modules = [
                'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
            ]
            inside_layer_modules = [
                ["attention.wqkv.linear"],
                ["attention.wo.linear"],
                ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                ["feed_forward.w2.linear"],
            ]
        model = InternLMXComposer2QForCausalLM.from_quantized(model_path , trust_remote_code=True, device="cuda:0").eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path , trust_remote_code=True)
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        temp_path = files_for_internlm / "temp.jpg"
        pil_image.save(temp_path)        
        text = f'<ImageHere>{question}'
        with torch.cuda.amp.autocast():     
            response, _ = model.chat(tokenizer, query=text, image=str(temp_path), history=[], do_sample=False) 
        return (response, )
        
# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"Internlm": Internlm}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Internlm": "Internlm Node"}
