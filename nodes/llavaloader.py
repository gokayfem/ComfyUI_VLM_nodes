import folder_paths
import os
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
from torchvision.transforms import ToPILImage


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
              "max_ctx": ("INT", {"default": 2048, "min": 300, "max": 100000, "step": 64}),
              "clip": ("CUSTOM", {"default": ""}),
                             }}
                
    
    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llava_checkpoint"

    CATEGORY = "VLM Nodes/LLava/LLavaLoader"
    def load_llava_checkpoint(self, ckpt_name, max_ctx, clip ):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        llm = Llama(model_path = ckpt_path, n_ctx = max_ctx, chat_handler=clip, logits_all=True, verbose=False) 
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

    CATEGORY = "VLM Nodes/LLava/LavaClipLoader"
    def load_clip_checkpoint(self, clip_name):
        clip_path = folder_paths.get_full_path("LLavacheckpoints", clip_name)
        clip = Llava15ChatHandler(clip_model_path = clip_path, verbose=False)        
        return (clip, ) 

class LLavaSampler:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",{"forceInput": True} ),
                "model": ("CUSTOM", {"default": ""}),              
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLava/LLavaSampler"

    def generate_text(self, image, prompt, model):
        

        # Assuming 'image' is a PyTorch tensor of shape [C, H, W]
        # Convert the PyTorch tensor to a PIL image
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))

        # Convert the PIL image to a bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")  # You can change the format if needed

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
            ]
        )

        return (f"{response['choices'][0]['message']['content']}", )

NODE_CLASS_MAPPINGS = {
    "LLava Loader Simple": LLavaLoader,
    "LLavaSampler": LLavaSampler,
    "LlavaClipLoader": LlavaClipLoader
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLava Loader Simple": "LLava Loader Simple",
    "LLavaSampler": "LLava Text Sampler",
    "LlavaClipLoader": "Llava Clip Loader"
}