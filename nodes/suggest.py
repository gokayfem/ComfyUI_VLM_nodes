import folder_paths
import os
from llama_cpp import Llama, LlamaGrammar
from .prompts import system_msg_prompts
from pydantic import BaseModel, Field
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation
import json
from openai import OpenAI
from .prompts import system_msg_prompts
from .prompts import system_msg_simple

supported_LLava_extensions = set(['.gguf'])

try:
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = (folder_paths.folder_names_and_paths["LLavacheckpoints"][0], supported_LLava_extensions)
except:
    # check if LLavacheckpoints exists otherwise create
    if not os.path.isdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints")):
        os.mkdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints"))
        
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = ([os.path.join(folder_paths.models_dir, "LLavacheckpoints")], supported_LLava_extensions)

class Analysis(BaseModel):
    """
    Represents entries about an analysis.
    """
    main_character: str = Field(..., description="Description of the main objects of the analysis")
    artform: list[str]  = Field(..., description="List of Artforms of the analysis")
    photo_type: list[str]  = Field(..., description="List of Types of the photo used in the analysis")
    color_with_objects: list[str]  = Field(..., description="List of Colors with objects of the analysis")
    digital_artform: list[str]  = Field(..., description="List of Digital artforms of the analysis")
    background: list[str]  = Field(..., description="List of Background of the analysis") 
    lighting: list[str]  = Field(..., description="List of Lighting settings of the analysis.")

class PromptGen(BaseModel):
    """
    Represents an entry about a prompt.
    """
    prompt : str = Field(..., description="Prompt for the analysis")

class Suggestion(BaseModel):
    """
    Represents an entry about a suggestion.
    """
    suggestion1 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion2 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion3 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion4 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion5 : str = Field(..., description="new Suggestion based on the inputs")

class PromptGenerateAPI:
    def __init__(self):
    	pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["ChatGPT-3.5", "ChatGPT-4", "DeepSeek"],
                    {
                        "default" : "ChatGPT-3.5"
                    }
                )
                , 
                "chat_type": 
                    ("BOOLEAN", 
                    {
                        "default": True, "label_on": "PromptGenerator", "label_off": "SimpleChat"
                    }
                )
                ,        
                "api_key": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    }
                ),
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

    FUNCTION = "generate_prompt"

    CATEGORY = "VLM Nodes/Prompt Generator"

    def generate_prompt(self, model_name, chat_type, api_key, description, question):

        if chat_type == True:
            system_msg = system_msg_prompts
        elif chat_type == False:
            system_msg = system_msg_simple


        # Define the user message
        if model_name == "DeepSeek":
            model = "deepseek-chat"
            base_url = "https://api.deepseek.com/v1"
        elif model_name == "ChatGPT-3.5":
            model = "gpt-3.5-turbo"
            base_url = None
        elif model_name == "ChatGPT-4":
            model = "gpt-4"
            base_url = None
        user_msg = f"""
        Description: {description}
	Optional Question: {question}

        Output: 
        """
        
        client = OpenAI(api_key = api_key, base_url=base_url)

        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        )

        prompt = completion.choices[0].message.content       
        return (prompt,)
    
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
        llm = Llama(model_path = ckpt_path, chat_format="chatml", offload_kqv=True, f16_kv=True, use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads,) 
        return (llm, ) 
    
class LLMPromptGenerator:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def generate_text_advanced(self,prompt, model, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty):
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
    
class LLMSampler:        
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
            {"role": "system", "content": system_msg},
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

# Example output model
    
class KeywordExtraction:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),                          
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "keyword_extract"
    CATEGORY = "VLM Nodes/LLM"
    
    def keyword_extract(self, prompt, model, temperature):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Analysis])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)


        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt="You are an advanced AI, tasked to create JSON database entries for analysis.\n\n\n" + documentation)

        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar)
        return (response, )
    
class LLavaPromptGenerator:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),                           
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompts"
    CATEGORY = "VLM Nodes/LLM"
    
    def generate_prompts(self, prompt, model, temperature):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([PromptGen])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt="You are an advanced AI, tasked to create JSON database entries for creative long prompts for image generation. \n\n\n" + documentation)
        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar, max_tokens=512, repeat_penalty=1.1)
        return (f"{response}", )
    

class Suggester:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "randomize": ("BOOLEAN", {"default": True, "label_on": "Consistent", "label_off": "Random"}),                           
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_suggestions"
    CATEGORY = "VLM Nodes/LLM"
    
    def generate_suggestions(self, prompt, model, temperature, randomize):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Suggestion])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)
        if randomize: 
            system_msg_suggester = f"You are an advanced AI, tasked to create JSON database entries for generating extremely similar to the prompt. \n\n\n" + documentation
        else:
            #you should suggest variation from prompts
            prompt = "Generate a prompt like: <A random character> <random action> <random place> <random object> <random color>"  
            system_msg_suggester = f"You are an advanced AI, tasked to create JSON database entries for suggesting completely different prompts.. \n\n\n" + documentation 
        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt=system_msg_suggester)
        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar, max_tokens=512, repeat_penalty=1.1)
    
        return (response, )


NODE_CLASS_MAPPINGS = {
    "LLMLoader": LLMLoader,
    "LLMSampler": LLMSampler,
    "LLMPromptGenerator": LLMPromptGenerator,
    "KeywordExtraction": KeywordExtraction,
    "LLavaPromptGenerator": LLavaPromptGenerator,
    "Suggester": Suggester,
    "PromptGenerateAPI": PromptGenerateAPI
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMLoader": "LLMLoader",
    "LLMSampler": "LLMSampler",
    "LLMPromptGenerator": "LLM PromptGenerator",
    "KeywordExtraction": "Get Keywords",
    "LLavaPromptGenerator": "LLava PromptGenerator",
    "Suggester": "Suggester",
    "PromptGenerateAPI": "API PromptGenerator"
}